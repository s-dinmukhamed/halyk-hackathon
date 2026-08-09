#!/usr/bin/env python3
"""Covenant-compliance agent — single entry point.

    python run.py --data ./agentic-bank-public \
                  --template ./agentic-bank-public/submission_template.json \
                  --out submission.json

Scenario-centric pipeline: map accounts to scenarios, route documents, extract each
scenario's covenant specs + related parties, classify its ledger transactions, then
compute verdicts/values/evidence deterministically and fill the template.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from agent import compute, decoy, emit, extract
from agent.config import settings
from agent.ingest import (
    AUDIT, CREDIT_AGREEMENT, KYC, Document,
    borrower_name, load_documents, load_ledger, load_scenario_index,
)
from agent.parallel import pmap
from agent.schemas import CovenantAnswer, Transaction, Verdict


def _to_usd(txs: list[Transaction]) -> None:
    """Normalize every amount to USD in place (covenant limits are in USD)."""
    for t in txs:
        if t.currency.upper() == "EUR" and t.amount is not None:
            t.amount = round(t.amount * settings.eur_usd_rate, 2)
            t.currency = "USD"


def _assign_scenarios(docs: list[Document], acc2scn: dict[str, str],
                      scenarios: set[str]) -> None:
    """Tag each document with its scenario, by account id first, then borrower name.

    Most documents cite the borrower's account id; some (e.g. certain KYC dossiers)
    name the borrower only by company name, so we resolve those against the names
    taken from the agreements.
    """
    for d in docs:
        hits = [s for a in d.account_ids if (s := acc2scn.get(a)) in scenarios]
        if hits:
            d.scenario_id = Counter(hits).most_common(1)[0][0]

    # Borrower name -> scenario, from the agreements that already got a scenario.
    name2scn: dict[str, str] = {}
    for d in docs:
        if d.kind == CREDIT_AGREEMENT and d.scenario_id:
            name = borrower_name(d.text)
            if name:
                name2scn[name] = d.scenario_id

    for d in docs:
        if d.scenario_id:
            continue
        for name, sid in name2scn.items():
            if name in d.text:
                d.scenario_id = sid
                break


def _pick_agreement(docs: list[Document]) -> Document | None:
    """The active (non-superseded) credit agreement; longest text breaks ties."""
    active = [d for d in docs if d.kind == CREDIT_AGREEMENT and not d.superseded]
    pool = active or [d for d in docs if d.kind == CREDIT_AGREEMENT]
    return max(pool, key=lambda d: len(d.text)) if pool else None


def _pick(docs: list[Document], kind: str, prefer: str = "") -> Document | None:
    """Pick an in-force document of `kind`, preferring one whose text has `prefer`."""
    pool = [d for d in docs if d.kind == kind and not d.superseded] \
        or [d for d in docs if d.kind == kind]
    if not pool:
        return None
    if prefer:
        preferred = [d for d in pool if prefer in d.text]
        if preferred:
            return preferred[0]
    return pool[0]


def _default_answer(sid: str, clause: str) -> CovenantAnswer:
    return CovenantAnswer(scenario_id=sid, clause=clause, verdict=Verdict.COMPLIANT, actual=None)


def process_scenario(sid: str, docs: list[Document], txs: list[Transaction],
                     clauses: list[str], fake_brands: set[str] | None = None) -> list[CovenantAnswer]:
    """Run extract -> classify -> compute for one scenario, returning its cells."""
    fake_brands = fake_brands or set()
    agreement = _pick_agreement(docs)
    kyc = _pick(docs, KYC, prefer="связанными сторонами")
    audit = _pick(docs, AUDIT, prefer="ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ")

    covenants = extract.extract_covenant_specs(agreement, sid, clauses) if agreement else []
    cov_by_clause = {c.clause: c for c in covenants}
    related = extract.extract_related_parties(kyc) if kyc else []
    note = extract.reclassification_note(audit)

    labels = extract.classify_transactions(sid, txs, related, note) if txs else {}
    for t in txs:
        lbl = labels.get(t.tx_id)
        if lbl:
            t.category = lbl["category"]
            t.related_party = lbl["related_party"]
        # Fabricated decoy lines never enter any covenant metric.
        if decoy.is_decoy(t.counterparty, fake_brands):
            t.category = "other"
            t.related_party = False

    # Apply in-force auditor reclassifications: move the named transaction into the
    # category the auditor assigned for covenant purposes, and remember it as the
    # decisive (evidence) line for the covenant its metric drives.
    reclassifications = extract.extract_reclassifications([d for d in docs if d.kind == AUDIT])
    reclassified: dict[str, str] = {}          # tx_id -> new category
    for r in reclassifications:
        for t in txs:
            if (t.amount and abs(abs(t.amount) - r["amount"]) < 0.01
                    and not decoy.is_decoy(t.counterparty, fake_brands)):
                t.category = r["new_category"]
                reclassified[t.tx_id] = r["new_category"]
                break

    answers: list[CovenantAnswer] = []
    for clause in clauses:
        cov = cov_by_clause.get(clause)
        if cov is None:
            answers.append(_default_answer(sid, clause))
            continue
        verdict, actual = compute.evaluate(cov, txs)
        evidence = compute.find_evidence(cov, txs, verdict)
        # A reclassified line is the auditor-defined decisive event for the covenant
        # whose formula references that category — prefer it as evidence on a breach.
        if verdict == Verdict.BREACH:
            for txid, cat in reclassified.items():
                if cat and cat in (cov.value_expr or ""):
                    evidence = txid
                    break
        answers.append(CovenantAnswer(
            scenario_id=sid, clause=clause, verdict=verdict, actual=actual,
            evidence_tx_id=evidence, threshold=cov.threshold, operator=cov.operator,
            value_expr=cov.value_expr,
        ))
    return answers


def run(data_dir: Path, template_path: Path, out_path: Path) -> Path:
    t0 = time.time()
    data_dir = Path(data_dir)
    ledger_csv = data_dir / "master_ledger_2025.csv"

    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    scenarios: list[str] = list(template.get("answers", {}).keys())

    txs = load_ledger(ledger_csv)
    _to_usd(txs)
    acc2scn = load_scenario_index(ledger_csv)
    docs = load_documents(data_dir / "documents")
    _assign_scenarios(docs, acc2scn, set(scenarios))
    print(f"[ingest] {len(docs)} PDFs, {len(txs)} txns, {len(scenarios)} scenarios",
          file=sys.stderr)

    txs_by_scn: dict[str, list[Transaction]] = {s: [] for s in scenarios}
    for t in txs:
        if t.scenario_id in txs_by_scn:
            txs_by_scn[t.scenario_id].append(t)
    docs_by_scn: dict[str, list[Document]] = {s: [] for s in scenarios}
    for d in docs:
        if d.scenario_id in docs_by_scn:
            docs_by_scn[d.scenario_id].append(d)

    clauses_by_scn = {s: list(template["answers"][s].keys()) for s in scenarios}
    fake_brands = decoy.build_fake_brands([t.counterparty for t in txs])
    print(f"[decoy] {len(fake_brands)} fabricated brand names identified", file=sys.stderr)

    def work(sid: str) -> list[CovenantAnswer]:
        return process_scenario(sid, docs_by_scn[sid], txs_by_scn[sid],
                                clauses_by_scn[sid], fake_brands)

    results = pmap(work, scenarios)
    answers = [a for sub in results for a in sub]
    print(f"[compute] {len(answers)} cells across {len(scenarios)} scenarios", file=sys.stderr)

    out = emit.write_submission(answers, template_path, out_path)
    print(f"[done] -> {out} in {time.time() - t0:.1f}s", file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Halyk covenant-compliance agent")
    ap.add_argument("--data", required=True, type=Path, help="dataset directory")
    ap.add_argument("--template", required=True, type=Path, help="submission_template.json")
    ap.add_argument("--out", default=Path("submission.json"), type=Path, help="output JSON")
    args = ap.parse_args()

    if not args.data.exists():
        ap.error(f"data dir not found: {args.data}")
    if not args.template.exists():
        ap.error(f"template not found: {args.template}")
    run(args.data, args.template, args.out)


if __name__ == "__main__":
    main()
