#!/usr/bin/env python3
"""Covenant-compliance agent — single entry point.

    python run.py --data ./data/public --out submission.json

One command, no manual steps: read the dataset, write submission.json. The rules
require answers to come from this agent, so on finals day we just point it at the
private dataset. Pipeline: ingest -> extract -> resolve -> compute -> verify -> emit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from agent import compute, emit, extract, resolve, verify
from agent.ingest import load_dataset
from agent.parallel import pflatmap
from agent.schemas import CovenantAnswer, CovenantType, Transaction, Verdict


def build_transactions(tabular_paths) -> list[Transaction]:
    """Load transaction-registry sidecars into Transaction objects.

    TODO: map the real column names once we see the registry schema.
    """
    from agent.ingest import load_tabular

    txs: list[Transaction] = []
    for path in tabular_paths:
        if "registr" not in path.name.lower() and "transact" not in path.name.lower():
            continue
        df = load_tabular(path)
        for _, row in df.iterrows():
            txs.append(Transaction(
                tx_id=str(row.get("tx_id", row.get("id", ""))),
                borrower_id=str(row.get("borrower_id", "")),
                date=str(row.get("date", "")) or None,
                amount=_num(row.get("amount")),
                tx_type=str(row.get("type", "")),
                description=str(row.get("description", "")),
            ))
    return txs


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run(data_dir: Path, out_path: Path) -> Path:
    t0 = time.time()

    docs, tabular = load_dataset(data_dir)
    print(f"[ingest] {len(docs)} PDFs, {len(tabular)} tabular files", file=sys.stderr)

    # One LLM call per doc, run concurrently — the main wall-clock cost.
    covenant_docs = [d for d in docs if d.kind in ("loan_agreement", "amendment", "unknown")]
    covenants = pflatmap(extract.extract_covenants, covenant_docs)
    financials = extract.extract_financials(
        [d for d in docs if d.kind == "financials"]
    )
    transactions = build_transactions(tabular)
    print(f"[extract] {len(covenants)} covenants, {len(transactions)} transactions",
          file=sys.stderr)

    # Drop covenant versions superseded by a later amendment.
    covenants = [c for c in resolve.resolve_covenants(covenants) if not c.superseded]
    print(f"[resolve] {len(covenants)} effective covenants", file=sys.stderr)

    # Deterministic verdict + number + evidence transaction per covenant.
    answers: list[CovenantAnswer] = []
    for c in covenants:
        if c.type == CovenantType.FINANCIAL:
            verdict, value = compute.evaluate_financial(c, financials.get(c.borrower_id, {}))
            answers.append(CovenantAnswer(
                borrower_id=c.borrower_id, covenant_id=c.covenant_id,
                verdict=verdict, value=value,
                threshold=c.threshold, operator=c.operator,
                evidence=c.source,
            ))
        else:
            offender = compute.find_offending_transaction(c, transactions)
            answers.append(CovenantAnswer(
                borrower_id=c.borrower_id, covenant_id=c.covenant_id,
                verdict=Verdict.BREACHED if offender else Verdict.COMPLIED,
                value=(offender.amount if offender else None),
                evidence_tx_id=(offender.tx_id if offender else None),
                threshold=c.threshold, operator=c.operator,
                evidence=c.source,
            ))

    answers, warnings = verify.verify_all(answers)
    for w in warnings:
        print(f"[verify] WARN {w}", file=sys.stderr)

    out = emit.write_submission(answers, out_path)
    print(f"[done] {len(answers)} answers -> {out} in {time.time() - t0:.1f}s",
          file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Halyk covenant-compliance agent")
    ap.add_argument("--data", required=True, type=Path, help="dataset directory")
    ap.add_argument("--out", default=Path("submission.json"), type=Path,
                    help="output submission JSON")
    args = ap.parse_args()

    if not args.data.exists():
        ap.error(f"data dir not found: {args.data}")
    run(args.data, args.out)


if __name__ == "__main__":
    main()
