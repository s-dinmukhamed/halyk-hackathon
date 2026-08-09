"""Fail fast on a malformed submission.json before uploading it.

The scorer wants the exact template shape: top-level team/contact_email/model, then
answers[scenario][clause] = {status, actual, evidence_txn_id}. A file that parses
but drifts from that — a null status, a non-numeric actual, a renamed/missing cell —
loses points silently. This turns those into loud errors.

    python -m agent.validate submission.json --template submission_template.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .emit import ACTUAL, EVIDENCE, STATUS

VALID_STATUS = {"COMPLIANT", "BREACH"}
TOP_LEVEL = ("team", "contact_email", "model", "answers")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)    # block upload
    warnings: list[str] = field(default_factory=list)   # likely point loss
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = ["=" * 60, "SUBMISSION VALIDATION", "=" * 60]
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        lines.append("-" * 60)
        if self.errors:
            lines.append(f"❌ {len(self.errors)} ERROR(S) — DO NOT UPLOAD:")
            lines += [f"   • {e}" for e in self.errors]
        if self.warnings:
            lines.append(f"⚠️  {len(self.warnings)} WARNING(S):")
            lines += [f"   • {w}" for w in self.warnings]
        if self.ok and not self.warnings:
            lines.append("✅ CLEAN — safe to upload.")
        elif self.ok:
            lines.append("✅ Structurally valid (review warnings above).")
        lines.append("=" * 60)
        return "\n".join(lines)


def validate_payload(payload, template=None) -> Report:
    rep = Report()
    if not isinstance(payload, dict):
        rep.errors.append("top level is not a JSON object")
        return rep

    for key in ("team", "contact_email", "model"):
        if not payload.get(key):
            rep.warnings.append(f"top-level '{key}' is empty")

    answers = payload.get("answers")
    if not isinstance(answers, dict) or not answers:
        rep.errors.append("'answers' missing or empty")
        return rep

    cells = status_counts = 0
    breaches = breaches_with_evidence = null_actual = 0
    for sid, clauses in answers.items():
        if not isinstance(clauses, dict):
            rep.errors.append(f"answers[{sid}] is not an object")
            continue
        for clause, cell in clauses.items():
            cells += 1
            loc = f"{sid}.{clause}"
            if not isinstance(cell, dict):
                rep.errors.append(f"{loc}: cell is not an object")
                continue
            for k in (STATUS, ACTUAL, EVIDENCE):
                if k not in cell:
                    rep.errors.append(f"{loc}: missing key '{k}'")

            status = cell.get(STATUS)
            if status not in VALID_STATUS:
                rep.errors.append(f"{loc}: status {status!r} not in {sorted(VALID_STATUS)}")
            else:
                status_counts += 1
                if status == "BREACH":
                    breaches += 1
                    if cell.get(EVIDENCE):
                        breaches_with_evidence += 1

            actual = cell.get(ACTUAL)
            if not isinstance(actual, (int, float)):
                rep.errors.append(f"{loc}: actual {actual!r} is not a number")
                null_actual += 1
            elif actual < 0:
                rep.warnings.append(f"{loc}: actual {actual} is negative (must be positive)")

    rep.stats.update({
        "scenarios": len(answers),
        "cells": cells,
        "valid_status": status_counts,
        "breaches": breaches,
        "breaches_with_evidence": breaches_with_evidence,
        "non_numeric_actual": null_actual,
    })

    if template is not None:
        _check_template(rep, answers, template)
    return rep


def _check_template(rep: Report, answers: dict, template: dict) -> None:
    """Every template cell must be present, with no added/renamed keys."""
    t_answers = template.get("answers", {})
    for sid, clauses in t_answers.items():
        if sid not in answers:
            rep.errors.append(f"missing scenario {sid}")
            continue
        for clause in clauses:
            if clause not in answers[sid]:
                rep.errors.append(f"missing cell {sid}.{clause}")
    extra = set(answers) - set(t_answers)
    if extra:
        rep.warnings.append(f"scenarios not in template: {sorted(extra)}")


def validate_file(path: str | Path, template: str | Path | None = None) -> Report:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Report(errors=[f"file not found: {path}"])
    except json.JSONDecodeError as e:
        return Report(errors=[f"not valid JSON: {e}"])

    tmpl = None
    if template:
        try:
            tmpl = json.loads(Path(template).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return Report(errors=[f"could not read template {template}: {e}"])
    return validate_payload(payload, tmpl)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate a Halyk submission.json")
    ap.add_argument("path", type=Path, help="submission.json to validate")
    ap.add_argument("--template", type=Path, default=None,
                    help="submission_template.json to check cells against")
    args = ap.parse_args(argv)

    rep = validate_file(args.path, args.template)
    print(rep.render(), file=sys.stderr)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
