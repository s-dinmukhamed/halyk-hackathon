"""Fail fast on a malformed submission.json before uploading it.

A file that parses but has the wrong shape — missing components, nulls where the
scorer wants a value, an unrecognised verdict label — loses points silently.
This turns those into loud errors. Two checks: structural (keys, labels, all
three components filled) and, when the organizers ship a sample, a key diff
against their template.

    python -m agent.validate submission.json
    python -m agent.validate submission.json --template sample_submission.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .emit import FIELD_EVIDENCE_TX, FIELD_VALUE, FIELD_VERDICT, VERDICT_LABELS

# The answers list may sit at the top level or under one of these wrapper keys.
ANSWERS_KEYS = ("answers", "results", "submissions", "data")
VALID_VERDICTS = {v for v in VERDICT_LABELS.values()}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)   # block upload
    warnings: list[str] = field(default_factory=list)  # likely point loss
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
            lines.append(f"⚠️  {len(self.warnings)} WARNING(S) — likely lost points:")
            lines += [f"   • {w}" for w in self.warnings]
        if self.ok and not self.warnings:
            lines.append("✅ CLEAN — safe to upload.")
        elif self.ok:
            lines.append("✅ Structurally valid (review warnings above).")
        lines.append("=" * 60)
        return "\n".join(lines)


def _find_answers(payload) -> tuple[list, str | None]:
    """Locate the answers list regardless of wrapper key."""
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        for key in ANSWERS_KEYS:
            if isinstance(payload.get(key), list):
                return payload[key], key
    return [], None


def validate_payload(payload) -> Report:
    """Validate an already-parsed submission object."""
    rep = Report()
    answers, wrapper = _find_answers(payload)

    if not answers:
        rep.errors.append("no answers found (empty list or unexpected top-level shape)")
        return rep

    rep.stats["answers"] = len(answers)
    rep.stats["wrapper_key"] = wrapper or "(top-level list)"

    seen: set[tuple] = set()
    borrowers: set = set()
    verdict_counts: dict[str, int] = {}
    missing_value = missing_tx_on_breach = 0

    for i, a in enumerate(answers):
        loc = f"answer[{i}]"
        if not isinstance(a, dict):
            rep.errors.append(f"{loc}: not an object")
            continue

        bid = a.get("borrower_id")
        cid = a.get("covenant_id")
        if not bid:
            rep.errors.append(f"{loc}: missing borrower_id")
        if not cid:
            rep.errors.append(f"{loc}: missing covenant_id")
        if bid:
            borrowers.add(bid)

        dup = (bid, cid)
        if dup in seen:
            rep.warnings.append(f"{loc}: duplicate (borrower={bid}, covenant={cid})")
        seen.add(dup)

        # Component 1 — verdict.
        verdict = a.get(FIELD_VERDICT)
        if verdict is None:
            rep.errors.append(f"{loc}: missing '{FIELD_VERDICT}'")
        elif verdict not in VALID_VERDICTS:
            rep.warnings.append(
                f"{loc}: verdict '{verdict}' not in {sorted(VALID_VERDICTS)} "
                "(scorer may not recognise it)"
            )
        else:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        # Component 2 — number. Null means a guaranteed lost component.
        if FIELD_VALUE not in a:
            rep.errors.append(f"{loc}: missing '{FIELD_VALUE}' key")
        elif a.get(FIELD_VALUE) is None:
            missing_value += 1

        # Component 3 — evidence transaction (only meaningful for breaches).
        if FIELD_EVIDENCE_TX not in a:
            rep.errors.append(f"{loc}: missing '{FIELD_EVIDENCE_TX}' key")
        if str(verdict).lower() == "breached" and not a.get(FIELD_EVIDENCE_TX):
            missing_tx_on_breach += 1

    rep.stats["distinct_borrowers"] = len(borrowers)
    rep.stats["verdicts"] = verdict_counts or "—"
    if missing_value:
        rep.warnings.append(
            f"{missing_value} answer(s) have null {FIELD_VALUE} → that component scores 0"
        )
    if missing_tx_on_breach:
        rep.warnings.append(
            f"{missing_tx_on_breach} breach(es) lack an evidence transaction id "
            "→ evidence component scores 0"
        )
    return rep


def validate_file(path: str | Path, template: str | Path | None = None) -> Report:
    """Load submission.json (and optional template) and validate."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Report(errors=[f"file not found: {path}"])
    except json.JSONDecodeError as e:
        return Report(errors=[f"not valid JSON: {e}"])

    rep = validate_payload(payload)
    if template:
        rep = _check_against_template(rep, payload, Path(template))
    return rep


def _check_against_template(rep: Report, payload, template_path: Path) -> Report:
    """Diff our output's answer keys against the organizers' sample template."""
    try:
        tmpl = json.loads(template_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        rep.warnings.append(f"could not read template {template_path}: {e}")
        return rep

    ours, _ = _find_answers(payload)
    theirs, _ = _find_answers(tmpl)
    if not ours or not theirs:
        rep.warnings.append("template check skipped (empty answers on one side)")
        return rep

    our_keys = set(ours[0]) if isinstance(ours[0], dict) else set()
    their_keys = set(theirs[0]) if isinstance(theirs[0], dict) else set()
    missing = their_keys - our_keys
    extra = our_keys - their_keys
    if missing:
        rep.errors.append(f"template requires keys we don't emit: {sorted(missing)}")
    if extra:
        rep.warnings.append(f"we emit keys not in template (ignored by scorer?): {sorted(extra)}")
    rep.stats["template_keys"] = sorted(their_keys)
    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate a Halyk submission.json")
    ap.add_argument("path", type=Path, help="submission.json to validate")
    ap.add_argument("--template", type=Path, default=None,
                    help="organizers' sample submission to diff keys against")
    args = ap.parse_args(argv)

    rep = validate_file(args.path, args.template)
    print(rep.render(), file=sys.stderr)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
