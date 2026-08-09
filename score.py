#!/usr/bin/env python3
"""Local scoring harness — measure a submission against ground_truth.json.

Implements the official per-cell rule so we can iterate offline:
  status   0.50  exact match; if wrong the whole cell scores 0
  actual   0.30  0.30 * max(0, 1 - e/0.05),  e = |ours - key| / |key|
  evidence 0.20  exact txn match when the key is non-null; when the key is null,
                 this 0.20 decays on the same scale as actual.

    python score.py submission.json ground_truth.json [--verbose]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _actual_fraction(ours, key) -> float:
    """Fraction in [0,1] of the decaying-scale credit for `actual`."""
    if not isinstance(ours, (int, float)):
        return 0.0
    if key == 0:
        return 1.0 if ours == 0 else 0.0
    e = abs(ours - key) / abs(key)
    return max(0.0, 1.0 - e / 0.05)


def score_cell(ours: dict | None, key: dict) -> tuple[float, str]:
    """Return (score in [0,1], one-line explanation)."""
    if not isinstance(ours, dict):
        return 0.0, "missing cell"

    key_status = key.get("status")
    if ours.get("status") != key_status:
        return 0.0, f"status {ours.get('status')!r} != {key_status!r} → cell 0"

    score = 0.50
    frac = _actual_fraction(ours.get("actual"), key.get("actual"))
    score += 0.30 * frac

    key_ev = key.get("evidence_txn_id")
    if key_ev is not None:
        ev_ok = ours.get("evidence_txn_id") == key_ev
        score += 0.20 if ev_ok else 0.0
        ev_note = "ev✓" if ev_ok else f"ev✗(want {key_ev})"
    else:
        score += 0.20 * frac
        ev_note = "ev=null"

    return score, (f"status✓ actual {ours.get('actual')} vs {key.get('actual')} "
                   f"({frac*100:.0f}%) {ev_note}")


def score(submission: dict, truth: dict, verbose: bool = False) -> float:
    ours = submission.get("answers", {})
    key_scn = truth.get("scenarios", {})

    total = 0.0
    n = 0
    rows: list[tuple[str, float, str]] = []
    status_hits = 0
    for sid in sorted(key_scn):
        covs = key_scn[sid].get("covenants", {})
        for clause in sorted(covs):
            n += 1
            cell_key = covs[clause]
            cell_ours = ours.get(sid, {}).get(clause)
            s, note = score_cell(cell_ours, cell_key)
            total += s
            if isinstance(cell_ours, dict) and cell_ours.get("status") == cell_key.get("status"):
                status_hits += 1
            rows.append((f"{sid}.{clause}", s, note))

    print("=" * 74)
    print(f"SCORE  {total:.3f} / {n}   ({total / n * 100:.1f}%)   "
          f"status correct: {status_hits}/{n}")
    print("=" * 74)
    if verbose:
        for name, s, note in rows:
            flag = "  " if s >= 0.999 else ("~ " if s >= 0.5 else "✗ ")
            print(f"{flag}{name:8} {s:4.2f}  {note}")
        print("-" * 74)
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a submission against ground truth")
    ap.add_argument("submission", type=Path)
    ap.add_argument("truth", type=Path)
    ap.add_argument("--verbose", "-v", action="store_true", help="per-cell breakdown")
    args = ap.parse_args(argv)

    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    score(submission, truth, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
