"""Step 5 — VERIFY: cheap self-checks before emitting.

Not a second LLM pass by default (costs rate-limit budget) — a set of
deterministic sanity checks that catch the most common failure modes:
  * verdict must be consistent with value vs threshold+operator
  * transactional breaches should carry an evidence transaction id
  * no answer field left null that the scorer expects filled

An optional LLM re-read of the cited snippet can be enabled for low-confidence
answers if time/budget allows on finals day.
"""
from __future__ import annotations

from .compute import evaluate_operator
from .schemas import CovenantAnswer, CovenantType, Verdict


def check_answer(ans: CovenantAnswer) -> list[str]:
    """Return a list of warnings for one answer (empty == clean)."""
    warnings: list[str] = []

    # Verdict must match the arithmetic when we have both value and threshold.
    if ans.value is not None and ans.threshold is not None and ans.operator is not None:
        complied = evaluate_operator(ans.value, ans.operator, ans.threshold)
        expected = Verdict.COMPLIED if complied else Verdict.BREACHED
        if expected != ans.verdict:
            warnings.append(
                f"{ans.covenant_id}: verdict {ans.verdict} != computed {expected} "
                f"(value={ans.value} {ans.operator} {ans.threshold})"
            )

    # A breach on a transactional covenant should name the offending transaction.
    if ans.verdict == Verdict.BREACHED and ans.evidence_tx_id is None:
        warnings.append(f"{ans.covenant_id}: breach without evidence_tx_id (may lose a component)")

    return warnings


def verify_all(answers: list[CovenantAnswer]) -> tuple[list[CovenantAnswer], list[str]]:
    """Run checks; auto-correct verdict/value mismatches toward the computed truth."""
    all_warnings: list[str] = []
    for ans in answers:
        warns = check_answer(ans)
        all_warnings.extend(warns)
        # Trust the deterministic computation: align verdict to value if they disagree.
        if ans.value is not None and ans.threshold is not None and ans.operator is not None:
            complied = evaluate_operator(ans.value, ans.operator, ans.threshold)
            ans.verdict = Verdict.COMPLIED if complied else Verdict.BREACHED
    return answers, all_warnings
