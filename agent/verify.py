"""Deterministic self-checks before emitting.

Catches the common failure modes: a verdict that disagrees with the arithmetic,
a breach with no evidence transaction, a field the scorer expects left null.
"""
from __future__ import annotations

from .compute import evaluate_operator
from .schemas import CovenantAnswer, Verdict


def check_answer(ans: CovenantAnswer) -> list[str]:
    """Warnings for one answer (empty list == clean)."""
    warnings: list[str] = []

    if ans.value is not None and ans.threshold is not None and ans.operator is not None:
        complied = evaluate_operator(ans.value, ans.operator, ans.threshold)
        expected = Verdict.COMPLIED if complied else Verdict.BREACHED
        if expected != ans.verdict:
            warnings.append(
                f"{ans.covenant_id}: verdict {ans.verdict} != computed {expected} "
                f"(value={ans.value} {ans.operator} {ans.threshold})"
            )

    if ans.verdict == Verdict.BREACHED and ans.evidence_tx_id is None:
        warnings.append(f"{ans.covenant_id}: breach without evidence_tx_id (may lose a component)")

    return warnings


def verify_all(answers: list[CovenantAnswer]) -> tuple[list[CovenantAnswer], list[str]]:
    """Run the checks and align each verdict to the computed value where they disagree."""
    all_warnings: list[str] = []
    for ans in answers:
        all_warnings.extend(check_answer(ans))
        if ans.value is not None and ans.threshold is not None and ans.operator is not None:
            complied = evaluate_operator(ans.value, ans.operator, ans.threshold)
            ans.verdict = Verdict.COMPLIED if complied else Verdict.BREACHED
    return answers, all_warnings
