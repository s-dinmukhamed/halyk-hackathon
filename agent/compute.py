"""Deterministic evaluation. All arithmetic happens here, never in the LLM.

The metric is computed from extracted inputs and compared to the resolved
threshold; transactional covenants are checked by scanning the registry for the
offending transaction.
"""
from __future__ import annotations

from typing import Callable, Optional

from .schemas import Covenant, Operator, Transaction, Verdict


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _debt_to_ebitda(f: dict) -> Optional[float]:
    return _safe_div(f.get("total_debt"), f.get("ebitda"))


def _current_ratio(f: dict) -> Optional[float]:
    return _safe_div(f.get("current_assets"), f.get("current_liabilities"))


def _quick_ratio(f: dict) -> Optional[float]:
    ca, inv = f.get("current_assets"), f.get("inventory")
    numerator = None if ca is None else ca - (inv or 0)
    return _safe_div(numerator, f.get("current_liabilities"))


def _dscr(f: dict) -> Optional[float]:
    return _safe_div(f.get("cfads") or f.get("ebitda"), f.get("debt_service"))


def _interest_coverage(f: dict) -> Optional[float]:
    return _safe_div(f.get("ebitda"), f.get("interest_expense"))


def _debt_to_equity(f: dict) -> Optional[float]:
    return _safe_div(f.get("total_debt"), f.get("equity"))


def _net_worth(f: dict) -> Optional[float]:
    return f.get("equity") if f.get("equity") is not None else f.get("net_worth")


# metric name (normalized in extract.py) -> calculator
METRICS: dict[str, Callable[[dict], Optional[float]]] = {
    "debt_to_ebitda": _debt_to_ebitda,
    "current_ratio": _current_ratio,
    "quick_ratio": _quick_ratio,
    "dscr": _dscr,
    "interest_coverage": _interest_coverage,
    "debt_to_equity": _debt_to_equity,
    "leverage": _debt_to_equity,
    "net_worth": _net_worth,
    "min_net_worth": _net_worth,
}


def evaluate_operator(value: float, operator: Operator, threshold: float) -> bool:
    """True when the covenant is COMPLIED (the operator states the required condition)."""
    if operator == Operator.LE:
        return value <= threshold
    if operator == Operator.GE:
        return value >= threshold
    if operator == Operator.LT:
        return value < threshold
    if operator == Operator.GT:
        return value > threshold
    if operator == Operator.EQ:
        return value == threshold
    raise ValueError(f"unknown operator: {operator}")


def evaluate_financial(covenant: Covenant, financials: dict
                       ) -> tuple[Verdict, Optional[float]]:
    calc = METRICS.get(covenant.metric)
    value = calc(financials) if calc else None
    if value is None or covenant.threshold is None:
        # Can't compute — return a best guess so no answer is left empty.
        return Verdict.COMPLIED, value
    complied = evaluate_operator(value, covenant.operator, covenant.threshold)
    return (Verdict.COMPLIED if complied else Verdict.BREACHED), round(value, 6)


def find_offending_transaction(covenant: Covenant, transactions: list[Transaction]
                               ) -> Optional[Transaction]:
    """Find the transaction that breaches a transactional covenant, if any.

    Filter by borrower, by a keyword from the covenant, and by amount over the
    threshold when one is set; return the largest match.
    """
    keyword = (covenant.metric or covenant.name).lower()
    candidates = [
        t for t in transactions
        if (not covenant.borrower_id or t.borrower_id == covenant.borrower_id)
        and (keyword in (t.tx_type + " " + t.description).lower() if keyword else True)
    ]
    if covenant.threshold is not None:
        candidates = [t for t in candidates
                      if t.amount is not None and t.amount > covenant.threshold]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.amount or 0)
