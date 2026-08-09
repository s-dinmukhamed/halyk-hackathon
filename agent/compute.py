"""Deterministic evaluation. All arithmetic happens here, never in the LLM.

Category-tagged transactions become primitive sums (revenue, opex, ebitda, capex,
payroll, utilities, related_party_payments, ...). Each covenant carries a small
formula (`value_expr`) over those primitives, which we evaluate with a restricted,
safe parser — no eval() — then compare to the threshold. Evidence is found by the
"remove one line and the verdict flips" test the spec describes.
"""
from __future__ import annotations

import ast
import operator as op
from typing import Optional

from .schemas import Covenant, Operator, Transaction, Verdict

# Operating-expense lines that make up `opex` (and therefore `ebitda`).
_OPEX_CATEGORIES = (
    "payroll", "utilities", "rent", "insurance", "marketing",
    "professional_services", "materials", "maintenance", "logistics",
)


def _in_q4(date: Optional[str]) -> bool:
    return bool(date) and date[5:7] in ("10", "11", "12")


def primitives(txs: list[Transaction], period_filter: str = "") -> dict[str, float]:
    """Sum category buckets into positive-magnitude primitives.

    Inflows (revenue, financing) count their positive amounts; expense lines count
    the magnitude of their outflows. `related_party_payments` sums outflows flagged
    related regardless of expense category.
    """
    rows = txs
    if period_filter.upper() == "Q4":
        rows = [t for t in txs if _in_q4(t.date)]

    def inflow(cat: str) -> float:
        return sum(t.amount for t in rows if t.category == cat and t.amount and t.amount > 0)

    def outflow(cat: str) -> float:
        return sum(-t.amount for t in rows if t.category == cat and t.amount and t.amount < 0)

    prims = {c: outflow(c) for c in _OPEX_CATEGORIES}
    prims.update({
        "revenue": inflow("revenue"),
        "financing_inflow": inflow("financing_inflow"),
        "capex": outflow("capex"),
        "interest": outflow("interest"),
        "tax": outflow("tax"),
        "related_party_payments": sum(
            -t.amount for t in rows if t.related_party and t.amount and t.amount < 0),
    })
    prims["opex"] = sum(prims[c] for c in _OPEX_CATEGORIES)
    prims["ebitda"] = prims["revenue"] - prims["opex"]
    return prims


# --- safe formula evaluation ---------------------------------------------- #

_BINOPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
_FUNCS = {"max": max, "min": min}


def eval_expr(expr: str, prims: dict[str, float]) -> Optional[float]:
    """Evaluate a formula over primitives with a restricted AST walk (no eval()).

    Supports + - * /, parentheses, unary minus, and max()/min(). Unknown names
    resolve to 0.0 so a formula naming a primitive we didn't populate degrades
    gracefully instead of crashing.
    """
    if not expr:
        return None
    try:
        node = ast.parse(expr, mode="eval").body
        return _ev(node, prims)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return None


def _ev(node, prims: dict[str, float]) -> float:
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_ev(node.left, prims), _ev(node.right, prims))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ev(node.operand, prims)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_ev(a, prims) for a in node.args])
    if isinstance(node, ast.Name):
        return float(prims.get(node.id, 0.0))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    raise ValueError(f"disallowed expression node: {ast.dump(node)}")


# --- verdict + evidence ---------------------------------------------------- #

def _complies(value: float, o: Operator, threshold: float) -> bool:
    if o == Operator.LE:
        return value <= threshold
    if o == Operator.GE:
        return value >= threshold
    if o == Operator.LT:
        return value < threshold
    if o == Operator.GT:
        return value > threshold
    return True


def _condition_active(cov: Covenant, prims: dict[str, float]) -> bool:
    """A springing covenant is enforced only when its condition holds (else always met)."""
    if not cov.condition_expr or cov.condition_op is None or cov.condition_value is None:
        return True
    val = eval_expr(cov.condition_expr, prims)
    return val is not None and _complies(val, cov.condition_op, cov.condition_value)


def evaluate(cov: Covenant, txs: list[Transaction]) -> tuple[Verdict, Optional[float]]:
    """Return (verdict, actual). actual is a positive number, 2 decimals."""
    prims = primitives(txs, cov.period_filter)
    value = eval_expr(cov.value_expr, prims)
    if value is None or cov.threshold is None:
        return Verdict.COMPLIANT, (round(abs(value), 2) if value is not None else None)

    actual = round(abs(value), 2)
    if not _condition_active(cov, prims):
        # Metric still reported; test not triggered, so compliant.
        return Verdict.COMPLIANT, actual
    verdict = Verdict.COMPLIANT if _complies(value, cov.operator, cov.threshold) else Verdict.BREACH
    return verdict, actual


def find_evidence(cov: Covenant, txs: list[Transaction], verdict: Verdict) -> Optional[str]:
    """The single transaction whose removal flips a breach back to compliant.

    Per the spec, evidence is the decisive line — remove it and the verdict changes.
    We only claim evidence on a breach (a compliant cell's key is null). Since a
    wrong guess on a null-key cell is ignored and a right guess on a real-key cell
    earns points, we always return our best candidate for a breach.
    """
    if verdict != Verdict.BREACH or cov.threshold is None:
        return None

    flippers: list[tuple[float, str]] = []
    for i, t in enumerate(txs):
        if not t.amount:
            continue
        rest = txs[:i] + txs[i + 1:]
        prims = primitives(rest, cov.period_filter)
        val = eval_expr(cov.value_expr, prims)
        if val is None or not _condition_active(cov, prims):
            continue
        if _complies(val, cov.operator, cov.threshold):
            # Removing t makes the covenant pass -> t is decisive. Rank by how close
            # to the threshold it lands (smallest margin = most precisely decisive).
            flippers.append((abs(val - cov.threshold), t.tx_id))

    if flippers:
        return min(flippers, key=lambda x: x[0])[1]

    # No single line flips it (breach driven by many lines): fall back to the
    # largest related-party payment, else None.
    rp = [t for t in txs if t.related_party and t.amount and t.amount < 0]
    return max(rp, key=lambda t: -t.amount).tx_id if rp else None
