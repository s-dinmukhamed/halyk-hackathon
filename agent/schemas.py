"""Data models for the covenant-compliance agent.

The pipeline is scenario-centric: each of the 12 scenarios (P1..P10, B1, B4) has
three covenant cells (6.1, 6.2, 6.3). Covenants vary a lot between borrowers —
ratios, aggregates, max-of-two-lines, sub-period and conditional ("springing")
tests — so the LLM extracts each one as a small formula over named category sums
(a `value_expr`), and Python evaluates it deterministically. The LLM only reads,
labels and maps to the formula vocabulary; every sum, ratio and verdict is Python.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Exact labels the scorer expects (uppercase)."""
    COMPLIANT = "COMPLIANT"
    BREACH = "BREACH"


class Operator(str, Enum):
    LE = "<="   # "must not exceed" / "at most"
    GE = ">="   # "at least" / "maintain"
    LT = "<"
    GT = ">"


# Transaction categories the classifier assigns. Finer than opex/capex so covenants
# that single out a line (payroll, utilities, rent, interest, tax...) are computable.
# `related_party` is a separate boolean flag — a payment can be e.g. payroll AND related.
CATEGORIES = (
    "revenue",                # operating revenue inflows (core sales) — positive
    "financing_inflow",       # loan/financing proceeds received — positive
    "payroll",                # wages, salaries, personnel costs
    "utilities",              # electricity, water, gas, similar supply
    "rent",                   # rent / operating lease of premises & equipment
    "insurance",              # insurance premiums
    "marketing",              # advertising, marketing, sponsorship
    "professional_services",  # advisory, audit, legal, consulting fees
    "materials",              # materials, supplies, inventory purchases
    "maintenance",            # servicing, repairs, maintenance
    "logistics",              # freight, courier, transport, customs handling
    "tax",                    # taxes, duties, withholding
    "interest",               # interest, debt service, financing costs
    "capex",                  # capital expenditure — asset purchases/construction
    "other",                  # refunds, credits, deposits, transfers — non-operating
)

# Primitive names a `value_expr` may reference. `opex` and `ebitda` are derived.
PRIMITIVES = (
    "revenue", "opex", "ebitda", "capex", "interest", "tax",
    "payroll", "utilities", "rent", "insurance", "marketing",
    "professional_services", "materials", "maintenance", "logistics",
    "financing_inflow", "related_party_payments",
)


class Covenant(BaseModel):
    """One covenant cell's normalized definition, extracted from the agreement."""
    scenario_id: str
    clause: str                                  # "6.1" | "6.2" | "6.3"
    operator: Operator = Operator.LE
    threshold: Optional[float] = None

    # The metric as a formula over PRIMITIVES, e.g. "capex / ebitda",
    # "revenue / (payroll + utilities)", "max(payroll, utilities)", "related_party_payments".
    value_expr: str = ""

    # Optional "springing"/conditional test: only enforced when condition holds.
    # e.g. condition_expr="financing_inflow", condition_op=">", condition_value=4_000_000.
    condition_expr: str = ""
    condition_op: Optional[Operator] = None
    condition_value: Optional[float] = None

    # Optional sub-period filter for the metric ("Q4" => Oct–Dec only).
    period_filter: str = ""

    description: str = ""
    source_doc: str = ""
    snippet: str = ""


class Transaction(BaseModel):
    tx_id: str
    scenario_id: str = ""
    date: Optional[str] = None
    amount: Optional[float] = None               # USD, sign preserved (expenses negative)
    currency: str = "USD"
    counterparty: str = ""
    description: str = ""

    # Assigned by the classifier / KYC step.
    category: str = "other"
    related_party: bool = False


class CovenantAnswer(BaseModel):
    scenario_id: str
    clause: str

    # The three scored components.
    verdict: Verdict
    actual: Optional[float] = None
    evidence_tx_id: Optional[str] = None

    # Kept for debug, not emitted.
    threshold: Optional[float] = None
    operator: Optional[Operator] = None
    value_expr: str = ""
    reasoning: str = ""
