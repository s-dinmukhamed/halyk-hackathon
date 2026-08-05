"""Data models.

Internal models carry evidence and intermediate values so we can self-check;
the final submission.json shape is built from them in emit.py.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    COMPLIED = "complied"
    BREACHED = "breached"


class CovenantType(str, Enum):
    FINANCIAL = "financial"          # threshold on a metric
    TRANSACTIONAL = "transactional"  # a prohibited action, breached by a transaction


class Operator(str, Enum):
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"
    EQ = "=="


class Evidence(BaseModel):
    doc: str = ""
    page: Optional[int] = None
    snippet: str = ""
    effective_date: Optional[str] = None  # ISO; used to order amendments in resolve.py


class Covenant(BaseModel):
    borrower_id: str
    covenant_id: str
    name: str = ""
    type: CovenantType = CovenantType.FINANCIAL
    metric: str = ""                    # e.g. "debt_to_ebitda"
    operator: Operator = Operator.LE
    threshold: Optional[float] = None   # effective value after resolve.py
    period: str = ""
    source: Evidence = Field(default_factory=Evidence)
    superseded: bool = False            # overridden by a later amendment


class Transaction(BaseModel):
    tx_id: str
    borrower_id: str = ""
    date: Optional[str] = None
    amount: Optional[float] = None
    currency: str = ""
    counterparty: str = ""
    tx_type: str = ""
    description: str = ""


class CovenantAnswer(BaseModel):
    borrower_id: str
    covenant_id: str

    # The three scored components (each graded independently).
    verdict: Verdict
    value: Optional[float] = None
    evidence_tx_id: Optional[str] = None

    # Kept for verify/debug, not emitted.
    threshold: Optional[float] = None
    operator: Optional[Operator] = None
    confidence: float = 0.0
    reasoning: str = ""
    evidence: Evidence = Field(default_factory=Evidence)


class Submission(BaseModel):
    answers: list[CovenantAnswer] = Field(default_factory=list)
