"""Core data models.

Two layers on purpose:
  * Rich internal models (Covenant, Transaction, CovenantAnswer) carry evidence
    and intermediate values so we can self-check and debug.
  * The final `submission.json` shape is produced in emit.py from these — the
    exact template keys are unknown until the public dataset drops (6 Aug),
    so keep the output mapping in one place, not scattered across the pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    COMPLIED = "complied"      # ковенант соблюдён
    BREACHED = "breached"      # ковенант нарушен


class CovenantType(str, Enum):
    FINANCIAL = "financial"          # порог по метрике (Debt/EBITDA, current ratio, ...)
    TRANSACTIONAL = "transactional"  # запрет действия, нарушается конкретной транзакцией


class Operator(str, Enum):
    LE = "<="   # не более
    GE = ">="   # не менее
    LT = "<"
    GT = ">"
    EQ = "=="


class Evidence(BaseModel):
    """Where a fact came from — for self-check and for the 'обоснование'."""
    doc: str = ""                       # имя файла-источника
    page: Optional[int] = None          # страница
    snippet: str = ""                   # процитированный фрагмент
    effective_date: Optional[str] = None  # ISO-дата вступления в силу (для RESOLVE)


class Covenant(BaseModel):
    borrower_id: str
    covenant_id: str
    name: str = ""                      # человекочитаемое описание
    type: CovenantType = CovenantType.FINANCIAL
    metric: str = ""                    # что измеряем, напр. "debt_to_ebitda"
    operator: Operator = Operator.LE
    threshold: Optional[float] = None   # актуальный порог (после RESOLVE)
    period: str = ""                    # период измерения / отчётная дата
    source: Evidence = Field(default_factory=Evidence)
    superseded: bool = False            # True если перекрыт более поздней версией


class Transaction(BaseModel):
    tx_id: str
    borrower_id: str = ""
    date: Optional[str] = None          # ISO-дата
    amount: Optional[float] = None
    currency: str = ""
    counterparty: str = ""
    tx_type: str = ""                   # тип операции (дивиденды, продажа актива, ...)
    description: str = ""


class CovenantAnswer(BaseModel):
    """Ответ по одному ковенанту — 3 оцениваемых компонента + служебные поля."""
    borrower_id: str
    covenant_id: str

    # --- 3 компонента ответа (оцениваются независимо, частичный зачёт) ---
    verdict: Verdict                         # 1. соблюдён / нарушен
    value: Optional[float] = None            # 2. число-обоснование
    evidence_tx_id: Optional[str] = None     # 3. транзакция-улика (если применимо)

    # --- служебное (в submission не идёт, для отладки/verify) ---
    threshold: Optional[float] = None
    operator: Optional[Operator] = None
    confidence: float = 0.0
    reasoning: str = ""
    evidence: Evidence = Field(default_factory=Evidence)


class Submission(BaseModel):
    answers: list[CovenantAnswer] = Field(default_factory=list)
