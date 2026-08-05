"""Pull structured covenants and financial inputs out of documents with the LLM.

The model only reads and extracts — thresholds, effective dates, line items — as
JSON. It never computes ratios or decides verdicts; that's compute.py. Every
item keeps its source (doc, page, snippet) so resolve.py can order amendments
and verify.py can re-check the citation.
"""
from __future__ import annotations

from .ingest import Document
from .llm_client import complete_json, complete_vision
from .schemas import Covenant, CovenantType, Evidence, Operator

_EXTRACT_SYSTEM = (
    "You are a credit-risk analyst extracting loan covenant definitions from "
    "banking documents. Extract ONLY what is stated. Do not compute anything. "
    "Return strict JSON."
)

_EXTRACT_PROMPT = """From the document below, extract every covenant as a JSON array.
For each covenant return:
- borrower_id: string (borrower identifier or name)
- covenant_id: string (covenant identifier if present, else a short slug)
- name: short human description
- type: "financial" (threshold on a metric) or "transactional" (a prohibited action)
- metric: normalized key when financial, one of:
  debt_to_ebitda, current_ratio, quick_ratio, dscr, interest_coverage,
  debt_to_equity, net_worth  (or "" for transactional)
- operator: one of "<=", ">=", "<", ">", "==" (the REQUIRED condition)
- threshold: number (or null)
- period: reporting period / measurement date if stated
- effective_date: ISO date this term takes effect, if stated (else null)
- page: page number you found it on
- snippet: the exact clause text (<=240 chars)

Document: {name}
---
{text}
---
Return only the JSON array."""

_OPERATOR_MAP = {"<=": Operator.LE, "≤": Operator.LE, ">=": Operator.GE, "≥": Operator.GE,
                 "<": Operator.LT, ">": Operator.GT, "==": Operator.EQ, "=": Operator.EQ}


def _coerce_covenant(raw: dict, doc_name: str) -> Covenant:
    ctype = (raw.get("type") or "financial").lower()
    return Covenant(
        borrower_id=str(raw.get("borrower_id", "")).strip(),
        covenant_id=str(raw.get("covenant_id") or raw.get("name") or "cov").strip(),
        name=str(raw.get("name", "")).strip(),
        type=CovenantType.TRANSACTIONAL if ctype.startswith("trans") else CovenantType.FINANCIAL,
        metric=str(raw.get("metric") or "").strip(),
        operator=_OPERATOR_MAP.get(str(raw.get("operator", "<=")).strip(), Operator.LE),
        threshold=_as_float(raw.get("threshold")),
        period=str(raw.get("period") or "").strip(),
        source=Evidence(
            doc=doc_name,
            page=raw.get("page"),
            snippet=str(raw.get("snippet") or "")[:240],
            effective_date=raw.get("effective_date"),
        ),
    )


def _as_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def extract_covenants(doc: Document) -> list[Covenant]:
    """Extract covenants from one loan agreement or amendment."""
    prompt = _EXTRACT_PROMPT.format(name=doc.path.name, text=doc.full_text[:120_000])
    try:
        raw = complete_json(_EXTRACT_SYSTEM, prompt)
    except Exception:
        return []
    items = raw if isinstance(raw, list) else raw.get("covenants", [])
    return [_coerce_covenant(i, doc.path.name) for i in items if isinstance(i, dict)]


def extract_covenants_vision(doc: Document, pages: list[int]) -> list[Covenant]:
    """Vision fallback for scanned pages: render to PNG and ask Gemini."""
    from .ingest import render_page_png

    images = [render_page_png(doc.path, p) for p in pages]
    prompt = _EXTRACT_PROMPT.format(name=doc.path.name, text="(see attached page images)")
    raw = complete_vision(_EXTRACT_SYSTEM, prompt, images=images)
    items = raw if isinstance(raw, list) else []
    return [_coerce_covenant(i, doc.path.name) for i in items if isinstance(i, dict)]


def extract_financials(docs: list[Document]) -> dict[str, dict]:
    """Financial line items per borrower: {borrower_id: {ebitda, total_debt, ...}}.

    Not implemented until we see the statement format; compute.METRICS consumes
    whatever keys this returns.
    """
    return {}
