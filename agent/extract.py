"""Step 2 — EXTRACT: documents -> structured covenants + financial inputs.

The LLM's job is *understanding and extraction only* — pull out covenant
definitions, thresholds, effective dates, and financial line items as JSON.
It never computes ratios or decides verdicts (that's compute.py).

Every extracted item must carry source evidence (doc + page + snippet) so
resolve.py can order amendments and verify.py can re-check the citation.

TODO(6 Aug): tune prompts and the metric-name normalization against real docs.
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
    """Extract covenants from one loan agreement / amendment document."""
    covenants: list[Covenant] = []
    # Chunk by page group to stay within context and keep page evidence precise.
    prompt = _EXTRACT_PROMPT.format(name=doc.path.name, text=doc.full_text[:120_000])
    try:
        raw = complete_json(_EXTRACT_SYSTEM, prompt)
    except Exception:
        return covenants
    items = raw if isinstance(raw, list) else raw.get("covenants", [])
    for item in items:
        if isinstance(item, dict):
            covenants.append(_coerce_covenant(item, doc.path.name))
    return covenants


def extract_covenants_vision(doc: Document, pages: list[int]) -> list[Covenant]:
    """Vision fallback for scanned pages: render pages to PNG and ask Gemini."""
    from .ingest import render_page_png

    images = [render_page_png(doc.path, p) for p in pages]
    prompt = _EXTRACT_PROMPT.format(name=doc.path.name, text="(see attached page images)")
    raw = complete_vision(_EXTRACT_SYSTEM, prompt, images=images)
    items = raw if isinstance(raw, list) else []
    return [_coerce_covenant(i, doc.path.name) for i in items if isinstance(i, dict)]


def extract_financials(docs: list[Document]) -> dict[str, dict]:
    """Extract financial line items per borrower for ratio computation.

    Returns {borrower_id: {ebitda, total_debt, current_assets, ...}}.
    TODO(6 Aug): decide text-vs-table-vs-vision path once statement format is known.
    """
    # Placeholder: real implementation fills inputs consumed by compute.METRICS.
    return {}
