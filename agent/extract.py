"""LLM extraction: covenant formulas, related parties, and transaction classification.

The model only reads and labels — it never sums or decides a verdict; compute.py
does that from the labels and the formula. Three jobs per scenario:
  * extract_covenant_specs — Article 6 clauses -> {operator, threshold, value_expr,
    optional springing condition, optional sub-period}.
  * extract_related_parties — KYC dossier -> counterparties at/above the ownership
    threshold.
  * classify_transactions — each ledger row -> {category, related_party}, applying
    the related-party list and any auditor reclassification note.
"""
from __future__ import annotations

import re
import sys

from .ingest import Document
from .llm_client import complete_json
from .schemas import PRIMITIVES, Covenant, Operator, Transaction

_OPERATOR_MAP = {"<=": Operator.LE, "≤": Operator.LE, ">=": Operator.GE, "≥": Operator.GE,
                 "<": Operator.LT, ">": Operator.GT}

# --------------------------------------------------------------------------- #
# 1. Covenant specs
# --------------------------------------------------------------------------- #

_COV_SYSTEM = (
    "You are a credit-risk analyst normalizing loan covenant definitions from a bank "
    "loan agreement into formulas. Extract ONLY what is written. Never compute anything. "
    "Return strict JSON."
)

_COV_PROMPT = """Bank loan/credit agreement (Russian or English) for borrower scenario {sid}.
The financial-covenants article ("Финансовые ковенанты" / "Financial Covenants" — it may be
Статья 5 or Статья 6, i.e. Article 5 or 6) defines the covenants. Extract EXACTLY these
clauses: {clauses}. Use the clause number as printed (e.g. "5.1", "6.4").
For EACH requested clause return one object:

{{
  "clause": "6.1",
  "operator": "<=" or ">=",   // "не превышать"/"не более"/"не выше" -> "<="; "не менее"/"не ниже"/"поддерживать"/"обеспечивать" -> ">="
  "threshold": number,        // strip $ , spaces and 'x'; "9.00x"->9.0; "$260,000.00"->260000.0
  "value_expr": "<formula>",  // the metric, as a formula over the primitives below
  "condition_expr": "",       // springing covenants only (see below), else ""
  "condition_op": "",         // ">" / "<" / ">=" / "<=" for the condition, else ""
  "condition_value": null,    // numeric trigger for the condition, else null
  "period_filter": "",        // "Q4" if the metric is limited to the 4th quarter, else ""
  "description": "short plain description",
  "snippet": "exact clause text, <=300 chars"
}}

value_expr is arithmetic (+ - * / , parentheses, and max()/min()) over these primitives ONLY:
  revenue, opex, ebitda, capex, interest, tax, payroll, utilities, rent, insurance,
  marketing, professional_services, materials, maintenance, logistics, financing_inflow,
  related_party_payments
Where: ebitda = revenue - opex; opex = the TOTAL of ALL operating-expense lines (it ALREADY
includes payroll, utilities, rent, insurance, marketing, etc. — never add those to opex).
financing_inflow = поступления по финансированию. related_party_payments = платежи связанным сторонам.

Mapping examples (from real clauses of this dataset):
- "отношение капитальных затрат к EBITDA не превышало 9.00x" -> "<=", 9.0, "capex / ebitda"
- "поддерживать поступления «Выручка» не менее $7,500,000" -> ">=", 7500000.0, "revenue"
- "платежи связанным сторонам не должны превышать $260,000" -> "<=", 260000.0, "related_party_payments"
- "Коэффициент покрытия процентов = EBITDA к Процентным расходам, не ниже 2.00x" -> ">=", 2.0, "ebitda / interest"
- "ни одна отдельная статья (расходы на оплату труда / коммунальные) не превышала $1,500,000; по наибольшей" -> "<=", 1500000.0, "max(payroll, utilities)"
- "капиталоёмкость = капзатраты к сумме операционных расходов и арендных платежей, не более 0.42x" -> "<=", 0.42, "capex / opex"   (opex already includes rent)
- "Выручка не менее 3.00x суммы Расходов на оплату труда и Коммунальных расходов" -> ">=", 3.0, "revenue / (payroll + utilities)"
- "платежи связанным сторонам не более 0.08x Операционных расходов" -> "<=", 0.08, "related_party_payments / opex"
- "сумма Налогов и Коммунальных расходов к EBITDA не более 0.30x" -> "<=", 0.30, "(tax + utilities) / ebitda"
- "Ограниченные платежи аффилированным лицам не более 0.05x от выручки" -> "<=", 0.05, "related_party_payments / revenue"
- "капитальные затраты не превышали $2,000,000" -> "<=", 2000000.0, "capex"
- "Выручка за четвёртый квартал не менее $3,500,000" -> ">=", 3500000.0, "revenue", period_filter "Q4"
- springing: "отношение поступлений по финансированию к EBITDA ≤ 1.70x применяется только если поступления по финансированию > $4,000,000"
  -> operator "<=", threshold 1.70, value_expr "financing_inflow / ebitda",
     condition_expr "financing_inflow", condition_op ">", condition_value 4000000.0

Return a JSON array with one object per requested clause ({clauses}). Agreement text:
---
{text}
---
Return only the JSON array."""


def _as_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(re.sub(r"[^0-9.\-]", "", str(v)))
    except ValueError:
        return None


def _san_expr(expr: str) -> str:
    """Keep only primitive names, numbers, operators and parens/commas."""
    expr = (expr or "").strip()
    for name in re.findall(r"[A-Za-z_][A-Za-z_]*", expr):
        if name not in PRIMITIVES and name not in ("max", "min"):
            return ""  # references something we don't compute -> treat as empty
    return expr if re.fullmatch(r"[\w\s+\-*/().,]*", expr) else ""


def _coerce_covenant(raw: dict, sid: str, doc_name: str) -> Covenant:
    cond_op = _OPERATOR_MAP.get(str(raw.get("condition_op", "")).strip())
    return Covenant(
        scenario_id=sid,
        clause=str(raw.get("clause") or "").strip(),
        operator=_OPERATOR_MAP.get(str(raw.get("operator", "<=")).strip(), Operator.LE),
        threshold=_as_float(raw.get("threshold")),
        value_expr=_san_expr(raw.get("value_expr")),
        condition_expr=_san_expr(raw.get("condition_expr")),
        condition_op=cond_op,
        condition_value=_as_float(raw.get("condition_value")),
        period_filter=str(raw.get("period_filter") or "").strip(),
        description=str(raw.get("description") or "").strip(),
        source_doc=doc_name,
        snippet=str(raw.get("snippet") or "")[:300],
    )


def extract_covenant_specs(agreement: Document, sid: str,
                           clauses: list[str] | None = None) -> list[Covenant]:
    clause_str = ", ".join(clauses) if clauses else "6.1, 6.2, 6.3"
    prompt = _COV_PROMPT.format(sid=sid, clauses=clause_str, text=agreement.text[:100_000])
    try:
        raw = complete_json(_COV_SYSTEM, prompt)
    except Exception as e:
        print(f"[extract] covenant_specs {sid} FAILED: {e}", file=sys.stderr)
        return []
    items = raw if isinstance(raw, list) else raw.get("covenants", raw.get("clauses", []))
    covs = [_coerce_covenant(i, sid, agreement.name) for i in items if isinstance(i, dict)]
    result = [c for c in covs if c.clause]
    print(f"[extract] covenant_specs {sid}: {len(result)} covenants extracted", file=sys.stderr)
    return result


# --------------------------------------------------------------------------- #
# 2. Related parties (KYC)
# --------------------------------------------------------------------------- #

_KYC_SYSTEM = (
    "You extract the related-party list from a KYC / compliance dossier. Return strict JSON."
)

_KYC_PROMPT = """This KYC dossier (Russian) lists counterparties with the Group's share of
voting rights, and states an ownership threshold at or above which a counterparty is a
RELATED PARTY ("связанная сторона") for the loan agreement (e.g. "35.0% и более ...
признаются связанными сторонами").

Return JSON:
{{
  "threshold_pct": number,               // the stated ownership threshold, e.g. 35.0
  "related_parties": ["exact counterparty name", ...]   // ONLY entities whose share >= threshold
}}

Be strict: an entity just below the threshold (e.g. 33.8% when the threshold is 35.0%) is
NOT a related party, even if its name looks affiliated. Dossier text:
---
{text}
---
Return only the JSON object."""


def extract_related_parties(kyc: Document) -> list[str]:
    prompt = _KYC_PROMPT.format(text=kyc.text[:60_000])
    try:
        raw = complete_json(_KYC_SYSTEM, prompt)
    except Exception as e:
        print(f"[extract] related_parties {kyc.name} FAILED: {e}", file=sys.stderr)
        return []
    if isinstance(raw, dict):
        names = raw.get("related_parties", [])
        result = [str(n).strip() for n in names if str(n).strip()]
        print(f"[extract] related_parties {kyc.name}: {len(result)} parties", file=sys.stderr)
        return result
    return []


# --------------------------------------------------------------------------- #
# 3. Auditor reclassification note
# --------------------------------------------------------------------------- #

_ADDENDUM_MARK = "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ"


def reclassification_note(audit: Document | None) -> str:
    """Pull the covenant-compliance addendum out of the audit notes, if present."""
    if audit is None:
        return ""
    text = audit.text
    i = text.find(_ADDENDUM_MARK)
    if i == -1:
        return ""
    tail = text[i:]
    end = re.search(r"За аудитора|Statutory\s+Auditors", tail)
    note = (tail[: end.start()] if end else tail[:2000]).strip()
    return "" if "не требовалось" in note else note


# Russian category label used in an auditor reclassification -> internal primitive.
_RECLASS_CATEGORY = {
    "процентные расходы": "interest",
    "коммунальные услуги": "utilities",
    "страховые премии": "insurance",
    "налоги": "tax",
    "расходы на оплату труда": "payroll",
    "капитальные затраты": "capex",
    "арендные платежи": "rent",
}


def extract_reclassifications(audit_docs: list[Document]) -> list[dict]:
    """Auditor reclassifications from IN-FORCE audit reports only.

    Each in-force "Отчёт о выполнении согласованных процедур" states final
    reclassifications ("Сумма ... переклассифицирована ... как <Категория>").
    Superseded/draft copies carry decoy reclassifications and are ignored — the
    ingest layer already flags them as `superseded`.
    """
    out: list[dict] = []
    seen: set[tuple[float, str]] = set()
    for d in audit_docs:
        if d.superseded:
            continue
        text = " ".join(d.text.split())
        for m in re.finditer(
            r"Сумма в размере\s+\$([\d,]+\.\d{2})[,)]?\s*"
            r"(?:выплаченная контрагенту\s+([^,]+?),)?"
            r"[^.]*?переклассифицирована[^.]*?как\s+([А-Яа-яё ]+?)\.",
            text,
        ):
            new_cat = _RECLASS_CATEGORY.get(m.group(3).strip().lower())
            if not new_cat:
                continue
            amount = float(m.group(1).replace(",", ""))
            key = (round(amount, 2), new_cat)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "amount": round(amount, 2),
                "counterparty": (m.group(2) or "").strip(),
                "new_category": new_cat,
            })
    return out


# --------------------------------------------------------------------------- #
# 4. Transaction classification
# --------------------------------------------------------------------------- #

_CLS_SYSTEM = (
    "You are a forensic accountant classifying ledger transactions for loan-covenant "
    "testing. Judge each line by the economic substance of its description, not just "
    "keywords. Return strict JSON."
)

_CLS_PROMPT = """Borrower scenario {sid}. Classify EVERY transaction below into exactly one
category and flag related-party payments. Amounts: negative = money out (payment/expense),
positive = money in (receipt). All amounts are already in USD.

Categories (pick the single best fit by economic substance):
- "revenue": OPERATING revenue from the borrower's core business (sales of its goods/services).
  A positive inflow. EXCLUDE refunds, rebates, credits, corrections, returned/released deposits,
  reimbursements, unclaimed-payroll returns, interest rebates, sublet/incidental income — those
  are "other".
- "financing_inflow": loan drawdowns / financing proceeds received (positive inflow).
- "payroll": wages, salaries, personnel costs, payroll top-ups, severance.
- "utilities": electricity, water, gas, heating and similar supply charges.
- "rent": rent or operating lease of premises / equipment (lease payments, yard/mast lease).
- "insurance": insurance premiums of any kind.
- "marketing": advertising, marketing, sponsorship, ad campaigns.
- "professional_services": advisory, audit, legal, consulting, management retainers.
- "materials": materials, supplies, stationery, inventory purchases.
- "maintenance": servicing, repairs, maintenance of plant/equipment.
- "logistics": freight, courier, transport, customs handling.
- "tax": taxes, duties, withholding, corporate income tax.
- "interest": interest, subordinated-note interest, debt service, financing costs.
- "capital_expenditure": purchase/construction of long-lived assets — equipment, plant,
  machinery, turbines, switchyard, property.
- "other": anything non-operating — refunds, credits, deposits released/returned, transfers,
  and every non-core inflow excluded from "revenue" above.

related_party: true only if the counterparty is one of these related parties:
{related_parties}
(match by company name; ignore the city in parentheses). Otherwise false.

{audit_note}
Return JSON: {{"classifications": [{{"tx_id": "...", "category": "...", "related_party": true|false}}, ...]}}
covering ALL {n} transactions. Use category name "capital_expenditure" exactly (not "capex").
Transactions:
{rows}
Return only the JSON object."""

# Map the verbose label used in the prompt to the internal category name.
_CATEGORY_ALIASES = {"capital_expenditure": "capex", "capex": "capex"}


def _format_rows(txs: list[Transaction]) -> str:
    return "\n".join(
        f"{t.tx_id} | {(t.amount or 0):.2f} | {t.counterparty} | {t.description}"
        for t in txs
    )


def classify_transactions(sid: str, txs: list[Transaction], related_parties: list[str],
                          audit_note: str = "") -> dict[str, dict]:
    """Return {tx_id: {"category": str, "related_party": bool}} for the scenario's rows."""
    rp = ", ".join(related_parties) if related_parties else "(none identified)"
    note = f"Auditor reclassification note (apply it when labeling): {audit_note}\n" if audit_note else ""
    prompt = _CLS_PROMPT.format(
        sid=sid, related_parties=rp, audit_note=note, n=len(txs), rows=_format_rows(txs),
    )
    try:
        raw = complete_json(_CLS_SYSTEM, prompt)
    except Exception as e:
        print(f"[extract] classify_transactions {sid} FAILED: {e}", file=sys.stderr)
        return {}
    items = raw.get("classifications", []) if isinstance(raw, dict) else raw
    out: dict[str, dict] = {}
    for it in items or []:
        if isinstance(it, dict) and it.get("tx_id"):
            cat = str(it.get("category") or "other").strip().lower()
            out[str(it["tx_id"]).strip()] = {
                "category": _CATEGORY_ALIASES.get(cat, cat),
                "related_party": bool(it.get("related_party")),
            }
    print(f"[extract] classify_transactions {sid}: {len(out)}/{len(txs)} classified", file=sys.stderr)
    return out
