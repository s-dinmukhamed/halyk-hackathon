"""Load the dataset: route documents to scenarios and load the ledger.

Filenames are opaque hashes, so a document's owner and type come from its text.
A document names the borrower by `account_id` (e.g. ACC-7805); the ledger links
`account_id -> scenario_id` because every `txn_id` starts with the scenario id of
the account it sits on (TXN-P5-0004 -> P5, ACC-7805). We keep only the scenarios
the submission template asks about and drop decoys, superseded editions and the
131 irrelevant PDFs.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import Transaction

# Document kinds we care about.
CREDIT_AGREEMENT = "credit_agreement"
AUDIT = "audit"
KYC = "kyc"
OTHER = "other"

_ACC_RE = re.compile(r"\b(?:ACC|TELE)-\d+\b")
# Scenario id is the first segment after TXN-. Most ids are TXN-<scn>-<digits>, but
# some (e.g. KC) use an extra category segment: TXN-KC-CAP-29, TXN-KC-FIN-19.
_SCENARIO_RE = re.compile(r"TXN-([A-Za-z0-9]+)-")
# Borrower name in an agreement: "... между <NAME> (акционерное общество ...".
_BORROWER_RE = re.compile(r"между\s+(.+?)\s*\(", re.DOTALL)

# Stamps marking a document as not-in-force: the superseded 2024 agreement edition
# and draft/interim audit workpapers. Any of these => ignore the document.
_SUPERSEDED_MARKERS = (
    "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ", "НЕ ПРИМЕНЯЕТСЯ", "SUPERSEDED", "NOT APPLICABLE",
    "ПРОЕКТ — ПРОМЕЖУТОЧНАЯ", "ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ",
    "НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ", "РАБОЧИЙ ДОКУМЕНТ",
)


@dataclass
class Document:
    path: Path
    text: str = ""
    kind: str = OTHER
    account_ids: list[str] = field(default_factory=list)
    scenario_id: str = ""          # filled once the account->scenario map is known
    superseded: bool = False

    @property
    def name(self) -> str:
        return self.path.name


def pdf_text(path: Path) -> str:
    """Extract the text layer with pypdf (all PDFs here are text, not scanned)."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def classify_document(text: str) -> str:
    head = text[:4000]
    low = head.lower()
    if "договор банковского займа" in low or "credit agreement" in low or "loan agreement" in low:
        return CREDIT_AGREEMENT
    if ("аудиторское дело" in low or ("аудитор" in low and "примечани" in low)
            or "agreed-upon procedures" in low or "auditor" in low):
        return AUDIT
    if ("знай своего клиента" in low or "kyc" in low or "связанных сторон" in low
            or "know your customer" in low or "related part" in low):
        return KYC
    return OTHER


def parse_document(path: Path) -> Document:
    text = pdf_text(path)
    doc = Document(path=path, text=text)
    doc.account_ids = sorted(set(_ACC_RE.findall(text)))
    doc.kind = classify_document(text)
    doc.superseded = any(m in text for m in _SUPERSEDED_MARKERS)
    return doc


def borrower_name(text: str) -> str:
    """Extract the borrower's company name from an agreement's preamble."""
    m = _BORROWER_RE.search(text[:2000])
    if not m:
        return ""
    name = " ".join(m.group(1).split())          # collapse whitespace/newlines
    return name if 3 <= len(name) <= 80 else ""


def load_ledger(csv_path: Path) -> list[Transaction]:
    """Load master_ledger_2025.csv into Transactions, tagging scenario_id from txn_id."""
    txs: list[Transaction] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tx_id = (row.get("txn_id") or "").strip()
            m = _SCENARIO_RE.match(tx_id)
            txs.append(Transaction(
                tx_id=tx_id,
                scenario_id=m.group(1) if m else "",
                date=(row.get("date") or "").strip() or None,
                amount=_num(row.get("amount")),
                currency=(row.get("currency") or "USD").strip() or "USD",
                counterparty=(row.get("counterparty") or "").strip(),
                description=(row.get("description") or "").strip(),
            ))
    return txs


def _num(v) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def load_scenario_index(csv_path: Path) -> dict[str, str]:
    """account_id -> scenario_id, recovered straight from the raw ledger rows.

    The ledger has no account->scenario column, but each real account only
    appears under one scenario's transactions (TXN-P5-* all sit on ACC-7805), so
    we take the majority scenario per account to be robust to any stray rows.
    """
    from collections import Counter, defaultdict

    votes: dict[str, Counter] = defaultdict(Counter)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            acc = (row.get("account_id") or "").strip()
            m = _SCENARIO_RE.match((row.get("txn_id") or "").strip())
            if acc and m:
                votes[acc][m.group(1)] += 1
    return {acc: c.most_common(1)[0][0] for acc, c in votes.items()}


def load_documents(documents_dir: Path) -> list[Document]:
    """Parse every PDF under documents_dir in parallel."""
    from .parallel import pmap

    pdfs = sorted(Path(documents_dir).rglob("*.pdf"))
    return pmap(parse_document, pdfs)
