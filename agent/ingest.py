"""Step 1 — INGEST: raw files -> structured, addressable content.

Emits per-document:
  * page text (with page numbers, so evidence can cite a page)
  * extracted tables (financial statements live in tables)
  * lazy page-image renderer for the vision fallback on scans / messy tables

Tabular sidecar files (transaction registry, financials as CSV/XLSX) are loaded
separately by load_tabular() — much more reliable than reading them out of a PDF.

NOTE: exact filenames / folder layout are unknown until the public dataset drops
(6 Aug). classify_document() uses filename+content heuristics; revisit then.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber


@dataclass
class Page:
    number: int
    text: str
    tables: list[list[list[Optional[str]]]] = field(default_factory=list)


@dataclass
class Document:
    path: Path
    kind: str = "unknown"      # loan_agreement | amendment | financials | registry | unknown
    pages: list[Page] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def parse_pdf(path: Path) -> Document:
    """Text + tables per page via pdfplumber (no OCR)."""
    doc = Document(path=path)
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            doc.pages.append(Page(
                number=i,
                text=(page.extract_text() or "").strip(),
                tables=page.extract_tables() or [],
            ))
    doc.kind = classify_document(path.name, doc.full_text)
    return doc


def render_page_png(path: Path, page_number: int, dpi: int = 150) -> bytes:
    """Render one page to PNG for the Gemini vision fallback.

    Imported lazily so a missing PyMuPDF never breaks the text path.
    """
    import fitz  # PyMuPDF

    with fitz.open(path) as pdf:
        page = pdf[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")


def classify_document(filename: str, text: str) -> str:
    """Heuristic router. TODO(6 Aug): tune against real filenames/headers."""
    name = filename.lower()
    head = text[:2000].lower()
    if "amendment" in name or "допсоглаш" in name or "дополнительное соглашение" in head:
        return "amendment"
    if "registry" in name or "transaction" in name or "реестр" in name or "транзакц" in head:
        return "registry"
    if any(k in name for k in ("financ", "statement", "баланс", "отчет", "отчёт")):
        return "financials"
    if any(k in head for k in ("ковенант", "covenant", "кредитн", "loan agreement")):
        return "loan_agreement"
    return "unknown"


def load_tabular(path: Path):
    """Load a CSV/XLSX sidecar (transaction registry, financials) into a DataFrame."""
    import pandas as pd

    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_dataset(data_dir: Path) -> tuple[list[Document], list[Path]]:
    """Discover and parse everything under data_dir.

    Returns (parsed PDFs, tabular sidecar paths). Kept separate so callers can
    decide how to consume structured tables vs document text.
    """
    data_dir = Path(data_dir)
    pdfs = [parse_pdf(p) for p in sorted(data_dir.rglob("*.pdf"))]
    tabular = sorted(
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in (".csv", ".xlsx", ".xls")
    )
    return pdfs, tabular
