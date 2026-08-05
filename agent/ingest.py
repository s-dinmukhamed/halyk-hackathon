"""Load raw files into structured, addressable content.

Each PDF becomes a Document with per-page text and tables (so evidence can cite
a page) plus a lazy page-image renderer for the vision fallback. Transaction
registries and financials shipped as CSV/XLSX are loaded separately by
load_tabular — far more reliable than pulling numbers out of a PDF.
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
    """Render one page to PNG for the vision fallback.

    fitz is imported lazily so a missing PyMuPDF never breaks the text path.
    """
    import fitz  # PyMuPDF

    with fitz.open(path) as pdf:
        page = pdf[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")


def classify_document(filename: str, text: str) -> str:
    """Route a document by filename and the first page of text."""
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
    """Load a CSV/XLSX sidecar into a DataFrame."""
    import pandas as pd

    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_dataset(data_dir: Path) -> tuple[list[Document], list[Path]]:
    """Parse every PDF under data_dir and collect the tabular sidecar paths."""
    from .parallel import pmap

    data_dir = Path(data_dir)
    pdfs = pmap(parse_pdf, sorted(data_dir.rglob("*.pdf")))
    tabular = sorted(
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in (".csv", ".xlsx", ".xls")
    )
    return pdfs, tabular
