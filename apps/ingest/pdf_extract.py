from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PdfPageText:
    page: int
    text: str


def extract_pdf_text_pages(path: Path) -> list[PdfPageText]:
    doc = fitz.open(str(path))
    out: list[PdfPageText] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            out.append(PdfPageText(page=i + 1, text=text))
    finally:
        doc.close()
    return out

