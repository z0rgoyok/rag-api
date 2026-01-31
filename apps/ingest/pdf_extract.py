from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import unicodedata

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PdfPageText:
    page: int
    text: str


_RE_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")


def _clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control chars (keep \n and \t semantics via normalization).
    text = _RE_CONTROL.sub("", text)
    # Remove soft hyphens that often appear in PDF text extraction.
    text = text.replace("\u00ad", "")
    # Normalize ligatures / compatibility chars.
    text = unicodedata.normalize("NFKC", text)
    # Collapse excessive whitespace while preserving paragraphs.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text_pages(path: Path) -> list[PdfPageText]:
    extractor = (os.getenv("PDF_TEXT_EXTRACTOR") or "pymupdf4llm").strip().lower()
    if extractor in {"pymupdf4llm", "markdown"}:
        try:
            import pymupdf4llm  # type: ignore[import-not-found]

            # Optional but recommended: improves page layout / reading order.
            try:
                import pymupdf_layout  # type: ignore[import-not-found]

                pymupdf_layout.activate()
            except Exception:
                pass

            pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
            out: list[PdfPageText] = []
            for i, p in enumerate(pages or []):
                # PyMuPDF4LLM returns per-page dicts; the content key may vary by version.
                text = ""
                if isinstance(p, dict):
                    text = (p.get("text") or p.get("markdown") or p.get("content") or "")  # type: ignore[assignment]
                out.append(PdfPageText(page=i + 1, text=_clean_extracted_text(text)))
            if out:
                return out
        except Exception:
            # Fallback to raw PyMuPDF extraction.
            pass

    doc = fitz.open(str(path))
    out: list[PdfPageText] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            out.append(PdfPageText(page=i + 1, text=_clean_extracted_text(text)))
    finally:
        doc.close()
    return out
