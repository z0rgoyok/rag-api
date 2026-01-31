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
_RE_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _repo_root() -> Path:
    env_root = (os.getenv("RAG_REPO_ROOT") or "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    def looks_like_repo_root(p: Path) -> bool:
        return (p / "apps").is_dir() and (p / "pyproject.toml").is_file()

    # Prefer CWD because inside Docker the package may be imported from site-packages,
    # but WORKDIR is still the repo root (`/app`).
    cwd = Path.cwd().resolve()
    for p in (cwd, *cwd.parents):
        if looks_like_repo_root(p):
            return p

    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if looks_like_repo_root(p):
            return p

    return cwd


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


def _dump_md_if_enabled(*, pdf_path: Path, pages: list[PdfPageText]) -> None:
    enabled = (os.getenv("PDF_DUMP_MD") or "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    root = _repo_root()
    default_dir = root / "var" / "extracted"

    dump_dir_env = (os.getenv("PDF_DUMP_DIR") or "").strip()
    if dump_dir_env:
        dump_dir = Path(dump_dir_env)
        dump_dir = (root / dump_dir) if not dump_dir.is_absolute() else dump_dir
        dump_dir = dump_dir.resolve()
        try:
            dump_dir.relative_to(root / "var")
        except ValueError:
            dump_dir = default_dir
    else:
        dump_dir = default_dir

    dump_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = _RE_SAFE_FILENAME.sub("_", pdf_path.stem).strip("._-")[:120] or "document"
    out_path = dump_dir / f"{safe_stem}.md"

    parts: list[str] = [f"# {pdf_path.name}"]
    for p in pages:
        if not p.text:
            continue
        parts.append(f"\n\n---\n\n## Page {p.page}\n\n{p.text}")

    out_path.write_text("".join(parts).strip() + "\n", encoding="utf-8")


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
                _dump_md_if_enabled(pdf_path=path, pages=out)
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
    if out:
        _dump_md_if_enabled(pdf_path=path, pages=out)
    return out
