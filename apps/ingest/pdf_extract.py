from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import unicodedata

@dataclass(frozen=True)
class PdfPageText:
    page: int
    text: str


_RE_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
_RE_SAFE_FILENAME = re.compile(r"[<>:\"/\\|?*\u0000-\u001F]+")
_PAGE_BREAK = "[[PAGE_BREAK]]"


def _repo_root() -> Path:
    env_root = (os.getenv("RAG_REPO_ROOT") or "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    def looks_like_repo_root(root_candidate: Path) -> bool:
        return (root_candidate / "apps").is_dir() and (root_candidate / "pyproject.toml").is_file()

    # Prefer CWD because inside Docker the package may be imported from site-packages,
    # but WORKDIR is still the repo root (`/app`).
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if looks_like_repo_root(candidate):
            return candidate

    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if looks_like_repo_root(candidate):
            return candidate

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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw.strip())
        except ValueError:
            value = default
    if min_value is not None and value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


def _env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw.strip())
        except ValueError:
            value = default
    if min_value is not None and value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


def _is_env_explicit(name: str) -> bool:
    raw = os.getenv(name)
    return raw is not None and bool(raw.strip())


def _sample_page_indexes(total_pages: int, sample_pages: int) -> list[int]:
    if total_pages <= 0:
        return []
    if sample_pages <= 0 or sample_pages >= total_pages:
        return list(range(total_pages))
    if sample_pages == 1:
        return [0]
    step = (total_pages - 1) / (sample_pages - 1)
    return sorted({int(round(i * step)) for i in range(sample_pages)})


def _estimate_text_layer_ratio(
    pdf_path: Path,
    *,
    min_chars: int,
    sample_pages: int,
) -> float | None:
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except (OSError, RuntimeError, ValueError, pdfium.PdfiumError):
        return None

    total_pages = len(doc)
    indexes = _sample_page_indexes(total_pages, sample_pages)
    if not indexes:
        return 0.0

    pages_with_text = 0
    scanned = 0
    for idx in indexes:
        try:
            page = doc[idx]
            text_page = page.get_textpage()
            text = text_page.get_text_range() or ""
            if len(text.strip()) >= min_chars:
                pages_with_text += 1
            scanned += 1
            text_page.close()
            page.close()
        except (OSError, RuntimeError, ValueError, pdfium.PdfiumError):
            continue

    if scanned == 0:
        return None
    return pages_with_text / scanned


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
    raw_stem = unicodedata.normalize("NFKC", pdf_path.stem)
    safe_stem = _RE_SAFE_FILENAME.sub("_", raw_stem).strip(" ._-")[:120] or "document"
    out_path = dump_dir / f"{safe_stem}.md"

    parts: list[str] = [f"# {pdf_path.name}"]
    for p in pages:
        if not p.text:
            continue
        parts.append(f"\n\n---\n\n## Page {p.page}\n\n{p.text}")

    out_path.write_text("".join(parts).strip() + "\n", encoding="utf-8")


def extract_pdf_text_pages(path: Path) -> list[PdfPageText]:
    extractor = (os.getenv("PDF_TEXT_EXTRACTOR") or "docling").strip().lower()
    if extractor != "docling":
        raise ValueError(
            f"Unsupported PDF_TEXT_EXTRACTOR={extractor!r}. "
            "Only 'docling' is supported."
        )

    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
        from docling.document_converter import PdfFormatOption  # type: ignore[import-not-found]
        from docling.datamodel.base_models import DocItemLabel  # type: ignore[import-not-found]
        from docling.datamodel.base_models import InputFormat  # type: ignore[import-not-found]
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "docling is required for PDF extraction but is not installed. "
            "Install project dependencies to continue."
        ) from e

    # Hybrid OCR by default; auto-disable OCR for PDFs with strong text layer.
    do_ocr = _env_bool("DOCLING_DO_OCR", True)
    do_table_structure = _env_bool("DOCLING_DO_TABLE_STRUCTURE", False)
    force_full_page_ocr = _env_bool("DOCLING_FORCE_FULL_PAGE_OCR", False)
    force_backend_text = _env_bool("DOCLING_FORCE_BACKEND_TEXT", False)
    include_pictures = _env_bool("DOCLING_INCLUDE_PICTURES", False)
    do_picture_classification = _env_bool("DOCLING_DO_PICTURE_CLASSIFICATION", False)
    do_picture_description = _env_bool("DOCLING_DO_PICTURE_DESCRIPTION", False)
    do_ocr_explicit = _is_env_explicit("DOCLING_DO_OCR")
    force_backend_text_explicit = _is_env_explicit("DOCLING_FORCE_BACKEND_TEXT")

    if _env_bool("DOCLING_OCR_AUTO", True) and not do_ocr_explicit:
        text_layer_ratio = _estimate_text_layer_ratio(
            path,
            min_chars=_env_int("DOCLING_OCR_AUTO_MIN_CHARS", 20, min_value=1, max_value=10000),
            sample_pages=_env_int("DOCLING_OCR_AUTO_SAMPLE_PAGES", 0, min_value=0),
        )
        threshold = _env_float("DOCLING_OCR_AUTO_TEXT_LAYER_THRESHOLD", 0.95, min_value=0.0, max_value=1.0)
        if text_layer_ratio is not None and text_layer_ratio >= threshold:
            if not force_backend_text_explicit:
                force_backend_text = True
            # Disable OCR entirely only when text layer is effectively complete.
            if text_layer_ratio >= 0.999:
                do_ocr = False

    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
        force_backend_text=force_backend_text,
        do_picture_classification=do_picture_classification,
        do_picture_description=do_picture_description,
    )
    # Avoid image generation overhead for text extraction flows.
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_table_images = False
    if getattr(pipeline_options, "ocr_options", None) is not None:
        pipeline_options.ocr_options.force_full_page_ocr = force_full_page_ocr

    export_labels = set(DocItemLabel)
    if not include_pictures:
        export_labels.discard(DocItemLabel.PICTURE)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(str(path))
    markdown = result.document.export_to_markdown(
        page_break_placeholder=f"\n\n{_PAGE_BREAK}\n\n",
        labels=export_labels,
        image_placeholder="<!-- image -->" if include_pictures else "",
    )
    parts = [part for part in markdown.split(_PAGE_BREAK)]

    out: list[PdfPageText] = []
    for i, part in enumerate(parts):
        # Keep original page numbering even for blank pages so citations stay accurate.
        out.append(PdfPageText(page=i + 1, text=_clean_extracted_text(part)))

    if out:
        _dump_md_if_enabled(pdf_path=path, pages=out)
    return out
