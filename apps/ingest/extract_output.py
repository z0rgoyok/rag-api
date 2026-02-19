from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata

from core.chunking import Chunk


_RE_SAFE_FILENAME = re.compile(r"[<>:\"/\\|?*\u0000-\u001F]+")


def _safe_stem(path: Path) -> str:
    raw_stem = unicodedata.normalize("NFKC", path.stem)
    return _RE_SAFE_FILENAME.sub("_", raw_stem).strip(" ._-")[:120] or "document"


def build_extract_output_path(*, pdf_path: Path, out_dir: Path, chunking_strategy: str) -> Path:
    safe_stem = _safe_stem(pdf_path)
    return out_dir / f"{safe_stem}.{chunking_strategy}.chunks.jsonl"


def is_extract_output_up_to_date(*, output_path: Path, source_sha256: str) -> bool:
    if not output_path.is_file():
        return False
    first_line = ""
    with output_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    if not first_line.strip():
        return False
    try:
        payload = json.loads(first_line)
    except json.JSONDecodeError:
        return False
    file_sha = str(payload.get("source_sha256") or "").strip()
    if not file_sha:
        return False
    return file_sha == source_sha256


def write_extract_output(
    *,
    output_path: Path,
    pdf_path: Path,
    source_sha256: str,
    chunking_strategy: str,
    chunks: list[Chunk],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            line = {
                "source_path": str(pdf_path),
                "source_sha256": source_sha256,
                "chunking_strategy": chunking_strategy,
                "ordinal": chunk.ordinal,
                "page": chunk.page,
                "content": chunk.content,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
