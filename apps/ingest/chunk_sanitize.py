from __future__ import annotations

from core.chunking import Chunk
import importlib
import os
import re
from typing import Any
import unicodedata

_ftfy: Any | None
try:
    _ftfy = importlib.import_module("ftfy")
except ImportError:  # pragma: no cover - optional safety net if env is stale
    _ftfy = None


_RE_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
_RE_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$")
_RE_DECORATIVE_RULE = re.compile(r"^\s*[-_=]{4,}\s*$")
_RE_NUMERIC_LIKE = re.compile(r"^[\d\s.,:;!?()\-–—]+$")
_RE_WORD = re.compile(r"\w+", flags=re.UNICODE)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 1000) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw.strip()) if raw is not None else default
    except (ValueError, AttributeError):
        value = default
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_chunk_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_CONTROL.sub("", text).replace("\u00ad", "")
    if _ftfy is not None:
        text = _ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if _RE_TABLE_SEPARATOR.fullmatch(stripped):
            continue
        if _RE_DECORATIVE_RULE.fullmatch(stripped):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"(?:\s*\.\s*){3,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_chunks(chunks: list[Chunk]) -> list[Chunk]:
    if not _env_bool("CHUNK_SANITIZE_ENABLED", True):
        return chunks

    min_words = _env_int("CHUNK_SANITIZE_MIN_WORDS", 3, min_value=0, max_value=100)
    dedup = _env_bool("CHUNK_SANITIZE_DEDUP", True)

    out: list[Chunk] = []
    seen: set[str] = set()

    for src in chunks:
        text = _clean_chunk_text(src.content)
        if not text:
            continue

        word_count = len(_RE_WORD.findall(text))
        is_heading = text.lstrip().startswith("#")
        if not is_heading and word_count < min_words:
            continue
        if _RE_NUMERIC_LIKE.fullmatch(text):
            continue

        key = _normalize_key(text)
        if dedup and key in seen:
            continue
        seen.add(key)

        out.append(Chunk(page=src.page, ordinal=len(out), content=text))

    return out
