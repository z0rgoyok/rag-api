from __future__ import annotations

from core.chunking import Chunk
import os
import re
import unicodedata

import ftfy
from wordfreq import zipf_frequency


_RE_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
_RE_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$")
_RE_DECORATIVE_RULE = re.compile(r"^\s*[-_=]{4,}\s*$")
_RE_NUMERIC_LIKE = re.compile(r"^[\d\s.,:;!?()\-–—]+$")
_RE_WORD = re.compile(r"\w+", flags=re.UNICODE)
_RE_LINEBREAK_HYPHEN = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)", flags=re.UNICODE)
_RE_INNER_HYPHEN_SPACES = re.compile(r"(?<=\w)\s*-\s*(?=\w)", flags=re.UNICODE)
_RE_WORD_HYPHEN_WORD = re.compile(r"(?<!\w)([A-Za-zА-Яа-яЁё]{2,})\s*-\s*([A-Za-zА-Яа-яЁё]{2,})(?!\w)")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%])", flags=re.UNICODE)
_RE_SPACE_BEFORE_CLOSE = re.compile(r"\s+([)\]\}»])", flags=re.UNICODE)
_RE_SPACE_AFTER_OPEN = re.compile(r"([(\[{«])\s+", flags=re.UNICODE)
_RE_CYR_LATIN_CYR = re.compile(r"(?<=[А-Яа-яЁё])\s*([AaBEeKkMmHhOoPpCcTtXxYy])\s*(?=[А-Яа-яЁё])")
_OCR_LANGS = ("ru", "en")
_LATIN_TO_CYR = {
    "a": "а",
    "b": "в",
    "c": "с",
    "e": "е",
    "h": "н",
    "k": "к",
    "m": "м",
    "o": "о",
    "p": "р",
    "t": "т",
    "x": "х",
    "y": "у",
}


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


def _zipf_max(token: str) -> float:
    norm = token.strip().lower()
    if not norm:
        return 0.0
    return max(float(zipf_frequency(norm, lang)) for lang in _OCR_LANGS)


def _normalize_hyphenated_words(match: re.Match[str]) -> str:
    raw = match.group(0)
    left = match.group(1)
    right = match.group(2)

    joined = f"{left}{right}"
    hyphenated = f"{left}-{right}"

    joined_score = _zipf_max(joined)
    hyphen_score = _zipf_max(hyphenated)
    has_spaces_around_hyphen = " " in raw

    # OCR often injects spaces around a wrap hyphen. Prefer joined lexical forms
    # when they are significantly more plausible in RU/EN frequency dictionaries.
    if has_spaces_around_hyphen:
        if joined_score >= hyphen_score + 0.35 and joined_score >= 2.2:
            return joined
        if hyphen_score >= joined_score and hyphen_score >= 2.2:
            return hyphenated
        if joined_score >= 1.8:
            return joined

    if hyphen_score >= joined_score and hyphen_score >= 1.8:
        return hyphenated
    if joined_score >= hyphen_score + 0.6 and joined_score >= 2.5:
        return joined
    return hyphenated


def _clean_chunk_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_CONTROL.sub("", text).replace("\u00ad", "")
    text = ftfy.fix_text(text)
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
    # De-hyphenate words split by line breaks: "отпу-\nскает" -> "отпускает".
    text = _RE_LINEBREAK_HYPHEN.sub("", text)
    # Language-aware RU/EN normalization for hyphen artifacts from OCR.
    text = _RE_WORD_HYPHEN_WORD.sub(_normalize_hyphenated_words, text)
    # Normalize intraword hyphen spacing: "что -то" -> "что-то", "северо - запад" -> "северо-запад".
    text = _RE_INNER_HYPHEN_SPACES.sub("-", text)
    # Fix OCR spacing around punctuation/brackets.
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _RE_SPACE_BEFORE_CLOSE.sub(r"\1", text)
    text = _RE_SPACE_AFTER_OPEN.sub(r"\1", text)
    # Fix OCR confusables: latin lookalike letter inside Cyrillic word.
    text = _RE_CYR_LATIN_CYR.sub(lambda m: _LATIN_TO_CYR.get(m.group(1).lower(), m.group(1)), text)
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
