from __future__ import annotations

from core.chunking import Chunk
from dataclasses import dataclass
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
_RE_SOFT_LINE_BREAK = re.compile(r"(?<=[^\n.!?…:;])\n(?=[a-zа-яё0-9])", flags=re.IGNORECASE)
_RE_LINEBREAK_HYPHEN = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)", flags=re.UNICODE)
_RE_INNER_HYPHEN_SPACES = re.compile(r"(?<=\w)\s*-\s*(?=\w)", flags=re.UNICODE)
_RE_WORD_HYPHEN_WORD = re.compile(r"(?<!\w)([A-Za-zА-Яа-яЁё]{2,})\s*-\s*([A-Za-zА-Яа-яЁё]{2,})(?!\w)")
_RE_WORD_HYPHEN_SHORT_SUFFIX = re.compile(r"(?<!\w)([A-Za-zА-Яа-яЁё]{3,})-([A-Za-zА-Яа-яЁё]{1,2})(?!\w)")
_RE_CYR_HYPHEN_LATIN_SUFFIX = re.compile(r"(?<!\w)([А-Яа-яЁё]{3,})-([A-Za-z])(?!\w)")
_RE_MIXED_SCRIPT_WORD = re.compile(
    r"(?<!\w)(?=[A-Za-zА-Яа-яЁёІі]*[A-Za-z])(?=[A-Za-zА-Яа-яЁёІі]*[А-Яа-яЁёІі])[A-Za-zА-Яа-яЁёІі]{3,}(?!\w)"
)
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%])", flags=re.UNICODE)
_RE_SPACE_BEFORE_CLOSE = re.compile(r"\s+([)\]\}»])", flags=re.UNICODE)
_RE_SPACE_AFTER_OPEN = re.compile(r"([(\[{«])\s+", flags=re.UNICODE)
_RE_CYR_LATIN_CYR = re.compile(r"(?<=[А-Яа-яЁё])\s*([AaBEeKkMmHhOoPpCcTtXxYy])\s*(?=[А-Яа-яЁё])")
_RE_HAS_CYR = re.compile(r"[А-Яа-яЁёІі]")
_RE_HAS_LAT = re.compile(r"[A-Za-z]")
_OCR_LANGS = ("ru", "en")
_LATIN_TO_CYR = {
    "a": "а",
    "b": "в",
    "c": "с",
    "d": "д",
    "e": "е",
    "h": "н",
    "i": "і",
    "k": "к",
    "m": "м",
    "n": "п",
    "o": "о",
    "p": "р",
    "t": "т",
    "x": "х",
    "y": "у",
    "z": "з",
}
_CYR_TO_LAT = {
    "а": "a",
    "в": "b",
    "с": "c",
    "д": "d",
    "е": "e",
    "н": "h",
    "і": "i",
    "к": "k",
    "м": "m",
    "о": "o",
    "п": "n",
    "р": "p",
    "т": "t",
    "у": "y",
    "х": "x",
    "з": "z",
}
_RU_JOINABLE_SINGLE_SUFFIX = set("аеиоуыэюяёйь")


@dataclass(frozen=True)
class SanitizedChunk:
    chunk: Chunk
    raw_content: str


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


def _normalize_short_suffix_hyphen(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)

    joined = f"{left}{right}"
    hyphenated = f"{left}-{right}"

    joined_score = _zipf_max(joined)
    hyphen_score = _zipf_max(hyphenated)

    if (
        len(right) == 1
        and right.lower() in _RU_JOINABLE_SINGLE_SUFFIX
        and _RE_HAS_CYR.search(left) is not None
        and _RE_HAS_CYR.search(right) is not None
        and joined_score >= 1.8
    ):
        return joined

    # OCR often inserts a hard hyphen before a short inflectional suffix.
    # Example: "носильщик-а" -> "носильщика".
    if joined_score >= hyphen_score + 0.45 and joined_score >= 2.2:
        return joined
    if hyphen_score >= joined_score and hyphen_score >= 2.0:
        return hyphenated
    return hyphenated


def _replace_mapped_char(ch: str, mapping: dict[str, str]) -> str:
    mapped = mapping.get(ch.lower())
    if mapped is None:
        return ch
    if ch.isupper():
        return mapped.upper()
    return mapped


def _normalize_cyr_hyphen_latin_suffix(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    mapped = _replace_mapped_char(right, _LATIN_TO_CYR)
    return f"{left}-{mapped}"


def _convert_token_by_mapping(token: str, mapping: dict[str, str]) -> str:
    return "".join(_replace_mapped_char(ch, mapping) for ch in token)


def _token_score(token: str) -> float:
    norm = token.strip("-").lower()
    if not norm:
        return 0.0
    has_cyr = _RE_HAS_CYR.search(norm) is not None
    has_lat = _RE_HAS_LAT.search(norm) is not None
    if has_cyr and not has_lat:
        return float(zipf_frequency(norm, "ru"))
    if has_lat and not has_cyr:
        return float(zipf_frequency(norm, "en"))
    return _zipf_max(norm) - 0.75


def _normalize_mixed_script_word(match: re.Match[str]) -> str:
    token = match.group(0)
    if len(token) < 3:
        return token

    original = token
    candidates = {
        original,
        _convert_token_by_mapping(original, _LATIN_TO_CYR),
        _convert_token_by_mapping(original, _CYR_TO_LAT),
    }

    best = original
    best_score = _token_score(original)
    original_score = best_score

    for candidate in candidates:
        if candidate == original:
            continue
        score = _token_score(candidate)
        if score > best_score:
            best = candidate
            best_score = score

    if best == original:
        return original
    if best_score >= original_score + 0.45 and best_score >= 1.6:
        return best
    return original


def _clean_chunk_text(text: str) -> str:
    return normalize_text_block(text)


def normalize_text_block(text: str) -> str:
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
    # Join soft-wrapped lines from OCR/layout where newline does not indicate
    # sentence/paragraph boundary.
    text = _RE_SOFT_LINE_BREAK.sub(" ", text)
    # Language-aware RU/EN normalization for hyphen artifacts from OCR.
    text = _RE_WORD_HYPHEN_WORD.sub(_normalize_hyphenated_words, text)
    # Normalize latin confusable one-letter suffix after Cyrillic hyphenated word.
    text = _RE_CYR_HYPHEN_LATIN_SUFFIX.sub(_normalize_cyr_hyphen_latin_suffix, text)
    # Join "word-<short suffix>" when lexical evidence strongly prefers a joined form.
    text = _RE_WORD_HYPHEN_SHORT_SUFFIX.sub(_normalize_short_suffix_hyphen, text)
    # Normalize intraword hyphen spacing: "что -то" -> "что-то", "северо - запад" -> "северо-запад".
    text = _RE_INNER_HYPHEN_SPACES.sub("-", text)
    # Fix mixed Cyrillic/Latin OCR confusions inside the same token.
    text = _RE_MIXED_SCRIPT_WORD.sub(_normalize_mixed_script_word, text)
    # Run suffix-join once more after mixed-script normalization
    # (e.g. "Носильщик-a" -> "Носильщик-а" -> "Носильщика").
    text = _RE_WORD_HYPHEN_SHORT_SUFFIX.sub(_normalize_short_suffix_hyphen, text)
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


def sanitize_chunks_with_raw(chunks: list[Chunk]) -> list[SanitizedChunk]:
    if not _env_bool("CHUNK_SANITIZE_ENABLED", True):
        return [SanitizedChunk(chunk=src, raw_content=src.content) for src in chunks]

    min_words = _env_int("CHUNK_SANITIZE_MIN_WORDS", 3, min_value=0, max_value=100)
    dedup = _env_bool("CHUNK_SANITIZE_DEDUP", True)

    out: list[SanitizedChunk] = []
    seen: set[str] = set()

    for src in chunks:
        raw_content = src.content
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

        out.append(
            SanitizedChunk(
                chunk=Chunk(page=src.page, ordinal=len(out), content=text),
                raw_content=raw_content,
            )
        )

    return out


def sanitize_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [row.chunk for row in sanitize_chunks_with_raw(chunks)]
