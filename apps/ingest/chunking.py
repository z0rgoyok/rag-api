from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    page: int | None
    ordinal: int
    content: str


def chunk_text_pages(pages: list[tuple[int, str]], *, max_chars: int = 2000, overlap_chars: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0

    for page_num, page_text in pages:
        text = (page_text or "").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(page=page_num, ordinal=ordinal, content=piece))
                ordinal += 1
            if end == len(text):
                break
            start = max(0, end - overlap_chars)

    return chunks

