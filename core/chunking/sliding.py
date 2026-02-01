from __future__ import annotations

from dataclasses import dataclass

from .protocol import Chunk, ChunkingStrategy, PageText


@dataclass(frozen=True)
class SlidingWindowStrategy(ChunkingStrategy):
    """Fixed-size sliding window chunker.

    Simple character-based chunking with overlap. Fast but may cut
    mid-sentence. Suitable for quick processing where semantic
    boundaries are not critical.
    """

    max_chars: int = 2000
    overlap_chars: int = 200

    def chunk(self, pages: list[PageText]) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0

        for page in pages:
            text = (page.text or "").strip()
            if not text:
                continue

            start = 0
            while start < len(text):
                end = min(len(text), start + self.max_chars)
                piece = text[start:end].strip()
                if piece:
                    chunks.append(Chunk(page=page.page, ordinal=ordinal, content=piece))
                    ordinal += 1
                if end == len(text):
                    break
                start = max(0, end - self.overlap_chars)

        return chunks
