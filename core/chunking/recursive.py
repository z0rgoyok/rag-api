from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .protocol import Chunk, ChunkingStrategy, PageText


@dataclass(frozen=True)
class RecursiveStrategy(ChunkingStrategy):
    """Recursive text splitter using semchunk.

    Splits text by trying separators in priority order:
    paragraphs -> newlines -> sentences -> words -> characters.
    Respects natural text boundaries without requiring embeddings.

    Good balance between speed and quality for books/documents.
    """

    chunk_size: int = 512
    _chunker: Callable[[str], list[str]] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._chunker is None:
            import semchunk

            chunker = semchunk.chunkerify(
                lambda text: len(text.split()),  # word-based tokenizer fallback
                chunk_size=self.chunk_size,
            )
            object.__setattr__(self, "_chunker", chunker)

    def chunk(self, pages: list[PageText]) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0

        for page in pages:
            text = (page.text or "").strip()
            if not text:
                continue

            for piece in self._chunker(text):
                piece = piece.strip()
                if piece:
                    chunks.append(Chunk(page=page.page, ordinal=ordinal, content=piece))
                    ordinal += 1

        return chunks
