from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .protocol import Chunk, ChunkingStrategy, PageText


@dataclass(frozen=True)
class SemanticStrategy(ChunkingStrategy):
    """Semantic chunker using chonkie.

    Splits text where embedding similarity between consecutive
    sentences drops below threshold. Best quality for books and
    long-form content where preserving semantic coherence matters.

    Requires: pip install chonkie[semantic]
    """

    chunk_size: int = 512
    similarity_threshold: float = 0.5
    _chunker: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._chunker is None:
            from chonkie import SemanticChunker

            chunker = SemanticChunker(
                chunk_size=self.chunk_size,
                similarity_threshold=self.similarity_threshold,
            )
            object.__setattr__(self, "_chunker", chunker)

    def chunk(self, pages: list[PageText]) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0

        for page in pages:
            text = (page.text or "").strip()
            if not text:
                continue

            for ch in self._chunker(text):
                content = ch.text.strip() if hasattr(ch, "text") else str(ch).strip()
                if content:
                    chunks.append(Chunk(page=page.page, ordinal=ordinal, content=content))
                    ordinal += 1

        return chunks
