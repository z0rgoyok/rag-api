from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PageText:
    """A single page of text from a document."""

    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A chunk of text extracted from a document."""

    page: int | None
    ordinal: int
    content: str


class ChunkingStrategy(Protocol):
    """Protocol for text chunking strategies."""

    def chunk(self, pages: list[PageText]) -> list[Chunk]:
        """Split pages into chunks.

        Args:
            pages: List of pages with their text content.

        Returns:
            List of chunks with page reference and ordinal.
        """
        ...
