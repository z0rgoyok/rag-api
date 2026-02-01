from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .protocol import ChunkingStrategy


ChunkingStrategyType = Literal["sliding", "recursive", "semantic"]


@dataclass(frozen=True)
class ChunkingSettings:
    """Configuration for chunking strategy."""

    strategy: ChunkingStrategyType = "recursive"
    chunk_size: int = 512
    overlap_chars: int = 200  # only for sliding
    similarity_threshold: float = 0.5  # only for semantic


def build_chunking_strategy(settings: ChunkingSettings) -> ChunkingStrategy:
    """Build a chunking strategy from settings.

    Args:
        settings: Chunking configuration.

    Returns:
        Configured chunking strategy instance.

    Raises:
        ValueError: If strategy type is unknown.
    """
    if settings.strategy == "sliding":
        from .sliding import SlidingWindowStrategy

        return SlidingWindowStrategy(
            max_chars=settings.chunk_size,
            overlap_chars=settings.overlap_chars,
        )

    if settings.strategy == "recursive":
        from .recursive import RecursiveStrategy

        return RecursiveStrategy(chunk_size=settings.chunk_size)

    if settings.strategy == "semantic":
        from .semantic import SemanticStrategy

        return SemanticStrategy(
            chunk_size=settings.chunk_size,
            similarity_threshold=settings.similarity_threshold,
        )

    raise ValueError(f"Unknown chunking strategy: {settings.strategy}")
