from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .protocol import RerankResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> "CrossEncoder":
    """Load cross-encoder model (cached to avoid reloading)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


@dataclass
class CrossEncoderReranker:
    """Reranker using sentence-transformers CrossEncoder.

    Uses a cross-encoder model to score query-document pairs.
    The model sees both texts together, enabling better relevance judgment
    than bi-encoder similarity.

    Popular models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better quality)
    - BAAI/bge-reranker-base (good multilingual support)
    - BAAI/bge-reranker-large (best quality, slower)
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _rerank_sync(
        self,
        query: str,
        documents: list[str],
        top_k: int | None,
    ) -> list[RerankResult]:
        """Synchronous reranking (runs in thread pool)."""
        if not documents:
            return []

        model = _load_model(self.model_name)

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get scores from cross-encoder
        scores = model.predict(pairs, batch_size=self.batch_size)

        # Create results with original indices
        results = [
            RerankResult(index=i, score=float(score))
            for i, score in enumerate(scores)
        ]

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents using cross-encoder model.

        Runs inference in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._rerank_sync,
            query,
            documents,
            top_k,
        )
