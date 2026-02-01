from __future__ import annotations

from .protocol import RerankResult


class NoOpReranker:
    """Passthrough reranker that preserves original order.

    Used when reranking is disabled. Returns documents in their
    original order with scores based on position (higher = better).
    """

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Return documents in original order with position-based scores."""
        n = len(documents)
        results = [
            RerankResult(index=i, score=1.0 - (i / max(n, 1)))
            for i in range(n)
        ]
        if top_k is not None:
            results = results[:top_k]
        return results
