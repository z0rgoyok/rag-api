from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RerankResult:
    """Result of reranking a single document."""

    index: int
    score: float


class Reranker(Protocol):
    """Protocol for reranking strategies.

    Rerankers take a query and a list of documents, then return
    relevance scores for each document, typically using a cross-encoder
    model that sees both query and document together.
    """

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: If specified, return only top K results. If None, return all.

        Returns:
            List of RerankResult sorted by score descending.
            Each result contains the original index and the relevance score.
        """
        ...
