from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .protocol import RerankResult

log = logging.getLogger(__name__)


@dataclass
class CohereReranker:
    """Reranker using Cohere Rerank API.

    Uses Cohere's hosted reranking service for high-quality relevance scoring.
    Requires a Cohere API key.

    Models:
    - rerank-english-v3.0 (English, best quality)
    - rerank-multilingual-v3.0 (100+ languages)
    - rerank-english-v2.0 (legacy)
    """

    api_key: str
    model: str = "rerank-english-v3.0"
    base_url: str = "https://api.cohere.ai"
    timeout: float = 30.0

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents using Cohere Rerank API."""
        if not documents:
            return []

        url = f"{self.base_url.rstrip('/')}/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if top_k is not None:
            payload["top_n"] = top_k

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = [
            RerankResult(
                index=item["index"],
                score=float(item["relevance_score"]),
            )
            for item in data.get("results", [])
        ]

        # API returns sorted by relevance, but ensure it
        results.sort(key=lambda r: r.score, reverse=True)
        return results
