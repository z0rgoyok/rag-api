from __future__ import annotations

from dataclasses import dataclass

import httpx

from .protocol import RerankResult


@dataclass
class HttpReranker:
    """Reranker proxy that calls an external HTTP `/v1/rerank` service."""

    base_url: str
    api_key: str | None
    model: str
    batch_size: int = 64
    timeout: float = 120.0

    async def _rerank_batch(
        self,
        *,
        query: str,
        documents: list[str],
        top_k: int | None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, object] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_k is not None:
            payload["top_n"] = top_k

        url = f"{self.base_url.rstrip('/')}/v1/rerank"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw_results: object = data.get("results") if isinstance(data, dict) else data
        if not isinstance(raw_results, list):
            raise RuntimeError("Invalid reranker response: expected list under `results`")

        out: list[RerankResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if isinstance(idx, int) and isinstance(score, (int, float)):
                out.append(RerankResult(index=idx, score=float(score)))

        out.sort(key=lambda r: r.score, reverse=True)
        return out

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        # Batch transport requests to keep payloads manageable for host services.
        if len(documents) <= max(1, self.batch_size):
            out = await self._rerank_batch(query=query, documents=documents, top_k=top_k)
            return out[:top_k] if top_k is not None else out

        merged: list[RerankResult] = []
        step = max(1, self.batch_size)
        for start in range(0, len(documents), step):
            part = documents[start : start + step]
            part_results = await self._rerank_batch(query=query, documents=part, top_k=None)
            for item in part_results:
                merged.append(RerankResult(index=start + item.index, score=item.score))

        merged.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            return merged[:top_k]
        return merged
