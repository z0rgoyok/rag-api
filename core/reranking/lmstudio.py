from __future__ import annotations

from dataclasses import dataclass
import math

from core.lmstudio import LmStudioClient

from .protocol import RerankResult


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    n = min(len(a), len(b))
    if n == 0:
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0.0 or nb <= 0.0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@dataclass(frozen=True)
class LmStudioReranker:
    """Rerank by cosine similarity using an LM Studio embedding model."""

    base_url: str
    api_key: str | None
    model: str
    batch_size: int = 16

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        client = LmStudioClient(base_url=self.base_url, api_key=self.api_key)
        query_vecs = await client.embeddings(model=self.model, input_texts=[query])
        if not query_vecs:
            return []
        query_vec = query_vecs[0]

        doc_vecs: list[list[float]] = []
        step = max(1, self.batch_size)
        for i in range(0, len(documents), step):
            batch = documents[i : i + step]
            part = await client.embeddings(model=self.model, input_texts=batch)
            doc_vecs.extend(part)

        results: list[RerankResult] = []
        for idx, vec in enumerate(doc_vecs):
            results.append(RerankResult(index=idx, score=_cosine(query_vec, vec)))

        results.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            return results[:top_k]
        return results

