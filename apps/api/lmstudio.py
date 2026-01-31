from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx


@dataclass(frozen=True)
class LmStudioClient:
    base_url: str

    async def embeddings(self, *, model: str, input_texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            r = await client.post("/embeddings", json={"model": model, "input": input_texts})
            r.raise_for_status()
            data = r.json()
            return [item["embedding"] for item in data["data"]]

    async def stream_chat_completions(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=None) as client:
            async with client.stream("POST", "/chat/completions", json=payload) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    if chunk:
                        yield chunk

    async def probe_embedding_dim(self, *, model: str) -> int:
        vectors = await self.embeddings(model=model, input_texts=["dim probe"])
        if not vectors or not vectors[0]:
            raise RuntimeError("LM Studio embeddings returned empty vector")
        return len(vectors[0])
