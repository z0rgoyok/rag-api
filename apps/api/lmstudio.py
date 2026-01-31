from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx


@dataclass(frozen=True)
class LmStudioClient:
    base_url: str
    api_key: str | None = None

    def _headers(self) -> dict[str, str] | None:
        if not self.api_key:
            return None
        return {"Authorization": f"Bearer {self.api_key}"}

    async def embeddings(self, *, model: str, input_texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(120.0, connect=5.0),
        ) as client:
            r = await client.post("/embeddings", json={"model": model, "input": input_texts})
            r.raise_for_status()
            data = r.json()
            return [item["embedding"] for item in data["data"]]

    async def chat_completions(self, payload: dict[str, Any], *, timeout_s: float = 120.0) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        ) as client:
            r = await client.post("/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()

    async def stream_chat_completions(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(None, connect=5.0),
        ) as client:
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
