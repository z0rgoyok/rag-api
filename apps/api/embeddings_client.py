from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings
from .lmstudio import LmStudioClient


class EmbeddingsClient(Protocol):
    async def embeddings(self, *, model: str, input_texts: list[str]) -> list[list[float]]: ...

    async def probe_embedding_dim(self, *, model: str) -> int: ...


def build_embeddings_client(settings: Settings) -> EmbeddingsClient:
    if settings.embeddings_backend == "litellm":
        return LiteLLMEmbeddingsClient(base_url=settings.embeddings_base_url, api_key=settings.embeddings_api_key)
    return LmStudioClient(settings.embeddings_base_url, api_key=settings.embeddings_api_key)


@dataclass(frozen=True)
class LiteLLMEmbeddingsClient:
    base_url: str
    api_key: str | None

    async def embeddings(self, *, model: str, input_texts: list[str]) -> list[list[float]]:
        try:
            import litellm  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("litellm is not installed (required when EMBEDDINGS_BACKEND=litellm)") from e

        # LiteLLM normalizes many providers behind OpenAI-style calls.
        # We still pass base_url/api_key explicitly to avoid hidden global auth.
        resp: Any = await litellm.aembedding(model=model, input=input_texts, api_base=self.base_url, api_key=self.api_key)

        data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
        if not data:
            raise RuntimeError("litellm embedding response missing 'data'")
        return [item["embedding"] for item in data]

    async def probe_embedding_dim(self, *, model: str) -> int:
        vectors = await self.embeddings(model=model, input_texts=["dim probe"])
        if not vectors or not vectors[0]:
            raise RuntimeError("Embeddings returned empty vector")
        return len(vectors[0])

