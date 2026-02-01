from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings
from .lmstudio import LmStudioClient


class EmbeddingsClient(Protocol):
    async def embeddings(self, *, model: str, input_texts: list[str], input_type: str | None = None) -> list[list[float]]: ...

    async def probe_embedding_dim(self, *, model: str) -> int: ...


def build_embeddings_client(settings: Settings) -> EmbeddingsClient:
    if settings.embeddings_backend == "litellm":
        return LiteLLMEmbeddingsClient(
            base_url=settings.embeddings_base_url,
            api_key=settings.embeddings_api_key,
            vertex_project=settings.embeddings_vertex_project,
            vertex_location=settings.embeddings_vertex_location,
            vertex_credentials=settings.embeddings_vertex_credentials,
        )
    return OpenAICompatEmbeddingsClient(settings.embeddings_base_url, api_key=settings.embeddings_api_key)


@dataclass(frozen=True)
class OpenAICompatEmbeddingsClient:
    base_url: str
    api_key: str | None

    async def embeddings(self, *, model: str, input_texts: list[str], input_type: str | None = None) -> list[list[float]]:
        _ = input_type  # OpenAI-compatible servers generally ignore embedding task type.
        client = LmStudioClient(self.base_url, api_key=self.api_key)
        return await client.embeddings(model=model, input_texts=input_texts)

    async def probe_embedding_dim(self, *, model: str) -> int:
        client = LmStudioClient(self.base_url, api_key=self.api_key)
        return await client.probe_embedding_dim(model=model)


@dataclass(frozen=True)
class LiteLLMEmbeddingsClient:
    base_url: str
    api_key: str | None
    vertex_project: str | None
    vertex_location: str | None
    vertex_credentials: str | None

    async def embeddings(self, *, model: str, input_texts: list[str], input_type: str | None = None) -> list[list[float]]:
        try:
            import litellm  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("litellm is not installed (required when EMBEDDINGS_BACKEND=litellm)") from e

        # LiteLLM normalizes many providers behind OpenAI-style calls.
        # We still pass provider config explicitly to avoid hidden global auth/config.
        kwargs: dict[str, Any] = {"model": model, "input": input_texts}
        if input_type:
            kwargs["input_type"] = input_type

        if model.startswith("vertex_ai/"):
            if not self.vertex_project or not self.vertex_location:
                raise RuntimeError(
                    "Vertex AI embeddings require EMBEDDINGS_VERTEX_PROJECT and EMBEDDINGS_VERTEX_LOCATION (or VERTEX_PROJECT/VERTEX_LOCATION)."
                )
            kwargs["vertex_project"] = self.vertex_project
            kwargs["vertex_location"] = self.vertex_location
            if self.vertex_credentials:
                kwargs["vertex_credentials"] = self.vertex_credentials
        else:
            kwargs["api_base"] = self.base_url
            kwargs["api_key"] = self.api_key

        resp = await litellm.aembedding(**kwargs)

        data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
        if not data:
            raise RuntimeError("litellm embedding response missing 'data'")
        return [item["embedding"] for item in data]

    async def probe_embedding_dim(self, *, model: str) -> int:
        vectors = await self.embeddings(model=model, input_texts=["dim probe"])
        if not vectors or not vectors[0]:
            raise RuntimeError("Embeddings returned empty vector")
        return len(vectors[0])
