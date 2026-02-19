from __future__ import annotations

from dataclasses import dataclass
import os


from typing import Literal

ChunkingStrategyType = Literal[
    "sliding",
    "recursive",
    "semantic",
    "docling_hierarchical",
    "docling_hybrid",
]
RerankingStrategyType = Literal["none", "cross_encoder", "cohere"]


@dataclass(frozen=True)
class Settings:
    database_url: str
    chat_backend: str
    chat_base_url: str
    chat_api_key: str | None
    chat_model: str
    chat_vertex_project: str | None
    chat_vertex_location: str | None
    chat_vertex_credentials: str | None
    embeddings_backend: str
    embeddings_base_url: str
    embeddings_api_key: str | None
    embeddings_model: str
    embeddings_vertex_project: str | None
    embeddings_vertex_location: str | None
    embeddings_vertex_credentials: str | None
    embedding_dim: int | None
    top_k: int
    max_context_chars: int
    retrieval_use_fts: bool
    allow_anonymous: bool
    # Chunking settings
    chunking_strategy: ChunkingStrategyType
    chunking_chunk_size: int
    chunking_overlap_chars: int
    chunking_similarity_threshold: float


def load_settings() -> Settings:
    def _bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    # Provider-neutral env (works for LM Studio, OpenAI, etc.) while keeping legacy LMSTUDIO_*.
    default_base_url = (os.getenv("INFERENCE_BASE_URL") or os.getenv("LMSTUDIO_BASE_URL") or "http://localhost:1234/v1").rstrip("/")
    default_api_key = os.getenv("INFERENCE_API_KEY") or os.getenv("LMSTUDIO_API_KEY") or None

    chat_backend = (os.getenv("CHAT_BACKEND") or "openai_compat").strip().lower()
    chat_base_url = (os.getenv("CHAT_BASE_URL") or default_base_url).rstrip("/")
    chat_api_key = os.getenv("CHAT_API_KEY") or default_api_key
    chat_model = os.getenv("CHAT_MODEL") or os.getenv("INFERENCE_CHAT_MODEL") or os.getenv("LMSTUDIO_CHAT_MODEL") or "local-model"
    chat_vertex_project = os.getenv("CHAT_VERTEX_PROJECT") or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or None
    chat_vertex_location = os.getenv("CHAT_VERTEX_LOCATION") or os.getenv("VERTEX_LOCATION") or None
    chat_vertex_credentials = os.getenv("CHAT_VERTEX_CREDENTIALS") or os.getenv("VERTEX_CREDENTIALS") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or None

    embeddings_base_url = (os.getenv("EMBEDDINGS_BASE_URL") or default_base_url).rstrip("/")
    embeddings_api_key = os.getenv("EMBEDDINGS_API_KEY") or default_api_key
    embeddings_model = os.getenv("EMBEDDINGS_MODEL") or os.getenv("INFERENCE_EMBEDDING_MODEL") or os.getenv("LMSTUDIO_EMBEDDING_MODEL") or "local-embedding-model"
    embeddings_backend = (os.getenv("EMBEDDINGS_BACKEND") or "openai_compat").strip().lower()
    embeddings_vertex_project = os.getenv("EMBEDDINGS_VERTEX_PROJECT") or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or None
    embeddings_vertex_location = os.getenv("EMBEDDINGS_VERTEX_LOCATION") or os.getenv("VERTEX_LOCATION") or None
    embeddings_vertex_credentials = os.getenv("EMBEDDINGS_VERTEX_CREDENTIALS") or os.getenv("VERTEX_CREDENTIALS") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or None

    # Chunking settings
    chunking_strategy_raw = os.getenv("CHUNKING_STRATEGY", "semantic").strip().lower()
    if chunking_strategy_raw not in (
        "sliding",
        "recursive",
        "semantic",
        "docling_hierarchical",
        "docling_hybrid",
    ):
        chunking_strategy_raw = "semantic"
    chunking_strategy: ChunkingStrategyType = chunking_strategy_raw  # type: ignore[assignment]

    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:56473/rag"),
        chat_backend=chat_backend,
        chat_base_url=chat_base_url,
        chat_api_key=chat_api_key,
        chat_model=chat_model,
        chat_vertex_project=chat_vertex_project,
        chat_vertex_location=chat_vertex_location,
        chat_vertex_credentials=chat_vertex_credentials,
        embeddings_backend=embeddings_backend,
        embeddings_base_url=embeddings_base_url,
        embeddings_api_key=embeddings_api_key,
        embeddings_model=embeddings_model,
        embeddings_vertex_project=embeddings_vertex_project,
        embeddings_vertex_location=embeddings_vertex_location,
        embeddings_vertex_credentials=embeddings_vertex_credentials,
        embedding_dim=int(os.environ["EMBEDDING_DIM"]) if os.getenv("EMBEDDING_DIM") else None,
        top_k=int(os.getenv("TOP_K", "6")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "24000")),
        retrieval_use_fts=_bool("RETRIEVAL_USE_FTS", True),
        allow_anonymous=_bool("ALLOW_ANONYMOUS", False),
        chunking_strategy=chunking_strategy,
        chunking_chunk_size=int(os.getenv("CHUNKING_CHUNK_SIZE", "512")),
        chunking_overlap_chars=int(os.getenv("CHUNKING_OVERLAP_CHARS", "200")),
        chunking_similarity_threshold=float(os.getenv("CHUNKING_SIMILARITY_THRESHOLD", "0.5")),
    )
