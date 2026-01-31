from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_url: str
    lmstudio_base_url: str
    lmstudio_chat_model: str
    lmstudio_embedding_model: str
    embedding_dim: int | None
    top_k: int
    max_context_chars: int
    allow_anonymous: bool


def load_settings() -> Settings:
    def _bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:56473/rag"),
        lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/"),
        lmstudio_chat_model=os.getenv("LMSTUDIO_CHAT_MODEL", "local-model"),
        lmstudio_embedding_model=os.getenv("LMSTUDIO_EMBEDDING_MODEL", "local-embedding-model"),
        embedding_dim=int(os.environ["EMBEDDING_DIM"]) if os.getenv("EMBEDDING_DIM") else None,
        top_k=int(os.getenv("TOP_K", "6")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "24000")),
        allow_anonymous=_bool("ALLOW_ANONYMOUS", False),
    )
