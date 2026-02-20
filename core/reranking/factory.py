from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .protocol import Reranker

RerankingStrategyType = Literal["none", "lmstudio", "cross_encoder", "cohere"]


@dataclass(frozen=True)
class RerankingSettings:
    """Configuration for reranking strategy."""

    strategy: RerankingStrategyType = "none"

    # How many candidates to fetch before reranking (retrieval_k)
    # Reranker will then pick top_k from these
    retrieval_k: int = 50

    # lmstudio settings
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str | None = None
    lmstudio_model: str = "text-embedding-bge-reranker-v2-m3"
    lmstudio_batch_size: int = 16

    # cross_encoder settings
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    cross_encoder_batch_size: int = 32

    # cohere settings
    cohere_api_key: str | None = None
    cohere_model: str = "rerank-english-v3.0"
    cohere_base_url: str = "https://api.cohere.ai"


def build_reranker(settings: RerankingSettings) -> Reranker:
    """Build a reranker from settings.

    Args:
        settings: Reranking configuration.

    Returns:
        Configured reranker instance.

    Raises:
        ValueError: If strategy type is unknown or required config is missing.
    """
    if settings.strategy == "none":
        from .none import NoOpReranker

        return NoOpReranker()

    if settings.strategy == "cross_encoder":
        from .cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(
            model_name=settings.cross_encoder_model,
            batch_size=settings.cross_encoder_batch_size,
        )

    if settings.strategy == "lmstudio":
        from .lmstudio import LmStudioReranker

        return LmStudioReranker(
            base_url=settings.lmstudio_base_url,
            api_key=settings.lmstudio_api_key,
            model=settings.lmstudio_model,
            batch_size=settings.lmstudio_batch_size,
        )

    if settings.strategy == "cohere":
        if not settings.cohere_api_key:
            raise ValueError("RERANKING_COHERE_API_KEY is required for cohere strategy")

        from .cohere import CohereReranker

        return CohereReranker(
            api_key=settings.cohere_api_key,
            model=settings.cohere_model,
            base_url=settings.cohere_base_url,
        )

    raise ValueError(f"Unknown reranking strategy: {settings.strategy}")
