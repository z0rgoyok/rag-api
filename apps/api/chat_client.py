from __future__ import annotations

from typing import Any, AsyncIterator, Protocol

from core.config import Settings
from core.lmstudio import LmStudioClient


class ChatClient(Protocol):
    async def chat_completions(self, payload: dict[str, Any], *, timeout_s: float = 120.0) -> dict[str, Any]: ...

    async def stream_chat_completions(self, payload: dict[str, Any]) -> AsyncIterator[bytes]: ...


def build_chat_client(settings: Settings) -> ChatClient:
    backend = (settings.chat_backend or "openai_compat").strip().lower()
    if backend == "litellm":
        from .litellm_chat import LiteLLMChatClient

        return LiteLLMChatClient(
            api_key=settings.chat_api_key,
            vertex_project=settings.chat_vertex_project,
            vertex_location=settings.chat_vertex_location,
            vertex_credentials=settings.chat_vertex_credentials,
        )
    return LmStudioClient(settings.chat_base_url, api_key=settings.chat_api_key)
