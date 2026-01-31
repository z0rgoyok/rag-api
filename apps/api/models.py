from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # RAG controls (client can ask; server enforces tier)
    rag: bool = True
    citations: bool = False


class ChatCompletionsResponse(BaseModel):
    raw: Any

