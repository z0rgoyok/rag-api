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


class AgentChatRequest(BaseModel):
    """Request for agentic RAG endpoint."""

    query: str = Field(..., min_length=1, description="The user's question")
    max_iterations: int = Field(default=3, ge=1, le=10, description="Max agent iterations")
    citations: bool = Field(default=False, description="Include source citations")


class AgentChatResponse(BaseModel):
    """Response from agentic RAG endpoint."""

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    search_count: int
    iterations: int

