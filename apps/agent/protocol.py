"""Protocol definitions and data types for the agentic RAG system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ToolName(str, Enum):
    """Available tools for the agent."""

    SEARCH = "search"
    REFINE_AND_SEARCH = "refine_and_search"
    FINAL_ANSWER = "final_answer"


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the agent."""

    name: ToolName
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Result of a tool execution."""

    tool_name: ToolName
    success: bool
    data: Any
    error: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """A single search result from the knowledge base."""

    content: str
    source: str
    page: int | None
    score: float


@dataclass
class AgentState:
    """Mutable state tracking the agent's reasoning process."""

    original_query: str
    search_history: list[tuple[str, list[SearchResult]]] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    iterations: int = 0
    final_answer: str | None = None

    def add_search(self, query: str, results: list[SearchResult]) -> None:
        self.search_history.append((query, results))

    def add_reasoning(self, step: str) -> None:
        self.reasoning_steps.append(step)

    def get_all_results(self) -> list[SearchResult]:
        """Deduplicated results from all searches, ordered by score."""
        seen: set[str] = set()
        all_results: list[SearchResult] = []
        for _, results in self.search_history:
            for r in results:
                key = f"{r.source}:{r.page}:{r.content[:100]}"
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)
        return sorted(all_results, key=lambda x: x.score, reverse=True)


@dataclass(frozen=True)
class AgentResult:
    """Final result from the agent."""

    answer: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]
    search_count: int
    iterations: int


class Tool(Protocol):
    """Protocol for agent tools."""

    @property
    def name(self) -> ToolName: ...

    @property
    def description(self) -> str: ...

    async def execute(self, arguments: dict[str, Any], state: AgentState) -> ToolResult: ...
