"""Tools available to the agentic RAG system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.embeddings_client import EmbeddingsClient
from core.qdrant import Qdrant
from core.vector_search import search_segments

from .protocol import AgentState, SearchResult, ToolName, ToolResult


async def _embed_and_search(
    *,
    qdrant: Qdrant,
    embed_client: EmbeddingsClient,
    embeddings_model: str,
    query_text: str,
    top_k: int,
    use_fts: bool,
) -> list[SearchResult]:
    embeddings = await embed_client.embeddings(
        model=embeddings_model,
        input_texts=[query_text],
        input_type="RETRIEVAL_QUERY",
    )
    query_vec = embeddings[0]
    rows = search_segments(
        qdrant,
        query_text=query_text,
        query_embedding=query_vec,
        k=top_k,
        use_fts=use_fts,
    )
    return [
        SearchResult(
            content=r.content,
            source=r.source_path,
            page=r.page,
            score=float(r.score),
        )
        for r in rows
    ]


async def _execute_query_search(
    *,
    tool_name: ToolName,
    query_text: str,
    state: AgentState,
    qdrant: Qdrant,
    embed_client: EmbeddingsClient,
    embeddings_model: str,
    top_k: int,
    use_fts: bool,
) -> ToolResult:
    try:
        results = await _embed_and_search(
            qdrant=qdrant,
            embed_client=embed_client,
            embeddings_model=embeddings_model,
            query_text=query_text,
            top_k=top_k,
            use_fts=use_fts,
        )
        state.add_search(query_text, results)
        return ToolResult(
            tool_name=tool_name,
            success=True,
            data=results,
        )
    except Exception as e:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            data=[],
            error=str(e),
        )


@dataclass
class SearchTool:
    """Search the knowledge base using semantic similarity."""

    qdrant: Qdrant
    embed_client: EmbeddingsClient
    embeddings_model: str
    top_k: int = 6
    use_fts: bool = True

    @property
    def name(self) -> ToolName:
        return ToolName.SEARCH

    @property
    def description(self) -> str:
        return (
            "Search the knowledge base for information relevant to a query. "
            "Use this when you need to find facts, definitions, or specific information."
        )

    async def execute(self, arguments: dict[str, Any], state: AgentState) -> ToolResult:
        query = arguments.get("query", "").strip()
        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=[],
                error="Query cannot be empty",
            )

        return await _execute_query_search(
            tool_name=self.name,
            query_text=query,
            state=state,
            qdrant=self.qdrant,
            embed_client=self.embed_client,
            embeddings_model=self.embeddings_model,
            top_k=self.top_k,
            use_fts=self.use_fts,
        )


@dataclass
class RefineAndSearchTool:
    """Refine query and search - useful when initial search was insufficient."""

    qdrant: Qdrant
    embed_client: EmbeddingsClient
    embeddings_model: str
    top_k: int = 6
    use_fts: bool = True

    @property
    def name(self) -> ToolName:
        return ToolName.REFINE_AND_SEARCH

    @property
    def description(self) -> str:
        return (
            "Reformulate the query to search from a different angle. "
            "Use when initial results don't fully answer the question. "
            "Provide a refined_query that approaches the topic differently."
        )

    async def execute(self, arguments: dict[str, Any], state: AgentState) -> ToolResult:
        refined_query = arguments.get("refined_query", "").strip()
        if not refined_query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=[],
                error="refined_query cannot be empty",
            )

        state.add_reasoning(f"Refining search with: {refined_query}")

        return await _execute_query_search(
            tool_name=self.name,
            query_text=refined_query,
            state=state,
            qdrant=self.qdrant,
            embed_client=self.embed_client,
            embeddings_model=self.embeddings_model,
            top_k=self.top_k,
            use_fts=self.use_fts,
        )


@dataclass
class FinalAnswerTool:
    """Provide the final answer to the user."""

    @property
    def name(self) -> ToolName:
        return ToolName.FINAL_ANSWER

    @property
    def description(self) -> str:
        return (
            "Provide the final answer to the user's question. "
            "Use this when you have gathered enough information to answer."
        )

    async def execute(self, arguments: dict[str, Any], state: AgentState) -> ToolResult:
        answer = arguments.get("answer", "").strip()
        if not answer:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=None,
                error="Answer cannot be empty",
            )

        state.final_answer = answer
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=answer,
        )


def get_tools_schema() -> list[dict[str, Any]]:
    """Return OpenAI-compatible tool definitions for the agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the knowledge base for information relevant to a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "refine_and_search",
                "description": "Search with a reformulated query when initial results are insufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "refined_query": {
                            "type": "string",
                            "description": "A reformulated query approaching the topic from a different angle",
                        },
                    },
                    "required": ["refined_query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Provide the final answer when you have enough information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The complete answer to the user's question",
                        },
                    },
                    "required": ["answer"],
                },
            },
        },
    ]
