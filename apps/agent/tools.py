"""Tools available to the agentic RAG system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.db import Db
from core.embeddings_client import EmbeddingsClient
from core.pgvector import vector_literal
from core.db import fetch_all

from .protocol import AgentState, SearchResult, ToolName, ToolResult


@dataclass
class SearchTool:
    """Search the knowledge base using semantic similarity."""

    db: Db
    embed_client: EmbeddingsClient
    embeddings_model: str
    top_k: int = 6

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

        try:
            # Embed the query
            embeddings = await self.embed_client.embeddings(
                model=self.embeddings_model,
                input_texts=[query],
                input_type="RETRIEVAL_QUERY",
            )
            query_vec = embeddings[0]

            # Search
            results = self._search(query_vec)

            # Update state
            state.add_search(query, results)

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=results,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=[],
                error=str(e),
            )

    def _search(self, query_embedding: list[float]) -> list[SearchResult]:
        with self.db.connect() as conn:
            q = vector_literal(query_embedding)
            rows = fetch_all(
                conn,
                """
                select
                  s.content,
                  d.source_path,
                  d.title,
                  s.page,
                  (1 - (e.embedding <=> %(q)s::vector)) as score
                from segment_embeddings e
                join segments s on s.id = e.segment_id
                join documents d on d.id = s.document_id
                order by (e.embedding <=> %(q)s::vector) + 0
                limit %(k)s
                """,
                {"q": q, "k": self.top_k},
            )
            return [
                SearchResult(
                    content=r["content"],
                    source=r["source_path"],
                    page=r["page"],
                    score=float(r["score"]),
                )
                for r in rows
            ]


@dataclass
class RefineAndSearchTool:
    """Refine query and search - useful when initial search was insufficient."""

    db: Db
    embed_client: EmbeddingsClient
    embeddings_model: str
    top_k: int = 6

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

        try:
            embeddings = await self.embed_client.embeddings(
                model=self.embeddings_model,
                input_texts=[refined_query],
                input_type="RETRIEVAL_QUERY",
            )
            query_vec = embeddings[0]
            results = self._search(query_vec)
            state.add_search(refined_query, results)

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=results,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=[],
                error=str(e),
            )

    def _search(self, query_embedding: list[float]) -> list[SearchResult]:
        with self.db.connect() as conn:
            q = vector_literal(query_embedding)
            rows = fetch_all(
                conn,
                """
                select
                  s.content,
                  d.source_path,
                  d.title,
                  s.page,
                  (1 - (e.embedding <=> %(q)s::vector)) as score
                from segment_embeddings e
                join segments s on s.id = e.segment_id
                join documents d on d.id = s.document_id
                order by (e.embedding <=> %(q)s::vector) + 0
                limit %(k)s
                """,
                {"q": q, "k": self.top_k},
            )
            return [
                SearchResult(
                    content=r["content"],
                    source=r["source_path"],
                    page=r["page"],
                    score=float(r["score"]),
                )
                for r in rows
            ]


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
