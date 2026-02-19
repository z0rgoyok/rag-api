"""Tools available to the agentic RAG system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.db import Db
from core.embeddings_client import EmbeddingsClient
from core.pgvector import vector_literal
from core.db import fetch_all

from .protocol import AgentState, SearchResult, ToolName, ToolResult

_HYBRID_CANDIDATE_MULTIPLIER = 8
_HYBRID_MIN_CANDIDATES = 50
_HYBRID_RRF_K = 60


async def _embed_and_search(
    *,
    db: Db,
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
    rows = _hybrid_search_rows(
        db=db,
        query_text=query_text,
        query_embedding=query_vec,
        k=top_k,
        use_fts=use_fts,
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


async def _execute_query_search(
    *,
    tool_name: ToolName,
    query_text: str,
    state: AgentState,
    db: Db,
    embed_client: EmbeddingsClient,
    embeddings_model: str,
    top_k: int,
    use_fts: bool,
) -> ToolResult:
    try:
        results = await _embed_and_search(
            db=db,
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


def _hybrid_search_rows(
    *,
    db: Db,
    query_text: str,
    query_embedding: list[float],
    k: int,
    use_fts: bool,
) -> list[dict[str, Any]]:
    with db.connect() as conn:
        q = vector_literal(query_embedding)
        return fetch_all(
            conn,
            """
            with cfg as (
              select
                greatest(%(k)s * %(cand_mult)s, %(cand_min)s)::integer as cand_k,
                %(rrf_k)s::float8 as rrf_k
            ),
            vector_hits as (
              select
                s.id as segment_id,
                row_number() over (order by (e.embedding <=> %(q)s::vector) + 0) as vec_rank,
                (1 - (e.embedding <=> %(q)s::vector))::float8 as vec_score
              from segment_embeddings e
              join segments s on s.id = e.segment_id
              order by (e.embedding <=> %(q)s::vector) + 0
              limit (select cand_k from cfg)
            ),
            fts_terms as (
              select term
              from regexp_split_to_table(lower(%(query_text)s), E'[^[:alnum:]а-яё]+') as term
              where %(use_fts)s
                and length(term) >= 4
                and term not in (
                  'это', 'этот', 'эта', 'эти',
                  'такое', 'какой', 'какая', 'какие',
                  'where', 'what', 'which', 'how', 'why', 'when',
                  'this', 'that', 'is', 'are', 'the'
                )
            ),
            fts_input as (
              select
                case
                  when count(*) = 0 then ''::tsquery
                  else to_tsquery('simple', string_agg(term, ' | '))
                end as tsq
              from fts_terms
            ),
            fts_hits as (
              select
                s.id as segment_id,
                row_number() over (
                  order by ts_rank_cd(s.tsv, f.tsq) desc, s.id
                ) as fts_rank
              from segments s
              cross join fts_input f
              where f.tsq <> ''::tsquery and s.tsv @@ f.tsq
              order by ts_rank_cd(s.tsv, f.tsq) desc, s.id
              limit (select cand_k from cfg)
            ),
            merged as (
              select
                coalesce(v.segment_id, f.segment_id) as segment_id,
                v.vec_rank,
                v.vec_score,
                f.fts_rank
              from vector_hits v
              full join fts_hits f on f.segment_id = v.segment_id
            ),
            ranked as (
              select
                m.segment_id,
                case
                  when %(use_fts)s then
                    coalesce(1.0 / ((select rrf_k from cfg) + m.vec_rank), 0.0)
                    + coalesce(1.0 / ((select rrf_k from cfg) + m.fts_rank), 0.0)
                  else coalesce(m.vec_score, -1.0)
                end as score
              from merged m
            )
            select
              s.content,
              d.source_path,
              d.title,
              s.page,
              r.score as score
            from ranked r
            join segments s on s.id = r.segment_id
            join documents d on d.id = s.document_id
            order by r.score desc, s.id
            limit %(k)s
            """,
            {
                "q": q,
                "k": k,
                "query_text": query_text,
                "use_fts": use_fts,
                "cand_mult": _HYBRID_CANDIDATE_MULTIPLIER,
                "cand_min": _HYBRID_MIN_CANDIDATES,
                "rrf_k": _HYBRID_RRF_K,
            },
        )


@dataclass
class SearchTool:
    """Search the knowledge base using semantic similarity."""

    db: Db
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
            db=self.db,
            embed_client=self.embed_client,
            embeddings_model=self.embeddings_model,
            top_k=self.top_k,
            use_fts=self.use_fts,
        )

@dataclass
class RefineAndSearchTool:
    """Refine query and search - useful when initial search was insufficient."""

    db: Db
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
            db=self.db,
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
