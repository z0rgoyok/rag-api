from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.db import Db, fetch_all
from core.pgvector import vector_literal

_HYBRID_CANDIDATE_MULTIPLIER = 8
_HYBRID_MIN_CANDIDATES = 50
_HYBRID_RRF_K = 60


@dataclass(frozen=True)
class RetrievedSegment:
    content: str
    source_path: str
    title: str
    page: int | None
    score: float


def retrieve_top_k(
    db: Db,
    *,
    query_text: str,
    query_embedding: list[float],
    k: int,
    use_fts: bool,
) -> list[RetrievedSegment]:
    with db.connect() as conn:
        q = vector_literal(query_embedding)
        rows = fetch_all(
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
                row_number() over (order by (e.embedding <=> %(q)s::vector) + 0) as vec_rank
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
                f.fts_rank
              from vector_hits v
              full join fts_hits f on f.segment_id = v.segment_id
            ),
            ranked as (
              select
                m.segment_id,
                coalesce(1.0 / ((select rrf_k from cfg) + m.vec_rank), 0.0)
                + coalesce(1.0 / ((select rrf_k from cfg) + m.fts_rank), 0.0) as hybrid_score
              from merged m
            )
            select
              s.content,
              d.source_path,
              d.title,
              s.page,
              r.hybrid_score as score
            from ranked r
            join segments s on s.id = r.segment_id
            join documents d on d.id = s.document_id
            order by r.hybrid_score desc, s.id
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
        out: list[RetrievedSegment] = []
        for r in rows:
            out.append(
                RetrievedSegment(
                    content=r["content"],
                    source_path=r["source_path"],
                    title=r["title"],
                    page=r["page"],
                    score=float(r["score"]),
                )
            )
        return out


def build_context(segments: list[RetrievedSegment], *, max_chars: int, include_sources: bool) -> tuple[str, list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    parts: list[str] = []
    remaining = max_chars

    for seg in segments:
        snippet = seg.content.strip()
        if not snippet:
            continue
        if len(snippet) > remaining:
            snippet = snippet[: max(0, remaining)]
        if not snippet:
            break

        if include_sources:
            label = seg.title
            if seg.page is not None:
                label = f"{label} (page {seg.page})"
            parts.append(f"[SOURCE] {label}\n{snippet}\n")
            sources.append(
                {
                    "title": seg.title,
                    "path": seg.source_path,
                    "page": seg.page,
                    "score": seg.score,
                }
            )
        else:
            parts.append(f"{snippet}\n")

        remaining -= len(snippet)
        if remaining <= 0:
            break

    return "\n".join(parts).strip(), sources
