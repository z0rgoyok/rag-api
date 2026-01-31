from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .db import Db, fetch_all
from .pgvector import vector_literal


@dataclass(frozen=True)
class RetrievedSegment:
    content: str
    source_path: str
    title: str
    page: int | None
    score: float


def retrieve_top_k(db: Db, *, query_embedding: list[float], k: int) -> list[RetrievedSegment]:
    with db.connect() as conn:
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
            order by e.embedding <=> %(q)s::vector
            limit %(k)s
            """,
            {"q": q, "k": k},
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
