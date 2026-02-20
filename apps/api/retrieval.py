from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.qdrant import Qdrant
from core.vector_search import search_segments


@dataclass(frozen=True)
class RetrievedSegment:
    content: str
    source_path: str
    title: str
    page: int | None
    score: float


def retrieve_top_k(
    qdrant: Qdrant,
    *,
    query_text: str,
    query_embedding: list[float],
    k: int,
    use_fts: bool,
) -> list[RetrievedSegment]:
    rows = search_segments(
        qdrant,
        query_text=query_text,
        query_embedding=query_embedding,
        k=k,
        use_fts=use_fts,
    )
    return [
        RetrievedSegment(
            content=row.content,
            source_path=row.source_path,
            title=row.title,
            page=row.page,
            score=row.score,
        )
        for row in rows
    ]


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
