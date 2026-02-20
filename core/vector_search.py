from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, cast

from core.qdrant import Qdrant

_HYBRID_CANDIDATE_MULTIPLIER = 8
_HYBRID_MIN_CANDIDATES = 50
_HYBRID_RRF_K = 60
_STOPWORDS = {
    "это",
    "этот",
    "эта",
    "эти",
    "такое",
    "какой",
    "какая",
    "какие",
    "where",
    "what",
    "which",
    "how",
    "why",
    "when",
    "this",
    "that",
    "is",
    "are",
    "the",
}


@dataclass(frozen=True)
class VectorSearchRow:
    content: str
    source_path: str
    title: str
    page: int | None
    score: float


def _extract_terms(query_text: str) -> list[str]:
    terms: list[str] = []
    for raw in re.split(r"[^0-9A-Za-zА-Яа-яЁё]+", query_text.lower()):
        token = raw.strip()
        if len(token) < 4:
            continue
        if token in _STOPWORDS:
            continue
        terms.append(token)
    return terms


def _lexical_score(content_lc: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    score = 0.0
    matched_terms = 0
    for term in terms:
        hits = content_lc.count(term)
        if hits > 0:
            matched_terms += 1
            score += 1.0 + min(4.0, float(hits - 1) * 0.25)
    if matched_terms == 0:
        return 0.0

    # Favor chunks that cover more query terms; partial one-term matches for
    # multi-term queries should not dominate the lexical side.
    coverage = float(matched_terms) / float(len(terms))
    score *= 0.35 + 0.65 * coverage
    if len(terms) >= 2 and matched_terms < len(terms):
        score *= 0.35
    if len(terms) >= 2 and " ".join(terms) in content_lc:
        score += 2.0
    return score


def search_segments(
    qdrant: Qdrant,
    *,
    query_text: str,
    query_embedding: list[float],
    k: int,
    use_fts: bool,
) -> list[VectorSearchRow]:
    candidate_limit = max(k * _HYBRID_CANDIDATE_MULTIPLIER, _HYBRID_MIN_CANDIDATES) if use_fts else k

    client = qdrant.connect()
    response = client.query_points(
        collection_name=qdrant.collection,
        query=cast(Any, query_embedding),
        limit=max(1, candidate_limit),
        with_payload=True,
        with_vectors=False,
    )
    hits = list(response.points or [])
    if not hits:
        return []

    terms = _extract_terms(query_text) if use_fts else []
    scored: dict[str, dict[str, Any]] = {}
    lexical_candidates: list[tuple[str, float]] = []

    for idx, point in enumerate(hits, start=1):
        point_id = str(point.id)
        payload = point.payload or {}
        content = str(payload.get("content") or "")
        content_lc = str(payload.get("content_lc") or content.lower())
        scored[point_id] = {
            "vec_rank": idx,
            "vec_score": float(point.score),
            "payload": payload,
        }

        if use_fts:
            lex_score = _lexical_score(content_lc, terms)
            if lex_score > 0.0:
                lexical_candidates.append((point_id, lex_score))

    if use_fts:
        lexical_candidates.sort(key=lambda item: item[1], reverse=True)
        for rank, (point_id, _) in enumerate(lexical_candidates, start=1):
            scored[point_id]["fts_rank"] = rank

    ranked_rows: list[tuple[float, dict[str, Any]]] = []
    for data in scored.values():
        vec_rank = data.get("vec_rank")
        fts_rank = data.get("fts_rank")

        if use_fts:
            score = 0.0
            if vec_rank is not None:
                score += 1.0 / (_HYBRID_RRF_K + float(vec_rank))
            if fts_rank is not None:
                score += 1.0 / (_HYBRID_RRF_K + float(fts_rank))
        else:
            score = float(data.get("vec_score", -1.0))

        ranked_rows.append((score, data))

    ranked_rows.sort(key=lambda row: row[0], reverse=True)

    out: list[VectorSearchRow] = []
    for score, data in ranked_rows[:k]:
        payload = data["payload"]
        page_raw = payload.get("page")
        page = int(page_raw) if isinstance(page_raw, int) else None
        out.append(
            VectorSearchRow(
                content=str(payload.get("content") or ""),
                source_path=str(payload.get("source_path") or ""),
                title=str(payload.get("title") or ""),
                page=page,
                score=float(score),
            )
        )

    return out
