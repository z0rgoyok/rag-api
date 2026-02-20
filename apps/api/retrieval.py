from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from core.reranking.protocol import Reranker
from core.qdrant import Qdrant
from core.vector_search import search_segments


@dataclass(frozen=True)
class RetrievedSegment:
    content: str
    source_path: str
    title: str
    page: int | None
    score: float


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _extract_query_terms(text: str, *, max_terms: int = 6) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _WORD_RE.finditer((text or "").lower()):
        token = match.group(0).strip("_")
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max_terms:
            break
    return out


def _term_rx(term: str) -> str:
    return rf"(?<!\w){re.escape(term)}(?!\w)"


def _select_focus_terms(*, query_text: str, segments: list["RetrievedSegment"], max_terms: int = 3) -> list[str]:
    tokens = _extract_query_terms(query_text, max_terms=12)
    if not tokens or not segments:
        return []

    corpus = [s.content.lower()[:1500] for s in segments if s.content]
    if not corpus:
        return []
    n = max(1, len(corpus))

    scored: list[tuple[float, str]] = []
    for token in tokens:
        rx = re.compile(_term_rx(token))
        df = sum(1 for text in corpus if rx.search(text))
        if df == 0:
            continue
        ratio = df / float(n)
        # Prefer longer and less frequent terms; language-agnostic.
        score = (1.0 - ratio) + min(12, len(token)) / 20.0
        if ratio <= 0.35:
            scored.append((score, token))

    if not scored:
        fallback: list[tuple[int, str]] = []
        for token in tokens:
            rx = re.compile(_term_rx(token))
            if any(rx.search(text) for text in corpus):
                fallback.append((len(token), token))
        fallback.sort(key=lambda item: item[0], reverse=True)
        return [token for _, token in fallback[:max_terms]]

    scored.sort(key=lambda item: item[0], reverse=True)
    out: list[str] = []
    for _, token in scored:
        if token in out:
            continue
        out.append(token)
        if len(out) >= max_terms:
            break
    return out


def _definition_pattern_boost(*, query_terms: list[str], content: str) -> float:
    if not query_terms or not content:
        return 0.0

    # Language-agnostic definition hints:
    # - term near the start
    # - term followed by separators (:, -, —, =)
    # - parenthesized alias/transliteration patterns
    head = content.lower()[:900]
    boost = 0.0
    for term in query_terms:
        rx_term = _term_rx(term)
        if not re.search(rx_term, head):
            continue

        if re.search(rf"(?:^|[\n\r\.\!\?]\s*){rx_term}\s*[\-:=\u2013\u2014]", head):
            boost += 0.030
        elif re.search(rf"{rx_term}\s*[\-:=\u2013\u2014]", head):
            boost += 0.018

        if re.search(rf"\({rx_term}\)[^\n\r]{{0,48}}[:=\u2013\u2014\-]", head):
            boost += 0.035
        elif re.search(rf"{rx_term}\s*\(", head) or re.search(rf"\({rx_term}\)", head):
            boost += 0.015

        if re.search(
            rf"[\x22\x27\u00ab\u00bb\u201c\u201d]{rx_term}[\x22\x27\u00ab\u00bb\u201c\u201d]\s*[\-:=\u2013\u2014]",
            head,
        ):
            boost += 0.025

    return min(boost, 0.090)


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


async def rerank_segments(
    *,
    query_text: str,
    segments: list[RetrievedSegment],
    reranker: Reranker,
    top_k: int,
) -> list[RetrievedSegment]:
    if not segments:
        return []

    # Get scores for all candidates first; apply lightweight pattern boost
    # before the final top-k cut.
    ranked = await reranker.rerank(query_text, [s.content for s in segments], top_k=None)
    query_terms = _select_focus_terms(query_text=query_text, segments=segments)
    out: list[RetrievedSegment] = []
    for item in ranked:
        if item.index < 0 or item.index >= len(segments):
            continue
        seg = segments[item.index]
        boost = _definition_pattern_boost(query_terms=query_terms, content=seg.content)
        out.append(
            RetrievedSegment(
                content=seg.content,
                source_path=seg.source_path,
                title=seg.title,
                page=seg.page,
                score=item.score + boost,
            )
        )
    out.sort(key=lambda s: s.score, reverse=True)
    return out[:top_k]
