from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import uuid

from qdrant_client import models

from core.chunking import Chunk
from core.qdrant import Qdrant


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DocumentSyncState:
    document_id: uuid.UUID
    sha256: str
    segment_count: int
    embedding_count: int


def _source_filter(source_path: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="source_path",
                match=models.MatchValue(value=source_path),
            )
        ]
    )


def get_document_sync_state(qdrant: Qdrant, *, source_path: str) -> DocumentSyncState | None:
    client = qdrant.connect()
    flt = _source_filter(source_path)

    count_res = client.count(
        collection_name=qdrant.collection,
        count_filter=flt,
        exact=True,
    )
    segment_count = int(count_res.count)
    if segment_count == 0:
        return None

    points, _ = client.scroll(
        collection_name=qdrant.collection,
        scroll_filter=flt,
        limit=1,
        with_vectors=False,
        with_payload=["document_id", "source_sha256"],
    )
    if not points:
        return None

    payload = points[0].payload or {}
    document_id_raw = payload.get("document_id")
    try:
        document_id = uuid.UUID(str(document_id_raw))
    except (TypeError, ValueError, AttributeError):
        document_id = uuid.uuid5(uuid.NAMESPACE_URL, source_path)

    return DocumentSyncState(
        document_id=document_id,
        sha256=str(payload.get("source_sha256") or ""),
        segment_count=segment_count,
        embedding_count=segment_count,
    )


def is_document_up_to_date(qdrant: Qdrant, *, source_path: str, sha256: str) -> bool:
    state = get_document_sync_state(qdrant, source_path=source_path)
    if state is None:
        return False
    if state.sha256 != sha256:
        return False
    return state.segment_count > 0 and state.segment_count == state.embedding_count


def replace_document_content(
    qdrant: Qdrant,
    *,
    source_path: str,
    title: str,
    sha256: str,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> uuid.UUID:
    if not chunks:
        raise ValueError("Cannot persist a document without chunks.")
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks/embeddings count mismatch.")

    client = qdrant.connect()

    client.delete(
        collection_name=qdrant.collection,
        points_selector=models.FilterSelector(filter=_source_filter(source_path)),
        wait=True,
    )

    document_id = uuid.uuid4()
    points: list[models.PointStruct] = []

    for chunk, embedding in zip(chunks, embeddings):
        segment_id = uuid.uuid4()
        points.append(
            models.PointStruct(
                id=str(segment_id),
                vector=[float(v) for v in embedding],
                payload={
                    "document_id": str(document_id),
                    "segment_id": str(segment_id),
                    "source_path": source_path,
                    "title": title,
                    "source_sha256": sha256,
                    "ordinal": int(chunk.ordinal),
                    "page": chunk.page,
                    "content": chunk.content,
                    "content_lc": chunk.content.lower(),
                },
            )
        )

    batch_size = 128
    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=qdrant.collection,
            points=points[start : start + batch_size],
            wait=True,
        )

    return document_id
