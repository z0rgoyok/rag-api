from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import uuid

from core.chunking import Chunk
from core.db import Db, execute, execute_many, fetch_one
from core.pgvector import vector_literal


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


def get_document_sync_state(db: Db, *, source_path: str) -> DocumentSyncState | None:
    with db.connect() as conn:
        row = fetch_one(
            conn,
            """
            select
              d.id as document_id,
              d.sha256 as sha256,
              coalesce((select count(*) from segments s where s.document_id = d.id), 0) as segment_count,
              coalesce(
                (
                  select count(*)
                  from segment_embeddings se
                  join segments s on s.id = se.segment_id
                  where s.document_id = d.id
                ),
                0
              ) as embedding_count
            from documents d
            where d.source_path = %(source_path)s
            """,
            {"source_path": source_path},
        )
        if not row:
            return None
        return DocumentSyncState(
            document_id=row["document_id"],
            sha256=str(row["sha256"]),
            segment_count=int(row["segment_count"]),
            embedding_count=int(row["embedding_count"]),
        )


def is_document_up_to_date(db: Db, *, source_path: str, sha256: str) -> bool:
    state = get_document_sync_state(db, source_path=source_path)
    if state is None:
        return False
    if state.sha256 != sha256:
        return False
    # Require non-empty successful index to avoid accepting legacy partial rows.
    return state.segment_count > 0 and state.segment_count == state.embedding_count


def replace_document_content(
    db: Db,
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

    with db.connect_tx() as conn:
        row = fetch_one(
            conn,
            "select id from documents where source_path = %(source_path)s",
            {"source_path": source_path},
        )
        if row:
            execute(conn, "delete from documents where id = %(id)s", {"id": row["id"]})

        document_id = uuid.uuid4()
        execute(
            conn,
            """
            insert into documents (id, source_path, title, sha256)
            values (%(id)s, %(source_path)s, %(title)s, %(sha256)s)
            """,
            {
                "id": document_id,
                "source_path": source_path,
                "title": title,
                "sha256": sha256,
            },
        )

        segment_ids: list[uuid.UUID] = []
        segment_rows: list[dict] = []
        for chunk in chunks:
            segment_id = uuid.uuid4()
            segment_ids.append(segment_id)
            segment_rows.append(
                {
                    "id": segment_id,
                    "document_id": document_id,
                    "ordinal": chunk.ordinal,
                    "page": chunk.page,
                    "content": chunk.content,
                }
            )

        execute_many(
            conn,
            """
            insert into segments (id, document_id, ordinal, page, content)
            values (%(id)s, %(document_id)s, %(ordinal)s, %(page)s, %(content)s)
            """,
            segment_rows,
        )

        embedding_rows: list[dict] = []
        for segment_id, embedding in zip(segment_ids, embeddings):
            embedding_rows.append(
                {
                    "segment_id": segment_id,
                    "embedding": vector_literal(embedding),
                }
            )

        execute_many(
            conn,
            """
            insert into segment_embeddings (segment_id, embedding)
            values (%(segment_id)s, %(embedding)s::vector)
            """,
            embedding_rows,
        )

        return document_id
