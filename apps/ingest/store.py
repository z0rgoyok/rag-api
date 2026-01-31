from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import uuid

from apps.api.db import Db, execute, execute_many, fetch_one
from apps.api.pgvector import vector_literal


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class StoredDocument:
    id: uuid.UUID
    up_to_date: bool


def upsert_document(db: Db, *, source_path: str, title: str, sha256: str) -> StoredDocument:
    with db.connect() as conn:
        row = fetch_one(conn, "select id, sha256 from documents where source_path = %(p)s", {"p": source_path})
        if row and row["sha256"] == sha256:
            return StoredDocument(id=row["id"], up_to_date=True)

        if row and row["sha256"] != sha256:
            # Replace by deleting and re-inserting (keeps MVP simple).
            execute(conn, "delete from documents where id = %(id)s", {"id": row["id"]})

        doc_id = uuid.uuid4()
        execute(
            conn,
            "insert into documents (id, source_path, title, sha256) values (%(id)s, %(p)s, %(t)s, %(s)s)",
            {"id": doc_id, "p": source_path, "t": title, "s": sha256},
        )
        return StoredDocument(id=doc_id, up_to_date=False)


def delete_document_by_source_path(db: Db, *, source_path: str) -> None:
    with db.connect() as conn:
        execute(conn, "delete from documents where source_path = %(p)s", {"p": source_path})


def insert_segments(db: Db, *, document_id: uuid.UUID, segments: Iterable[dict]) -> None:
    with db.connect() as conn:
        execute_many(
            conn,
            """
            insert into segments (id, document_id, ordinal, page, content)
            values (%(id)s, %(document_id)s, %(ordinal)s, %(page)s, %(content)s)
            """,
            segments,
        )


def insert_embeddings(db: Db, *, rows: Iterable[dict]) -> None:
    adapted = []
    for r in rows:
        adapted.append(
            {
                "segment_id": r["segment_id"],
                "embedding": vector_literal(r["embedding"]),
            }
        )
    with db.connect() as conn:
        execute_many(
            conn,
            """
            insert into segment_embeddings (segment_id, embedding)
            values (%(segment_id)s, %(embedding)s::vector)
            """,
            adapted,
        )
