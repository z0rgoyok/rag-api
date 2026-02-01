from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import psycopg

from .db import Db, execute, fetch_one


@dataclass(frozen=True)
class SchemaInfo:
    embedding_dim: int


def get_schema_info(db: Db) -> Optional[SchemaInfo]:
    try:
        with db.connect() as conn:
            row = fetch_one(conn, "select embedding_dim from rag_meta limit 1")
            if not row:
                return None
            return SchemaInfo(embedding_dim=int(row["embedding_dim"]))
    except psycopg.errors.UndefinedTable:
        return None


def ensure_schema(db: Db, *, embedding_dim: int) -> SchemaInfo:
    with db.connect() as conn:
        execute(conn, "create extension if not exists vector;")

        execute(
            conn,
            """
            create table if not exists rag_meta (
              embedding_dim integer not null,
              created_at timestamptz not null default now()
            );
            """,
        )
        row = fetch_one(conn, "select embedding_dim from rag_meta limit 1;")
        if row:
            existing = int(row["embedding_dim"])
            if existing != embedding_dim:
                raise RuntimeError(f"Embedding dimension mismatch: db={existing} env/probe={embedding_dim}")
            return SchemaInfo(embedding_dim=existing)

        execute(conn, "insert into rag_meta (embedding_dim) values (%(d)s);", {"d": embedding_dim})

        execute(
            conn,
            """
            create table if not exists documents (
              id uuid primary key,
              source_path text not null,
              title text not null,
              sha256 text not null,
              created_at timestamptz not null default now()
            );
            """,
        )
        execute(
            conn,
            """
            create unique index if not exists documents_source_path_idx on documents(source_path);
            create unique index if not exists documents_sha256_idx on documents(sha256);
            """,
        )
        execute(
            conn,
            """
            create table if not exists segments (
              id uuid primary key,
              document_id uuid not null references documents(id) on delete cascade,
              ordinal integer not null,
              page integer,
              content text not null,
              tsv tsvector generated always as (to_tsvector('simple', content)) stored,
              created_at timestamptz not null default now(),
              unique(document_id, ordinal)
            );
            """,
        )
        execute(conn, "create index if not exists segments_document_id_idx on segments(document_id);")
        execute(conn, "create index if not exists segments_tsv_idx on segments using gin(tsv);")

        execute(
            conn,
            f"""
            create table if not exists segment_embeddings (
              segment_id uuid primary key references segments(id) on delete cascade,
              embedding vector({embedding_dim}) not null
            );
            """,
        )
        execute(conn, "create index if not exists segment_embeddings_ivfflat_cosine on segment_embeddings using ivfflat (embedding vector_cosine_ops) with (lists = 100);")

        execute(
            conn,
            """
            create table if not exists api_keys (
              api_key text primary key,
              tier text not null,
              citations_enabled boolean not null default false,
              created_at timestamptz not null default now()
            );
            """,
        )

        return SchemaInfo(embedding_dim=embedding_dim)
