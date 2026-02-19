from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import psycopg

from .db import Db, execute, fetch_one


@dataclass(frozen=True)
class SchemaInfo:
    embedding_dim: int
    embedding_model: str | None


def get_schema_info(db: Db) -> Optional[SchemaInfo]:
    try:
        with db.connect() as conn:
            row = fetch_one(conn, "select embedding_dim, embedding_model from rag_meta limit 1")
            if not row:
                return None
            return SchemaInfo(
                embedding_dim=int(row["embedding_dim"]),
                embedding_model=row.get("embedding_model"),
            )
    except psycopg.errors.UndefinedColumn:
        with db.connect() as conn:
            row = fetch_one(conn, "select embedding_dim from rag_meta limit 1")
            if not row:
                return None
            return SchemaInfo(
                embedding_dim=int(row["embedding_dim"]),
                embedding_model=None,
            )
    except psycopg.errors.UndefinedTable:
        return None


def _ensure_ingest_task_tables(conn: psycopg.Connection) -> None:
    execute(
        conn,
        """
        create table if not exists ingest_tasks (
          id uuid primary key,
          pdf_dir text not null,
          input_mode text not null default 'pdf' check (input_mode in ('pdf', 'chunks')),
          chunking_strategy text not null,
          error_strategy text not null,
          pipeline_mode text not null default 'full' check (pipeline_mode in ('full', 'extract_only')),
          extract_output_dir text,
          force boolean not null default false,
          status text not null check (status in ('pending', 'running', 'completed', 'failed', 'interrupted')),
          created_at timestamptz not null default now(),
          started_at timestamptz,
          finished_at timestamptz,
          heartbeat_at timestamptz,
          last_error text
        );
        """,
    )
    # Lightweight migration for previously created table shape.
    execute(
        conn,
        """
        alter table if exists ingest_tasks
        add column if not exists input_mode text not null default 'pdf';
        """,
    )
    execute(
        conn,
        """
        do $$
        begin
          if not exists (
            select 1
            from pg_constraint
            where conname = 'ingest_tasks_input_mode_chk'
          ) then
            alter table ingest_tasks
            add constraint ingest_tasks_input_mode_chk
            check (input_mode in ('pdf', 'chunks')) not valid;
          end if;
        end
        $$;
        """,
    )
    execute(
        conn,
        """
        alter table if exists ingest_tasks
        validate constraint ingest_tasks_input_mode_chk;
        """,
    )
    execute(
        conn,
        """
        alter table if exists ingest_tasks
        add column if not exists pipeline_mode text not null default 'full';
        """,
    )
    execute(
        conn,
        """
        alter table if exists ingest_tasks
        add column if not exists extract_output_dir text;
        """,
    )
    execute(conn, "create index if not exists ingest_tasks_status_idx on ingest_tasks(status);")
    execute(conn, "create index if not exists ingest_tasks_created_at_idx on ingest_tasks(created_at);")

    execute(
        conn,
        """
        create table if not exists ingest_task_items (
          id uuid primary key,
          task_id uuid not null references ingest_tasks(id) on delete cascade,
          ordinal integer not null,
          source_path text not null,
          status text not null check (status in ('pending', 'running', 'completed', 'failed', 'skipped')),
          attempt integer not null default 0,
          created_at timestamptz not null default now(),
          started_at timestamptz,
          finished_at timestamptz,
          heartbeat_at timestamptz,
          last_error text,
          unique(task_id, ordinal),
          unique(task_id, source_path)
        );
        """,
    )
    execute(conn, "create index if not exists ingest_task_items_task_status_idx on ingest_task_items(task_id, status);")
    execute(conn, "create index if not exists ingest_task_items_task_ordinal_idx on ingest_task_items(task_id, ordinal);")


def ensure_ingest_task_schema(db: Db) -> None:
    with db.connect() as conn:
        _ensure_ingest_task_tables(conn)


def ensure_schema(db: Db, *, embedding_dim: int, embedding_model: str) -> SchemaInfo:
    with db.connect() as conn:
        execute(conn, "create extension if not exists vector;")

        execute(
            conn,
            """
            create table if not exists rag_meta (
              embedding_dim integer not null,
              embedding_model text,
              created_at timestamptz not null default now()
            );
            """,
        )
        execute(
            conn,
            """
            alter table if exists rag_meta
            add column if not exists embedding_model text;
            """,
        )
        row = fetch_one(conn, "select embedding_dim, embedding_model from rag_meta limit 1;")
        effective_embedding_dim: int
        effective_embedding_model: str
        if row:
            existing = int(row["embedding_dim"])
            if existing != embedding_dim:
                raise RuntimeError(f"Embedding dimension mismatch: db={existing} env/probe={embedding_dim}")
            effective_embedding_dim = existing
            existing_model = row.get("embedding_model")
            if existing_model is None:
                execute(
                    conn,
                    "update rag_meta set embedding_model = %(m)s where embedding_model is null;",
                    {"m": embedding_model},
                )
                effective_embedding_model = embedding_model
            elif str(existing_model) != embedding_model:
                raise RuntimeError(
                    f"Embedding model mismatch: db={existing_model} env={embedding_model}"
                )
            else:
                effective_embedding_model = str(existing_model)
        else:
            execute(
                conn,
                "insert into rag_meta (embedding_dim, embedding_model) values (%(d)s, %(m)s);",
                {"d": embedding_dim, "m": embedding_model},
            )
            effective_embedding_dim = embedding_dim
            effective_embedding_model = embedding_model

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
              embedding vector({effective_embedding_dim}) not null
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

        _ensure_ingest_task_tables(conn)

        return SchemaInfo(
            embedding_dim=effective_embedding_dim,
            embedding_model=effective_embedding_model,
        )
