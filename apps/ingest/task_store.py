from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import uuid

from core.db import Db, execute, execute_many, fetch_one

RunStrategy = Literal["fail", "skip"]
PipelineMode = Literal["full", "extract_only"]
InputMode = Literal["pdf", "chunks"]


@dataclass(frozen=True)
class IngestTask:
    id: uuid.UUID
    source_dir: str
    input_mode: str
    chunking_strategy: str
    error_strategy: str
    pipeline_mode: str
    extract_output_dir: str | None
    force: bool
    status: str
    last_error: str | None


@dataclass(frozen=True)
class IngestTaskItem:
    id: uuid.UUID
    task_id: uuid.UUID
    ordinal: int
    source_path: str
    status: str
    attempt: int


@dataclass(frozen=True)
class IngestTaskStats:
    total_items: int
    pending_items: int
    running_items: int
    completed_items: int
    failed_items: int
    skipped_items: int


def _row_to_task(row: dict) -> IngestTask:
    return IngestTask(
        id=row["id"],
        source_dir=str(row["pdf_dir"]),
        input_mode=str(row["input_mode"]),
        chunking_strategy=str(row["chunking_strategy"]),
        error_strategy=str(row["error_strategy"]),
        pipeline_mode=str(row["pipeline_mode"]),
        extract_output_dir=row["extract_output_dir"],
        force=bool(row["force"]),
        status=str(row["status"]),
        last_error=row["last_error"],
    )


def _row_to_task_item(row: dict) -> IngestTaskItem:
    return IngestTaskItem(
        id=row["id"],
        task_id=row["task_id"],
        ordinal=int(row["ordinal"]),
        source_path=str(row["source_path"]),
        status=str(row["status"]),
        attempt=int(row["attempt"]),
    )


def create_ingest_task(
    db: Db,
    *,
    source_dir: Path,
    source_paths: list[Path],
    input_mode: InputMode,
    chunking_strategy: str,
    error_strategy: RunStrategy,
    pipeline_mode: PipelineMode,
    extract_output_dir: str | None,
    force: bool,
) -> IngestTask:
    task_id = uuid.uuid4()
    with db.connect() as conn:
        execute(
            conn,
            """
            insert into ingest_tasks (
              id,
              pdf_dir,
              input_mode,
              chunking_strategy,
              error_strategy,
              pipeline_mode,
              extract_output_dir,
              force,
              status,
              heartbeat_at
            )
            values (
              %(id)s,
              %(pdf_dir)s,
              %(input_mode)s,
              %(chunking_strategy)s,
              %(error_strategy)s,
              %(pipeline_mode)s,
              %(extract_output_dir)s,
              %(force)s,
              'pending',
              now()
            )
            """,
            {
                "id": task_id,
                "pdf_dir": str(source_dir),
                "input_mode": input_mode,
                "chunking_strategy": chunking_strategy,
                "error_strategy": error_strategy,
                "pipeline_mode": pipeline_mode,
                "extract_output_dir": extract_output_dir,
                "force": force,
            },
        )

        item_rows = []
        for ordinal, path in enumerate(source_paths):
            item_rows.append(
                {
                    "id": uuid.uuid4(),
                    "task_id": task_id,
                    "ordinal": ordinal,
                    "source_path": str(path),
                    "status": "pending",
                }
            )

        execute_many(
            conn,
            """
            insert into ingest_task_items (
              id,
              task_id,
              ordinal,
              source_path,
              status
            )
            values (
              %(id)s,
              %(task_id)s,
              %(ordinal)s,
              %(source_path)s,
              %(status)s
            )
            """,
            item_rows,
        )

        row = fetch_one(conn, "select * from ingest_tasks where id = %(id)s", {"id": task_id})
        if not row:
            raise RuntimeError("Failed to load created ingest task.")
        return _row_to_task(row)


def get_ingest_task(db: Db, *, task_id: uuid.UUID) -> IngestTask | None:
    with db.connect() as conn:
        row = fetch_one(conn, "select * from ingest_tasks where id = %(id)s", {"id": task_id})
        if not row:
            return None
        return _row_to_task(row)


def prepare_ingest_task_run(
    db: Db,
    *,
    task_id: uuid.UUID,
    error_strategy: RunStrategy,
    force: bool,
    expected_pipeline_mode: PipelineMode | None = None,
    expected_input_mode: InputMode | None = None,
    extract_output_dir: str | None = None,
) -> IngestTask:
    with db.connect() as conn:
        row = fetch_one(conn, "select * from ingest_tasks where id = %(id)s", {"id": task_id})
        if not row:
            raise RuntimeError(f"Ingest task not found: {task_id}")
        status = str(row["status"])
        if status == "completed":
            raise RuntimeError(f"Ingest task is already completed: {task_id}")
        if expected_pipeline_mode and str(row["pipeline_mode"]) != expected_pipeline_mode:
            raise RuntimeError(
                f"Ingest task mode mismatch: task={row['pipeline_mode']} run={expected_pipeline_mode}"
            )
        if expected_input_mode and str(row["input_mode"]) != expected_input_mode:
            raise RuntimeError(
                f"Ingest task input mismatch: task={row['input_mode']} run={expected_input_mode}"
            )
        if expected_pipeline_mode == "extract_only" and extract_output_dir:
            existing_out = str(row.get("extract_output_dir") or "")
            if existing_out and existing_out != extract_output_dir:
                raise RuntimeError(
                    f"Ingest task output mismatch: task={existing_out} run={extract_output_dir}"
                )

        execute(
            conn,
            """
            update ingest_task_items
            set
              status = 'pending',
              finished_at = null,
              heartbeat_at = now(),
              last_error = coalesce(last_error, 'Recovered for retry')
            where task_id = %(task_id)s and status in ('running', 'failed')
            """,
            {"task_id": task_id},
        )

        run_row = fetch_one(
            conn,
            """
            update ingest_tasks
            set
              status = 'running',
              started_at = coalesce(started_at, now()),
              finished_at = null,
              heartbeat_at = now(),
              error_strategy = %(error_strategy)s,
              force = %(force)s,
              extract_output_dir = coalesce(extract_output_dir, %(extract_output_dir)s),
              last_error = null
            where id = %(task_id)s
            returning *
            """,
            {
                "task_id": task_id,
                "error_strategy": error_strategy,
                "force": force,
                "extract_output_dir": extract_output_dir,
            },
        )
        if not run_row:
            raise RuntimeError(f"Failed to start ingest task: {task_id}")
        return _row_to_task(run_row)


def claim_next_ingest_task_item(db: Db, *, task_id: uuid.UUID) -> IngestTaskItem | None:
    with db.connect() as conn:
        row = fetch_one(
            conn,
            """
            with next_item as (
              select id
              from ingest_task_items
              where task_id = %(task_id)s and status = 'pending'
              order by ordinal
              limit 1
            )
            update ingest_task_items as it
            set
              status = 'running',
              attempt = it.attempt + 1,
              started_at = now(),
              finished_at = null,
              heartbeat_at = now(),
              last_error = null
            from next_item
            where it.id = next_item.id
            returning it.*
            """,
            {"task_id": task_id},
        )
        if not row:
            return None
        return _row_to_task_item(row)


def touch_ingest_task(db: Db, *, task_id: uuid.UUID) -> None:
    with db.connect() as conn:
        execute(
            conn,
            """
            update ingest_tasks
            set heartbeat_at = now()
            where id = %(task_id)s
            """,
            {"task_id": task_id},
        )


def touch_ingest_task_item(db: Db, *, task_item_id: uuid.UUID) -> None:
    with db.connect() as conn:
        execute(
            conn,
            """
            update ingest_task_items
            set heartbeat_at = now()
            where id = %(task_item_id)s
            """,
            {"task_item_id": task_item_id},
        )


def mark_ingest_task_item_completed(db: Db, *, task_item_id: uuid.UUID) -> None:
    with db.connect() as conn:
        execute(
            conn,
            """
            update ingest_task_items
            set
              status = 'completed',
              finished_at = now(),
              heartbeat_at = now(),
              last_error = null
            where id = %(task_item_id)s and status = 'running'
            """,
            {"task_item_id": task_item_id},
        )


def mark_ingest_task_item_failed(db: Db, *, task_item_id: uuid.UUID, error: str) -> None:
    with db.connect() as conn:
        execute(
            conn,
            """
            update ingest_task_items
            set
              status = 'failed',
              finished_at = now(),
              heartbeat_at = now(),
              last_error = %(error)s
            where id = %(task_item_id)s and status = 'running'
            """,
            {
                "task_item_id": task_item_id,
                "error": error,
            },
        )


def mark_ingest_task_item_skipped(db: Db, *, task_item_id: uuid.UUID, error: str) -> None:
    with db.connect() as conn:
        execute(
            conn,
            """
            update ingest_task_items
            set
              status = 'skipped',
              finished_at = now(),
              heartbeat_at = now(),
              last_error = %(error)s
            where id = %(task_item_id)s and status = 'running'
            """,
            {
                "task_item_id": task_item_id,
                "error": error,
            },
        )


def mark_ingest_task_failed(db: Db, *, task_id: uuid.UUID, error: str) -> None:
    with db.connect() as conn:
        execute(
            conn,
            """
            update ingest_tasks
            set
              status = 'failed',
              finished_at = now(),
              heartbeat_at = now(),
              last_error = %(error)s
            where id = %(task_id)s
            """,
            {
                "task_id": task_id,
                "error": error,
            },
        )


def mark_ingest_task_interrupted(db: Db, *, task_id: uuid.UUID, error: str) -> None:
    with db.connect() as conn:
        execute(
            conn,
            """
            update ingest_task_items
            set
              status = 'pending',
              finished_at = null,
              heartbeat_at = now(),
              last_error = coalesce(last_error, 'Interrupted while processing')
            where task_id = %(task_id)s and status = 'running'
            """,
            {"task_id": task_id},
        )
        execute(
            conn,
            """
            update ingest_tasks
            set
              status = 'interrupted',
              finished_at = now(),
              heartbeat_at = now(),
              last_error = %(error)s
            where id = %(task_id)s
            """,
            {
                "task_id": task_id,
                "error": error,
            },
        )


def mark_ingest_task_completed_if_done(db: Db, *, task_id: uuid.UUID) -> bool:
    with db.connect() as conn:
        row = fetch_one(
            conn,
            """
            update ingest_tasks
            set
              status = 'completed',
              finished_at = now(),
              heartbeat_at = now(),
              last_error = null
            where
              id = %(task_id)s
              and status = 'running'
              and not exists (
                select 1
                from ingest_task_items i
                where
                  i.task_id = %(task_id)s
                  and i.status in ('pending', 'running')
              )
              and not exists (
                select 1
                from ingest_task_items i
                where
                  i.task_id = %(task_id)s
                  and i.status = 'failed'
              )
            returning id
            """,
            {"task_id": task_id},
        )
        return row is not None


def get_ingest_task_stats(db: Db, *, task_id: uuid.UUID) -> IngestTaskStats:
    with db.connect() as conn:
        row = fetch_one(
            conn,
            """
            select
              count(*)::int as total_items,
              count(*) filter (where status = 'pending')::int as pending_items,
              count(*) filter (where status = 'running')::int as running_items,
              count(*) filter (where status = 'completed')::int as completed_items,
              count(*) filter (where status = 'failed')::int as failed_items,
              count(*) filter (where status = 'skipped')::int as skipped_items
            from ingest_task_items
            where task_id = %(task_id)s
            """,
            {"task_id": task_id},
        )
        if not row:
            return IngestTaskStats(
                total_items=0,
                pending_items=0,
                running_items=0,
                completed_items=0,
                failed_items=0,
                skipped_items=0,
            )
        return IngestTaskStats(
            total_items=int(row["total_items"]),
            pending_items=int(row["pending_items"]),
            running_items=int(row["running_items"]),
            completed_items=int(row["completed_items"]),
            failed_items=int(row["failed_items"]),
            skipped_items=int(row["skipped_items"]),
        )
