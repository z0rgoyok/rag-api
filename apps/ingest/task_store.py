from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import uuid

from sqlalchemy import case, func, select, update

from core.db import Db
from core.db_models import IngestTask as IngestTaskModel
from core.db_models import IngestTaskItem as IngestTaskItemModel

RunStrategy = Literal["fail", "skip"]
PipelineMode = Literal["full", "extract_only"]
InputMode = Literal["pdf", "chunks"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


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


def _to_task(row: IngestTaskModel) -> IngestTask:
    row_id = uuid.UUID(str(getattr(row, "id")))
    pdf_dir = str(getattr(row, "pdf_dir"))
    input_mode = str(getattr(row, "input_mode"))
    chunking_strategy = str(getattr(row, "chunking_strategy"))
    error_strategy = str(getattr(row, "error_strategy"))
    pipeline_mode = str(getattr(row, "pipeline_mode"))
    extract_output_dir_raw = getattr(row, "extract_output_dir")
    extract_output_dir = None if extract_output_dir_raw is None else str(extract_output_dir_raw)
    force = bool(getattr(row, "force"))
    status = str(getattr(row, "status"))
    last_error_raw = getattr(row, "last_error")
    last_error = None if last_error_raw is None else str(last_error_raw)

    return IngestTask(
        id=row_id,
        source_dir=pdf_dir,
        input_mode=input_mode,
        chunking_strategy=chunking_strategy,
        error_strategy=error_strategy,
        pipeline_mode=pipeline_mode,
        extract_output_dir=extract_output_dir,
        force=force,
        status=status,
        last_error=last_error,
    )


def _to_task_item(row: IngestTaskItemModel) -> IngestTaskItem:
    row_id = uuid.UUID(str(getattr(row, "id")))
    task_id = uuid.UUID(str(getattr(row, "task_id")))
    ordinal = int(getattr(row, "ordinal"))
    source_path = str(getattr(row, "source_path"))
    status = str(getattr(row, "status"))
    attempt = int(getattr(row, "attempt"))

    return IngestTaskItem(
        id=row_id,
        task_id=task_id,
        ordinal=ordinal,
        source_path=source_path,
        status=status,
        attempt=attempt,
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
    with db.session() as session:
        task = IngestTaskModel(
            id=task_id,
            pdf_dir=str(source_dir),
            input_mode=input_mode,
            chunking_strategy=chunking_strategy,
            error_strategy=error_strategy,
            pipeline_mode=pipeline_mode,
            extract_output_dir=extract_output_dir,
            force=force,
            status="pending",
            heartbeat_at=_now(),
        )
        session.add(task)

        for ordinal, path in enumerate(source_paths):
            session.add(
                IngestTaskItemModel(
                    id=uuid.uuid4(),
                    task_id=task_id,
                    ordinal=ordinal,
                    source_path=str(path),
                    status="pending",
                )
            )

        session.commit()
        session.refresh(task)
        return _to_task(task)


def get_ingest_task(db: Db, *, task_id: uuid.UUID) -> IngestTask | None:
    with db.session() as session:
        row = session.get(IngestTaskModel, task_id)
        if row is None:
            return None
        return _to_task(row)


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
    with db.session() as session:
        task = session.get(IngestTaskModel, task_id)
        if task is None:
            raise RuntimeError(f"Ingest task not found: {task_id}")

        if str(task.status) == "completed":
            raise RuntimeError(f"Ingest task is already completed: {task_id}")

        if expected_pipeline_mode and str(task.pipeline_mode) != expected_pipeline_mode:
            raise RuntimeError(
                f"Ingest task mode mismatch: task={task.pipeline_mode} run={expected_pipeline_mode}"
            )
        if expected_input_mode and str(task.input_mode) != expected_input_mode:
            raise RuntimeError(
                f"Ingest task input mismatch: task={task.input_mode} run={expected_input_mode}"
            )
        if expected_pipeline_mode == "extract_only" and extract_output_dir:
            existing_out = str(task.extract_output_dir or "")
            if existing_out and existing_out != extract_output_dir:
                raise RuntimeError(
                    f"Ingest task output mismatch: task={existing_out} run={extract_output_dir}"
                )

        now = _now()
        session.execute(
            update(IngestTaskItemModel)
            .where(
                IngestTaskItemModel.task_id == task_id,
                IngestTaskItemModel.status.in_(["running", "failed"]),
            )
            .values(
                status="pending",
                finished_at=None,
                heartbeat_at=now,
                last_error=case(
                    (IngestTaskItemModel.last_error.is_(None), "Recovered for retry"),
                    else_=IngestTaskItemModel.last_error,
                ),
            )
        )

        task.status = "running"
        if task.started_at is None:
            task.started_at = now
        task.finished_at = None
        task.heartbeat_at = now
        task.error_strategy = error_strategy
        task.force = force
        if task.extract_output_dir is None and extract_output_dir is not None:
            task.extract_output_dir = extract_output_dir
        task.last_error = None

        session.commit()
        session.refresh(task)
        return _to_task(task)


def claim_next_ingest_task_item(db: Db, *, task_id: uuid.UUID) -> IngestTaskItem | None:
    with db.session() as session:
        stmt = (
            select(IngestTaskItemModel)
            .where(IngestTaskItemModel.task_id == task_id, IngestTaskItemModel.status == "pending")
            .order_by(IngestTaskItemModel.ordinal)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        row = session.execute(stmt).scalar_one_or_none()
        if row is None:
            return None

        now = _now()
        row.status = "running"
        row.attempt = int(row.attempt) + 1
        row.started_at = now
        row.finished_at = None
        row.heartbeat_at = now
        row.last_error = None

        session.commit()
        session.refresh(row)
        return _to_task_item(row)


def touch_ingest_task(db: Db, *, task_id: uuid.UUID) -> None:
    with db.session() as session:
        session.execute(
            update(IngestTaskModel)
            .where(IngestTaskModel.id == task_id)
            .values(heartbeat_at=_now())
        )
        session.commit()


def touch_ingest_task_item(db: Db, *, task_item_id: uuid.UUID) -> None:
    with db.session() as session:
        session.execute(
            update(IngestTaskItemModel)
            .where(IngestTaskItemModel.id == task_item_id)
            .values(heartbeat_at=_now())
        )
        session.commit()


def mark_ingest_task_item_completed(db: Db, *, task_item_id: uuid.UUID) -> None:
    now = _now()
    with db.session() as session:
        session.execute(
            update(IngestTaskItemModel)
            .where(IngestTaskItemModel.id == task_item_id, IngestTaskItemModel.status == "running")
            .values(
                status="completed",
                finished_at=now,
                heartbeat_at=now,
                last_error=None,
            )
        )
        session.commit()


def mark_ingest_task_item_failed(db: Db, *, task_item_id: uuid.UUID, error: str) -> None:
    now = _now()
    with db.session() as session:
        session.execute(
            update(IngestTaskItemModel)
            .where(IngestTaskItemModel.id == task_item_id, IngestTaskItemModel.status == "running")
            .values(
                status="failed",
                finished_at=now,
                heartbeat_at=now,
                last_error=error,
            )
        )
        session.commit()


def mark_ingest_task_item_skipped(db: Db, *, task_item_id: uuid.UUID, error: str) -> None:
    now = _now()
    with db.session() as session:
        session.execute(
            update(IngestTaskItemModel)
            .where(IngestTaskItemModel.id == task_item_id, IngestTaskItemModel.status == "running")
            .values(
                status="skipped",
                finished_at=now,
                heartbeat_at=now,
                last_error=error,
            )
        )
        session.commit()


def mark_ingest_task_failed(db: Db, *, task_id: uuid.UUID, error: str) -> None:
    now = _now()
    with db.session() as session:
        session.execute(
            update(IngestTaskModel)
            .where(IngestTaskModel.id == task_id)
            .values(
                status="failed",
                finished_at=now,
                heartbeat_at=now,
                last_error=error,
            )
        )
        session.commit()


def mark_ingest_task_interrupted(db: Db, *, task_id: uuid.UUID, error: str) -> None:
    now = _now()
    with db.session() as session:
        session.execute(
            update(IngestTaskItemModel)
            .where(IngestTaskItemModel.task_id == task_id, IngestTaskItemModel.status == "running")
            .values(
                status="pending",
                finished_at=None,
                heartbeat_at=now,
                last_error=case(
                    (IngestTaskItemModel.last_error.is_(None), "Interrupted while processing"),
                    else_=IngestTaskItemModel.last_error,
                ),
            )
        )
        session.execute(
            update(IngestTaskModel)
            .where(IngestTaskModel.id == task_id)
            .values(
                status="interrupted",
                finished_at=now,
                heartbeat_at=now,
                last_error=error,
            )
        )
        session.commit()


def mark_ingest_task_completed_if_done(db: Db, *, task_id: uuid.UUID) -> bool:
    with db.session() as session:
        task = session.get(IngestTaskModel, task_id)
        if task is None or str(task.status) != "running":
            return False

        pending_or_running = session.execute(
            select(func.count())
            .select_from(IngestTaskItemModel)
            .where(
                IngestTaskItemModel.task_id == task_id,
                IngestTaskItemModel.status.in_(["pending", "running"]),
            )
        ).scalar_one()
        failed_items = session.execute(
            select(func.count())
            .select_from(IngestTaskItemModel)
            .where(
                IngestTaskItemModel.task_id == task_id,
                IngestTaskItemModel.status == "failed",
            )
        ).scalar_one()

        if int(pending_or_running) > 0 or int(failed_items) > 0:
            return False

        now = _now()
        task.status = "completed"
        task.finished_at = now
        task.heartbeat_at = now
        task.last_error = None
        session.commit()
        return True


def get_ingest_task_stats(db: Db, *, task_id: uuid.UUID) -> IngestTaskStats:
    with db.session() as session:
        row = session.execute(
            select(
                func.count(IngestTaskItemModel.id),
                func.sum(case((IngestTaskItemModel.status == "pending", 1), else_=0)),
                func.sum(case((IngestTaskItemModel.status == "running", 1), else_=0)),
                func.sum(case((IngestTaskItemModel.status == "completed", 1), else_=0)),
                func.sum(case((IngestTaskItemModel.status == "failed", 1), else_=0)),
                func.sum(case((IngestTaskItemModel.status == "skipped", 1), else_=0)),
            ).where(IngestTaskItemModel.task_id == task_id)
        ).one()

    total_items = int(row[0] or 0)
    return IngestTaskStats(
        total_items=total_items,
        pending_items=int(row[1] or 0),
        running_items=int(row[2] or 0),
        completed_items=int(row[3] or 0),
        failed_items=int(row[4] or 0),
        skipped_items=int(row[5] or 0),
    )
