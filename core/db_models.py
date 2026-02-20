from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RagMeta(Base):
    __tablename__ = "rag_meta"

    embedding_dim: Mapped[int] = mapped_column(Integer, primary_key=True)
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ApiKey(Base):
    __tablename__ = "api_keys"

    api_key: Mapped[str] = mapped_column(Text, primary_key=True)
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    citations_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class IngestTask(Base):
    __tablename__ = "ingest_tasks"
    __table_args__ = (
        CheckConstraint("input_mode in ('pdf', 'chunks')", name="ingest_tasks_input_mode_chk"),
        CheckConstraint("pipeline_mode in ('full', 'extract_only')", name="ingest_tasks_pipeline_mode_chk"),
        CheckConstraint(
            "status in ('pending', 'running', 'completed', 'failed', 'interrupted')",
            name="ingest_tasks_status_chk",
        ),
        Index("ingest_tasks_status_idx", "status"),
        Index("ingest_tasks_created_at_idx", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pdf_dir: Mapped[str] = mapped_column(Text, nullable=False)
    input_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="pdf", server_default="pdf")
    chunking_strategy: Mapped[str] = mapped_column(Text, nullable=False)
    error_strategy: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="full", server_default="full")
    extract_output_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    force: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class IngestTaskItem(Base):
    __tablename__ = "ingest_task_items"
    __table_args__ = (
        CheckConstraint(
            "status in ('pending', 'running', 'completed', 'failed', 'skipped')",
            name="ingest_task_items_status_chk",
        ),
        UniqueConstraint("task_id", "ordinal", name="ingest_task_items_task_ordinal_uq"),
        UniqueConstraint("task_id", "source_path", name="ingest_task_items_task_source_path_uq"),
        Index("ingest_task_items_task_status_idx", "task_id", "status"),
        Index("ingest_task_items_task_ordinal_idx", "task_id", "ordinal"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest_tasks.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
