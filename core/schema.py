from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qdrant_client import models
from sqlalchemy import inspect, select

from .db import Db
from .db_models import ApiKey, Base, IngestTask, IngestTaskItem, RagMeta
from .qdrant import Qdrant


@dataclass(frozen=True)
class SchemaInfo:
    embedding_dim: int
    embedding_model: str | None


def get_schema_info(db: Db) -> Optional[SchemaInfo]:
    inspector = inspect(db.engine)
    if not inspector.has_table("rag_meta"):
        return None

    with db.session() as session:
        row = session.execute(select(RagMeta).limit(1)).scalar_one_or_none()
        if row is None:
            return None
        return SchemaInfo(
            embedding_dim=int(row.embedding_dim),
            embedding_model=row.embedding_model,
        )


def ensure_ingest_task_schema(db: Db) -> None:
    Base.metadata.create_all(bind=db.engine, tables=[IngestTask.__table__, IngestTaskItem.__table__])


def _extract_collection_dim(collection_info: models.CollectionInfo) -> int | None:
    vectors_cfg = collection_info.config.params.vectors
    if isinstance(vectors_cfg, models.VectorParams):
        return int(vectors_cfg.size)
    if isinstance(vectors_cfg, dict):
        first = next(iter(vectors_cfg.values()), None)
        if isinstance(first, models.VectorParams):
            return int(first.size)
    size = getattr(vectors_cfg, "size", None)
    if size is None:
        return None
    return int(size)


def _ensure_qdrant_collection(qdrant: Qdrant, *, embedding_dim: int) -> None:
    client = qdrant.connect()

    if not client.collection_exists(qdrant.collection):
        client.create_collection(
            collection_name=qdrant.collection,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE),
            on_disk_payload=True,
        )
        client.create_payload_index(
            collection_name=qdrant.collection,
            field_name="source_path",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=qdrant.collection,
            field_name="source_sha256",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=qdrant.collection,
            field_name="content_lc",
            field_schema=models.PayloadSchemaType.TEXT,
        )
        return

    info = client.get_collection(qdrant.collection)
    existing_dim = _extract_collection_dim(info)
    if existing_dim is None:
        raise RuntimeError(f"Cannot determine vector size for Qdrant collection: {qdrant.collection}")
    if int(existing_dim) != int(embedding_dim):
        raise RuntimeError(
            f"Embedding dimension mismatch: qdrant={existing_dim} env/probe={embedding_dim}"
        )


def ensure_schema(db: Db, qdrant: Qdrant, *, embedding_dim: int, embedding_model: str) -> SchemaInfo:
    Base.metadata.create_all(
        bind=db.engine,
        tables=[
            RagMeta.__table__,
            ApiKey.__table__,
            IngestTask.__table__,
            IngestTaskItem.__table__,
        ],
    )

    with db.session() as session:
        row = session.execute(select(RagMeta).limit(1)).scalar_one_or_none()

        effective_embedding_dim: int
        effective_embedding_model: str

        if row is None:
            row = RagMeta(embedding_dim=embedding_dim, embedding_model=embedding_model)
            session.add(row)
            effective_embedding_dim = embedding_dim
            effective_embedding_model = embedding_model
        else:
            existing_dim = int(row.embedding_dim)
            if existing_dim != embedding_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: db={existing_dim} env/probe={embedding_dim}"
                )
            effective_embedding_dim = existing_dim

            existing_model = row.embedding_model
            if existing_model is None:
                row.embedding_model = embedding_model
                effective_embedding_model = embedding_model
            elif str(existing_model) != embedding_model:
                raise RuntimeError(
                    f"Embedding model mismatch: db={existing_model} env={embedding_model}"
                )
            else:
                effective_embedding_model = str(existing_model)

        session.commit()

    _ensure_qdrant_collection(qdrant, embedding_dim=effective_embedding_dim)

    return SchemaInfo(
        embedding_dim=effective_embedding_dim,
        embedding_model=effective_embedding_model,
    )
