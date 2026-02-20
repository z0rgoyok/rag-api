from __future__ import annotations

import os
import secrets

from dotenv import load_dotenv
from sqlalchemy import select

from core.config import load_settings
from core.db import Db
from core.db_models import ApiKey
from core.embeddings_client import build_embeddings_client
from core.qdrant import Qdrant
from core.schema import ensure_schema, get_schema_info


def main() -> None:
    load_dotenv()
    settings = load_settings()
    db = Db(settings.database_url)
    qdrant = Qdrant(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
    )
    embed_client = build_embeddings_client(settings)

    info = get_schema_info(db)
    if info is None:
        # If schema not initialized yet, initialize with the embedding dim probe.
        import asyncio

        dim = asyncio.run(embed_client.probe_embedding_dim(model=settings.embeddings_model))
        ensure_schema(
            db,
            qdrant,
            embedding_dim=dim,
            embedding_model=settings.embeddings_model,
        )
    else:
        ensure_schema(
            db,
            qdrant,
            embedding_dim=info.embedding_dim,
            embedding_model=settings.embeddings_model,
        )

    api_key = os.getenv("API_KEY") or secrets.token_urlsafe(32)
    tier = os.getenv("TIER") or "pro"
    citations_enabled = (os.getenv("CITATIONS_ENABLED") or "false").strip().lower() in {"1", "true", "yes", "y", "on"}

    with db.session() as session:
        existing = session.execute(select(ApiKey).where(ApiKey.api_key == api_key)).scalar_one_or_none()
        if existing is None:
            session.add(ApiKey(api_key=api_key, tier=tier, citations_enabled=citations_enabled))
            session.commit()

    print(api_key)


if __name__ == "__main__":
    main()
