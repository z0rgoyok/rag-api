from __future__ import annotations

import os
import secrets

from dotenv import load_dotenv

from ..config import load_settings
from ..db import Db, execute
from ..schema import ensure_schema, get_schema_info
from ..lmstudio import LmStudioClient


def main() -> None:
    load_dotenv()
    settings = load_settings()
    db = Db(settings.database_url)
    lm = LmStudioClient(settings.lmstudio_base_url)

    info = get_schema_info(db)
    if info is None:
        # If schema not initialized yet, initialize with the embedding dim probe.
        import asyncio

        dim = asyncio.run(lm.probe_embedding_dim(model=settings.lmstudio_embedding_model))
        ensure_schema(db, embedding_dim=dim)

    api_key = os.getenv("API_KEY") or secrets.token_urlsafe(32)
    tier = os.getenv("TIER") or "pro"
    citations_enabled = (os.getenv("CITATIONS_ENABLED") or "true").strip().lower() in {"1", "true", "yes", "y", "on"}

    with db.connect() as conn:
        execute(
            conn,
            "insert into api_keys (api_key, tier, citations_enabled) values (%(k)s, %(t)s, %(c)s) on conflict (api_key) do nothing",
            {"k": api_key, "t": tier, "c": citations_enabled},
        )

    print(api_key)


if __name__ == "__main__":
    main()

