from __future__ import annotations

import logging
import os
import time


def _parse_level(name: str) -> int:
    name = (name or "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


def configure_logging() -> None:
    level = _parse_level(os.getenv("LOG_LEVEL", "INFO"))
    fmt = (os.getenv("LOG_FORMAT") or "pretty").strip().lower()

    if fmt in {"json", "jsonl"}:
        logging.basicConfig(level=level, handlers=[_JsonStreamHandler()], force=True)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
            force=True,
        )

    # Avoid noisy third-party logs (and reduce chances of leaking secrets).
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


class _JsonStreamHandler(logging.StreamHandler):
    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(_JsonFormatter())


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json

        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)

