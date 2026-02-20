#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RERANK_HOST_BIND="${RERANK_HOST_BIND:-0.0.0.0}"
RERANK_HOST_PORT="${RERANK_HOST_PORT:-18123}"

exec "$PYTHON_BIN" -m uvicorn apps.rerank_host.main:app --host "$RERANK_HOST_BIND" --port "$RERANK_HOST_PORT"
