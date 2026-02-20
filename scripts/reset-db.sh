#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

echo "This will delete local metadata DB data under infra/pgdata/ and vector index data under var/qdrant/." >&2
echo "Proceeding (non-interactive mode)." >&2

dc down --remove-orphans
rm -rf "$(repo_root)/infra/pgdata"
rm -rf "$(repo_root)/var/qdrant"
dc up -d
dc ps
