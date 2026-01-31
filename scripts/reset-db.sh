#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

echo "This will delete local Postgres data under infra/pgdata/." >&2
echo "Proceeding (non-interactive mode)." >&2

dc down --remove-orphans
rm -rf "$(repo_root)/infra/pgdata"
dc up -d
dc ps

