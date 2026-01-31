#!/usr/bin/env bash
set -euo pipefail

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

compose_file() {
  echo "$(repo_root)/infra/compose.yml"
}

dc() {
  docker compose -f "$(compose_file)" "$@"
}

api_port() {
  echo "${API_PORT:-18080}"
}

pg_port() {
  echo "${PG_PORT:-56473}"
}

api_url() {
  echo "http://localhost:$(api_port)"
}

wait_http_ok() {
  local url="$1"
  local seconds="${2:-60}"
  local i
  for ((i = 1; i <= seconds; i++)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

