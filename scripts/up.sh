#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

dc up -d --build
dc ps

if wait_http_ok "$(api_url)/healthz" 60; then
  curl -sS "$(api_url)/healthz"
  echo
else
  echo "api not ready on $(api_url) after 60s" >&2
  dc logs --no-color --tail=100 api || true
  exit 1
fi

