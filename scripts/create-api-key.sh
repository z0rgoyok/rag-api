#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# Env overrides:
# - API_KEY=... (optional)
# - TIER=... (optional)
# - CITATIONS_ENABLED=true|false (optional)
dc run --rm -e API_KEY="${API_KEY:-}" -e TIER="${TIER:-pro}" -e CITATIONS_ENABLED="${CITATIONS_ENABLED:-false}" api \
  python -m apps.api.scripts.create_api_key
