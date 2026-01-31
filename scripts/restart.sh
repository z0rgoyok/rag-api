#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

dc restart
dc ps

wait_http_ok "$(api_url)/healthz" 60
curl -sS "$(api_url)/healthz"
echo

