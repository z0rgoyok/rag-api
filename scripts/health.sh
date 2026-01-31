#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

curl -sS "$(api_url)/healthz"
echo

