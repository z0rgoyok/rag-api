#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

dc exec db psql -U rag -d rag

