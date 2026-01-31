#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# Reads PDFs from repo `var/pdfs/` (mounted into container as `/data/pdfs`).
dc run --rm api rag-ingest ingest --pdf-dir /data/pdfs

