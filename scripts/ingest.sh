#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# Explicit run mode (no implicit mode guessing by arg presence).
# Default: index PDFs from repo `var/pdfs/` (mounted as `/data/pdfs`).
mode="${INGEST_MODE:-pdf_full}"

args=(
  rag-ingest ingest
  --mode "${mode}"
  --on-error "${INGEST_ON_ERROR:-fail}"
)

case "${mode}" in
  pdf_full|pdf_extract)
    args+=(--pdf-dir /data/pdfs)
    ;;
  chunks_full)
    args+=(--chunks-dir "${INGEST_CHUNKS_DIR:-/app/var/extracted}")
    ;;
  resume)
    if [[ -z "${INGEST_TASK_ID:-}" ]]; then
      echo "INGEST_TASK_ID is required when INGEST_MODE=resume" >&2
      exit 2
    fi
    args+=(--task-id "${INGEST_TASK_ID}")
    ;;
  *)
    echo "Unsupported INGEST_MODE: ${mode}. Expected: pdf_full|pdf_extract|chunks_full|resume" >&2
    exit 2
    ;;
esac

if [[ "${mode}" == "pdf_extract" ]]; then
  args+=(--extract-output-dir "${INGEST_EXTRACT_OUTPUT_DIR:-var/extracted}")
fi
if [[ -n "${INGEST_FORCE:-}" ]]; then
  args+=(--force)
fi

dc run --rm api "${args[@]}"
