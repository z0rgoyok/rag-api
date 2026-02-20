#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

mode_raw="${INGEST_MODE:-ingest}"

# Backward compatible aliases:
# - pdf_extract  -> extract
# - chunks_full  -> ingest
# - resume       -> resume_ingest
case "${mode_raw}" in
  extract|ingest|resume_extract|resume_ingest)
    mode="${mode_raw}"
    ;;
  pdf_extract)
    mode="extract"
    ;;
  chunks_full)
    mode="ingest"
    ;;
  resume)
    mode="resume_ingest"
    ;;
  *)
    echo "Unsupported INGEST_MODE: ${mode_raw}. Expected: extract|ingest|resume_extract|resume_ingest" >&2
    exit 2
    ;;
esac

run_local=0
ingest_args=(ingest --on-error "${INGEST_ON_ERROR:-fail}")

case "${mode}" in
  extract)
    run_local=1
    ingest_args+=(
      --mode pdf_extract
      --pdf-dir "${INGEST_PDF_DIR:-var/pdfs}"
      --extract-output-dir "${INGEST_EXTRACT_OUTPUT_DIR:-var/extracted}"
    )
    ;;
  ingest)
    run_local=0
    ingest_args+=(
      --mode chunks_full
      --chunks-dir "${INGEST_CHUNKS_DIR:-/app/var/extracted}"
    )
    ;;
  resume_extract)
    run_local=1
    if [[ -z "${INGEST_TASK_ID:-}" ]]; then
      echo "INGEST_TASK_ID is required when INGEST_MODE=resume_extract" >&2
      exit 2
    fi
    ingest_args+=(--mode resume --task-id "${INGEST_TASK_ID}")
    ;;
  resume_ingest)
    run_local=0
    if [[ -z "${INGEST_TASK_ID:-}" ]]; then
      echo "INGEST_TASK_ID is required when INGEST_MODE=resume_ingest" >&2
      exit 2
    fi
    ingest_args+=(--mode resume --task-id "${INGEST_TASK_ID}")
    ;;
esac

if [[ -n "${INGEST_FORCE:-}" ]]; then ingest_args+=(--force); fi

if [[ "${run_local}" -eq 1 ]]; then
  repo="$(repo_root)"
  python_bin="${PYTHON_BIN:-${repo}/.venv/bin/python}"
  if [[ ! -x "${python_bin}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python_bin="$(command -v python3)"
    else
      echo "Python not found. Set PYTHON_BIN or install python3." >&2
      exit 1
    fi
  fi
  (
    cd "${repo}"
    "${python_bin}" -m apps.ingest.cli "${ingest_args[@]}"
  )
  exit 0
fi

dc run --rm api rag-ingest "${ingest_args[@]}"
