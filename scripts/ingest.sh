#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

format_elapsed() {
  local total="${1:-0}"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  if ((h > 0)); then
    printf '%dh %02dm %02ds' "${h}" "${m}" "${s}"
  elif ((m > 0)); then
    printf '%dm %02ds' "${m}" "${s}"
  else
    printf '%ds' "${s}"
  fi
}

started_at="${SECONDS}"
repo="$(repo_root)"

# Load local .env for wrapper-level INGEST_* variables too (not only for docker compose).
env_file="${repo}/.env"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

mode_raw="${INGEST_MODE:-ingest}"

# Backward compatible aliases:
# - pdf_extract -> extract
# - chunks_full -> ingest
case "${mode_raw}" in
  extract|ingest|resume)
    mode="${mode_raw}"
    ;;
  pdf_extract)
    mode="extract"
    ;;
  chunks_full)
    mode="ingest"
    ;;
  *)
    echo "Unsupported INGEST_MODE: ${mode_raw}. Expected: extract|ingest|resume" >&2
    exit 2
    ;;
esac

ingest_args=(ingest --on-error "${INGEST_ON_ERROR:-fail}")

case "${mode}" in
  extract)
    ingest_args+=(
      --mode pdf_extract
      --pdf-dir "${INGEST_PDF_DIR:-var/pdfs}"
      --extract-output-dir "${INGEST_EXTRACT_OUTPUT_DIR:-var/extracted}"
    )
    ;;
  ingest)
    ingest_args+=(
      --mode chunks_full
      --chunks-dir "${INGEST_CHUNKS_DIR:-var/extracted}"
    )
    ;;
  resume)
    if [[ -z "${INGEST_TASK_ID:-}" ]]; then
      echo "INGEST_TASK_ID is required when INGEST_MODE=resume" >&2
      exit 2
    fi
    ingest_args+=(--mode resume --task-id "${INGEST_TASK_ID}")
    ;;
esac

if [[ -n "${INGEST_FORCE:-}" ]]; then ingest_args+=(--force); fi

python_bin="${PYTHON_BIN:-${repo}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  else
    echo "Python not found. Set PYTHON_BIN or install python3." >&2
    exit 1
  fi
fi
set +e
(
  cd "${repo}"
  echo "ingest_start mode=${mode} cmd=${python_bin} -m apps.ingest.cli ${ingest_args[*]}"
  "${python_bin}" -m apps.ingest.cli "${ingest_args[@]}"
)
status=$?
set -e

elapsed=$((SECONDS - started_at))
if [[ ${status} -eq 0 ]]; then
  echo "ingest_done mode=${mode} elapsed=$(format_elapsed "${elapsed}") (${elapsed}s)"
else
  echo "ingest_fail mode=${mode} elapsed=$(format_elapsed "${elapsed}") (${elapsed}s) exit_code=${status}" >&2
fi
exit "${status}"
