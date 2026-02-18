#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/pdf_to_md_local.sh var/pdfs/file.pdf [var/extracted/out.md]
  ./scripts/pdf_to_md_local.sh /abs/path/file.pdf [var/extracted/out.md]

Notes:
  - Runs locally (no Docker), using Python from `.venv` by default.
  - Output is written under `var/extracted/` by default (repo hygiene).
  - Optional:
      PYTHON_BIN=.venv/bin/python
      PDF_TEXT_EXTRACTOR=docling
      DOCLING_DO_OCR=1
      DOCLING_DO_TABLE_STRUCTURE=0
      DOCLING_FORCE_FULL_PAGE_OCR=0
      DOCLING_FORCE_BACKEND_TEXT=0
      PDF_DUMP_MD=1
EOF
}

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

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

pdf_input="${1:-}"
out_rel="${2:-}"
if [[ -z "${pdf_input}" || "${pdf_input}" == "-h" || "${pdf_input}" == "--help" ]]; then
  usage
  exit 2
fi
started_at="${SECONDS}"

repo="$(repo_root)"

if [[ "${pdf_input}" = /* ]]; then
  pdf_abs="${pdf_input}"
else
  pdf_abs="${repo}/${pdf_input}"
fi
if [[ ! -f "${pdf_abs}" ]]; then
  echo "PDF not found: ${pdf_input}" >&2
  exit 1
fi

mkdir -p "${repo}/var/extracted"

out_abs=""
if [[ -n "${out_rel}" ]]; then
  out_abs="${repo}/${out_rel}"
  if [[ "${out_abs}" != "${repo}/var/"* ]]; then
    echo "Refusing to write outside var/: ${out_rel}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${out_abs}")"
fi

default_name="$(basename "${pdf_abs}")"
default_name="${default_name%.pdf}"
default_name="${default_name%.PDF}"
default_name="${default_name}.md"
default_out="${repo}/var/extracted/${default_name}"
rm -f "${default_out}" || true

python_bin="${PYTHON_BIN:-${repo}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  else
    echo "Python not found. Set PYTHON_BIN or install python3." >&2
    exit 1
  fi
fi

pdf_dump_md="${PDF_DUMP_MD:-1}"
pdf_dump_dir="${PDF_DUMP_DIR:-var/extracted}"
pdf_text_extractor="${PDF_TEXT_EXTRACTOR:-docling}"
docling_do_ocr="${DOCLING_DO_OCR:-1}"
docling_do_table_structure="${DOCLING_DO_TABLE_STRUCTURE:-0}"
docling_force_full_page_ocr="${DOCLING_FORCE_FULL_PAGE_OCR:-0}"
docling_force_backend_text="${DOCLING_FORCE_BACKEND_TEXT:-0}"
heartbeat_sec="${PDF_LOG_HEARTBEAT_SEC:-15}"
if ! [[ "${heartbeat_sec}" =~ ^[0-9]+$ ]] || ((heartbeat_sec < 1)); then
  heartbeat_sec=15
fi

log "Start PDF extraction"
log "Input: ${pdf_abs}"
if [[ -n "${out_abs}" ]]; then
  log "Output: ${out_rel}"
else
  log "Output: var/extracted/${default_name}"
fi
log "Python: ${python_bin}"
log "Settings: OCR=${docling_do_ocr}, TABLES=${docling_do_table_structure}, FORCE_FULL_PAGE_OCR=${docling_force_full_page_ocr}, FORCE_BACKEND_TEXT=${docling_force_backend_text}"
log "Heartbeat interval: ${heartbeat_sec}s"

worker_pid=""
trap 'if [[ -n "${worker_pid}" ]]; then kill "${worker_pid}" 2>/dev/null || true; fi' INT TERM

(
  cd "${repo}"
  PDF_DUMP_MD="${pdf_dump_md}" \
  PDF_DUMP_DIR="${pdf_dump_dir}" \
  PDF_TEXT_EXTRACTOR="${pdf_text_extractor}" \
  DOCLING_DO_OCR="${docling_do_ocr}" \
  DOCLING_DO_TABLE_STRUCTURE="${docling_do_table_structure}" \
  DOCLING_FORCE_FULL_PAGE_OCR="${docling_force_full_page_ocr}" \
  DOCLING_FORCE_BACKEND_TEXT="${docling_force_backend_text}" \
  RAG_REPO_ROOT="${repo}" \
  INPUT_PDF_PATH="${pdf_abs}" \
  "${python_bin}" -u - <<'PY'
import os
import time
from pathlib import Path

from apps.ingest.pdf_extract import extract_pdf_text_pages

pdf_path = Path(os.environ["INPUT_PDF_PATH"]).expanduser().resolve()
started = time.monotonic()
print(f"extract_start path={pdf_path}", flush=True)
pages = extract_pdf_text_pages(pdf_path)
print(f"pages={len(pages)}")
elapsed = time.monotonic() - started
print(f"extract_done elapsed={elapsed:.2f}s", flush=True)
PY
) &
worker_pid="$!"

while kill -0 "${worker_pid}" 2>/dev/null; do
  sleep "${heartbeat_sec}"
  if kill -0 "${worker_pid}" 2>/dev/null; then
    elapsed_now=$((SECONDS - started_at))
    log "Still running... elapsed $(format_elapsed "${elapsed_now}")"
  fi
done

wait "${worker_pid}"
trap - INT TERM

if [[ -n "${out_abs}" ]]; then
  mv -f "${default_out}" "${out_abs}"
  echo "Wrote: ${out_rel}"
else
  echo "Wrote: var/extracted/${default_name}"
fi

elapsed=$((SECONDS - started_at))
echo "Elapsed: $(format_elapsed "${elapsed}")"
