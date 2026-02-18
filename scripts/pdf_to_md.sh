#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/pdf_to_md.sh var/pdfs/file.pdf [var/extracted/out.md]
  ./scripts/pdf_to_md.sh /abs/path/file.pdf [var/extracted/out.md]

Notes:
  - Runs inside Docker (same env as ingest).
  - Output is written under `var/extracted/` by default (repo hygiene).
  - You can also set:
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

repo_pdfs="${repo}/var/pdfs"
extra_mount=""
if [[ "${pdf_abs}" == "${repo_pdfs}/"* ]]; then
  pdf_rel_in_repo="${pdf_abs#"${repo_pdfs}/"}"
  pdf_container_path="/data/pdfs/${pdf_rel_in_repo}"
else
  pdf_dir="$(dirname "${pdf_abs}")"
  pdf_file="$(basename "${pdf_abs}")"
  extra_mount="${pdf_dir}:/data/extpdf:ro"
  pdf_container_path="/data/extpdf/${pdf_file}"
fi

default_name="$(basename "${pdf_abs}")"
default_name="${default_name%.pdf}"
default_name="${default_name%.PDF}"
default_name="${default_name}.md"
default_out="${repo}/var/extracted/${default_name}"
rm -f "${default_out}" || true

# Reuse the api image because it already contains all PDF deps.
dc_args=(
  run --rm
  -e RAG_REPO_ROOT=/app
  -e PDF_DUMP_MD
  -e PDF_TEXT_EXTRACTOR
  -e DOCLING_DO_OCR
  -e DOCLING_DO_TABLE_STRUCTURE
  -e DOCLING_FORCE_FULL_PAGE_OCR
  -e DOCLING_FORCE_BACKEND_TEXT
  -e PDF_DUMP_DIR
  -e INPUT_PDF_PATH
)
if [[ -n "${extra_mount}" ]]; then
  dc_args+=(-v "${extra_mount}")
fi
dc_args+=(
  api
  python -c "import os; from pathlib import Path; from apps.ingest.pdf_extract import extract_pdf_text_pages; p=Path(os.environ[\"INPUT_PDF_PATH\"]); pages=extract_pdf_text_pages(p); print(f\"pages={len(pages)}\")"
)
PDF_DUMP_MD="${PDF_DUMP_MD:-1}" \
PDF_DUMP_DIR="${PDF_DUMP_DIR:-var/extracted}" \
DOCLING_DO_OCR="${DOCLING_DO_OCR:-1}" \
DOCLING_DO_TABLE_STRUCTURE="${DOCLING_DO_TABLE_STRUCTURE:-0}" \
DOCLING_FORCE_FULL_PAGE_OCR="${DOCLING_FORCE_FULL_PAGE_OCR:-0}" \
DOCLING_FORCE_BACKEND_TEXT="${DOCLING_FORCE_BACKEND_TEXT:-0}" \
INPUT_PDF_PATH="${pdf_container_path}" \
dc "${dc_args[@]}"

if [[ -n "${out_abs}" ]]; then
  mv -f "${default_out}" "${out_abs}"
  echo "Wrote: ${out_rel}"
else
  echo "Wrote: var/extracted/${default_name}"
fi

elapsed=$((SECONDS - started_at))
echo "Elapsed: $(format_elapsed "${elapsed}")"
