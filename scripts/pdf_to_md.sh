#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/pdf_to_md.sh var/pdfs/file.pdf [var/extracted/out.md]

Notes:
  - Runs inside Docker (same env as ingest).
  - Output is written under `var/extracted/` by default (repo hygiene).
  - You can also set:
      PDF_TEXT_EXTRACTOR=pymupdf4llm|pymupdf
      PDF_DUMP_MD=1
EOF
}

pdf_rel="${1:-}"
out_rel="${2:-}"
if [[ -z "${pdf_rel}" || "${pdf_rel}" == "-h" || "${pdf_rel}" == "--help" ]]; then
  usage
  exit 2
fi

repo="$(repo_root)"
pdf_abs="${repo}/${pdf_rel}"
if [[ ! -f "${pdf_abs}" ]]; then
  echo "PDF not found: ${pdf_rel}" >&2
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

pdf_inside="${pdf_rel#var/pdfs/}"
default_name="$(basename "${pdf_inside%.pdf}").md"
default_out="${repo}/var/extracted/${default_name}"
rm -f "${default_out}" || true

# Reuse the api image because it already contains all PDF deps.
PDF_DUMP_MD="${PDF_DUMP_MD:-1}" \
PDF_DUMP_DIR="${PDF_DUMP_DIR:-var/extracted}" \
dc run --rm \
  -e RAG_REPO_ROOT=/app \
  -e PDF_DUMP_MD \
  -e PDF_TEXT_EXTRACTOR \
  -e PDF_DUMP_DIR \
  api \
  python -c "from pathlib import Path; from apps.ingest.pdf_extract import extract_pdf_text_pages; p=Path('/data/pdfs')/'${pdf_inside}'; pages=extract_pdf_text_pages(p); print(f'pages={len(pages)}')"

if [[ -n "${out_abs}" ]]; then
  mv -f "${default_out}" "${out_abs}"
  echo "Wrote: ${out_rel}"
else
  echo "Wrote: var/extracted/${default_name}"
fi
