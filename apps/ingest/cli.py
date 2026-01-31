from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import uuid

from dotenv import load_dotenv

from apps.api.config import load_settings
from apps.api.db import Db
from apps.api.lmstudio import LmStudioClient
from apps.api.schema import ensure_schema, get_schema_info
from .chunking import chunk_text_pages
from .pdf_extract import extract_pdf_text_pages
from .store import insert_embeddings, insert_segments, sha256_file, upsert_document


async def ingest_pdf(db: Db, lm: LmStudioClient, *, pdf_path: Path, embedding_model: str) -> None:
    file_hash = sha256_file(pdf_path)
    doc = upsert_document(db, source_path=str(pdf_path), title=pdf_path.name, sha256=file_hash)
    if doc.up_to_date:
        return

    pages = extract_pdf_text_pages(pdf_path)
    page_pairs = [(p.page, p.text) for p in pages]
    chunks = chunk_text_pages(page_pairs)
    if not chunks:
        return

    segment_rows = []
    segment_ids: list[uuid.UUID] = []
    for c in chunks:
        seg_id = uuid.uuid4()
        segment_ids.append(seg_id)
        segment_rows.append(
            {
                "id": seg_id,
                "document_id": doc.id,
                "ordinal": c.ordinal,
                "page": c.page,
                "content": c.content,
            }
        )

    insert_segments(db, document_id=doc.id, segments=segment_rows)

    embeddings = await lm.embeddings(model=embedding_model, input_texts=[c.content for c in chunks])
    embed_rows = []
    for seg_id, vec in zip(segment_ids, embeddings):
        embed_rows.append({"segment_id": seg_id, "embedding": vec})
    insert_embeddings(db, rows=embed_rows)


def main() -> None:
    load_dotenv()
    settings = load_settings()
    db = Db(settings.database_url)
    lm = LmStudioClient(settings.lmstudio_base_url)

    parser = argparse.ArgumentParser(prog="rag-ingest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDFs from a directory")
    p_ingest.add_argument("--pdf-dir", required=True, type=str)

    args = parser.parse_args()

    info = get_schema_info(db)
    if info is None:
        dim = asyncio.run(lm.probe_embedding_dim(model=settings.lmstudio_embedding_model))
        ensure_schema(db, embedding_dim=dim)

    if args.cmd == "ingest":
        pdf_dir = Path(args.pdf_dir).expanduser().resolve()
        pdfs = sorted([p for p in pdf_dir.glob("**/*.pdf") if p.is_file()])
        if not pdfs:
            return

        async def _run() -> None:
            for p in pdfs:
                await ingest_pdf(db, lm, pdf_path=p, embedding_model=settings.lmstudio_embedding_model)

        asyncio.run(_run())


if __name__ == "__main__":
    main()
