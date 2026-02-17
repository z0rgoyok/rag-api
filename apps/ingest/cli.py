from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import uuid

from dotenv import load_dotenv
import httpx

from core.config import load_settings
from core.db import Db
from core.embeddings_client import EmbeddingsClient, build_embeddings_client
from core.schema import ensure_schema, get_schema_info
from core.chunking import ChunkingStrategy, PageText, build_chunking_strategy
from core.chunking.factory import ChunkingSettings
from .pdf_extract import extract_pdf_text_pages
from .store import delete_document_by_source_path, insert_embeddings, insert_segments, sha256_file, upsert_document


async def ingest_pdf(
    db: Db,
    lm: EmbeddingsClient,
    chunker: ChunkingStrategy,
    *,
    pdf_path: Path,
    embedding_model: str,
    force: bool,
) -> None:
    file_hash = sha256_file(pdf_path)
    source_path = str(pdf_path)
    if force:
        delete_document_by_source_path(db, source_path=source_path)
    doc = upsert_document(db, source_path=source_path, title=pdf_path.name, sha256=file_hash)
    if doc.up_to_date:
        return

    pages = extract_pdf_text_pages(pdf_path)
    page_texts = [PageText(page=p.page, text=p.text) for p in pages]
    chunks = chunker.chunk(page_texts)
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

    insert_segments(db, segments=segment_rows)

    embeddings = await lm.embeddings(model=embedding_model, input_texts=[c.content for c in chunks], input_type="RETRIEVAL_DOCUMENT")
    embed_rows = []
    for seg_id, vec in zip(segment_ids, embeddings):
        embed_rows.append({"segment_id": seg_id, "embedding": vec})
    insert_embeddings(db, rows=embed_rows)


def main() -> None:
    load_dotenv()
    settings = load_settings()
    db = Db(settings.database_url)
    embed_client = build_embeddings_client(settings)
    chunker = build_chunking_strategy(
        ChunkingSettings(
            strategy=settings.chunking_strategy,
            chunk_size=settings.chunking_chunk_size,
            overlap_chars=settings.chunking_overlap_chars,
            similarity_threshold=settings.chunking_similarity_threshold,
        )
    )

    parser = argparse.ArgumentParser(prog="rag-ingest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDFs from a directory")
    p_ingest.add_argument("--pdf-dir", required=True, type=str)
    p_ingest.add_argument("--force", action="store_true", help="Re-ingest even if file hash unchanged")

    args = parser.parse_args()

    info = get_schema_info(db)
    if info is None:
        try:
            dim = asyncio.run(embed_client.probe_embedding_dim(model=settings.embeddings_model))
        except httpx.ConnectError as e:
            raise SystemExit(
                "\n".join(
                    [
                        "Failed to connect to the OpenAI-compatible inference server.",
                        f"- EMBEDDINGS_BASE_URL={settings.embeddings_base_url}",
                        "",
                        "Fix:",
                        "- Start your OpenAI-compatible server (LM Studio or any external provider base URL).",
                        "- If you run ingest via Docker and target LM Studio on the host, set EMBEDDINGS_BASE_URL to http://host.docker.internal:1234/v1",
                        "  (localhost inside a container points to the container itself).",
                        "- Ensure EMBEDDINGS_MODEL is set to an embedding-capable model name.",
                        "",
                        f"Details: {e}",
                    ]
                )
            ) from e
        except httpx.HTTPError as e:
            raise SystemExit(
                "\n".join(
                    [
                        "Inference server request failed while probing embedding dimension.",
                        f"- EMBEDDINGS_BASE_URL={settings.embeddings_base_url}",
                        f"- EMBEDDINGS_MODEL={settings.embeddings_model}",
                        "",
                        "Fix:",
                        "- Ensure the server is running and the model name matches exactly.",
                        "- Ensure the selected model supports the /embeddings endpoint.",
                        "",
                        f"Details: {e}",
                    ]
                )
            ) from e
        ensure_schema(db, embedding_dim=dim)

    if args.cmd == "ingest":
        pdf_dir = Path(args.pdf_dir).expanduser().resolve()
        pdfs = sorted([p for p in pdf_dir.glob("**/*.pdf") if p.is_file()])
        if not pdfs:
            return

        async def _run() -> None:
            for p in pdfs:
                await ingest_pdf(db, embed_client, chunker, pdf_path=p, embedding_model=settings.embeddings_model, force=bool(args.force))

        asyncio.run(_run())


if __name__ == "__main__":
    main()
