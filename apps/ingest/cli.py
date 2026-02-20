from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from typing import Callable, Literal
import uuid

from dotenv import load_dotenv
import httpx

from core.chunking import Chunk, ChunkingStrategy, PageText, build_chunking_strategy
from core.chunking.factory import ChunkingSettings
from core.config import Settings, load_settings
from core.db import Db
from core.embeddings_client import EmbeddingsClient, build_embeddings_client
from core.qdrant import Qdrant
from core.schema import ensure_ingest_task_schema, ensure_schema, get_schema_info

from .chunk_sanitize import SanitizedChunk, sanitize_chunks, sanitize_chunks_with_raw
from .extract_output import build_extract_output_path, is_extract_output_up_to_date, write_extract_output
from .pdf_extract import describe_pdf_extraction_mode, extract_pdf_docling_chunks, extract_pdf_text_pages
from .store import is_document_up_to_date, replace_document_content, sha256_file
from .task_store import (
    InputMode,
    PipelineMode,
    RunStrategy,
    claim_next_ingest_task_item,
    create_ingest_task,
    get_ingest_task,
    get_ingest_task_stats,
    mark_ingest_task_completed_if_done,
    mark_ingest_task_failed,
    mark_ingest_task_interrupted,
    mark_ingest_task_item_completed,
    mark_ingest_task_item_failed,
    mark_ingest_task_item_skipped,
    prepare_ingest_task_run,
    touch_ingest_task,
    touch_ingest_task_item,
)


def _exception_text(exc: BaseException) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return message[:2000]


def _resolve_extract_output_dir(raw: str) -> Path:
    out_dir = Path(raw).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    var_root = (Path.cwd() / "var").resolve()
    try:
        out_dir.relative_to(var_root)
    except ValueError as e:
        raise SystemExit(
            f"Refusing to write extract chunks outside var/: {out_dir} (expected under {var_root})"
        ) from e
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_input_dir(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _collect_pdf_files(pdf_dir: Path) -> list[Path]:
    return sorted([p for p in pdf_dir.glob("**/*.pdf") if p.is_file()])


def _collect_chunk_files(chunks_dir: Path) -> list[Path]:
    return sorted([p for p in chunks_dir.glob("**/*.chunks.jsonl") if p.is_file()])


def _load_chunks_from_jsonl(chunks_path: Path) -> tuple[str, str, str, list[Chunk]]:
    source_path: str | None = None
    source_sha256: str | None = None
    chunking_strategy: str | None = None
    chunks: list[Chunk] = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at {chunks_path}:{line_no}") from e

            row_source_path = str(payload.get("source_path") or "").strip()
            row_sha = str(payload.get("source_sha256") or "").strip()
            row_strategy = str(payload.get("chunking_strategy") or "").strip()
            if not row_source_path or not row_sha or not row_strategy:
                raise RuntimeError(f"Missing required metadata at {chunks_path}:{line_no}")

            if source_path is None:
                source_path = row_source_path
                source_sha256 = row_sha
                chunking_strategy = row_strategy
            elif row_source_path != source_path or row_sha != source_sha256 or row_strategy != chunking_strategy:
                raise RuntimeError(
                    f"Inconsistent metadata in {chunks_path}:{line_no} "
                    "(source_path/source_sha256/chunking_strategy mismatch)"
                )

            try:
                ordinal = int(payload["ordinal"])
            except (KeyError, TypeError, ValueError) as e:
                raise RuntimeError(f"Invalid ordinal at {chunks_path}:{line_no}") from e

            page_raw = payload.get("page")
            page = None if page_raw is None else int(page_raw)
            content = str(payload.get("content") or "")
            chunks.append(Chunk(page=page, ordinal=ordinal, content=content))

    if source_path is None or source_sha256 is None or chunking_strategy is None:
        raise RuntimeError(f"Chunks file is empty: {chunks_path}")

    ordered = sorted(chunks, key=lambda c: c.ordinal)
    return source_path, source_sha256, chunking_strategy, sanitize_chunks(ordered)


def _build_chunker(
    *,
    chunking_strategy: str,
    chunk_size: int,
    overlap_chars: int,
    similarity_threshold: float,
) -> ChunkingStrategy | None:
    if chunking_strategy in {"docling_hierarchical", "docling_hybrid"}:
        return None
    return build_chunking_strategy(
        ChunkingSettings(
            strategy=chunking_strategy,  # type: ignore[arg-type]
            chunk_size=chunk_size,
            overlap_chars=overlap_chars,
            similarity_threshold=similarity_threshold,
        )
    )


def _extract_chunks(*, pdf_path: Path, chunker: ChunkingStrategy | None, chunking_strategy: str) -> list[SanitizedChunk]:
    chunks: list[Chunk]
    if chunking_strategy in {"docling_hierarchical", "docling_hybrid"}:
        docling_chunks = extract_pdf_docling_chunks(pdf_path, strategy=chunking_strategy)
        chunks = [
            Chunk(
                page=c.page,
                ordinal=c.ordinal,
                content=c.text,
            )
            for c in docling_chunks
        ]
    else:
        if chunker is None:
            raise RuntimeError("Chunker is not configured for non-docling strategy.")
        pages = extract_pdf_text_pages(pdf_path)
        page_texts = [PageText(page=p.page, text=p.text) for p in pages]
        chunks = chunker.chunk(page_texts)
    return sanitize_chunks_with_raw(chunks)


async def ingest_pdf(
    qdrant: Qdrant,
    lm: EmbeddingsClient | None,
    chunker: ChunkingStrategy | None,
    *,
    pdf_path: Path,
    embedding_model: str,
    chunking_strategy: str,
    force: bool,
    pipeline_mode: PipelineMode,
    extract_output_dir: Path | None,
    touch: Callable[[], None] | None = None,
) -> Literal["indexed", "extracted", "up_to_date"]:
    file_hash = sha256_file(pdf_path)
    source_path = str(pdf_path)
    if touch:
        touch()

    output_path: Path | None = None
    if pipeline_mode == "extract_only":
        if extract_output_dir is None:
            raise RuntimeError("Extract output directory is required in extract-only mode.")
        output_path = build_extract_output_path(
            pdf_path=pdf_path,
            out_dir=extract_output_dir,
            chunking_strategy=chunking_strategy,
        )
        if not force and is_extract_output_up_to_date(output_path=output_path, source_sha256=file_hash):
            return "up_to_date"
    else:
        if not force and is_document_up_to_date(qdrant, source_path=source_path, sha256=file_hash):
            return "up_to_date"

    sanitized_rows = _extract_chunks(
        pdf_path=pdf_path,
        chunker=chunker,
        chunking_strategy=chunking_strategy,
    )
    chunks = [row.chunk for row in sanitized_rows]
    raw_contents = [row.raw_content for row in sanitized_rows]
    if touch:
        touch()
    if not chunks:
        raise RuntimeError(f"No chunks extracted after sanitization: {source_path}")

    if pipeline_mode == "extract_only":
        if output_path is None:
            raise RuntimeError("Internal error: missing output path in extract-only mode.")
        write_extract_output(
            output_path=output_path,
            pdf_path=pdf_path,
            source_sha256=file_hash,
            chunking_strategy=chunking_strategy,
            chunks=chunks,
            raw_contents=raw_contents,
        )
        if touch:
            touch()
        return "extracted"

    return await _embed_and_store(
        qdrant=qdrant,
        lm=lm,
        source_path=source_path,
        source_title=pdf_path.name,
        source_sha256=file_hash,
        chunks=chunks,
        embedding_model=embedding_model,
        touch=touch,
    )


async def ingest_chunks_jsonl(
    qdrant: Qdrant,
    lm: EmbeddingsClient | None,
    *,
    chunks_path: Path,
    embedding_model: str,
    expected_chunking_strategy: str,
    force: bool,
    touch: Callable[[], None] | None = None,
) -> Literal["indexed", "up_to_date"]:
    source_path, source_sha256, chunking_strategy, chunks = _load_chunks_from_jsonl(chunks_path)
    if chunking_strategy != expected_chunking_strategy:
        raise RuntimeError(
            f"Chunking strategy mismatch for {chunks_path}: "
            f"file={chunking_strategy} task={expected_chunking_strategy}"
        )
    if touch:
        touch()

    if not force and is_document_up_to_date(qdrant, source_path=source_path, sha256=source_sha256):
        return "up_to_date"
    if not chunks:
        raise RuntimeError(f"No chunks loaded after sanitization: {chunks_path}")
    return await _embed_and_store(
        qdrant=qdrant,
        lm=lm,
        source_path=source_path,
        source_title=Path(source_path).name or chunks_path.stem,
        source_sha256=source_sha256,
        chunks=chunks,
        embedding_model=embedding_model,
        touch=touch,
    )


async def _embed_and_store(
    *,
    qdrant: Qdrant,
    lm: EmbeddingsClient | None,
    source_path: str,
    source_title: str,
    source_sha256: str,
    chunks: list[Chunk],
    embedding_model: str,
    touch: Callable[[], None] | None = None,
) -> Literal["indexed"]:
    if lm is None:
        raise RuntimeError("Embeddings client is required for full ingest mode.")

    batch_size_raw = (os.getenv("INGEST_EMBED_BATCH_SIZE") or "").strip()
    try:
        batch_size = int(batch_size_raw) if batch_size_raw else 128
    except ValueError:
        batch_size = 128
    if batch_size <= 0:
        batch_size = 128

    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        end = min(len(chunks), start + batch_size)
        part = await lm.embeddings(
            model=embedding_model,
            input_texts=[c.content for c in chunks[start:end]],
            input_type="RETRIEVAL_DOCUMENT",
        )
        embeddings.extend(part)
        if touch:
            touch()

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch for {source_path}: chunks={len(chunks)} embeddings={len(embeddings)}"
        )

    replace_document_content(
        qdrant,
        source_path=source_path,
        title=source_title,
        sha256=source_sha256,
        chunks=chunks,
        embeddings=embeddings,
    )
    if touch:
        touch()
    return "indexed"


async def _run_ingest_task(
    *,
    db: Db,
    qdrant: Qdrant,
    lm: EmbeddingsClient | None,
    task_id: uuid.UUID,
    chunker: ChunkingStrategy | None,
    embedding_model: str,
    chunking_strategy: str,
    input_mode: InputMode,
    error_strategy: RunStrategy,
    pipeline_mode: PipelineMode,
    extract_output_dir: Path | None,
    force: bool,
) -> None:
    if pipeline_mode == "extract_only":
        mode_label = "extract"
    elif input_mode == "chunks":
        mode_label = "ingest"
    else:
        mode_label = "pdf_full"

    task_started_at = time.monotonic()
    while True:
        touch_ingest_task(db, task_id=task_id)
        item = claim_next_ingest_task_item(db, task_id=task_id)
        if item is None:
            break

        source_file = Path(item.source_path)

        def touch() -> None:
            touch_ingest_task(db, task_id=task_id)
            touch_ingest_task_item(db, task_item_id=item.id)

        if not source_file.is_file():
            kind = "PDF" if input_mode == "pdf" else "Chunks file"
            error = f"{kind} not found: {source_file}"
            if error_strategy == "skip":
                mark_ingest_task_item_skipped(db, task_item_id=item.id, error=error)
                print(f"item_skip task_id={task_id} ordinal={item.ordinal} path={source_file} reason={error}")
                continue
            mark_ingest_task_item_failed(db, task_item_id=item.id, error=error)
            mark_ingest_task_failed(db, task_id=task_id, error=error)
            raise RuntimeError(error)

        item_started_at = time.monotonic()
        if input_mode == "pdf":
            mode_text = describe_pdf_extraction_mode(source_file)
            print(
                f"item_start task_id={task_id} ordinal={item.ordinal} attempt={item.attempt} "
                f"path={source_file} mode={mode_label} {mode_text}"
            )
        else:
            print(
                f"item_start task_id={task_id} ordinal={item.ordinal} attempt={item.attempt} "
                f"path={source_file} mode={mode_label}"
            )
        try:
            touch()
            if input_mode == "pdf":
                outcome = await ingest_pdf(
                    qdrant,
                    lm,
                    chunker,
                    pdf_path=source_file,
                    embedding_model=embedding_model,
                    chunking_strategy=chunking_strategy,
                    force=force,
                    pipeline_mode=pipeline_mode,
                    extract_output_dir=extract_output_dir,
                    touch=touch,
                )
            else:
                if pipeline_mode != "full":
                    raise RuntimeError("Chunks input supports only full mode.")
                outcome = await ingest_chunks_jsonl(
                    qdrant,
                    lm,
                    chunks_path=source_file,
                    embedding_model=embedding_model,
                    expected_chunking_strategy=chunking_strategy,
                    force=force,
                    touch=touch,
                )
            mark_ingest_task_item_completed(db, task_item_id=item.id)
            stats = get_ingest_task_stats(db, task_id=task_id)
            out_hint = ""
            if input_mode == "pdf" and outcome == "extracted" and extract_output_dir is not None:
                out_hint = (
                    f" out={build_extract_output_path(pdf_path=source_file, out_dir=extract_output_dir, chunking_strategy=chunking_strategy)}"
                )
            print(
                f"item_done task_id={task_id} ordinal={item.ordinal} outcome={outcome} "
                f"done={stats.completed_items} failed={stats.failed_items} skipped={stats.skipped_items} total={stats.total_items}"
                f"{out_hint} mode={mode_label} elapsed_s={time.monotonic() - item_started_at:.2f}"
            )
        except Exception as e:
            error = _exception_text(e)
            if error_strategy == "skip":
                mark_ingest_task_item_skipped(db, task_item_id=item.id, error=error)
                stats = get_ingest_task_stats(db, task_id=task_id)
                print(
                    f"item_skip task_id={task_id} ordinal={item.ordinal} reason={error} "
                    f"done={stats.completed_items} failed={stats.failed_items} skipped={stats.skipped_items} total={stats.total_items} "
                    f"mode={mode_label} elapsed_s={time.monotonic() - item_started_at:.2f}"
                )
                continue
            mark_ingest_task_item_failed(db, task_item_id=item.id, error=error)
            mark_ingest_task_failed(db, task_id=task_id, error=error)
            print(
                f"item_fail task_id={task_id} ordinal={item.ordinal} reason={error} "
                f"mode={mode_label} elapsed_s={time.monotonic() - item_started_at:.2f}"
            )
            raise

    mark_ingest_task_completed_if_done(db, task_id=task_id)
    task = get_ingest_task(db, task_id=task_id)
    stats = get_ingest_task_stats(db, task_id=task_id)
    if task is not None:
        print(
            f"task_done id={task.id} status={task.status} "
            f"done={stats.completed_items} failed={stats.failed_items} skipped={stats.skipped_items} total={stats.total_items} "
            f"elapsed_s={time.monotonic() - task_started_at:.2f}"
        )


def main() -> None:
    load_dotenv()
    settings = load_settings()
    db = Db(settings.database_url)
    qdrant = Qdrant(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
    )
    embed_client: EmbeddingsClient | None = None

    parser = argparse.ArgumentParser(prog="rag-ingest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Run ingest task (new or resume)")
    p_ingest.add_argument(
        "--mode",
        choices=("pdf_full", "pdf_extract", "chunks_full", "resume"),
        required=True,
        help=(
            "Explicit run mode: "
            "pdf_full=PDF->chunks->embeddings, "
            "pdf_extract=PDF->chunks.jsonl only, "
            "chunks_full=chunks.jsonl->embeddings, "
            "resume=resume existing task by --task-id."
        ),
    )
    p_ingest.add_argument("--pdf-dir", type=str, help="Directory with PDF files (**/*.pdf)")
    p_ingest.add_argument(
        "--chunks-dir",
        type=str,
        help="Directory with extracted chunk files (**/*.chunks.jsonl)",
    )
    p_ingest.add_argument("--task-id", type=str, help="Existing task id (required for --mode resume)")
    p_ingest.add_argument("--force", action="store_true", help="Re-ingest even if file hash unchanged")
    p_ingest.add_argument(
        "--on-error",
        choices=("fail", "skip"),
        default="fail",
        help="How to handle per-file failures during this run.",
    )
    p_ingest.add_argument(
        "--extract-output-dir",
        type=str,
        default="var/extracted",
        help="Output directory for pdf_extract mode chunks (must be under var/).",
    )

    args = parser.parse_args()

    mode = str(args.mode)
    ensure_ingest_task_schema(db)

    if args.cmd != "ingest":
        return

    task_id: uuid.UUID
    requested_input_mode: InputMode | None = None
    requested_pipeline_mode: PipelineMode | None = None
    extract_output_dir: Path | None = None
    if mode == "resume":
        if args.task_id is None:
            raise SystemExit("--mode resume requires --task-id")
        if args.pdf_dir:
            raise SystemExit("--pdf-dir is not allowed with --mode resume")
        if args.chunks_dir:
            raise SystemExit("--chunks-dir is not allowed with --mode resume")
        try:
            task_id = uuid.UUID(str(args.task_id))
        except (TypeError, ValueError) as e:
            raise SystemExit(f"Invalid --task-id: {args.task_id}") from e
        task = get_ingest_task(db, task_id=task_id)
        if task is None:
            raise SystemExit(f"Ingest task not found: {task_id}")
        if task.pipeline_mode == "extract_only":
            extract_dir_raw = task.extract_output_dir or args.extract_output_dir
            extract_output_dir = _resolve_extract_output_dir(extract_dir_raw)
        print(f"task_resume id={task_id} prev_status={task.status}")
    else:
        if args.task_id:
            raise SystemExit("--task-id is allowed only with --mode resume")
        source_dir: Path
        source_paths: list[Path]

        if mode == "pdf_full":
            if not args.pdf_dir:
                raise SystemExit("--mode pdf_full requires --pdf-dir")
            if args.chunks_dir:
                raise SystemExit("--chunks-dir is not allowed with --mode pdf_full")
            requested_input_mode = "pdf"
            requested_pipeline_mode = "full"
            source_dir = _resolve_input_dir(args.pdf_dir)
            source_paths = _collect_pdf_files(source_dir)
            if not source_paths:
                print(f"No PDF files found in: {source_dir}")
                return
        elif mode == "pdf_extract":
            if not args.pdf_dir:
                raise SystemExit("--mode pdf_extract requires --pdf-dir")
            if args.chunks_dir:
                raise SystemExit("--chunks-dir is not allowed with --mode pdf_extract")
            requested_input_mode = "pdf"
            requested_pipeline_mode = "extract_only"
            source_dir = _resolve_input_dir(args.pdf_dir)
            source_paths = _collect_pdf_files(source_dir)
            if not source_paths:
                print(f"No PDF files found in: {source_dir}")
                return
            extract_output_dir = _resolve_extract_output_dir(args.extract_output_dir)
        elif mode == "chunks_full":
            if not args.chunks_dir:
                raise SystemExit("--mode chunks_full requires --chunks-dir")
            if args.pdf_dir:
                raise SystemExit("--pdf-dir is not allowed with --mode chunks_full")
            requested_input_mode = "chunks"
            requested_pipeline_mode = "full"
            source_dir = _resolve_input_dir(args.chunks_dir)
            source_paths = _collect_chunk_files(source_dir)
            if not source_paths:
                print(f"No chunk files found in: {source_dir}")
                return
        else:
            raise SystemExit(f"Unsupported --mode: {mode}")

        if requested_input_mode is None or requested_pipeline_mode is None:
            raise RuntimeError(f"Internal mode resolution error for --mode={mode}")

        task = create_ingest_task(
            db,
            source_dir=source_dir,
            source_paths=source_paths,
            input_mode=requested_input_mode,
            chunking_strategy=settings.chunking_strategy,
            error_strategy=args.on_error,
            pipeline_mode=requested_pipeline_mode,
            extract_output_dir=str(extract_output_dir) if extract_output_dir else None,
            force=bool(args.force),
        )
        task_id = task.id
        print(
            f"task_created id={task.id} files={len(source_paths)} chunking={task.chunking_strategy} "
            f"on_error={args.on_error} mode={task.pipeline_mode} input={task.input_mode} force={bool(args.force)}"
        )

    task = prepare_ingest_task_run(
        db,
        task_id=task_id,
        error_strategy=args.on_error,
        force=bool(args.force),
        expected_pipeline_mode=requested_pipeline_mode,
        expected_input_mode=requested_input_mode,
        extract_output_dir=str(extract_output_dir) if extract_output_dir else None,
    )
    pipeline_mode: PipelineMode = "extract_only" if task.pipeline_mode == "extract_only" else "full"
    input_mode: InputMode = "chunks" if task.input_mode == "chunks" else "pdf"
    if pipeline_mode == "extract_only" and extract_output_dir is None:
        extract_dir_raw = task.extract_output_dir or args.extract_output_dir
        extract_output_dir = _resolve_extract_output_dir(extract_dir_raw)
    print(
        f"task_start id={task.id} chunking={task.chunking_strategy} on_error={args.on_error} "
        f"mode={pipeline_mode} input={input_mode} force={bool(args.force)}"
    )

    if pipeline_mode == "full":
        embed_client = build_embeddings_client(settings)
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
            ensure_schema(
                db,
                qdrant,
                embedding_dim=dim,
                embedding_model=settings.embeddings_model,
            )
        else:
            ensure_schema(
                db,
                qdrant,
                embedding_dim=info.embedding_dim,
                embedding_model=settings.embeddings_model,
            )

    chunker: ChunkingStrategy | None = None
    if input_mode == "pdf":
        chunker = _build_chunker(
            chunking_strategy=task.chunking_strategy,
            chunk_size=settings.chunking_chunk_size,
            overlap_chars=settings.chunking_overlap_chars,
            similarity_threshold=settings.chunking_similarity_threshold,
        )

    try:
        asyncio.run(
            _run_ingest_task(
                db=db,
                qdrant=qdrant,
                lm=embed_client,
                task_id=task.id,
                chunker=chunker,
                embedding_model=settings.embeddings_model,
                chunking_strategy=task.chunking_strategy,
                input_mode=input_mode,
                error_strategy=args.on_error,
                pipeline_mode=pipeline_mode,
                extract_output_dir=extract_output_dir,
                force=bool(args.force),
            )
        )
    except KeyboardInterrupt as e:
        mark_ingest_task_interrupted(
            db,
            task_id=task.id,
            error="Interrupted by user (KeyboardInterrupt). Resume with --task-id.",
        )
        raise SystemExit(130) from e
    except Exception as e:
        raise SystemExit(f"Ingest task failed ({task.id}): {_exception_text(e)}") from e


if __name__ == "__main__":
    main()
