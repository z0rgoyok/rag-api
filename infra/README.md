# infra

```bash
docker compose -f compose.yml up -d
docker compose -f compose.yml ps
```

Ingest PDFs (reads from `../var/pdfs` mounted into the container as `/data/pdfs`):

```bash
docker compose -f compose.yml run --rm api rag-ingest ingest --pdf-dir /data/pdfs
```
