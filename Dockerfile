FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal); leave build tooling out.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY core /app/core
COPY apps /app/apps

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .

EXPOSE 8080

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
