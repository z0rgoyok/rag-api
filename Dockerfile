FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for API runtime only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY requirements-docker.txt /app/requirements-docker.txt

RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -r /app/requirements-docker.txt

COPY core /app/core
COPY apps /app/apps

RUN pip install --no-cache-dir --no-deps -e /app

EXPOSE 8080

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
