FROM python:3.13-slim AS app-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY serper_mcp ./serper_mcp
COPY queue_handler.py minio_handler.py ./
COPY mcp_servers.json ./mcp_servers.json

RUN mkdir -p /app/data /app/models

FROM app-base AS worker
CMD [".venv/bin/python", "-m", "app.worker"]

FROM app-base AS api
EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]