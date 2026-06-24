FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLMROUTER_SERVER__HOST=0.0.0.0 \
    LLMROUTER_SERVER__PORT=8000 \
    LLMROUTER_PROVIDERS__OLLAMA__BASE_URL=http://host.docker.internal:11434

WORKDIR /app

RUN addgroup --system llmrouter \
    && adduser --system --ingroup llmrouter --home /app llmrouter

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install --upgrade pip \
    && pip install .

RUN mkdir -p /app/data \
    && chown -R llmrouter:llmrouter /app

USER llmrouter

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["python", "-m", "uvicorn", "llmrouter.main:app", "--host", "0.0.0.0", "--port", "8000"]
