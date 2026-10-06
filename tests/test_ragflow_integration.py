"""Integration tests — RagflowClient real contra RAGFlow local (E5-B2).

Corre contra o stack docker validado no spike E5-A1:
  base_url: http://172.17.0.1:9380  (RAGFlow v0.19.1-full)

Setup exigido (documentado no épico):
  - RAGFlow up (docker compose em /opt/data/ragflow/repo/docker)
  - dataset com doc parseado (smoke_test2.py cria um novo a cada execução)
  - RAGFLOW_TEST_API_KEY no env (API key ragflow-... do tenant)

Skip automático quando o RAGFlow não está acessível (CI, dev sem stack).
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Any

import httpx
import pytest

from llmrouter.config import RagflowConfig
from llmrouter.core.ragflow_client import (
    RagflowClient,
    RagflowRecord,
    RagflowUnavailable,
)

BASE_URL = os.environ.get("RAGFLOW_TEST_BASE_URL", "http://172.17.0.1:9380")
API_KEY = os.environ.get("RAGFLOW_TEST_API_KEY", "")
# dataset criado no smoke E5-A1 (doc llmrouter.txt parseado, embd BAAI)
DATASET_ID = os.environ.get("RAGFLOW_TEST_DATASET_ID", "")


def _ragflow_up() -> bool:
    try:
        response = httpx.get(f"{BASE_URL}/v1/system/version", timeout=2.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _ready() -> bool:
    return _ragflow_up() and bool(API_KEY) and bool(DATASET_ID)


pytestmark = pytest.mark.skipif(
    not _ready(),
    reason="RAGFlow local indisponível ou sem API key/dataset de teste",
)


def _config(**overrides: Any) -> RagflowConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "timeout_seconds": 5.0,
        "retries": 0,
    }
    defaults.update(overrides)
    return RagflowConfig(**defaults)


@pytest.mark.asyncio
async def test_real_retrieval_returns_records_from_indexed_doc() -> None:
    """Happy path real: query sobre o doc llmrouter.txt indexado no smoke."""
    client = RagflowClient(_config())
    try:
        result = await client.retrieval(
            query="O que é o LLMRouter e quais caches ele tem?",
            dataset_id=DATASET_ID,
            top_k=3,
        )
    finally:
        await client.close()

    assert not isinstance(result, RagflowUnavailable), f"degraded: {result.error}"
    assert isinstance(result, list)
    assert len(result) >= 1
    top = result[0]
    assert isinstance(top, RagflowRecord)
    assert "llmrouter" in top.content.lower() or "gateway" in top.content.lower()
    assert top.score > 0.0


@pytest.mark.asyncio
async def test_real_retrieval_degrades_on_bad_dataset() -> None:
    """dataset inexistente → a rota RAGFlow responde erro de code → fail-open."""
    client = RagflowClient(_config())
    try:
        result = await client.retrieval(
            query="qualquer coisa",
            dataset_id="dataset-que-nao-existe-000",
        )
    finally:
        await client.close()

    assert isinstance(result, RagflowUnavailable)
    assert result.degraded is True


@pytest.mark.asyncio
async def test_real_retrieval_latency_p50_under_3s() -> None:
    """Probe de latência: p50 < 3s sobre 10 retrievals reais (meta B2)."""
    client = RagflowClient(_config(timeout_seconds=10.0))
    timings: list[float] = []
    try:
        for _ in range(10):
            started = time.perf_counter()
            result = await client.retrieval(
                query="retrieval híbrido do RAGFlow",
                dataset_id=DATASET_ID,
                top_k=3,
            )
            timings.append(time.perf_counter() - started)
            assert not isinstance(result, RagflowUnavailable), f"degraded: {result.error}"
    finally:
        await client.close()

    p50 = statistics.median(timings)
    p95 = sorted(timings)[int(len(timings) * 0.95) - 1]
    print(f"\nlatency: p50={p50*1000:.0f}ms p95={p95*1000:.0f}ms n={len(timings)}")
    assert p50 < 3.0, f"p50 {p50:.2f}s > 3s"
    assert p95 < 10.0, f"p95 {p95:.2f}s > 10s"


@pytest.mark.asyncio
async def test_real_health_endpoint_shape() -> None:
    """Client stats expõem o que o /v1/llmrouter/rag/health consome."""
    client = RagflowClient(_config())
    try:
        await client.retrieval(query="prova de vida", dataset_id=DATASET_ID, top_k=1)
        stats = client.stats
    finally:
        await client.close()

    assert stats["enabled"] is True
    assert stats["circuit_open"] is False
    assert stats["consecutive_failures"] == 0
