"""Tests do endpoint RAG query (E5-B1) — Cenário A Lite do ADR-0001.

Contrato:
- POST /v1/llmrouter/rag/query
- Auth igual às demais rotas /v1/llmrouter (x-api-key ou Bearer)
- Body: {query, dataset_ids, top_k?, similarity_threshold?}
- 200: {query, results: [{content, score, document_id, dataset_id, title?}], total, elapsed_ms}
- 503 quando RAGFlow off/unconfigured; 502 em erro upstream; 400 em payload inválido
- Config: bloco rag (enabled, base_url, api_key, timeout_seconds, default_top_k)
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from llmrouter.api.routes import create_app
from llmrouter.config import RagConfig


class FakeRagflowClient:
    """Stub do RagflowClient p/ testes de rota (unidade)."""

    def __init__(
        self,
        *,
        records: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.records = records or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        dataset_ids: list[str],
        *,
        top_k: int,
        similarity_threshold: float,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "dataset_ids": dataset_ids,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
            }
        )
        if self.error:
            raise self.error
        return self.records


def _make_client(rag: RagConfig, client: FakeRagflowClient | None = None) -> TestClient:
    app = create_app(api_key="secret", rag_config=rag, ragflow_client=client)
    return TestClient(app)


def _cfg(**overrides: Any) -> RagConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "base_url": "http://ragflow:9380",
        "api_key": "ragflow-test",
        "timeout_seconds": 3.0,
        "default_top_k": 4,
        "default_similarity_threshold": 0.2,
    }
    defaults.update(overrides)
    return RagConfig(**defaults)


AUTH = {"Authorization": "Bearer secret"}


def test_rag_query_requires_auth() -> None:
    client = _make_client(_cfg(), FakeRagflowClient())
    response = client.post("/v1/llmrouter/rag/query", json={"query": "q", "dataset_ids": ["d1"]})
    assert response.status_code == 401


def test_rag_query_disabled_returns_503() -> None:
    client = _make_client(_cfg(enabled=False), FakeRagflowClient())
    response = client.post(
        "/v1/llmrouter/rag/query",
        headers=AUTH,
        json={"query": "q", "dataset_ids": ["d1"]},
    )
    assert response.status_code == 503
    assert "disabled" in response.json()["detail"].lower()


def test_rag_query_invalid_payload_returns_422_or_400() -> None:
    client = _make_client(_cfg(), FakeRagflowClient())
    # sem query
    r1 = client.post("/v1/llmrouter/rag/query", headers=AUTH, json={"dataset_ids": ["d1"]})
    # sem dataset_ids
    r2 = client.post("/v1/llmrouter/rag/query", headers=AUTH, json={"query": "q"})
    assert r1.status_code in (400, 422)
    assert r2.status_code in (400, 422)


def test_rag_query_happy_path_maps_dify_records() -> None:
    fake = FakeRagflowClient(
        records=[
            {
                "content": "LLMRouter é um gateway",
                "score": 0.91,
                "title": "llmrouter.txt",
                "metadata": {"document_id": "doc-1", "dataset_id": "kb-1"},
            },
            {
                "content": "cache semântico E1",
                "score": 0.55,
                "title": "cache.md",
                "metadata": {"document_id": "doc-2", "dataset_id": "kb-1"},
            },
        ]
    )
    client = _make_client(_cfg(), fake)
    response = client.post(
        "/v1/llmrouter/rag/query",
        headers=AUTH,
        json={"query": "o que é o llmrouter?", "dataset_ids": ["kb-1"], "top_k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "o que é o llmrouter?"
    assert body["total"] == 2
    assert len(body["results"]) == 2
    first = body["results"][0]
    assert first["content"] == "LLMRouter é um gateway"
    assert first["score"] == pytest.approx(0.91)
    assert first["document_id"] == "doc-1"
    assert first["title"] == "llmrouter.txt"
    assert body["elapsed_ms"] >= 0
    # chamada passou params corretos ao client
    assert fake.calls[0]["query"] == "o que é o llmrouter?"
    assert fake.calls[0]["dataset_ids"] == ["kb-1"]
    assert fake.calls[0]["top_k"] == 2


def test_rag_query_uses_defaults_when_omitted() -> None:
    fake = FakeRagflowClient(records=[])
    client = _make_client(_cfg(default_top_k=7, default_similarity_threshold=0.33), fake)
    response = client.post(
        "/v1/llmrouter/rag/query",
        headers=AUTH,
        json={"query": "q", "dataset_ids": ["kb"]},
    )
    assert response.status_code == 200
    assert fake.calls[0]["top_k"] == 7
    assert fake.calls[0]["similarity_threshold"] == pytest.approx(0.33)


def test_rag_query_upstream_error_returns_502() -> None:
    fake = FakeRagflowClient(error=RuntimeError("connection refused"))
    client = _make_client(_cfg(), fake)
    response = client.post(
        "/v1/llmrouter/rag/query",
        headers=AUTH,
        json={"query": "q", "dataset_ids": ["d1"]},
    )
    assert response.status_code == 502
    detail = response.json()["detail"].lower()
    assert "upstream" in detail or "ragflow" in detail


def test_rag_query_no_client_configured_returns_503() -> None:
    client = _make_client(_cfg(), None)
    response = client.post(
        "/v1/llmrouter/rag/query",
        headers=AUTH,
        json={"query": "q", "dataset_ids": ["d1"]},
    )
    assert response.status_code == 503


def test_rag_query_deduplicates_and_preserves_order() -> None:
    fake = FakeRagflowClient(
        records=[
            {"content": "a", "score": 0.9, "title": "t1", "metadata": {"document_id": "x"}},
            {"content": "a", "score": 0.8, "title": "t1", "metadata": {"document_id": "x"}},
            {"content": "b", "score": 0.7, "title": "t2", "metadata": {"document_id": "y"}},
        ]
    )
    client = _make_client(_cfg(), fake)
    body = client.post(
        "/v1/llmrouter/rag/query",
        headers=AUTH,
        json={"query": "q", "dataset_ids": ["kb"]},
    ).json()
    contents = [r["content"] for r in body["results"]]
    assert contents == ["a", "b"]  # dedup por (content, document_id), ordem preservada
    assert body["total"] == 2
