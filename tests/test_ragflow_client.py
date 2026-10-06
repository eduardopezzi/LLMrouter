"""Tests for the RAGFlow coupling (ADR-0001 — Cenário A Lite / E5-B1).

Mocking strategy
----------------
The :class:`llmrouter.core.ragflow_client.RagflowClient` accepts an injected
``client`` object that must expose the ``post()`` / ``aclose()`` methods of
``httpx.AsyncClient``.  Tests use a light shim that records calls and serves
scripted responses — close enough to the real contract to catch regressions
in URL / headers / payload mapping without standing up a RAGFlow fixture.
"""

from __future__ import annotations

import types
from typing import Any

from fastapi.testclient import TestClient

from llmrouter.api.routes import create_app
from llmrouter.config import RagflowConfig, Settings


class _ScriptedHTTPXClient:
    """Shim for ``httpx.AsyncClient`` that returns canned responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.closed = False

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> Any:
        self.calls.append((url, json, headers))
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self) -> None:
        self.closed = True

    __aenter__ = None  # not used
    __aexit__ = None
    timeout: Any = None


def _payload(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"code": 0, "records": list(records or [])}


def _response(
    body: dict[str, Any],
    *,
    status_code: int = 200,
    json_error: Exception | None = None,
) -> Any:
    return types.SimpleNamespace(
        status_code=status_code,
        json=(lambda: body) if json_error is None else _raise(json_error),
        text=str(body)[:200],
    )


def _raise(exc: Exception) -> Any:
    def _inner() -> Any:
        raise exc

    return _inner


def _make_client(config: RagflowConfig, fake_http: _ScriptedHTTPXClient) -> Any:
    from llmrouter.core.ragflow_client import RagflowClient

    return RagflowClient(config, client=fake_http)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Route-level tests
# ---------------------------------------------------------------------------


async def test_route_returns_404_when_flag_disabled_at_runtime() -> None:
    """``ragflow.enabled=False`` → /v1/llmrouter/rag/query 404s."""
    app = create_app(ragflow_client=None)

    with TestClient(app) as tc:
        response = tc.post(
            "/v1/llmrouter/rag/query",
            json={"query": "hello"},
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 404
    assert "disabled" in response.json()["detail"]


async def test_route_returns_records_on_success() -> None:
    """Happy path: RAGFlow returns records → route returns them."""
    config = RagflowConfig(
        enabled=True,
        base_url="http://ragflow.example",
        api_key="api-key",
    )
    fake = _ScriptedHTTPXClient(
        [
            _response(
                _payload(
                    [
                        {
                            "content": "chunk1",
                            "score": 0.92,
                            "title": "doc1",
                            "metadata": {"page": 3},
                        }
                    ]
                )
            )
        ]
    )
    app = create_app(ragflow_client=_make_client(config, fake))

    with TestClient(app) as tc:
        response = tc.post(
            "/v1/llmrouter/rag/query",
            json={"query": "what is a mutex", "dataset_id": "d1"},
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is False
    assert len(body["records"]) == 1
    assert body["records"][0]["content"] == "chunk1"
    assert body["records"][0]["score"] == 0.92
    assert body["records"][0]["title"] == "doc1"
    assert body["records"][0]["metadata"] == {"page": 3}
    assert body["query"] == "what is a mutex"
    assert body["dataset_id"] == "d1"

    url, payload, headers = fake.calls[0]
    assert url.endswith("/api/v1/dify/retrieval")
    assert payload["knowledge_id"] == "d1"
    assert payload["query"] == "what is a mutex"
    assert headers["Authorization"] == "Bearer api-key"
    assert headers["Content-Type"] == "application/json"


async def test_route_degrades_gracefully_on_5xx() -> None:
    """5xx from RAGFlow → 200 with degraded=True, never raise."""
    config = RagflowConfig(
        enabled=True,
        base_url="http://ragflow.example",
        api_key="api-key",
        default_dataset_id="default-ds",
        retries=1,
    )
    fake = _ScriptedHTTPXClient(
        [_response({"message": "boom"}, status_code=500)] * 2
    )
    app = create_app(ragflow_client=_make_client(config, fake))

    with TestClient(app) as tc:
        response = tc.post(
            "/v1/llmrouter/rag/query",
            json={"query": "q"},
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["records"] == []
    assert "status 500" in body["error"]


async def test_route_degrades_gracefully_on_network_error() -> None:
    """Network error from httpx → degraded=True."""
    import httpx

    config = RagflowConfig(
        enabled=True,
        base_url="http://ragflow.example",
        default_dataset_id="d",
        retries=0,
    )
    fake = _ScriptedHTTPXClient([httpx.TimeoutException("timeout boom")])
    app = create_app(ragflow_client=_make_client(config, fake))

    with TestClient(app) as tc:
        response = tc.post(
            "/v1/llmrouter/rag/query",
            json={"query": "q"},
            headers={"Authorization": "Bearer test"}, )
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert "TimeoutException" in body["error"]


async def test_route_missing_dataset_returns_degraded() -> None:
    """Missing dataset_id + no default → degraded; HTTP not touched."""
    config = RagflowConfig(
        enabled=True,
        base_url="http://ragflow.example",
        api_key="k",
    )
    fake = _ScriptedHTTPXClient([])
    app = create_app(ragflow_client=_make_client(config, fake))

    with TestClient(app) as tc:
        response = tc.post(
            "/v1/llmrouter/rag/query",
            json={"query": "q"},
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["error"] == "no_dataset_id"
    assert fake.calls == []


async def test_route_health_returns_client_stats() -> None:
    config = RagflowConfig(
        enabled=True, base_url="http://ragflow.example", api_key="k"
    )
    fake = _ScriptedHTTPXClient([])
    app = create_app(ragflow_client=_make_client(config, fake))

    with TestClient(app) as tc:
        response = tc.get(
            "/v1/llmrouter/rag/health",
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["base_url"] == "http://ragflow.example"
    assert body["circuit_open"] is False
    assert body["consecutive_failures"] == 0
    assert body["timeout_seconds"] > 0
    assert body["retries"] >= 0


async def test_route_health_404_when_disabled() -> None:
    app = create_app(ragflow_client=None)
    with TestClient(app) as tc:
        response = tc.get(
            "/v1/llmrouter/rag/health",
            headers={"Authorization": "Bearer test"},
        )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Client-level tests (RagflowClient contract)
# ---------------------------------------------------------------------------


async def test_client_success_returns_parsed_records() -> None:
    from llmrouter.core.ragflow_client import RagflowClient

    config = RagflowConfig(
        enabled=True,
        base_url="http://ragflow.example",
        default_dataset_id="d",
    )
    fake = _ScriptedHTTPXClient(
        [
            _response(
                _payload(
                    [
                        {
                            "content": "c1",
                            "score": 0.9,
                            "title": "t",
                            "metadata": {"k": "v"},
                        }
                    ]
                )
            )
        ]
    )
    client = RagflowClient(config, client=fake)
    records = await client.retrieval(query="q")
    assert len(records) == 1
    assert records[0].content == "c1"
    assert records[0].score == 0.9
    assert records[0].title == "t"
    assert records[0].metadata == {"k": "v"}
    assert not client.circuit_open
    await client.close()


async def test_client_disabled_does_not_perform_http_call() -> None:
    from llmrouter.core.ragflow_client import RagflowClient

    config = RagflowConfig(
        enabled=False,
        base_url="http://ragflow.example",
        default_dataset_id="d",
    )
    fake = _ScriptedHTTPXClient([])
    client = RagflowClient(config, client=fake)
    result = await client.retrieval(query="q")
    assert result.error == "ragflow_disabled"
    assert fake.calls == []
    await client.close()


async def test_client_circuit_opens_and_recovers() -> None:
    from llmrouter.core.ragflow_client import RagflowClient

    config = RagflowConfig(
        enabled=True,
        base_url="http://ragflow.example",
        default_dataset_id="d",
        retries=0,
        circuit_breaker_failures=2,
        circuit_breaker_cooldown_seconds=60,
    )
    fake = _ScriptedHTTPXClient(
        [_response({"message": "500"}, status_code=500)] * 2
    )

    client = RagflowClient(config, client=fake)

    await client.retrieval(query="q1")
    assert client.circuit_open is False

    await client.retrieval(query="q2")
    assert client.circuit_open is True

    # Third call short-circuits without HTTP traffic.
    result = await client.retrieval(query="q3")
    assert result.error == "circuit_open"
    assert len(fake.calls) == 2
    await client.close()


async def test_client_4xx_does_not_trip_circuit() -> None:
    from llmrouter.core.ragflow_client import RagflowClient

    config = RagflowConfig(
        enabled=True,
        base_url="http://ragflow.example",
        default_dataset_id="d",
        retries=0,
        circuit_breaker_failures=2,
    )
    fake = _ScriptedHTTPXClient(
        [_response({"message": "bad request"}, status_code=400)]
    )
    client = RagflowClient(config, client=fake)

    result = await client.retrieval(query="q")
    assert "http_400" in result.error
    assert client.circuit_open is False
    assert client.stats["consecutive_failures"] == 0
    await client.close()


# ---------------------------------------------------------------------------
# Wiring tests (runtime builder)
# ---------------------------------------------------------------------------


async def test_runtime_builder_disabled_returns_none() -> None:
    from llmrouter.runtime import _build_ragflow_client

    settings = Settings(ragflow={"enabled": False})
    assert _build_ragflow_client(settings) is None


async def test_runtime_builder_enabled_returns_client() -> None:
    from llmrouter.core.ragflow_client import RagflowClient
    from llmrouter.runtime import _build_ragflow_client

    settings = Settings(
        ragflow={
            "enabled": True,
            "base_url": "http://x",
            "api_key": "k",
        }
    )
    client = _build_ragflow_client(settings)
    assert isinstance(client, RagflowClient)
    await client.close()
