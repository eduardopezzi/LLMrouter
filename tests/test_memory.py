from __future__ import annotations

from typing import Any

import httpx

from llmrouter.memory import PrecogMemoryConfig, PrecogMemoryStore


def test_precog_memory_store_retrieves_entries(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        seen["url"] = url
        seen["json"] = kwargs["json"]
        seen["headers"] = kwargs["headers"]
        return httpx.Response(
            200,
            json={
                "memories": [
                    {
                        "id": 42,
                        "project": "precog",
                        "prompt": "How are contracts versioned?",
                        "response": "Contracts live in phoenix_versions.",
                        "score": 0.91,
                        "metadata": {"source": "rag"},
                    }
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    store = PrecogMemoryStore(
        PrecogMemoryConfig(
            enabled=True,
            base_url="http://precog.test",
            api_key="secret",
            top_k=3,
            min_score=0.2,
        )
    )

    entries = store.retrieve(project="precog", query="contracts versioning")

    assert seen["url"] == "http://precog.test/internal/rag/query"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["json"]["project"] == "precog"
    assert seen["json"]["query"] == "contracts versioning"
    assert seen["json"]["top_k"] == 3
    assert seen["json"]["min_score"] == 0.2
    assert entries[0].id == 42
    assert entries[0].response == "Contracts live in phoenix_versions."
    assert entries[0].score == 0.91


def test_precog_memory_store_records_interaction(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        seen["url"] = url
        seen["json"] = kwargs["json"]
        seen["headers"] = kwargs["headers"]
        return httpx.Response(202, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    store = PrecogMemoryStore(
        PrecogMemoryConfig(
            enabled=True,
            base_url="http://precog.test/",
            api_key="secret",
        )
    )

    recorded = store.record_interaction(
        project="precog",
        prompt="Remember the API contract.",
        response="The API contract uses /internal/rag/query.",
        metadata={"request_id": "req-1", "selected_model": "ollama/test"},
    )

    assert recorded is True
    assert seen["url"] == "http://precog.test/internal/llmrouter/observations"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["json"]["request_id"] == "req-1"
    assert seen["json"]["project"] == "precog"
    assert seen["json"]["source"] == "llmrouter"
    assert seen["json"]["prompt"] == "Remember the API contract."
    assert seen["json"]["response"] == "The API contract uses /internal/rag/query."
