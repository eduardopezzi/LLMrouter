"""RAGFlow SDK client (E5-B1) — retrieval-only p/ acoplamento Lite (ADR-0001).

Usa POST /api/v1/retrieval (Bearer ragflow-...) que suporta múltiplos
dataset_ids num único request e devolve chunks com similaridade híbrida.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from llmrouter.config import RagConfig


@dataclass(slots=True)
class RagChunk:
    """Chunk normalizado devolvido pelo gateway."""

    content: str
    score: float
    document_id: str | None = None
    dataset_id: str | None = None
    title: str | None = None


class RagflowError(RuntimeError):
    """Falha de comunicação/resposta com o RAGFlow."""


class RagflowClient:
    """Cliente HTTP mínimo do RAGFlow (retrieval)."""

    def __init__(self, config: RagConfig, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._http = http_client

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._config.timeout_seconds)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def retrieve(
        self,
        query: str,
        dataset_ids: list[str],
        *,
        top_k: int = 4,
        similarity_threshold: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Retrieval no RAGFlow; devolve records no formato dify-like normalizado."""
        if not query.strip():
            raise RagflowError("query vazia")
        if not dataset_ids:
            raise RagflowError("dataset_ids vazio")

        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        payload = {
            "question": query,
            "dataset_ids": dataset_ids,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "page": 1,
            "page_size": max(top_k, 1) * len(dataset_ids),
        }

        url = f"{self._config.base_url.rstrip('/')}/api/v1/retrieval"
        started = time.perf_counter()
        try:
            client = await self._client()
            response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise RagflowError(f"RAGFlow inacessível: {exc}") from exc

        if response.status_code != 200:
            raise RagflowError(f"RAGFlow HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise RagflowError("RAGFlow devolveu corpo não-JSON") from exc

        # RAGFlow empacota erros como {"code": <n>, "message": ...} com HTTP 200
        code = body.get("code")
        if code not in (0, None):
            raise RagflowError(f"RAGFlow code={code}: {body.get('message', '?')}")

        chunks = (body.get("data") or {}).get("chunks") or []
        records: list[dict[str, Any]] = []
        for chunk in chunks:
            content = chunk.get("content") or chunk.get("content_with_weight") or ""
            if not content:
                continue
            records.append(
                {
                    "content": content,
                    "score": float(chunk.get("similarity") or 0.0),
                    "title": chunk.get("document_keyword") or None,
                    "metadata": {
                        "document_id": chunk.get("document_id"),
                        "dataset_id": chunk.get("dataset_id") or chunk.get("kb_id"),
                        "term_similarity": chunk.get("term_similarity"),
                        "vector_similarity": chunk.get("vector_similarity"),
                    },
                }
            )
        _ = time.perf_counter() - started  # elapsed medido pela rota
        return records
