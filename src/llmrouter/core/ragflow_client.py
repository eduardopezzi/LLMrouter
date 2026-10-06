"""RAGFlow retrieval client (ADR-0001 — Cenário A Lite).

Wraps the Dify-compatible retrieval endpoint of RAGFlow
(``POST /api/v1/dify/retrieval``) behind an :class:`httpx.AsyncClient` with
deadline-aware retries and a small circuit breaker.

Design notes
------------
- **Strictly opt-in**: the runtime only builds a client when
  ``settings.ragflow.enabled`` is ``True``.
- **Minimal payload**: the Dify endpoint accepts ``{knowledge_id, query,
  retrieval_setting:{top_k, score_threshold}}`` and returns
  ``{records:[{content, score, title, metadata}]}`` — smaller than the
  full ``/api/v1/retrieval`` payload, easier to proxy.
- **No SDK dependency**: the ragflow SDK Python is internal to the
  RAGFlow server, so this client speaks HTTP directly.
- **Fail-open by design**: a transient error from RAGFlow returns a
  structured ``RagflowUnavailable`` payload (HTTP 200 with ``degraded``
  flag) instead of failing the caller's request — the LLM still has
  its baseline context to work with.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from llmrouter.config import RagflowConfig


@dataclass(frozen=True)
class RagflowRecord:
    """A single record returned by the Dify retrieval endpoint."""

    content: str
    score: float
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> RagflowRecord:
        return cls(
            content=str(payload.get("content") or ""),
            score=float(payload.get("score") or 0.0),
            title=str(payload.get("title") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RagflowUnavailable:
    """Sentinel returned when RAGFlow is down / circuit is open.

    The ``degraded=True`` flag is part of the public contract — they tell
    the caller that no retrieval happened and they should fall back to
    baseline LLM context.
    """

    error: str
    degraded: bool = True
    retry_after_seconds: float = 0.0


RagflowResult = list[RagflowRecord] | RagflowUnavailable


class RagflowClient:
    """Async HTTP client for the RAGFlow Dify-compatible retrieval endpoint.

    The client is cheap to instantiate (just holds config + counters), so
    the runtime builds a single instance at startup and reuses it across
    requests.  ``close()`` releases the underlying ``httpx.AsyncClient``.
    """

    def __init__(
        self,
        config: RagflowConfig,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Any = asyncio.sleep,
        now: Any = time.monotonic,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._now = now
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds),
        )
        # Circuit breaker counters
        self._consecutive_failures: int = 0
        self._circuit_opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def config(self) -> RagflowConfig:
        return self._config

    @property
    def circuit_open(self) -> bool:
        if self._circuit_opened_at is None:
            return False
        if (
            self._now() - self._circuit_opened_at
            >= self._config.circuit_breaker_cooldown_seconds
        ):
            return False
        return True

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "base_url": self._config.base_url,
            "circuit_open": self.circuit_open,
            "consecutive_failures": self._consecutive_failures,
            "timeout_seconds": self._config.timeout_seconds,
            "retries": self._config.retries,
        }

    async def close(self) -> None:
        await self._client.aclose()

    async def retrieval(
        self,
        *,
        query: str,
        dataset_id: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> RagflowResult:
        """Call ``POST /api/v1/dify/retrieval`` and return parsed records.

        Returns ``RagflowUnavailable`` on transient failures (network,
        timeout, 5xx, open-circuit) so the request fails-open.
        """
        if not self._config.enabled:
            return RagflowUnavailable(error="ragflow_disabled")

        target_dataset = dataset_id or self._config.default_dataset_id
        if not target_dataset:
            return RagflowUnavailable(error="no_dataset_id")

        if self.circuit_open:
            return RagflowUnavailable(
                error="circuit_open",
                retry_after_seconds=max(
                    0.0,
                    self._config.circuit_breaker_cooldown_seconds
                    - (self._now() - (self._circuit_opened_at or self._now())),
                ),
            )

        payload = {
            "knowledge_id": target_dataset,
            "query": query,
            "retrieval_setting": {
                "top_k": int(top_k or self._config.default_top_k),
                "score_threshold": float(
                    score_threshold
                    if score_threshold is not None
                    else self._config.default_score_threshold
                ),
            },
        }
        url = self._config.base_url.rstrip("/") + "/api/v1/dify/retrieval"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        last_error = ""
        attempts = self._config.retries + 1
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    url, json=payload, headers=headers
                )
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:200]
                if attempt < attempts - 1:
                    await self._sleep(0.2 * (attempt + 1))
                continue

            if response.status_code >= 500:
                last_error = f"status {response.status_code}"
                if attempt < attempts - 1:
                    await self._sleep(0.2 * (attempt + 1))
                continue

            if response.status_code >= 400:
                # 4xx is a client error — don't retry, don't trip the breaker
                return RagflowUnavailable(
                    error=f"http_{response.status_code}: {response.text[:160]}",
                )

            try:
                data = response.json()
            except ValueError as exc:
                last_error = f"json_decode: {exc}"
                continue

            if not isinstance(data, dict):
                last_error = f"unexpected_payload: {type(data).__name__}"
                continue

            code = data.get("code", 0)
            if code not in (0, 200, None):
                last_error = str(data.get("message") or f"code={code}")[:200]
                if attempt < attempts - 1:
                    await self._sleep(0.2 * (attempt + 1))
                continue

            await self._record_success()
            raw_records = data.get("records") or []
            records: list[RagflowRecord] = []
            if isinstance(raw_records, list):
                records = [
                    RagflowRecord.from_api(r)
                    for r in raw_records
                    if isinstance(r, dict)
                ]
            return records

        await self._record_failure()
        return RagflowUnavailable(error=last_error or "unknown_failure")

    async def _record_success(self) -> None:
        async with self._lock:
            self._consecutive_failures = 0
            self._circuit_opened_at = None

    async def _record_failure(self) -> None:
        async with self._lock:
            self._consecutive_failures += 1
            if (
                self._consecutive_failures
                >= self._config.circuit_breaker_failures
            ):
                self._circuit_opened_at = self._now()


__all__ = ["RagflowClient", "RagflowRecord", "RagflowUnavailable"]
