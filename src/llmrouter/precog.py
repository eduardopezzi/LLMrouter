"""Best-effort PRecog observation publishing."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.precog")


class PrecogPublisher:
    """Publish LLMrouter observations and outcomes to PRecog internal APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 3.0,
        auth_failure_cooldown_seconds: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._auth_failure_cooldown_seconds = auth_failure_cooldown_seconds
        self._auth_blocked_until = 0.0

    def record_observation(self, payload: dict[str, Any]) -> None:
        """Schedule an observation POST without blocking the chat response."""
        self._schedule("POST", "/internal/llmrouter/observations", payload)

    def update_observation(self, request_id: str, outcome: dict[str, Any]) -> None:
        """Schedule an outcome PATCH without blocking the caller."""
        self._schedule(
            "PATCH",
            f"/internal/llmrouter/observations/{request_id}",
            {"outcome": outcome},
        )

    def _schedule(self, method: str, path: str, payload: dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._send(method, path, payload))

    async def _send(self, method: str, path: str, payload: dict[str, Any]) -> None:
        if self._auth_blocked_until > time.monotonic():
            _logger.debug("PRecog publication suppressed while auth circuit is open")
            return
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(method, url, json=payload, headers=headers)
                response.raise_for_status()
                self._auth_blocked_until = 0.0
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code in {401, 403}:
                self._auth_blocked_until = time.monotonic() + self._auth_failure_cooldown_seconds
            request_id = payload.get("request_id") or path.rsplit("/", 1)[-1]
            _logger.warning(
                "Failed to publish PRecog observation request_id=%s method=%s path=%s: %s",
                request_id,
                method,
                path,
                exc,
            )
            return
        except Exception as exc:
            request_id = payload.get("request_id") or path.rsplit("/", 1)[-1]
            _logger.warning(
                "Failed to publish PRecog observation request_id=%s method=%s path=%s: %s",
                request_id,
                method,
                path,
                exc,
            )
            return

        _logger.debug("Published PRecog observation method=%s path=%s", method, path)
