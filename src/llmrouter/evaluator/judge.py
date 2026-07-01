"""Local LLM quality judge backed by Ollama."""

from __future__ import annotations

import json
from typing import Any

import httpx

from llmrouter.evaluator.types import ComparisonResult, QualityScore


def _clamp_score(value: object) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 3
    return min(max(score, 1), 5)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from model output, tolerating fenced or prefixed text."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("judge response must be a JSON object")
    return parsed


class QualityJudge:
    """Evaluate response quality using a local Ollama model."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        model: str = "qwen2.5-coder:3b",
        timeout: float = 60.0,
        temperature: float = 0.1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._client = client

    async def evaluate(self, prompt: str, response: str, model_used: str) -> QualityScore:
        """Return a quality score for a completed model response."""
        system = (
            "You are a strict evaluator for an LLM gateway. Return only JSON with "
            "integer fields relevance, accuracy, completeness, concision, safety "
            "from 1 to 5, plus a short rationale string."
        )
        user = (
            f"Prompt:\n{prompt}\n\nModel used: {model_used}\n\n"
            f"Response to evaluate:\n{response}"
        )
        data = await self._chat_json(system, user)
        return QualityScore(
            relevance=_clamp_score(data.get("relevance")),
            accuracy=_clamp_score(data.get("accuracy")),
            completeness=_clamp_score(data.get("completeness")),
            concision=_clamp_score(data.get("concision")),
            safety=_clamp_score(data.get("safety")),
            rationale=str(data.get("rationale", "")),
        )

    async def compare(self, prompt: str, response_a: str, response_b: str) -> ComparisonResult:
        """Compare two responses for the same prompt."""
        system = (
            "You compare two LLM responses. Return only JSON with winner as 'a', 'b', "
            "or 'tie', confidence from 0 to 1, and rationale."
        )
        user = f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}"
        data = await self._chat_json(system, user)
        winner = str(data.get("winner", "tie")).lower()
        if winner not in {"a", "b", "tie"}:
            winner = "tie"
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return ComparisonResult(
            winner=winner,
            confidence=min(max(confidence, 0.0), 1.0),
            rationale=str(data.get("rationale", "")),
        )

    async def _chat_json(self, system: str, user: str) -> dict[str, Any]:
        payload: dict[str, object] = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": self._temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._client is not None:
            response = await self._client.post("/api/chat", json=payload, headers=self._headers())
            response.raise_for_status()
            return _extract_json(_ollama_content(response.json()))

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        ) as client:
            response = await client.post("/api/chat", json=payload, headers=self._headers())
            response.raise_for_status()
            return _extract_json(_ollama_content(response.json()))

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}


def _ollama_content(body: dict[str, Any]) -> str:
    message = body.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    response = body.get("response")
    if isinstance(response, str):
        return response
    raise ValueError("Ollama response did not include message.content")
