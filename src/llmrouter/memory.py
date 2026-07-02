"""Project-scoped local memory for lightweight RAG."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from llmrouter.logging_config import get_logger

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{3,}")
_MAX_STORED_TEXT_CHARS = 12000
_logger = get_logger("llmrouter.memory")


@dataclass(frozen=True)
class MemoryConfig:
    """Runtime knobs for local project memory."""

    enabled: bool = False
    backend: str = "local"
    db_path: str = "data/llmrouter_memory.db"
    default_project: str = "default"
    top_k: int = 4
    min_score: float = 0.12
    max_context_chars: int = 2400
    min_prompt_chars: int = 80
    min_response_chars: int = 40


@dataclass(frozen=True)
class PrecogMemoryConfig:
    """PRecog memory/RAG backend settings."""

    enabled: bool = False
    base_url: str = "http://localhost:8888"
    api_key: str | None = None
    timeout: float = 3.0
    default_project: str = "default"
    top_k: int = 4
    min_score: float = 0.12
    max_context_chars: int = 2400
    query_path: str = "/internal/rag/query"
    record_path: str = "/internal/llmrouter/observations"


@dataclass(frozen=True)
class MemoryEntry:
    """A retrieved memory snippet."""

    id: int
    project: str
    prompt: str
    response: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        prompt = _compact(self.prompt, 420)
        response = _compact(self.response, 720)
        return f"Previous request: {prompt}\nUseful outcome: {response}"


class MemoryStore(Protocol):
    """Common interface for memory/RAG stores."""

    @property
    def config(self) -> MemoryConfig | PrecogMemoryConfig:
        """Return runtime config for context rendering and project defaults."""
        ...

    def retrieve(self, *, project: str, query: str) -> list[MemoryEntry]:
        """Return memories relevant to a query within one project."""
        ...

    def record_interaction(
        self,
        *,
        project: str,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Persist an interaction when useful for future requests."""
        ...


class SQLiteMemoryStore:
    """SQLite-backed memory store with simple lexical retrieval."""

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._initialized = False

    @property
    def config(self) -> MemoryConfig:
        return self._config

    def retrieve(self, *, project: str, query: str) -> list[MemoryEntry]:
        """Return memories relevant to a query within one project."""
        if not self._config.enabled or not query.strip():
            return []
        self._ensure_initialized()
        query_terms = _token_weights(query)
        if not query_terms:
            return []

        candidates: list[MemoryEntry] = []
        with sqlite3.connect(self._config.db_path) as db:
            rows = db.execute(
                """
                SELECT id, project, prompt, response, metadata_json, token_json
                FROM memories
                WHERE project = ?
                ORDER BY id DESC
                LIMIT 400
                """,
                (project,),
            ).fetchall()

        for row in rows:
            token_weights = _json_dict(row[5])
            score = _cosine_score(query_terms, token_weights)
            if score < self._config.min_score:
                continue
            candidates.append(
                MemoryEntry(
                    id=int(row[0]),
                    project=str(row[1]),
                    prompt=str(row[2]),
                    response=str(row[3]),
                    score=score,
                    metadata=_json_dict(row[4]),
                )
            )

        candidates.sort(key=lambda item: (-item.score, -item.id))
        return candidates[: max(self._config.top_k, 0)]

    def record_interaction(
        self,
        *,
        project: str,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Persist an interaction when it is substantial enough to be useful later."""
        if not self._config.enabled:
            return False
        prompt = prompt.strip()
        response = response.strip()
        if len(prompt) < self._config.min_prompt_chars:
            return False
        if len(response) < self._config.min_response_chars:
            return False

        self._ensure_initialized()
        prompt = _compact(prompt, _MAX_STORED_TEXT_CHARS)
        response = _compact(response, _MAX_STORED_TEXT_CHARS)
        token_weights = _token_weights(f"{prompt}\n{response}")
        if not token_weights:
            return False
        now = int(time.time())
        with sqlite3.connect(self._config.db_path) as db:
            db.execute(
                """
                INSERT INTO memories (
                    project, prompt, response, metadata_json, token_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project,
                    prompt,
                    response,
                    json.dumps(metadata or {}, sort_keys=True),
                    json.dumps(token_weights, sort_keys=True),
                    now,
                ),
            )
            db.commit()
        return True

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        Path(self._config.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._config.db_path) as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    token_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_project_id
                    ON memories(project, id DESC);
                """
            )
            db.commit()
        self._initialized = True


class PrecogMemoryStore:
    """PRecog-backed project memory/RAG store."""

    def __init__(self, config: PrecogMemoryConfig) -> None:
        self._config = config

    @property
    def config(self) -> PrecogMemoryConfig:
        return self._config

    def retrieve(self, *, project: str, query: str) -> list[MemoryEntry]:
        if not self._config.enabled or not query.strip():
            return []
        payload = {
            "project": project,
            "query": query,
            "top_k": self._config.top_k,
            "min_score": self._config.min_score,
            "max_context_chars": self._config.max_context_chars,
            "source": "llmrouter",
        }
        try:
            response = httpx.post(
                f"{self._config.base_url.rstrip('/')}{self._config.query_path}",
                json=payload,
                headers=self._headers(),
                timeout=self._config.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            _logger.warning("PRecog memory retrieval failed project=%s: %s", project, exc)
            return []
        raw_entries = body.get("memories") or body.get("results") or []
        if not isinstance(raw_entries, list):
            return []
        entries: list[MemoryEntry] = []
        for index, item in enumerate(raw_entries, 1):
            if not isinstance(item, dict):
                continue
            entry = _precog_entry(item, project=project, fallback_id=index)
            if entry is not None:
                entries.append(entry)
        return entries[: max(self._config.top_k, 0)]

    def record_interaction(
        self,
        *,
        project: str,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if not self._config.enabled:
            return False
        prompt = prompt.strip()
        response = response.strip()
        if not prompt or not response:
            return False
        payload = {
            "project": project,
            "source": "llmrouter",
            "prompt": _compact(prompt, _MAX_STORED_TEXT_CHARS),
            "response": _compact(response, _MAX_STORED_TEXT_CHARS),
            "metadata": metadata or {},
        }
        request_id = (metadata or {}).get("request_id")
        if request_id:
            payload["request_id"] = request_id
        try:
            response_obj = httpx.post(
                f"{self._config.base_url.rstrip('/')}{self._config.record_path}",
                json=payload,
                headers=self._headers(),
                timeout=self._config.timeout,
            )
            response_obj.raise_for_status()
        except Exception as exc:
            _logger.warning("PRecog memory record failed project=%s: %s", project, exc)
            return False
        return True

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers


def render_memory_context(entries: list[MemoryEntry], *, max_chars: int) -> str:
    """Render retrieved memories as a compact system prompt block."""
    if not entries or max_chars <= 0:
        return ""
    lines = [
        "Relevant project memory retrieved from previous interactions.",
        "Use it only when it helps answer the current request; prefer current user context.",
    ]
    remaining = max_chars - sum(len(line) + 1 for line in lines)
    for index, entry in enumerate(entries, 1):
        chunk = f"[memory {index} | score={entry.score:.2f}]\n{entry.text}"
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = _compact(chunk, remaining)
        lines.append(chunk)
        remaining -= len(chunk) + 1
    return "\n\n".join(lines)


def _precog_entry(
    item: dict[str, Any],
    *,
    project: str,
    fallback_id: int,
) -> MemoryEntry | None:
    text = item.get("text") or item.get("content") or item.get("response") or ""
    prompt = item.get("prompt") or item.get("title") or item.get("summary") or ""
    response = item.get("response") or text
    if not isinstance(prompt, str):
        prompt = str(prompt)
    if not isinstance(response, str):
        response = str(response)
    if not prompt and not response:
        return None
    raw_score = item.get("score") or item.get("similarity") or item.get("relevance") or 0.0
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    entry_id = item.get("id") or item.get("memory_id") or fallback_id
    try:
        parsed_id = int(entry_id)
    except (TypeError, ValueError):
        parsed_id = fallback_id
    return MemoryEntry(
        id=parsed_id,
        project=str(item.get("project") or project),
        prompt=prompt,
        response=response,
        score=score,
        metadata=metadata,
    )


def _token_weights(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in _TOKEN_RE.findall(text.lower()):
        if token.isdigit():
            continue
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _cosine_score(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = 0.0
    for token, weight in left.items():
        dot += weight * right.get(token, 0.0)
    if dot == 0:
        return 0.0
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _json_dict(raw: object) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _compact(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    if limit <= 1:
        return normalized[:limit]
    return normalized[: limit - 1].rstrip() + "…"
