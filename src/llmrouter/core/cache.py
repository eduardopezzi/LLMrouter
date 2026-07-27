"""Response cache with SQLite backend for exact-match caching.

Provides a ``CacheManager`` that stores non-streaming chat completion
responses keyed by a normalized hash of (prompt, model, temperature,
top_p, max_tokens).  TTL is configurable per tier.  Streaming requests
always bypass the cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmrouter.core.types import ChatRequest, ChatResponse, Tier, Usage
from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.cache")

# Default TTL per tier (seconds)
_DEFAULT_TTL_BY_TIER: dict[int, float] = {
    Tier.T1: 30.0,
    Tier.T2: 60.0,
    Tier.T3: 90.0,
}


@dataclass
class CacheStats:
    """Aggregated cache statistics."""

    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0
    entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "tokens_saved": self.tokens_saved,
            "cost_saved_usd": round(self.cost_saved_usd, 6),
            "entries": self.entries,
        }


def _cache_key(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
) -> str:
    """Generate a deterministic cache key from request parameters."""
    normalized_prompt = " ".join(prompt.split())
    payload = json.dumps(
        {
            "p": normalized_prompt,
            "m": model,
            "t": temperature,
            "tp": top_p,
            "mt": max_tokens or 0,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request_cache_key(request: ChatRequest, model: str) -> str:
    """Return a collision-resistant key for a complete non-streaming request."""
    messages = [
        {
            "role": message.role,
            "content": message.content,
            "name": message.name,
            "tool_calls": message.tool_calls,
            "tool_call_id": message.tool_call_id,
        }
        for message in request.messages
    ]
    payload = {
        "messages": messages,
        "model": model,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stop": request.stop,
        "extra": request.extra,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SQLiteCacheBackend:
    """SQLite-backed cache storage.

    Creates the table on first use.  Expired entries are pruned lazily
    on read and periodically on write.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self._last_prune: float = 0.0
        self._prune_interval: float = 300.0  # 5 min

    async def _ensure_table(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tier INTEGER NOT NULL,
                    tokens_total INTEGER NOT NULL,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    ttl_seconds REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_created
                ON cache_entries (created_at)
            """)
            conn.commit()

    async def get(self, key: str) -> dict[str, Any] | None:
        """Return cached response dict if entry exists and is not expired."""
        await self._ensure_table()
        async with self._lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute(
                    """SELECT response_json, created_at, ttl_seconds
                       FROM cache_entries
                       WHERE key = ?""",
                    (key,),
                ).fetchone()
            if row is None:
                return None
            response_json, created_at, ttl_seconds = row
            if time.time() - created_at > ttl_seconds:
                # Expired — delete lazily
                with sqlite3.connect(str(self._db_path)) as conn:
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                return None
            return json.loads(response_json)

    async def set(
        self,
        key: str,
        response: dict[str, Any],
        model: str,
        tier: int,
        tokens_total: int,
        cost_usd: float,
        ttl_seconds: float,
    ) -> None:
        """Store a response in the cache."""
        await self._ensure_table()
        async with self._lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                        (key, response_json, model, tier, tokens_total, cost_usd,
                         created_at, ttl_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        json.dumps(response),
                        model,
                        tier,
                        tokens_total,
                        cost_usd,
                        time.time(),
                        ttl_seconds,
                    ),
                )
                conn.commit()
            await self._prune_if_needed()

    async def count(self) -> int:
        """Return the number of non-expired entries."""
        await self._ensure_table()
        async with self._lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM cache_entries WHERE created_at + ttl_seconds > ?",
                    (time.time(),),
                ).fetchone()
            return row[0] if row else 0

    async def _prune_if_needed(self) -> None:
        now = time.time()
        if now - self._last_prune < self._prune_interval:
            return
        self._last_prune = now
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "DELETE FROM cache_entries WHERE created_at + ttl_seconds <= ?",
                (now,),
            )
            conn.commit()


class CacheManager:
    """High-level cache manager with TTL-per-tier and stats tracking."""

    def __init__(
        self,
        backend: SQLiteCacheBackend,
        *,
        ttl_by_tier: dict[int, float] | None = None,
    ) -> None:
        self._backend = backend
        self._ttl_by_tier = ttl_by_tier or _DEFAULT_TTL_BY_TIER
        self._stats = CacheStats()
        self._stats_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(
        self,
        request: ChatRequest,
        model_name: str,
        tier: int,
    ) -> ChatResponse | None:
        """Return a cached response or None on miss/expiry."""
        key = _request_cache_key(request, model_name)
        cached = await self._backend.get(key)
        if cached is None:
            async with self._stats_lock:
                self._stats.misses += 1
            return None

        async with self._stats_lock:
            self._stats.hits += 1
            self._stats.tokens_saved += cached.get("usage", {}).get("total_tokens", 0)
            self._stats.cost_saved_usd += cached.get("cost_usd", 0.0)

        return ChatResponse(
            id=cached.get("id", ""),
            model=cached.get("model", model_name),
            choices=cached.get("choices", []),
            usage=Usage(
                prompt_tokens=cached.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=cached.get("usage", {}).get("completion_tokens", 0),
                total_tokens=cached.get("usage", {}).get("total_tokens", 0),
                cached_tokens=cached.get("usage", {}).get("prompt_tokens", 0),
                cache_status="local_hit",
            ),
            created=cached.get("created"),
            latency_ms=0.0,  # cache hit = zero latency
        )

    async def set(
        self,
        request: ChatRequest,
        model_name: str,
        tier: int,
        response: ChatResponse,
        cost_usd: float = 0.0,
    ) -> None:
        """Store a response in the cache."""
        key = _request_cache_key(request, model_name)
        ttl = self._ttl_by_tier.get(tier, _DEFAULT_TTL_BY_TIER[Tier.T3])
        response_dict = {
            "id": response.id,
            "model": response.model,
            "choices": response.choices,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cached_tokens": response.usage.cached_tokens,
                "cache_status": response.usage.cache_status,
            },
            "created": response.created,
            "cost_usd": cost_usd,
        }
        await self._backend.set(
            key=key,
            response=response_dict,
            model=model_name,
            tier=tier,
            tokens_total=response.usage.total_tokens,
            cost_usd=cost_usd,
            ttl_seconds=ttl,
        )

    async def stats(self) -> CacheStats:
        """Return current cache statistics including entry count."""
        async with self._stats_lock:
            self._stats.entries = await self._backend.count()
            return self._stats
