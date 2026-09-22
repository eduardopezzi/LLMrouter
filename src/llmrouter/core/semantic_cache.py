"""Semantic (embedding-similarity) response cache.

Complements the exact-match cache in :mod:`llmrouter.core.cache` with a
similarity cache: prompts whose embeddings are near-duplicates (cosine
similarity >= threshold) reuse the stored response.  Entries are scoped by
model, tier, and sampling parameters (temperature/top_p/max_tokens).

Storage is additive: a dedicated ``semantic_cache_entries`` table is created
with ``CREATE TABLE IF NOT EXISTS`` — the exact cache's ``cache_entries``
table is never altered, and both can share one SQLite file.

All public entry points are fault tolerant: an unavailable or failing
embedder never raises — lookups return ``None`` and stores return ``False``,
with the ``semantic_unavailable`` counter incremented.  Streaming requests
never reach this module (enforced by the proxy).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Coroutine
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from llmrouter.core.types import ChatResponse, Usage
from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.semantic_cache")

_DEFAULT_TTL_SECONDS: float = 3600.0


def _is_awaitable(value: Any) -> bool:
    """Return True when ``value`` can be awaited (coroutine/future-like)."""
    return isinstance(value, Coroutine) or isinstance(value, Future)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Adapted from the semantic scorer's implementation (kept local on purpose:
    this module must not import from ``semantic_scorer``).
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _prompt_hash(prompt: str) -> str:
    """Return a stable hash of the whitespace-normalized prompt."""
    normalized = " ".join(prompt.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _restrictions_key(
    prompt_hash: str,
    model: str,
    tier: int,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
) -> str:
    """Return a deterministic key for an entry's full restriction tuple."""
    payload = json.dumps(
        {
            "p": prompt_hash,
            "m": model,
            "t": tier,
            "temp": temperature,
            "tp": top_p,
            "mt": max_tokens or 0,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SemanticCache:
    """Similarity-based response cache backed by a dedicated SQLite table.

    Args:
        db_path: SQLite database file.  May be shared with the exact cache —
            this class only creates its own ``semantic_cache_entries`` table.
        embedder: object able to produce embeddings.  Two interfaces are
            accepted: ``await embedder.embed(texts) -> list[list[float]]`` or
            the synchronous ``embedder.encode(texts) -> list[list[float]] | None``
            used by :class:`llmrouter.core.semantic_scorer.HybridScorer`'s
            ``embedder`` property.  ``None`` disables embedding (raw-vector
            lookup/store still works).
        threshold: minimum cosine similarity for a hit (default 0.95).
        ttl_seconds: default per-entry TTL (default 3600s).
    """

    def __init__(
        self,
        db_path: str,
        embedder: Any | None = None,
        *,
        threshold: float = 0.95,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._db_path = Path(db_path)
        self._embedder = embedder
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._semantic_hits = 0
        self._semantic_misses = 0
        self._semantic_unavailable = 0

    @property
    def threshold(self) -> float:
        """Minimum cosine similarity required for a semantic hit."""
        return self._threshold

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float] | None:
        """Embed one text via the configured embedder; ``None`` on failure.

        Never raises: any failure (missing embedder, exception, empty or
        malformed result) increments ``semantic_unavailable`` and returns
        ``None``.
        """
        if self._embedder is None:
            self._semantic_unavailable += 1
            return None
        try:
            embed_method = getattr(self._embedder, "embed", None)
            if callable(embed_method):
                result: Any = embed_method([text])
                vectors = await result if _is_awaitable(result) else result
            else:
                encode_method = getattr(self._embedder, "encode", None)
                if not callable(encode_method):
                    self._semantic_unavailable += 1
                    return None
                vectors = encode_method([text])
            if not isinstance(vectors, list) or not vectors:
                self._semantic_unavailable += 1
                return None
            vector = vectors[0]
            if not isinstance(vector, list) or not vector:
                self._semantic_unavailable += 1
                return None
            return [float(value) for value in vector]
        except Exception as exc:
            _logger.warning("Semantic embedding unavailable: %s", exc)
            self._semantic_unavailable += 1
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def lookup(
        self,
        embedding: list[float],
        *,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> ChatResponse | None:
        """Return the best matching response if similarity >= threshold.

        Scans entries restricted to the same model/tier/sampling tuple and
        returns the response with the highest cosine similarity, or ``None``.
        Never raises: storage failures are logged and treated as a miss.
        """
        try:
            return await self._lookup_locked(
                embedding,
                model=model,
                tier=tier,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            _logger.warning("Semantic cache lookup failed (treated as miss): %s", exc)
            self._semantic_misses += 1
            return None

    async def lookup_prompt(
        self,
        prompt: str,
        *,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> ChatResponse | None:
        """Embed ``prompt`` and perform a semantic lookup.

        Convenience path used by the proxy.  Embedding failures never
        propagate: they return ``None`` and bump ``semantic_unavailable``.
        """
        embedding = await self._embed(prompt)
        if embedding is None:
            return None
        return await self.lookup(
            embedding,
            model=model,
            tier=tier,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    async def store(
        self,
        response: ChatResponse,
        prompt: str,
        *,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
        embedding: list[float] | None = None,
        ttl_seconds: float | None = None,
    ) -> bool:
        """Store a response with its embedding and restriction tuple.

        When ``embedding`` is not supplied, the prompt is embedded via the
        configured embedder; on failure the store is skipped and ``False``
        is returned (``semantic_unavailable`` incremented).  Never raises.
        """
        if embedding is None:
            embedding = await self._embed(prompt)
            if embedding is None:
                return False
        try:
            await self._store_locked(
                response=response,
                prompt_hash=_prompt_hash(prompt),
                embedding=embedding,
                model=model,
                tier=tier,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                ttl_seconds=self._ttl_seconds if ttl_seconds is None else ttl_seconds,
            )
            return True
        except Exception as exc:
            _logger.warning("Semantic cache store failed (skipped): %s", exc)
            return False

    def stats(self) -> dict[str, int]:
        """Return semantic cache counters (hits, misses, unavailability)."""
        return {
            "semantic_hits": self._semantic_hits,
            "semantic_misses": self._semantic_misses,
            "semantic_unavailable": self._semantic_unavailable,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _lookup_locked(
        self,
        embedding: list[float],
        *,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> ChatResponse | None:
        await self._ensure_table()
        async with self._lock:
            rows = await self._fetch_candidates(
                model=model,
                tier=tier,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        best_key: str | None = None
        best_similarity = -1.0
        best_cached: dict[str, Any] | None = None
        for key, embedding_json, response_json in rows:
            try:
                candidate = [float(value) for value in json.loads(embedding_json)]
                cached = dict(json.loads(response_json))
            except (TypeError, ValueError):
                continue
            similarity = _cosine_similarity(embedding, candidate)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cached = cached
                best_key = key
        if best_cached is None or best_key is None or best_similarity < self._threshold:
            self._semantic_misses += 1
            return None
        self._semantic_hits += 1
        return self._to_response(best_cached, model)

    async def _store_locked(
        self,
        *,
        response: ChatResponse,
        prompt_hash: str,
        embedding: list[float],
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
        ttl_seconds: float,
    ) -> None:
        await self._ensure_table()
        key = _restrictions_key(prompt_hash, model, tier, temperature, top_p, max_tokens)
        response_dict = {
            "id": response.id,
            "model": response.model,
            "choices": response.choices,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "created": response.created,
        }
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO semantic_cache_entries
                        (key, prompt_hash, embedding_json, response_json,
                         model, tier, temperature, top_p, max_tokens,
                         created_at, ttl_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        prompt_hash,
                        json.dumps(embedding),
                        json.dumps(response_dict),
                        model,
                        tier,
                        temperature,
                        top_p,
                        max_tokens or 0,
                        time.time(),
                        ttl_seconds,
                    ),
                )
                conn.commit()

    async def _fetch_candidates(
        self,
        *,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> list[tuple[str, str, str]]:
        now = time.time()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, embedding_json, response_json
                FROM semantic_cache_entries
                WHERE model = ? AND tier = ? AND temperature = ?
                  AND top_p = ? AND max_tokens = ?
                  AND created_at + ttl_seconds > ?
                """,
                (model, tier, temperature, top_p, max_tokens or 0, now),
            ).fetchall()
            conn.execute(
                "DELETE FROM semantic_cache_entries WHERE created_at + ttl_seconds <= ?",
                (now,),
            )
            conn.commit()
        return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]

    @staticmethod
    def _to_response(cached: dict[str, Any], fallback_model: str) -> ChatResponse:
        usage = cached.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        return ChatResponse(
            id=str(cached.get("id", "")),
            model=str(cached.get("model", fallback_model)),
            choices=list(cached.get("choices", [])),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                cached_tokens=prompt_tokens,
                cache_status="semantic_hit",
            ),
            created=int(cached.get("created", 0)),
            latency_ms=0.0,  # cache hit = zero latency
        )

    def _connect(self) -> sqlite3.Connection:
        """Open a low-latency connection (see ``cache.SQLiteCacheBackend``)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    async def _ensure_table(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache_entries (
                    key TEXT PRIMARY KEY,
                    prompt_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tier INTEGER NOT NULL,
                    temperature REAL NOT NULL,
                    top_p REAL NOT NULL,
                    max_tokens INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_seconds REAL NOT NULL
                )
                """
            )
            conn.commit()
