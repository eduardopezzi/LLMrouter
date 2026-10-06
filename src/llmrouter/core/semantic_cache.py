"""Semantic (embedding-similarity) response cache.

Complements the exact-match cache in :mod:`llmrouter.core.cache` with a
similarity cache: prompts whose embeddings are near-duplicates (cosine
similarity >= threshold) reuse the stored response.  Entries are scoped by
model, tier, and sampling parameters (temperature/top_p/max_tokens).

Storage is additive: a dedicated ``semantic_cache_entries`` table is created
with ``CREATE TABLE IF NOT EXISTS`` — the exact cache's ``cache_entries``
table is never altered, and both can share one SQLite file.

P-CHR hit log (ROADMAP_TOKEN_OPTIMIZATION E1.3): when ``hit_log_enabled`` is
set, every semantic hit appends one row to the sibling
``semantic_cache_hit_log`` table (prompt/response/restriction tuple,
similarity and threshold) so precision can be audited offline by
:meth:`SemanticCache.verify_pending`.  Hit-log writes are best-effort: they
never raise and never change the served response.

All public entry points are fault tolerant: an unavailable or failing
embedder never raises — lookups return ``None`` and stores return ``False``,
with the ``semantic_unavailable`` counter incremented.  Streaming requests
never reach this module (enforced by the proxy).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import sqlite3
import time
from collections.abc import Coroutine
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import httpx

from llmrouter.core.types import ChatResponse, Usage
from llmrouter.logging_config import get_logger

_logger = get_logger("llmrouter.semantic_cache")

_DEFAULT_TTL_SECONDS: float = 3600.0

# P-CHR similarity buckets (ROADMAP_TOKEN_OPTIMIZATION E1.3): [low, high).
_PCHR_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
    (0.95, 1.01),
)
_PCHR_BUCKET_LABELS: tuple[str, ...] = tuple(
    f"{low:.2f}-{high:.2f}" if high <= 1.0 else f"{low:.2f}+"
    for low, high in _PCHR_BUCKETS
)


def _bucket_label(similarity: float) -> str | None:
    """Return the P-CHR bucket label for ``similarity``, or ``None``."""
    for low, high in _PCHR_BUCKETS:
        if low <= similarity < high:
            return f"{low:.2f}-{high:.2f}" if high <= 1.0 else f"{low:.2f}+"
    return None


def _response_text(response_json: str | None) -> str | None:
    """Extract the served text from a stored response JSON payload.

    Prefers ``choices[0].message.content``; falls back to the JSON encoding
    of the whole ``choices`` array.  Returns ``None`` when the payload cannot
    be decoded (the column stays NULL rather than raising).
    """
    if not response_json:
        return None
    try:
        choices = json.loads(response_json).get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if content is not None:
                        return str(content)
        return json.dumps(choices) if choices is not None else None
    except (TypeError, ValueError):
        return None


# Verified flags in ``semantic_cache_hit_log``.
_VERIFIED_PENDING = 0
_VERIFIED_OK = 1
_VERIFIED_MISMATCH = -1
_VERIFIED_ERROR = 2


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


_TERMINAL_FINISH_REASONS: frozenset[str] = frozenset({"stop", "length", "tool_calls"})


def _stream_has_terminal_chunk(chunks: list[dict[str, Any]]) -> bool:
    """Return True iff at least one chunk carries a terminal ``finish_reason``.

    PRD §4.3 (E2, S1): a stream is only persisted when it ends cleanly —
    i.e. the provider emitted a final chunk with ``choices[0].finish_reason``
    in ``{stop, length, tool_calls}``.  Missing terminal chunk = provider cut
    before ``[DONE]``; we never persist those (defensive: the route layer
    is supposed to skip storage for abnormal terminations).
    """
    if not chunks:
        return False
    for chunk in chunks:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        reason = first.get("finish_reason")
        if isinstance(reason, str) and reason in _TERMINAL_FINISH_REASONS:
            return True
    return False


def _first_k_tokens(chunks: list[dict[str, Any]], *, k: int) -> str:
    """Concatenate ``choices[0].delta.content`` deltas until ``k`` tokens.

    Token count is approximated by whitespace-split tokens; this is the same
    heuristic the live path uses for `stream_tokens_saved_total`.  We cap at
    the first ``k`` tokens of content (PRD §2.2) — subsequent tokens are
    irrelevant for the probe comparison.
    """
    buffer: list[str] = []
    collected = 0
    for chunk in chunks:
        if collected >= k:
            break
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        delta = first.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if not isinstance(content, str) or not content:
            continue
        # Split current content on whitespace; emit as many tokens as fit.
        tokens = content.split()
        for token in tokens:
            if collected >= k:
                break
            buffer.append(token)
            collected += 1
    return " ".join(buffer)


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
        hit_log_enabled: bool = False,
        hit_log_retention_days: int = 45,
        stream_cache_enabled: bool = True,
        stream_probe_k: int = 8,
        stream_probe_timeout_seconds: float = 10.0,
        stream_probe_soft_circuit_threshold: int = 3,
        stream_probe_soft_circuit_seconds: float = 3600.0,
    ) -> None:
        self._db_path = Path(db_path)
        self._embedder = embedder
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        self._hit_log_enabled = hit_log_enabled
        self._hit_log_retention_days = max(0, int(hit_log_retention_days))
        self._stream_cache_enabled = bool(stream_cache_enabled)
        self._stream_probe_k = max(1, int(stream_probe_k))
        self._stream_probe_timeout_seconds = float(stream_probe_timeout_seconds)
        self._stream_probe_soft_circuit_threshold = max(1, int(stream_probe_soft_circuit_threshold))
        self._stream_probe_soft_circuit_seconds = float(stream_probe_soft_circuit_seconds)
        self._lock = asyncio.Lock()
        self._semantic_hits = 0
        self._semantic_misses = 0
        self._semantic_unavailable = 0
        # ROADMAP_TOKEN_OPTIMIZATION E2 — streaming cache replay counters.
        # This module (Dev E2-A) only writes the four "storage" counters;
        # the probe/replay counters (stream_probes_ok_total, etc.) are
        # populated by the route-layer code in Dev E2-B's work and exposed
        # here with a zero default so ``stream_stats()`` always returns a
        # stable shape.
        self._stream_counters: dict[str, int] = {
            "stream_stored_total": 0,
            "stream_purged_total": 0,
            "stream_lookup_hit_total": 0,
            "stream_lookup_miss_total": 0,
            "stream_replays_total": 0,
            "stream_probes_ok_total": 0,
            "stream_probes_fail_total": 0,
            "stream_tokens_saved_total": 0,
            "stream_replay_bytes_served_total": 0,
            "stream_replay_error_total": 0,
        }

    @property
    def threshold(self) -> float:
        """Minimum cosine similarity required for a semantic hit."""
        return self._threshold

    @property
    def stream_cache_enabled(self) -> bool:
        """Master toggle for the streaming replay path (ROADMAP E2)."""
        return self._stream_cache_enabled

    @property
    def stream_probe_k(self) -> int:
        """Number of output tokens to request for the k-token probe."""
        return self._stream_probe_k

    @property
    def stream_probe_timeout_seconds(self) -> float:
        """Per-request timeout for the k-token probe, in seconds."""
        return self._stream_probe_timeout_seconds

    @property
    def stream_probe_soft_circuit_threshold(self) -> int:
        """Consecutive probe failures that trip the per-model soft-circuit."""
        return self._stream_probe_soft_circuit_threshold

    @property
    def stream_probe_soft_circuit_seconds(self) -> float:
        """How long the per-model probe soft-circuit stays open, in seconds."""
        return self._stream_probe_soft_circuit_seconds

    def bump_stream_counter(self, name: str, by: int = 1) -> None:
        """Increment a stream_* counter (route layer writes replay/probe).

        Unknown counter names are silently ignored — the storage layer is
        not the source of truth for the probe/replay counters.
        """
        if name in self._stream_counters:
            self._stream_counters[name] += by

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
        prompt: str | None = None,
    ) -> ChatResponse | None:
        """Return the best matching response if similarity >= threshold.

        Scans entries restricted to the same model/tier/sampling tuple and
        returns the response with the highest cosine similarity, or ``None``.
        Never raises: storage failures are logged and treated as a miss.

        When ``prompt`` is supplied and the hit log is enabled, the served
        prompt text is persisted in the P-CHR hit log alongside the entry.
        """
        try:
            return await self._lookup_locked(
                embedding,
                model=model,
                tier=tier,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                prompt=prompt,
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
            prompt=prompt,
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

    def stats(self) -> dict[str, Any]:
        """Return semantic cache counters.

        The legacy in-memory counters (``semantic_hits``/``semantic_misses``/
        ``semantic_unavailable``) are kept for compatibility.  When the hit
        log is enabled, the P-CHR counters are derived from the persisted log
        (so they survive process restarts): ``pchr_pending``,
        ``pchr_verified_ok``, ``pchr_verified_mismatch``, ``pchr_precision``
        (``None`` until at least one verification exists) and
        ``pchr_last_verified_ts``.
        """
        counters: dict[str, Any] = {
            "semantic_hits": self._semantic_hits,
            "semantic_misses": self._semantic_misses,
            "semantic_unavailable": self._semantic_unavailable,
        }
        if not self._hit_log_enabled:
            return counters
        try:
            counters.update(self._pchr_stats())
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning("P-CHR hit-log stats unavailable: %s", exc)
        return counters

    def _pchr_stats(self) -> dict[str, Any]:
        """Derive P-CHR counters from the hit log via SQL aggregation."""
        self._ensure_schema()
        with self._connect() as conn:
            pending = int(
                conn.execute(
                    "SELECT COUNT(*) FROM semantic_cache_hit_log WHERE verified = ?",
                    (_VERIFIED_PENDING,),
                ).fetchone()[0]
            )
            ok = int(
                conn.execute(
                    "SELECT COUNT(*) FROM semantic_cache_hit_log WHERE verified = ?",
                    (_VERIFIED_OK,),
                ).fetchone()[0]
            )
            mismatch = int(
                conn.execute(
                    "SELECT COUNT(*) FROM semantic_cache_hit_log WHERE verified = ?",
                    (_VERIFIED_MISMATCH,),
                ).fetchone()[0]
            )
            last_verified = conn.execute(
                "SELECT MAX(verify_ts) FROM semantic_cache_hit_log"
            ).fetchone()[0]
        precision: float | None = None
        if ok + mismatch > 0:
            precision = ok / (ok + mismatch)
        return {
            "pchr_pending": pending,
            "pchr_verified_ok": ok,
            "pchr_verified_mismatch": mismatch,
            "pchr_precision": precision,
            "pchr_last_verified_ts": (
                float(last_verified) if last_verified is not None else None
            ),
        }

    async def verify_pending(
        self,
        sample_size: int,
        judge: Any,
        method: str = "local_llm",
    ) -> dict[str, Any]:
        """Audit up to ``sample_size`` pending hit-log rows with ``judge``.

        The judge receives ``(prompt_text, response_text)`` and returns
        ``(verdict, score, note)``.  Each checked row is updated in place:
        verdict True -> ``verified=1``, False -> ``verified=-1``; a raising
        judge marks the row ``verified=2`` (verify_error) with the exception
        name in ``verify_note``.  Rows with a final verdict (ok/mismatch) are
        never re-checked; ``verify_error`` rows are re-selected on the next
        run (orchestrator amendment — a judge outage must not burn the
        sample permanently).  Returns ``{checked, ok, mismatch, error,
        buckets}`` where ``buckets`` groups results by similarity range.
        """
        buckets: dict[str, dict[str, int]] = {
            label: {"checked": 0, "ok": 0, "mismatch": 0}
            for label in _PCHR_BUCKET_LABELS
        }
        result: dict[str, Any] = {
            "checked": 0,
            "ok": 0,
            "mismatch": 0,
            "error": 0,
            "buckets": buckets,
        }
        try:
            await self._ensure_table()
            await asyncio.to_thread(self.purge_expired_hit_log)
            # ROADMAP_TOKEN_OPTIMIZATION E2 (S1, Dev E2-A) — purge the
            # streaming table at the same hook.  Best-effort: a failure
            # here is logged by ``purge_expired_stream_responses`` and
            # does NOT short-circuit verify_pending (the hit-log sweep
            # already ran above).
            await asyncio.to_thread(self.purge_expired_stream_responses)
            rows = await asyncio.to_thread(self._fetch_pending_rows, sample_size)
        except Exception as exc:
            _logger.warning("P-CHR verify: could not read pending rows: %s", exc)
            return result
        for row_id, prompt_text, response_text, similarity in rows:
            result["checked"] += 1
            label = _bucket_label(similarity)
            if label is not None:
                buckets[label]["checked"] += 1
            try:
                outcome = judge(prompt_text, response_text)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                verdict, score, note = outcome
                verified = _VERIFIED_OK if verdict else _VERIFIED_MISMATCH
                if verdict:
                    result["ok"] += 1
                else:
                    result["mismatch"] += 1
                if label is not None:
                    key = "ok" if verdict else "mismatch"
                    buckets[label][key] += 1
            except Exception as exc:
                verified = _VERIFIED_ERROR
                score = None
                note = f"{type(exc).__name__}: {exc}"[:500]
                result["error"] += 1
            try:
                async with self._lock:
                    await asyncio.to_thread(
                        self._write_verdict,
                        row_id,
                        verified,
                        method,
                        score,
                        note,
                    )
            except Exception as exc:
                _logger.warning("P-CHR verify: could not persist verdict: %s", exc)
        return result

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
        prompt: str | None = None,
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
        best_response_json: str | None = None
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
                best_response_json = response_json
        if best_cached is None or best_key is None or best_similarity < self._threshold:
            self._semantic_misses += 1
            return None
        self._semantic_hits += 1
        if self._hit_log_enabled:
            # P-CHR: persist one audit row per hit, best-effort (a failure
            # logs a warning and never breaks the served response).
            try:
                await asyncio.to_thread(
                    self._write_hit_log_row,
                    {
                        "ts": time.time(),
                        "key": best_key,
                        "prompt_hash": _prompt_hash(prompt) if prompt else None,
                        "prompt_text": prompt,
                        "response_text": _response_text(best_response_json),
                        "model": model,
                        "tier": int(tier),
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "max_tokens": int(max_tokens) if max_tokens is not None else 0,
                        "similarity": float(best_similarity),
                        "threshold": float(self._threshold),
                    },
                )
            except Exception as exc:
                _logger.warning("P-CHR hit-log write failed (ignored): %s", exc)
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

    def _write_hit_log_row(self, row: dict[str, Any]) -> None:
        """Insert one P-CHR audit row (synchronous; runs in a worker thread)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO semantic_cache_hit_log
                    (ts, key, prompt_hash, prompt_text, response_text,
                     model, tier, temperature, top_p, max_tokens,
                     similarity, threshold, verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["ts"],
                    row["key"],
                    row["prompt_hash"],
                    row["prompt_text"],
                    row["response_text"],
                    row["model"],
                    row["tier"],
                    row["temperature"],
                    row["top_p"],
                    row["max_tokens"],
                    row["similarity"],
                    row["threshold"],
                    _VERIFIED_PENDING,
                ),
            )
            conn.commit()

    def purge_expired_hit_log(self) -> int:
        """Delete hit-log rows older than ``hit_log_retention_days`` (R2).

        Returns the number of rows removed.  ``hit_log_retention_days=0``
        disables retention (no-op).  Best-effort by design: called at the
        start of ``verify_pending``; a failure raises only to the caller
        there, which already guards the whole block.
        """
        if self._hit_log_retention_days <= 0:
            return 0
        cutoff = time.time() - self._hit_log_retention_days * 86400.0
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM semantic_cache_hit_log WHERE ts < ?",
                (cutoff,),
            )
            conn.commit()
            return int(cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0)

    # ------------------------------------------------------------------
    # ROADMAP_TOKEN_OPTIMIZATION E2 — streaming cache replay (S1, Dev E2-A)
    # ------------------------------------------------------------------
    #
    # Public surface (storage-only; probe/replay instrumentation is owned
    # by Dev E2-B):
    #
    #   * store_stream_response(chunks, *, prompt, usage, model, tier,
    #                            temperature, top_p, max_tokens,
    #                            k=8, ttl_seconds=None) -> bool
    #       Persist a finished SSE stream's chunks + final usage to
    #       ``semantic_stream_responses``.  Best-effort: any failure
    #       (embedder down, DB error, missing final chunk) returns False
    #       and never raises.  Counters via ``stream_stats()``.
    #
    #   * lookup_stream_response(prompt, *, model, tier, temperature,
    #                             top_p, max_tokens)
    #                            -> (chunks, completion_tokens, first_k) | None
    #       Embed-and-lookup against the streaming table.  Same threshold
    #       and restriction tuple semantics as the non-stream path; returns
    #       the matching ``(chunks, completion_tokens, first_k)`` triple or
    #       ``None``.  Increments ``stream_lookup_hit_total`` /
    #       ``stream_lookup_miss_total``.
    #
    #   * purge_expired_stream_responses() -> int
    #       Remove rows with ``expires_at < now`` (same TTL semantics as the
    #       non-stream cache).  Idempotent.  Returns the # removed and
    #       bumps ``stream_purged_total``.
    #
    #   * stream_stats() -> dict[str, int]
    #       Snapshot of the in-memory counters.  ``stream_probes_*`` /
    #       ``stream_replays_*`` / ``stream_tokens_saved_total`` /
    #       ``stream_replay_bytes_served_total`` / ``stream_replay_error_total``
    #       are managed by Dev E2-B's route-layer code; this module
    #       exposes them with a stable zero default so the contract shape
    #       never changes once both halves land.

    async def store_stream_response(
        self,
        chunks: list[dict[str, Any]],
        *,
        prompt: str,
        usage: Usage,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
        k: int = 8,
        ttl_seconds: float | None = None,
    ) -> bool:
        """Persist a finished SSE stream's chunks + final usage.

        Best-effort: never raises.  Returns ``False`` when:

        - the embedder is unavailable or fails (``semantic_unavailable`` ++);
        - the input chunks lack a terminal ``finish_reason`` (PRD §4.3:
          provider cut before ``[DONE]`` is never persisted);
        - the SQLite write fails (logged at WARNING).

        On success returns ``True`` and bumps ``stream_stored_total``.
        """
        if not _stream_has_terminal_chunk(chunks):
            _logger.info(
                "Stream cache store skipped: no terminal finish_reason (provider cut?)"
            )
            return False
        embedding = await self._embed(prompt)
        if embedding is None:
            return False
        first_k = _first_k_tokens(chunks, k=k)
        usage_payload = {
            "prompt_tokens": int(usage.prompt_tokens),
            "completion_tokens": int(usage.completion_tokens),
            "total_tokens": int(usage.total_tokens),
        }
        ttl = self._ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        now = time.time()
        expires_at = now + ttl
        prompt_hash = _prompt_hash(prompt)
        # QA MEDIUM-8: SQLite treats NULLs as distinct in UNIQUE constraints,
        # so an absent max_tokens would never collapse duplicate rows.  Use a
        # sentinel of -1 in the UNIQUE key while storing NULL in the column.
        max_tokens_key = int(max_tokens) if max_tokens is not None else -1
        embedding_blob = json.dumps(embedding).encode("utf-8")
        try:
            await self._store_stream_locked(
                prompt_hash=prompt_hash,
                embedding_blob=embedding_blob,
                prompt_text=prompt,
                model=model,
                tier=int(tier),
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=max_tokens_key,
                chunks_json=json.dumps(list(chunks)),
                first_k_tokens=first_k,
                usage_json=json.dumps(usage_payload),
                created_at=now,
                expires_at=expires_at,
            )
            self._stream_counters["stream_stored_total"] += 1
            return True
        except Exception as exc:
            _logger.warning("Semantic stream cache store failed (skipped): %s", exc)
            return False

    async def lookup_stream_response(
        self,
        prompt: str,
        *,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> tuple[list[dict[str, Any]], int, str] | None:
        """Return ``(chunks, completion_tokens, first_k)`` for a similar prompt.

        Embeds the prompt and scans the streaming table for the best
        matching candidate (same restriction tuple as the non-stream path,
        threshold applied).  ``None`` means no semantic hit (counter
        ``stream_lookup_miss_total`` bumped); on hit, ``stream_lookup_hit_total``
        advances and the triple is returned for replay.
        """
        embedding = await self._embed(prompt)
        if embedding is None:
            return None
        try:
            return await self._lookup_stream_locked(
                embedding,
                model=model,
                tier=tier,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            _logger.warning(
                "Semantic stream cache lookup failed (treated as miss): %s", exc
            )
            self._stream_counters["stream_lookup_miss_total"] += 1
            return None

    def purge_expired_stream_responses(self) -> int:
        """Remove rows with ``expires_at < now``; returns the # removed.

        Best-effort by design: the caller (``verify_pending``) already
        guards the wider block.  ``stream_purged_total`` is bumped by the
        number actually deleted.
        """
        now = time.time()
        try:
            self._ensure_schema()
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM semantic_stream_responses WHERE expires_at < ?",
                    (now,),
                )
                conn.commit()
                removed = int(cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0)
        except Exception as exc:
            _logger.warning(
                "Semantic stream cache purge failed (skipped): %s", exc
            )
            return 0
        if removed:
            self._stream_counters["stream_purged_total"] += removed
        return removed

    def stream_stats(self) -> dict[str, int]:
        """Snapshot of the in-memory streaming cache counters.

        Counter ownership:

        - ``stream_stored_total`` / ``stream_purged_total`` /
          ``stream_lookup_hit_total`` / ``stream_lookup_miss_total`` are
          maintained here (Dev E2-A — storage).
        - ``stream_replays_total`` / ``stream_probes_ok_total`` /
          ``stream_probes_fail_total`` / ``stream_tokens_saved_total`` /
          ``stream_replay_bytes_served_total`` / ``stream_replay_error_total``
          are bumped by Dev E2-B's route-layer code; this view returns
          whatever was last set, defaulting to zero so the snapshot shape
          is stable from day one.
        """
        return dict(self._stream_counters)

    async def _store_stream_locked(
        self,
        *,
        prompt_hash: str,
        embedding_blob: bytes,
        prompt_text: str,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
        chunks_json: str,
        first_k_tokens: str,
        usage_json: str,
        created_at: float,
        expires_at: float,
    ) -> None:
        """Insert one streaming response (synchronous SQLite write)."""
        await self._ensure_table()
        async with self._lock:
            await asyncio.to_thread(
                self._insert_stream_row,
                prompt_hash,
                embedding_blob,
                prompt_text,
                model,
                tier,
                temperature,
                top_p,
                max_tokens,
                chunks_json,
                first_k_tokens,
                usage_json,
                created_at,
                expires_at,
            )

    def _insert_stream_row(
        self,
        prompt_hash: str,
        embedding_blob: bytes,
        prompt_text: str,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
        chunks_json: str,
        first_k_tokens: str,
        usage_json: str,
        created_at: float,
        expires_at: float,
    ) -> None:
        """Worker-thread INSERT (UNIQUE key collapses duplicates)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO semantic_stream_responses
                    (prompt_hash, embedding, prompt_text, model, tier,
                     temperature, top_p, max_tokens, response_chunks_json,
                     first_k_tokens, usage_json, created_at, expires_at,
                     purge_pending)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    prompt_hash,
                    embedding_blob,
                    prompt_text,
                    model,
                    tier,
                    temperature,
                    top_p,
                    max_tokens,
                    chunks_json,
                    first_k_tokens,
                    usage_json,
                    created_at,
                    expires_at,
                ),
            )
            conn.commit()

    async def _lookup_stream_locked(
        self,
        embedding: list[float],
        *,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> tuple[list[dict[str, Any]], int, str] | None:
        """Best-candidate lookup against ``semantic_stream_responses``."""
        await self._ensure_table()
        # QA MEDIUM-8: mirror the store-side sentinel so lookups match rows
        # persisted with an absent max_tokens (-1 instead of NULL).
        max_tokens_key = int(max_tokens) if max_tokens is not None else -1
        async with self._lock:
            rows = await asyncio.to_thread(
                self._fetch_stream_candidates,
                model=model,
                tier=tier,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens_key,
            )
        best_similarity = -1.0
        best_chunks: list[dict[str, Any]] | None = None
        best_completion_tokens = 0
        best_first_k = ""
        best_row_id = -1
        for row_id, embedding_blob, chunks_json, usage_json in rows:
            try:
                candidate = [
                    float(value) for value in json.loads(embedding_blob.decode("utf-8"))
                ]
                chunks = json.loads(chunks_json)
                usage = json.loads(usage_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(chunks, list) or not isinstance(usage, dict):
                continue
            similarity = _cosine_similarity(embedding, candidate)
            if similarity > best_similarity:
                best_similarity = similarity
                best_chunks = chunks
                best_completion_tokens = int(usage.get("completion_tokens", 0))
                best_first_k = ""
                best_row_id = row_id
        if (
            best_chunks is None
            or best_similarity < self._threshold
            or best_row_id < 0
        ):
            self._stream_counters["stream_lookup_miss_total"] += 1
            return None
        # Re-resolve first_k from the winning row **by id** so the value and
        # the chunks provably belong to the same stored candidate (QA HIGH-3:
        # an ``ORDER BY id DESC`` fallback could pair chunks of the newest
        # row with usage of the similarity winner).
        await self._ensure_table()
        with self._connect() as conn:
            winning_row = conn.execute(
                """
                SELECT response_chunks_json, first_k_tokens
                FROM semantic_stream_responses
                WHERE id = ?
                  AND expires_at > ?
                """,
                (
                    best_row_id,
                    time.time(),
                ),
            ).fetchone()
        if winning_row is None:
            self._stream_counters["stream_lookup_miss_total"] += 1
            return None
        try:
            chunks = json.loads(str(winning_row[0]))
            best_first_k = str(winning_row[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            self._stream_counters["stream_lookup_miss_total"] += 1
            return None
        self._stream_counters["stream_lookup_hit_total"] += 1
        return (chunks, best_completion_tokens, best_first_k)

    def _fetch_stream_candidates(
        self,
        *,
        model: str,
        tier: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> list[tuple[int, bytes, str, str]]:
        """Return (id, embedding_blob, chunks_json, usage_json) for matching rows."""
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT id, embedding, response_chunks_json, usage_json
                FROM semantic_stream_responses
                WHERE model = ? AND tier = ? AND temperature = ?
                  AND top_p = ? AND max_tokens = ?
                  AND expires_at > ?
                """,
                (
                    model,
                    int(tier),
                    float(temperature),
                    float(top_p),
                    max_tokens,
                    now,
                ),
            )
            rows = cur.fetchall()
            # Opportunistic TTL purge of the streaming table; piggy-backs
            # on the same hook as the hit-log retention purge (called from
            # ``verify_pending`` for the canonical sweep).
            conn.execute(
                "DELETE FROM semantic_stream_responses WHERE expires_at <= ?",
                (now,),
            )
            conn.commit()
        return [
            (int(row[0]), bytes(row[1]), str(row[2]), str(row[3])) for row in rows
        ]

    def _fetch_pending_rows(
        self, sample_size: int
    ) -> list[tuple[int, str | None, str | None, float]]:
        """Select up to ``sample_size`` most-recent pending/erroneous rows for audit.

        Per the orchestrator amendment, ``verify_error`` rows (verified=2) are
        re-selected on the next run — only ok (1) and mismatch (-1) are final.
        This prevents a judge outage from permanently burning the sample.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, prompt_text, response_text, similarity
                FROM semantic_cache_hit_log
                WHERE verified IN (?, ?)
                ORDER BY ts DESC, id DESC
                LIMIT ?
                """,
                (_VERIFIED_PENDING, _VERIFIED_ERROR, int(sample_size)),
            ).fetchall()
        return [
            (int(row[0]), row[1], row[2], float(row[3] if row[3] is not None else 0.0))
            for row in rows
        ]

    def _write_verdict(
        self,
        row_id: int,
        verified: int,
        method: str,
        score: float | None,
        note: str | None,
    ) -> None:
        """Persist one judge verdict onto a hit-log row (worker thread).

        The ``WHERE verified IN (?, ?)`` guard keeps the write idempotent
        against rows that reached a final verdict concurrently, while still
        allowing the orchestrator's verify_error (2) rows to be overwritten
        by the retry.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE semantic_cache_hit_log
                SET verified = ?, verify_ts = ?, verify_method = ?,
                    verify_score = ?, verify_note = ?
                WHERE id = ? AND verified IN (?, ?)
                """,
                (
                    verified,
                    time.time(),
                    method,
                    score,
                    note,
                    row_id,
                    _VERIFIED_PENDING,
                    _VERIFIED_ERROR,
                ),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Open a low-latency connection (see ``cache.SQLiteCacheBackend``)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    async def _ensure_table(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the cache and hit-log tables/indexes if missing (sync)."""
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
            # P-CHR hit log (ROADMAP_TOKEN_OPTIMIZATION E1.3): sibling table,
            # never alters semantic_cache_entries.  ``verified`` codes:
            # 0=pending, 1=ok, -1=mismatch, 2=verify_error.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache_hit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    key TEXT,
                    prompt_hash TEXT,
                    prompt_text TEXT,
                    response_text TEXT,
                    model TEXT,
                    tier INTEGER,
                    temperature REAL,
                    top_p REAL,
                    max_tokens INTEGER,
                    similarity REAL,
                    threshold REAL,
                    verified INTEGER DEFAULT 0,
                    verify_ts REAL,
                    verify_method TEXT,
                    verify_score REAL,
                    verify_note TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pchr_pending "
                "ON semantic_cache_hit_log (verified, ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pchr_bucket "
                "ON semantic_cache_hit_log (threshold, similarity)"
            )
            # ROADMAP_TOKEN_OPTIMIZATION E2 — streaming cache replay (S1,
            # Dev E2-A).  Sister table to ``semantic_cache_entries``; the
            # non-stream cache is never altered.  See PRD §4.3 + TL §2.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_stream_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    prompt_text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tier INTEGER NOT NULL,
                    temperature REAL NOT NULL,
                    top_p REAL NOT NULL,
                    max_tokens INTEGER,
                    response_chunks_json TEXT NOT NULL,
                    first_k_tokens TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    purge_pending INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(prompt_hash, model, tier, temperature, top_p, max_tokens)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stream_lookup "
                "ON semantic_stream_responses(expires_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stream_purge "
                "ON semantic_stream_responses(purge_pending)"
            )
            conn.commit()


class OllamaJudge:
    """Binary yes/no judge backed by the native Ollama ``/api/chat`` endpoint.

    Used by :meth:`SemanticCache.verify_pending` to audit cached hits
    (P-CHR).  The prompt asks for a strict ``yes``/``no`` verdict plus a
    short reason; parsing is tolerant (first word, case-insensitive, with
    common punctuation stripped).  Unparseable replies raise ``ValueError``
    (mapped to ``verified=2`` by ``verify_pending``).  The ``transport``
    argument is an ``httpx`` escape hatch used by tests.
    """

    _PROMPT_TEMPLATE = (
        "You are auditing a semantic cache hit. Decide whether the CACHED "
        "RESPONSE is a correct answer to the ORIGINAL PROMPT.\n"
        "Reply with exactly one word first — yes or no — followed by a short "
        "reason (one sentence).\n\n"
        "ORIGINAL PROMPT:\n{prompt}\n\n"
        "CACHED RESPONSE:\n{response}"
    )

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "glm-5.2",
        timeout_seconds: float = 5.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/api/chat"
        self._model = model
        self._timeout_seconds = float(timeout_seconds)
        self._transport = transport

    async def __call__(
        self, prompt_text: str | None, response_text: str | None
    ) -> tuple[bool, float | None, str | None]:
        payload = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user",
                    "content": self._PROMPT_TEMPLATE.format(
                        prompt=prompt_text or "(unknown prompt)",
                        response=response_text or "(unknown response)",
                    ),
                }
            ],
        }
        async with httpx.AsyncClient(
            timeout=self._timeout_seconds, transport=self._transport
        ) as client:
            reply = await client.post(self._url, json=payload)
            reply.raise_for_status()
            data = reply.json()
        content = str(data.get("message", {}).get("content", "")).strip()
        if not content:
            raise ValueError("empty judge reply")
        verdict_word = content.split()[0].strip(".,:;!\"'()[]").lower()
        if verdict_word not in {"yes", "no"}:
            raise ValueError(f"unparseable judge reply: {content[:120]!r}")
        return verdict_word == "yes", None, content
