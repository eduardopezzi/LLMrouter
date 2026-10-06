#!/usr/bin/env python3
"""P-CHR daily verification job (ROADMAP E1.3).

Calls ``POST /v1/llmrouter/cache/verify`` on a running LLMrouter gateway and
then reads ``GET /v1/llmrouter/cache/stats`` to report the rolling P-CHR
precision. Exit codes (alarm semantics per the product decision in
docs/ROADMAP.md#e1):

- 0: verification ran, precision healthy (or not yet measurable)
- 2: verification ran but the API rejected the call (HTTP 4xx/5xx)
- 3: could not reach the gateway at all
- 4: precision below 0.99 with >= 30 verified samples in the window
  (advisory alarm: raise threshold to 0.98 or disable semantic_cache via
  config — no redeploy needed)

The judge itself runs on the gateway side (local Ollama by default), so this
script spends ~0 subscription tokens.

Usage:
    python scripts/pchr_daily.py [--base-url URL] [--api-key KEY]
                                 [--sample-size N] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx

VERIFY_PATH = "/v1/llmrouter/cache/verify"
STATS_PATH = "/v1/llmrouter/cache/stats"
PRECISION_FLOOR = 0.99
PRECISION_MIN_SAMPLES = 30
JOB_TIMEOUT_SECONDS = 300.0


def _resolve_api_key(explicit: str | None) -> str | None:
    """Explicit flag wins, then env, then the gateway's own settings."""
    if explicit:
        return explicit
    env_key = os.environ.get("LLMROUTER_API_KEY")
    if env_key:
        return env_key
    try:
        from llmrouter.config import get_settings

        key: str | None = get_settings().server.api_key
        return key
    except Exception:
        return None


def run_verify(
    base_url: str = "http://127.0.0.1:12345",
    api_key: str | None = None,
    sample_size: int | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Run one verification cycle against the gateway and return the report.

    ``client`` is injectable for tests (httpx.MockTransport).
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=JOB_TIMEOUT_SECONDS)
    try:
        payload: dict[str, Any] = {}
        if sample_size is not None:
            payload["sample_size"] = sample_size
        verify_resp = client.post(
            f"{base_url.rstrip('/')}{VERIFY_PATH}", json=payload, headers=headers
        )
        if verify_resp.status_code >= 400:
            return {
                "ok": False,
                "stage": "verify",
                "status": verify_resp.status_code,
                "detail": verify_resp.text[:500],
            }

        stats_resp = client.get(
            f"{base_url.rstrip('/')}{STATS_PATH}", headers=headers
        )
        stats: dict[str, Any] = {}
        if stats_resp.status_code < 400:
            body = stats_resp.json()
            stats = {
                k: body.get(k)
                for k in (
                    "pchr_precision",
                    "pchr_verified_ok",
                    "pchr_verified_mismatch",
                    "pchr_pending",
                    "pchr_last_verified_ts",
                )
            }

        return {"ok": True, "verify": verify_resp.json(), "stats_pchr": stats}
    except httpx.HTTPError as exc:
        return {"ok": False, "stage": "connect", "detail": str(exc)[:500]}
    finally:
        if owns_client and client is not None:
            client.close()


def evaluate(report: dict[str, Any]) -> int:
    """Map a report from :func:`run_verify` to a job exit code."""
    if not report.get("ok"):
        if report.get("stage") == "connect":
            return 3
        return 2

    stats = report.get("stats_pchr") or {}
    precision = stats.get("pchr_precision")
    verified = (stats.get("pchr_verified_ok") or 0) + (
        stats.get("pchr_verified_mismatch") or 0
    )
    if precision is not None and verified >= PRECISION_MIN_SAMPLES:
        if float(precision) < PRECISION_FLOOR:
            return 4
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLMROUTER_BASE_URL", "http://127.0.0.1:12345"),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="dump full report")
    args = parser.parse_args(argv)

    api_key = _resolve_api_key(args.api_key)
    report = run_verify(
        base_url=args.base_url,
        api_key=api_key,
        sample_size=args.sample_size,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    elif report.get("ok"):
        verify = report.get("verify", {})
        stats = report.get("stats_pchr", {})
        print(
            "P-CHR run: checked={checked} ok={ok} mismatch={mismatch} "
            "error={error}".format(
                checked=verify.get("checked"),
                ok=verify.get("ok"),
                mismatch=verify.get("mismatch"),
                error=verify.get("error"),
            )
        )
        print(f"window: {stats}")
    else:
        stage = report.get("stage")
        reason = report.get("detail") or report.get("status")
        print(f"P-CHR run FAILED at {stage}: {reason}")

    return evaluate(report)


if __name__ == "__main__":
    sys.exit(main())
