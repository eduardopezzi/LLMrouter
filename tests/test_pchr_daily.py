"""Tests for the P-CHR daily job (scripts/pchr_daily.py).

Uses httpx.MockTransport to fake the gateway. Stdlib + httpx only (httpx is
already a direct dependency of the project).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "pchr_daily.py"

spec = importlib.util.spec_from_file_location("pchr_daily", SCRIPT)
pchr = importlib.util.module_from_spec(spec)
sys.modules.setdefault("pchr_daily", pchr)
spec.loader.exec_module(pchr)  # type: ignore[union-attr]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


VERIFY_PAYLOAD = {
    "checked": 20,
    "ok": 20,
    "mismatch": 0,
    "error": 0,
    "buckets": {
        "0.80-0.85": {"checked": 1, "ok": 1, "mismatch": 0},
        "0.85-0.90": {"checked": 4, "ok": 4, "mismatch": 0},
        "0.90-0.95": {"checked": 7, "ok": 7, "mismatch": 0},
        "0.95-1.01": {"checked": 8, "ok": 8, "mismatch": 0},
    },
}

STATS_HEALTHY = {
    "pchr_precision": 1.0,
    "pchr_verified_ok": 35,
    "pchr_verified_mismatch": 0,
    "pchr_pending": 4,
    "pchr_last_verified_ts": 1790000000.0,
}


def test_run_verify_happy_path_posts_and_reads_stats():
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.method] = request
        if request.url.path == "/v1/llmrouter/cache/verify":
            return httpx.Response(200, json=VERIFY_PAYLOAD)
        return httpx.Response(200, json=STATS_HEALTHY)

    report = pchr.run_verify(
        base_url="http://fake",
        api_key="k1",
        client=_client(handler),
    )
    assert report["ok"] is True
    assert report["verify"] == VERIFY_PAYLOAD
    assert report["stats_pchr"]["pchr_precision"] == 1.0
    assert seen["POST"].url.path == "/v1/llmrouter/cache/verify"
    assert seen["POST"].headers["x-api-key"] == "k1"


def test_run_verify_sends_sample_size_when_given():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/verify"):
            assert b"sample_size" in request.read()
            return httpx.Response(200, json=VERIFY_PAYLOAD)
        return httpx.Response(200, json=STATS_HEALTHY)

    report = pchr.run_verify(
        base_url="http://fake", api_key=None, sample_size=5, client=_client(handler)
    )
    assert report["ok"] is True


def test_run_verify_http_error_returns_stage_verify():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="cache unavailable")

    report = pchr.run_verify(base_url="http://fake", client=_client(handler))
    assert report["ok"] is False
    assert report["stage"] == "verify"
    assert report["status"] == 503


def test_run_verify_connect_error_returns_stage_connect():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    report = pchr.run_verify(base_url="http://fake", client=_client(handler))
    assert report["ok"] is False
    assert report["stage"] == "connect"


def test_evaluate_healthy_returns_zero():
    report = {
        "ok": True,
        "verify": VERIFY_PAYLOAD,
        "stats_pchr": STATS_HEALTHY,
    }
    assert pchr.evaluate(report) == 0


def test_evaluate_precision_below_floor_alarms_only_with_enough_samples():
    low = {**STATS_HEALTHY, "pchr_precision": 0.95, "pchr_verified_ok": 35}
    report = {"ok": True, "verify": VERIFY_PAYLOAD, "stats_pchr": low}
    assert pchr.evaluate(report) == 4

    # Same bad precision but below the minimum sample count -> no alarm yet.
    few = {**low, "pchr_verified_ok": 2, "pchr_verified_mismatch": 0}
    report_few = {"ok": True, "verify": VERIFY_PAYLOAD, "stats_pchr": few}
    assert pchr.evaluate(report_few) == 0


def test_evaluate_maps_failures_to_exit_codes():
    assert pchr.evaluate({"ok": False, "stage": "verify", "status": 500}) == 2
    assert pchr.evaluate({"ok": False, "stage": "connect"}) == 3


def test_main_end_to_end_exit_zero(capsys, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/verify"):
            return httpx.Response(200, json=VERIFY_PAYLOAD)
        return httpx.Response(200, json=STATS_HEALTHY)

    real_client = httpx.Client  # capture BEFORE patching (pchr.httpx IS httpx)

    def fake_client_factory(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(pchr.httpx, "Client", fake_client_factory)
    code = pchr.main(["--base-url", "http://fake", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out


def test_script_has_no_execution_paths_constitutional_guard():
    """Dry-constitucional: o job só fala HTTP com o gateway, nunca executa código."""
    text = SCRIPT.read_text(encoding="utf-8")
    for banned in ("subprocess", "os.system", "popen", "eval(", "exec("):
        assert banned not in text, f"banned token in pchr_daily.py: {banned}"
