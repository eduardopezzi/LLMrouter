"""CLI panel for routing configuration and local statistics."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from llmrouter.config import Settings
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import Provider, RoutingStrategy

ROUTING_STRATEGY_ENV = "LLMROUTER_ROUTING__STRATEGY"
FALLBACK_COUNT_ENV = "LLMROUTER_ROUTING__FALLBACK_COUNT"
PROVIDER_COST_ORDER_ENV = "LLMROUTER_ROUTING__PROVIDER_COST_ORDER"
DEBUG_ENV = "LLMROUTER_DEBUG"
LOG_LEVEL_ENV = "LLMROUTER_LOG_LEVEL"
DEFAULT_LOG_PATH = "logs/llmrouter.log"


@dataclass(frozen=True)
class RoutingPanelConfig:
    """Current routing configuration shown by the CLI panel."""

    strategy: str
    fallback_count: int
    provider_cost_order: tuple[str, ...]


def routing_panel_config(settings: Settings) -> RoutingPanelConfig:
    """Build display-friendly routing settings."""
    return RoutingPanelConfig(
        strategy=settings.routing.strategy.value,
        fallback_count=settings.routing.fallback_count,
        provider_cost_order=tuple(settings.routing.provider_cost_order),
    )


def set_routing_strategy(env_path: str | Path, strategy: str) -> None:
    """Persist routing strategy in the .env file."""
    parsed_strategy = RoutingStrategy(strategy)
    update_env_file(env_path, {ROUTING_STRATEGY_ENV: parsed_strategy.value})


def set_fallback_count(env_path: str | Path, fallback_count: int) -> None:
    """Persist fallback count in the .env file."""
    if fallback_count < 0:
        raise ValueError("fallback count must be zero or greater")
    update_env_file(env_path, {FALLBACK_COUNT_ENV: str(fallback_count)})


def set_provider_cost_order(env_path: str | Path, providers: list[str]) -> None:
    """Persist provider cost tie-break order in the .env file."""
    normalized = _normalize_provider_order(providers)
    update_env_file(env_path, {PROVIDER_COST_ORDER_ENV: json.dumps(normalized)})


def update_env_file(env_path: str | Path, updates: dict[str, str]) -> None:
    """Update or append keys in a .env file while preserving unrelated lines."""
    path = Path(env_path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    updated_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            updated_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            updated_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            updated_lines.append(f"{key}={value}")

    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def catalog_stats(registry: ModelRegistry) -> dict[str, object]:
    """Return model catalog statistics."""
    models = registry.all()
    provider_counts = Counter(model.provider.value for model in models)
    tier_counts = Counter(f"T{model.tier.value}" for model in models)
    role_counts = Counter(role for model in models for role in model.capabilities)
    return {
        "models": len(models),
        "providers": dict(sorted(provider_counts.items())),
        "tiers": dict(sorted(tier_counts.items())),
        "roles": dict(sorted(role_counts.items())),
    }


def observation_stats(db_path: str | Path) -> dict[str, object]:
    """Return persisted routing observation statistics."""
    path = Path(db_path)
    if not path.exists():
        return {"database": str(path), "observations": 0, "reviews": 0}

    with sqlite3.connect(path) as db:
        if not _table_exists(db, "observations"):
            return {"database": str(path), "observations": 0, "reviews": 0}
        observations = int(db.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
        reviews = (
            int(db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0])
            if _table_exists(db, "reviews")
            else 0
        )
        row = db.execute(
            """
            SELECT
                COALESCE(AVG(latency_ms), 0),
                COALESCE(SUM(cost_usd), 0),
                COALESCE(SUM(prompt_tokens), 0),
                COALESCE(SUM(completion_tokens), 0)
            FROM observations
            """
        ).fetchone()
        model_rows = db.execute(
            """
            SELECT chosen_model, COUNT(*), COALESCE(AVG(latency_ms), 0)
            FROM observations
            GROUP BY chosen_model
            ORDER BY COUNT(*) DESC, chosen_model ASC
            LIMIT 5
            """
        ).fetchall()

    return {
        "database": str(path),
        "observations": observations,
        "reviews": reviews,
        "avg_latency_ms": round(float(row[0]), 2),
        "total_cost_usd": round(float(row[1]), 6),
        "prompt_tokens": int(row[2]),
        "completion_tokens": int(row[3]),
        "top_models": [
            {"model": str(model), "requests": int(count), "avg_latency_ms": round(float(avg), 2)}
            for model, count, avg in model_rows
        ],
    }


def render_panel_summary(settings: Settings, registry: ModelRegistry) -> str:
    """Render current routing configuration and stats."""
    routing = routing_panel_config(settings)
    catalog = catalog_stats(registry)
    observations = observation_stats(settings.evaluator.db_path)
    lines = [
        "LLMrouter CLI Panel",
        "",
        "Routing",
        f"  strategy: {routing.strategy}",
        f"  fallback_count: {routing.fallback_count}",
        f"  provider_cost_order: {', '.join(routing.provider_cost_order)}",
        "",
        "Catalog",
        f"  models: {catalog['models']}",
        f"  providers: {_format_mapping(catalog['providers'])}",
        f"  tiers: {_format_mapping(catalog['tiers'])}",
        "",
        "Observations",
        f"  database: {observations['database']}",
        f"  observations: {observations['observations']}",
        f"  reviews: {observations['reviews']}",
    ]
    if observations.get("observations"):
        lines.extend(
            [
                f"  avg_latency_ms: {observations['avg_latency_ms']}",
                f"  total_cost_usd: {observations['total_cost_usd']}",
                f"  tokens: {observations['prompt_tokens']} prompt, "
                f"{observations['completion_tokens']} completion",
            ]
        )
        top_models = observations.get("top_models", [])
        if isinstance(top_models, list) and top_models:
            lines.append("  top_models:")
            for item in top_models:
                if isinstance(item, dict):
                    lines.append(
                        f"    - {item['model']}: {item['requests']} requests, "
                        f"{item['avg_latency_ms']}ms avg"
                    )
    return "\n".join(lines)


def run_interactive_panel(
    settings: Settings,
    registry: ModelRegistry,
    *,
    env_path: str | Path = ".env",
) -> None:
    """Run an interactive terminal panel."""
    while True:
        print()
        print(render_panel_summary(settings, registry))
        print()
        print("1. Set routing strategy")
        print("2. Set fallback count")
        print("3. Set provider cost order")
        print("4. Show current settings")
        print("5. View logs")
        print("6. Toggle debug mode")
        print("7. Refresh stats")
        print("0. Exit")
        try:
            choice = input("Select an option: ").strip()
        except EOFError:
            return

        if choice == "1":
            _prompt_routing_strategy(env_path, settings)
            settings = _reload(settings)
        elif choice == "2":
            _prompt_fallback_count(env_path, settings)
            settings = _reload(settings)
        elif choice == "3":
            _prompt_provider_cost_order(env_path, settings, registry)
            settings = _reload(settings)
        elif choice == "4":
            print()
            print(render_current_settings(settings, registry))
            _pause_for_enter()
        elif choice == "5":
            _prompt_view_logs(settings)
        elif choice == "6":
            _prompt_toggle_debug(env_path, settings)
            settings = _reload(settings)
        elif choice in {"7", ""}:
            continue
        elif choice == "0":
            return
        else:
            print("Invalid option")


def render_current_settings(settings: Settings, registry: ModelRegistry) -> str:
    """Render all current configuration values for inspection."""
    routing = routing_panel_config(settings)
    catalog = catalog_stats(registry)
    models = registry.all()
    providers_in_catalog = sorted({model.provider.value for model in models})

    lines = [
        "=== Current Settings ===",
        "",
        "Routing",
        f"  strategy:           {routing.strategy}",
        f"  fallback_count:     {routing.fallback_count}",
        f"  provider_cost_order: {', '.join(routing.provider_cost_order)}",
        f"  max_cost_per_request: {settings.routing.max_cost_per_request or 'unlimited'}",
        "",
        "Scorer weights",
    ]
    for name, weight in sorted(settings.routing.scorer_weights.items()):
        lines.append(f"  {name}: {weight}")

    lines.extend(
        [
            "",
            "Server",
            f"  host: {settings.server.host}",
            f"  port: {settings.server.port}",
            f"  api_key: {'(set)' if settings.server.api_key else '(not set)'}",
            "",
            "Evaluator",
            f"  enabled: {settings.evaluator.enabled}",
            f"  db_path: {settings.evaluator.db_path}",
            f"  ollama_model: {settings.evaluator.ollama.model}",
            f"  ollama_base_url: {settings.evaluator.ollama.base_url}",
            "",
        "Debug",
        f"  debug: {settings.debug}",
        f"  log_level: {settings.log_level.value}",
            "",
            "Catalog summary",
            f"  total_models: {catalog['models']}",
            f"  providers_in_catalog: {', '.join(providers_in_catalog)}",
            f"  tiers: {_format_mapping(catalog['tiers'])}",
            f"  roles: {_format_mapping(catalog['roles'])}",
        ]
    )
    return "\n".join(lines)


def _prompt_routing_strategy(env_path: str | Path, settings: Settings) -> None:
    """Interactive prompt for routing strategy selection."""
    strategies = list(RoutingStrategy)
    current = settings.routing.strategy.value

    print()
    print(f"Current strategy: {current}")
    print("Available strategies:")
    for idx, strategy in enumerate(strategies, 1):
        marker = " (current)" if strategy.value == current else ""
        print(f"  {idx}) {strategy.value}{marker}")

    value = input(f"Select strategy [1-{len(strategies)}] or press Enter to keep current: ").strip()

    if not value:
        print("No changes.")
        return

    try:
        idx = int(value)
        if 1 <= idx <= len(strategies):
            selected = strategies[idx - 1].value
            set_routing_strategy(env_path, selected)
            print(f"Updated {ROUTING_STRATEGY_ENV}={selected}")
            return
    except ValueError:
        pass

    try:
        set_routing_strategy(env_path, value)
        print(f"Updated {ROUTING_STRATEGY_ENV}={value}")
    except Exception as exc:
        print(f"Error: {exc}")


def _prompt_fallback_count(env_path: str | Path, settings: Settings) -> None:
    """Interactive prompt for fallback count."""
    current = settings.routing.fallback_count
    print()
    print(f"Current fallback count: {current}")
    value = input("New fallback count (>=0) or press Enter to keep current: ").strip()

    if not value:
        print("No changes.")
        return

    try:
        count = int(value)
        set_fallback_count(env_path, count)
        print(f"Updated {FALLBACK_COUNT_ENV}={count}")
    except ValueError as exc:
        print(f"Error: {exc}")


def _prompt_provider_cost_order(
    env_path: str | Path,
    settings: Settings,
    registry: ModelRegistry,
) -> None:
    """Interactive prompt for provider cost order with numbered selection."""
    models = registry.all()
    available = sorted({model.provider.value for model in models})
    current = list(settings.routing.provider_cost_order)

    print()
    print(f"Current order: {', '.join(current) if current else '(empty)'}")
    print("Available providers in catalog:")
    for idx, provider in enumerate(available, 1):
        marker = ""
        if provider in current:
            marker = f" (position {current.index(provider) + 1})"
        print(f"  {idx}) {provider}{marker}")

    all_providers = sorted(provider.value for provider in Provider)
    missing = [p for p in all_providers if p not in available]
    if missing:
        print("Other supported providers (not in catalog):")
        offset = len(available)
        for idx, provider in enumerate(missing, 1):
            marker = ""
            if provider in current:
                marker = f" (position {current.index(provider) + 1})"
            print(f"  {offset + idx}) {provider}{marker}")
        available = available + missing

    print()
    value = input(
        "Enter numbers comma-separated (e.g. 2,5,3) or names (e.g. nvidia,zai,ollama).\n"
        "Press Enter to keep current: "
    ).strip()

    if not value:
        print("No changes.")
        return

    providers = _parse_provider_selection(value, available)
    if not providers:
        print("No valid providers parsed. No changes.")
        return

    try:
        set_provider_cost_order(env_path, providers)
        print(f"Updated {PROVIDER_COST_ORDER_ENV}={providers}")
    except Exception as exc:
        print(f"Error: {exc}")


def _prompt_view_logs(settings: Settings) -> None:
    """Follow recent log entries until interrupted."""
    import os

    log_path = os.environ.get("LLMROUTER_LOG_FILE")
    journal_unit = os.environ.get("LLMROUTER_JOURNAL_UNIT", "llmrouter")

    print()
    print("Showing last 25 lines. Press Ctrl+C to return to the panel.")

    try:
        if log_path:
            print(f"Log file: {log_path}")
            follow_log_file(log_path, lines_count=25)
        elif _journalctl_available():
            follow_journal_logs(journal_unit, lines_count=25)
        else:
            print(f"Log file: {DEFAULT_LOG_PATH}")
            follow_log_file(DEFAULT_LOG_PATH, lines_count=25)
    except KeyboardInterrupt:
        print()
        print("Stopped following logs.")


def _prompt_toggle_debug(env_path: str | Path, settings: Settings) -> None:
    """Toggle debug mode in the .env file."""
    current = settings.debug
    new_value = not current
    update_env_file(env_path, {DEBUG_ENV: "true" if new_value else "false"})
    print(f"Debug mode: {'ENABLED' if new_value else 'DISABLED'}")
    print(f"Updated {DEBUG_ENV}={'true' if new_value else 'false'}")


def _parse_provider_selection(value: str, available: list[str]) -> list[str]:
    """Parse user input into a list of provider names.

    Accepts either comma-separated numbers (indices into ``available``) or
    comma-separated provider names.
    """
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return []

    # Detect numeric selection
    if all(token.isdigit() for token in tokens):
        result: list[str] = []
        for token in tokens:
            idx = int(token)
            if 1 <= idx <= len(available):
                provider = available[idx - 1]
                if provider not in result:
                    result.append(provider)
        return result

    # Name-based selection
    return _normalize_provider_order(tokens)


def _read_log_tail(log_path: str, lines_count: int) -> str | None:
    """Read the last N lines from a log file. Returns None if file missing."""
    path = Path(log_path)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-lines_count:])


def follow_log_file(
    log_path: str,
    *,
    lines_count: int = 25,
    poll_interval_seconds: float = 1.0,
) -> None:
    """Print the last lines of a log file and follow appended content."""
    path = Path(log_path)
    if not path.exists():
        print(f"Log file not found: {log_path}")
        print("Tip: logs are written to stderr by default. Set LLMROUTER_LOG_FILE to persist.")
        return

    content = _read_log_tail(log_path, lines_count)
    print()
    print(f"=== Last {lines_count} lines of {log_path} ===")
    if content:
        print(content)
    else:
        print("(log is empty)")
    print("=== Following log ===")

    offset = _log_file_end_offset(path)
    while True:
        chunk, offset = _read_log_since(path, offset)
        if chunk:
            print(chunk, end="" if chunk.endswith("\n") else "\n", flush=True)
        time.sleep(max(poll_interval_seconds, 0.1))


def _log_file_end_offset(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        file.seek(0, 2)
        return file.tell()


def _read_log_since(path: Path, offset: int) -> tuple[str, int]:
    if not path.exists():
        return "", 0
    with path.open("r", encoding="utf-8", errors="replace") as file:
        file.seek(0, 2)
        end = file.tell()
        if end < offset:
            offset = 0
        file.seek(offset)
        chunk = file.read()
        return chunk, file.tell()


def follow_journal_logs(unit: str = "llmrouter", *, lines_count: int = 25) -> None:
    """Follow systemd journal logs for the LLMrouter service."""
    command = _journalctl_follow_command(unit, lines_count)
    print(f"Journal unit: {unit}")
    print("$ " + " ".join(command))
    subprocess.run(command, check=False)


def _journalctl_follow_command(unit: str, lines_count: int = 25) -> list[str]:
    return ["journalctl", "-u", unit, "-n", str(max(lines_count, 1)), "-f"]


def _journalctl_available() -> bool:
    return shutil.which("journalctl") is not None


def _pause_for_enter() -> None:
    try:
        input("\nPress Enter to return to the menu...")
    except EOFError:
        return


def _reload(settings: Settings) -> Settings:
    """Reload settings after env file changes."""
    from llmrouter.config import reload_settings

    return reload_settings()


def _normalize_provider_order(providers: list[str]) -> list[str]:
    if not providers:
        raise ValueError("provider order cannot be empty")
    normalized: list[str] = []
    for provider in providers:
        value = Provider(provider.strip().lower()).value
        if value not in normalized:
            normalized.append(value)
    return normalized


def _table_exists(db: sqlite3.Connection, table: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _format_mapping(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return ", ".join(f"{key}={item}" for key, item in value.items())
