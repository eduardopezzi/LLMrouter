"""CLI panel for routing configuration and local statistics."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from llmrouter.config import Settings
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import Provider, RoutingStrategy

ROUTING_STRATEGY_ENV = "LLMROUTER_ROUTING__STRATEGY"
FALLBACK_COUNT_ENV = "LLMROUTER_ROUTING__FALLBACK_COUNT"
PROVIDER_COST_ORDER_ENV = "LLMROUTER_ROUTING__PROVIDER_COST_ORDER"


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
        print("4. Refresh stats")
        print("5. Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            values = ", ".join(strategy.value for strategy in RoutingStrategy)
            value = input(f"Strategy ({values}): ").strip()
            set_routing_strategy(env_path, value)
            print(f"Updated {ROUTING_STRATEGY_ENV}={value}")
        elif choice == "2":
            value = int(input("Fallback count: ").strip())
            set_fallback_count(env_path, value)
            print(f"Updated {FALLBACK_COUNT_ENV}={value}")
        elif choice == "3":
            value = input("Provider order, comma-separated (example: nvidia,zai,ollama): ")
            providers = [item.strip() for item in value.split(",") if item.strip()]
            set_provider_cost_order(env_path, providers)
            print(f"Updated {PROVIDER_COST_ORDER_ENV}={providers}")
        elif choice == "4":
            continue
        elif choice == "5":
            return
        else:
            print("Invalid option")


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
