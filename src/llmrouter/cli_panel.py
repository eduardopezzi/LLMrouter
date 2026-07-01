"""CLI panel for routing configuration and local statistics."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from llmrouter.config import ProviderConfig, Settings
from llmrouter.core.registry import ModelRegistry
from llmrouter.core.types import ChatMessage, ChatRequest, ModelInfo, Provider, RoutingStrategy
from llmrouter.providers.base import BaseProvider
from llmrouter.providers.nvidia_provider import NvidiaProvider
from llmrouter.providers.ollama_provider import OllamaProvider
from llmrouter.providers.openai_provider import OpenAIProvider
from llmrouter.providers.zai_provider import ZaiProvider

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


@dataclass(frozen=True)
class ModelPriority:
    """Display row for catalog priority ordering."""

    rank: int
    priority: int
    name: str
    provider: str
    tier: str
    roles: tuple[str, ...]


@dataclass
class _ModelBlock:
    name: str
    priority: int
    name_line_index: int
    priority_line_index: int | None


@dataclass(frozen=True)
class _RankerModel:
    display_name: str
    provider: Provider
    provider_model_name: str


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


def model_priorities(registry: ModelRegistry, *, limit: int = 10) -> list[ModelPriority]:
    """Return models ordered by catalog priority."""
    ordered = sorted(registry.all(), key=lambda model: (model.priority, model.name))
    rows: list[ModelPriority] = []
    for rank, model in enumerate(ordered[: max(limit, 0)], 1):
        rows.append(
            ModelPriority(
                rank=rank,
                priority=model.priority,
                name=model.name,
                provider=model.provider.value,
                tier=f"T{model.tier.value}",
                roles=tuple(sorted(model.capabilities)),
            )
        )
    return rows


def render_model_priorities(registry: ModelRegistry, *, limit: int = 10) -> str:
    """Render the highest-priority models in the catalog."""
    rows = model_priorities(registry, limit=limit)
    lines = [f"Top {len(rows)} model priorities"]
    if not rows:
        lines.append("  (catalog is empty)")
        return "\n".join(lines)
    for row in rows:
        roles = ", ".join(row.roles) if row.roles else "-"
        lines.append(
            f"  {row.rank:>2}. priority={row.priority:<3} {row.name} "
            f"provider={row.provider} tier={row.tier} roles={roles}"
        )
    return "\n".join(lines)


def promote_model_priority(models_file: str | Path, model_name: str) -> None:
    """Move a model to priority 1 and shift the remaining catalog priorities down."""
    path = Path(models_file)
    blocks = _model_blocks(path)
    if not blocks:
        raise ValueError("models file does not contain model entries")
    if model_name not in {block.name for block in blocks}:
        raise ValueError(f"model not found in catalog: {model_name}")

    ordered = sorted(blocks, key=lambda block: (block.priority, block.name))
    promoted = next(block for block in ordered if block.name == model_name)
    reordered = [promoted] + [block for block in ordered if block.name != model_name]
    new_priorities = {block.name: index for index, block in enumerate(reordered, 1)}

    lines = path.read_text(encoding="utf-8").splitlines()
    for block in blocks:
        priority = new_priorities[block.name]
        if block.priority_line_index is None:
            name_line = lines[block.name_line_index]
            indent = _line_indent(name_line)
            lines.insert(block.name_line_index + 1, f"{indent}  priority: {priority}")
            for other in blocks:
                if other.name_line_index > block.name_line_index:
                    other.name_line_index += 1
                if (
                    other.priority_line_index is not None
                    and other.priority_line_index > block.name_line_index
                ):
                    other.priority_line_index += 1
            continue
        current = lines[block.priority_line_index]
        indent = _line_indent(current)
        lines[block.priority_line_index] = f"{indent}priority: {priority}"

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_model_priority_order(models_file: str | Path, ordered_model_names: list[str]) -> None:
    """Apply a complete priority order to the catalog."""
    path = Path(models_file)
    blocks = _model_blocks(path)
    if not blocks:
        raise ValueError("models file does not contain model entries")

    catalog_names = [block.name for block in blocks]
    _validate_model_order(ordered_model_names, catalog_names)
    new_priorities = {name: index for index, name in enumerate(ordered_model_names, 1)}

    lines = path.read_text(encoding="utf-8").splitlines()
    for block in blocks:
        priority = new_priorities[block.name]
        if block.priority_line_index is None:
            name_line = lines[block.name_line_index]
            indent = _line_indent(name_line)
            lines.insert(block.name_line_index + 1, f"{indent}  priority: {priority}")
            for other in blocks:
                if other.name_line_index > block.name_line_index:
                    other.name_line_index += 1
                if (
                    other.priority_line_index is not None
                    and other.priority_line_index > block.name_line_index
                ):
                    other.priority_line_index += 1
            continue
        current = lines[block.priority_line_index]
        indent = _line_indent(current)
        lines[block.priority_line_index] = f"{indent}priority: {priority}"

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reset_model_priorities_to_catalog_order(models_file: str | Path) -> None:
    """Reset priorities to the original model order in the YAML catalog."""
    blocks = _model_blocks(Path(models_file))
    if not blocks:
        raise ValueError("models file does not contain model entries")
    set_model_priority_order(models_file, [block.name for block in blocks])


def demote_model_priority(models_file: str | Path, model_name: str) -> bool:
    """Move a model to the lowest catalog priority.

    Returns ``True`` when the model was found and rewritten, or ``False`` when
    it was already the lowest-priority model.
    """
    blocks = _model_blocks(Path(models_file))
    if not blocks:
        raise ValueError("models file does not contain model entries")
    if model_name not in {block.name for block in blocks}:
        raise ValueError(f"model not found in catalog: {model_name}")

    ordered = sorted(blocks, key=lambda block: (block.priority, block.name))
    if ordered[-1].name == model_name:
        return False
    reordered = [block.name for block in ordered if block.name != model_name] + [model_name]
    set_model_priority_order(models_file, reordered)
    return True


async def request_llm_model_priority_order(
    settings: Settings,
    registry: ModelRegistry,
    *,
    llm_model: str | None = None,
    ranker_model: _RankerModel | None = None,
) -> list[str]:
    """Ask the evaluator LLM for a complete model priority order."""
    models = registry.all()
    if not models:
        raise ValueError("models file does not contain model entries")
    if ranker_model is None:
        ranker_model = _RankerModel(
            display_name=llm_model or settings.evaluator.ollama.model,
            provider=Provider.OLLAMA,
            provider_model_name=llm_model or settings.evaluator.ollama.model,
        )

    provider = _build_ranker_provider(settings, ranker_model.provider)
    try:
        response = await provider.chat_completion(
            ChatRequest(
                model=ranker_model.provider_model_name,
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "You rank LLM router model catalogs. Return only strict JSON, "
                            "with no markdown or commentary."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=_build_llm_priority_prompt(
                            models,
                            strategy=settings.routing.strategy.value,
                            provider_cost_order=settings.routing.provider_cost_order,
                        ),
                    ),
                ],
                temperature=settings.evaluator.ollama.temperature,
                max_tokens=4096,
                extra={"response_format": {"type": "json_object"}},
            ),
            ranker_model.provider_model_name,
        )
    finally:
        await provider.close()

    content = _response_text(response.choices)
    ordered_names = _parse_llm_priority_order(content)
    _validate_model_order(ordered_names, [model.name for model in models])
    return ordered_names


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
        print("7. Show model priorities")
        print("8. Promote model priority")
        print("9. Refresh stats")
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
        elif choice == "7":
            print()
            print(render_model_priorities(registry, limit=10))
            _pause_for_enter()
        elif choice == "8":
            registry = _prompt_model_priority_panel(settings, registry)
        elif choice in {"9", ""}:
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


def _prompt_model_priority_panel(settings: Settings, registry: ModelRegistry) -> ModelRegistry:
    """Submenu for model priority operations."""
    while True:
        print()
        print("=== Promote model priority ===")
        print(render_model_priorities(registry, limit=20))
        print()
        print("1. Promote a model to priority 1")
        print("2. Reset priorities to original catalog order")
        print("3. Ask evaluator LLM to reorder for current strategy")
        print("0. Return to main menu")
        try:
            choice = input("Select an option: ").strip()
        except EOFError:
            return registry

        if choice == "1":
            _prompt_promote_model_priority(settings.models_file, registry)
            registry = _reload_registry(settings.models_file)
        elif choice == "2":
            _prompt_reset_model_priorities(settings.models_file)
            registry = _reload_registry(settings.models_file)
        elif choice == "3":
            _prompt_llm_model_priority_order(settings, registry)
            registry = _reload_registry(settings.models_file)
        elif choice in {"0", ""}:
            return registry
        else:
            print("Invalid option")


def _prompt_promote_model_priority(models_file: str | Path, registry: ModelRegistry) -> None:
    """Interactive prompt for moving a model to the top catalog priority."""
    rows = model_priorities(registry, limit=20)
    print()
    value = input("Model number or exact name to promote, Enter to cancel: ").strip()
    if not value:
        print("No changes.")
        return

    model_name = value
    if value.isdigit():
        index = int(value)
        if 1 <= index <= len(rows):
            model_name = rows[index - 1].name
        else:
            print("Invalid model number.")
            return

    try:
        promote_model_priority(models_file, model_name)
        print(f"Promoted {model_name} to priority 1 in {models_file}")
    except Exception as exc:
        print(f"Error: {exc}")


def _prompt_reset_model_priorities(models_file: str | Path) -> None:
    """Interactive prompt for resetting priorities to catalog order."""
    value = input("Reset priorities to the original YAML order? [y/N]: ").strip().lower()
    if value not in {"y", "yes", "s", "sim"}:
        print("No changes.")
        return

    try:
        reset_model_priorities_to_catalog_order(models_file)
        print(f"Reset priorities to original catalog order in {models_file}")
    except Exception as exc:
        print(f"Error: {exc}")


def _prompt_llm_model_priority_order(settings: Settings, registry: ModelRegistry) -> None:
    """Ask the configured evaluator LLM to reorder model priorities."""
    default_llm_model = settings.evaluator.ollama.model
    print()
    print(
        "This will ask an LLM "
        f"(default: {default_llm_model} at {settings.evaluator.ollama.base_url}) "
        f"to reorder models for strategy '{settings.routing.strategy.value}'."
    )
    ranker_model = _prompt_ranker_model(settings, registry, default_llm_model)
    if ranker_model is None:
        print("No changes.")
        return
    value = input("Apply the LLM-suggested order if valid? [y/N]: ").strip().lower()
    if value not in {"y", "yes", "s", "sim"}:
        print("No changes.")
        return

    try:
        ordered_names = asyncio.run(
            request_llm_model_priority_order(settings, registry, ranker_model=ranker_model)
        )
        set_model_priority_order(settings.models_file, ordered_names)
        print(f"Applied LLM-suggested priority order to {settings.models_file}")
        print("New top priorities:")
        print(render_model_priorities(_reload_registry(settings.models_file), limit=10))
    except Exception as exc:
        print(f"Error: {exc}")


def _prompt_ranker_model(
    settings: Settings,
    registry: ModelRegistry,
    default_model: str,
) -> _RankerModel | None:
    """Prompt for the model/API used to rank priorities."""
    rankers = _available_ranker_models(settings, registry)
    if rankers:
        print("Available models/APIs for the ranking request:")
        for idx, ranker in enumerate(rankers, 1):
            print(f"  {idx}) {ranker.display_name} provider={ranker.provider.value}")
    value = input(
        f"LLM model to ask, number/name, Enter for {default_model}, or 0 to cancel: "
    ).strip()
    if value == "0":
        return None
    if not value:
        return _RankerModel(
            display_name=default_model,
            provider=Provider.OLLAMA,
            provider_model_name=default_model,
        )
    if value.isdigit():
        index = int(value)
        if 1 <= index <= len(rankers):
            return rankers[index - 1]
        print("Invalid model number.")
        return None
    matched = next((ranker for ranker in rankers if ranker.display_name == value), None)
    if matched is not None:
        return matched
    return _RankerModel(display_name=value, provider=Provider.OLLAMA, provider_model_name=value)


def _available_ranker_models(settings: Settings, registry: ModelRegistry) -> list[_RankerModel]:
    rankers: list[_RankerModel] = []
    for model in sorted(registry.all(), key=lambda item: (item.priority, item.name)):
        if not _ranker_provider_available(settings, model.provider):
            continue
        if model.provider == Provider.GEMINI:
            continue
        rankers.append(
            _RankerModel(
                display_name=model.name,
                provider=model.provider,
                provider_model_name=model.provider_model_name,
            )
        )
    return rankers


def _ranker_provider_available(settings: Settings, provider: Provider) -> bool:
    if provider == Provider.OLLAMA:
        return settings.providers.ollama.enabled
    if provider == Provider.OPENAI:
        return (
            settings.providers.openai.enabled
            and _api_key(settings.providers.openai, "OPENAI_API_KEY") is not None
        )
    if provider == Provider.NVIDIA:
        return settings.providers.nvidia.enabled and _api_key(
            settings.providers.nvidia,
            "NVIDIA_NIM_API_KEY",
            "NVIDIA_API_KEY",
        ) is not None
    if provider == Provider.ZAI:
        return (
            settings.providers.zai.enabled
            and _api_key(settings.providers.zai, "ZAI_API_KEY") is not None
        )
    return False


def _build_ranker_provider(settings: Settings, provider: Provider) -> BaseProvider:
    if provider == Provider.OLLAMA:
        return OllamaProvider(
            api_key=_api_key(settings.providers.ollama, "OLLAMA_API_KEY"),
            base_url=settings.providers.ollama.base_url,
            timeout=settings.providers.ollama.timeout,
            max_retries=settings.providers.ollama.max_retries,
        )
    if provider == Provider.OPENAI:
        api_key = _api_key(settings.providers.openai, "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not configured")
        return OpenAIProvider(
            api_key=api_key,
            base_url=settings.providers.openai.base_url,
            timeout=settings.providers.openai.timeout,
            max_retries=settings.providers.openai.max_retries,
        )
    if provider == Provider.NVIDIA:
        api_key = _api_key(settings.providers.nvidia, "NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA API key is not configured")
        return NvidiaProvider(
            api_key=api_key,
            base_url=settings.providers.nvidia.base_url,
            timeout=settings.providers.nvidia.timeout,
            max_retries=settings.providers.nvidia.max_retries,
        )
    if provider == Provider.ZAI:
        api_key = _api_key(settings.providers.zai, "ZAI_API_KEY")
        if not api_key:
            raise ValueError("ZAI API key is not configured")
        return ZaiProvider(
            api_key=api_key,
            base_url=settings.providers.zai.base_url,
            timeout=settings.providers.zai.timeout,
            max_retries=settings.providers.zai.max_retries,
        )
    raise ValueError(f"Provider {provider.value} cannot be used for priority ranking")


def _api_key(config: ProviderConfig, *env_names: str) -> str | None:
    if config.api_key:
        return config.api_key
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


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


def _build_llm_priority_prompt(
    models: list[ModelInfo],
    *,
    strategy: str,
    provider_cost_order: list[str],
) -> str:
    strategy_guidance = {
        "cost": (
            "Prefer the lowest total token cost. Use provider_cost_order as a tie-breaker "
            "when costs are equal, then prefer smaller/fast models."
        ),
        "quality": (
            "Prefer the strongest and most capable models first, even when that changes the "
            "current order substantially. Use roles, context window, tier, provider, and "
            "descriptions as quality signals. You may move models from any provider/API "
            "(Ollama, NVIDIA, ZAI, OpenAI, etc.) to the top when their metadata suggests "
            "better quality."
        ),
        "balanced": (
            "Balance quality, breadth of roles, context window, and cost. Avoid putting an "
            "expensive specialist ahead of a strong generalist unless the metadata justifies it."
        ),
        "latency": (
            "Prefer likely faster models first. Favor local Ollama models and smaller models "
            "when quality is similar."
        ),
    }.get(strategy, "Rank models according to the selected routing strategy.")
    rows = []
    for model in sorted(models, key=lambda item: (item.priority, item.name)):
        rows.append(
            {
                "name": model.name,
                "provider": model.provider.value,
                "tier": model.tier.value,
                "priority": model.priority,
                "roles": sorted(model.capabilities),
                "context_window": model.context_window,
                "max_tokens": model.max_tokens,
                "cost_ratio": model.cost_ratio,
                "description": model.description,
            }
        )
    return (
        f"Current routing strategy: {strategy}\n"
        f"Provider cost tie-break order: {', '.join(provider_cost_order)}\n"
        f"Strategy guidance: {strategy_guidance}\n\n"
        "Return a complete priority order for every model below. Lower position means higher "
        "priority. The order may differ from the current priority and may promote any "
        "provider/API. Do not add, remove, rename, or duplicate models.\n\n"
        "Respond exactly as JSON with this shape:\n"
        '{"models":["exact model name","exact model name"]}\n\n'
        f"Models:\n{json.dumps(rows, ensure_ascii=False, indent=2)}"
    )


def _response_text(choices: object) -> str:
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response did not include choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("LLM response choice is not a mapping")
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise ValueError("LLM response did not include text content")


def _parse_llm_priority_order(content: str) -> list[str]:
    parsed: object
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match is None:
            raise ValueError("LLM response was not valid JSON") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON") from exc

    if isinstance(parsed, dict):
        models = parsed.get("models")
    else:
        models = parsed
    if not isinstance(models, list) or not all(isinstance(item, str) for item in models):
        raise ValueError('LLM response must contain a "models" list of strings')
    return list(models)


def _validate_model_order(ordered_model_names: list[str], catalog_names: list[str]) -> None:
    expected = set(catalog_names)
    received = set(ordered_model_names)
    if len(ordered_model_names) != len(catalog_names):
        raise ValueError(
            f"model order must include exactly {len(catalog_names)} models; "
            f"received {len(ordered_model_names)}"
        )
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unknown: {', '.join(extra)}")
        raise ValueError("model order does not match catalog (" + "; ".join(details) + ")")
    if len(received) != len(ordered_model_names):
        raise ValueError("model order contains duplicate models")


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


def _reload_registry(models_file: str | Path) -> ModelRegistry:
    from llmrouter.core.registry import load_model_registry

    return load_model_registry(models_file)


def _normalize_provider_order(providers: list[str]) -> list[str]:
    if not providers:
        raise ValueError("provider order cannot be empty")
    normalized: list[str] = []
    for provider in providers:
        value = Provider(provider.strip().lower()).value
        if value not in normalized:
            normalized.append(value)
    return normalized


def _model_blocks(path: Path) -> list[_ModelBlock]:
    lines = path.read_text(encoding="utf-8").splitlines()
    starts: list[int] = []
    name_pattern = re.compile(r"^(\s*)-\s+name:\s+(.+?)\s*$")
    priority_pattern = re.compile(r"^\s+priority:\s+(\d+)\s*$")

    for index, line in enumerate(lines):
        if name_pattern.match(line):
            starts.append(index)

    blocks: list[_ModelBlock] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        name_match = name_pattern.match(lines[start])
        if name_match is None:
            continue
        name = _strip_yaml_scalar(name_match.group(2))
        priority = 10
        priority_line_index: int | None = None
        for index in range(start + 1, end):
            priority_match = priority_pattern.match(lines[index])
            if priority_match is not None:
                priority = int(priority_match.group(1))
                priority_line_index = index
                break
        blocks.append(
            _ModelBlock(
                name=name,
                priority=priority,
                name_line_index=start,
                priority_line_index=priority_line_index,
            )
        )
    return blocks


def _strip_yaml_scalar(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _line_indent(value: str) -> str:
    match = re.match(r"^(\s*)", value)
    return match.group(1) if match is not None else ""


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
