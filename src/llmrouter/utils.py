"""Shared utility functions used across LLMrouter modules."""

from __future__ import annotations

import os
from typing import Protocol


class _ApiKeyConfig(Protocol):
    api_key: str | None


def resolve_api_key(config: _ApiKeyConfig, *env_names: str) -> str | None:
    """Resolve an API key from config or environment variables.

    Checks the config object first, then falls back to environment variables
    in the order they are provided.
    """
    if config.api_key:
        return config.api_key
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return None