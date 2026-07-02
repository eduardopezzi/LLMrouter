from __future__ import annotations

from llmrouter.core.types import Provider
from llmrouter.providers.deepseek_provider import DeepSeekProvider


def test_deepseek_provider_default_base_url() -> None:
    provider = DeepSeekProvider(api_key="sk-test")
    assert provider._base_url == "https://api.deepseek.com/v1"
    assert provider._name == "deepseek"


def test_deepseek_provider_custom_base_url() -> None:
    provider = DeepSeekProvider(api_key="sk-test", base_url="https://custom.deepseek.com/v1")
    assert provider._base_url == "https://custom.deepseek.com/v1"


def test_deepseek_provider_builds_bearer_headers() -> None:
    provider = DeepSeekProvider(api_key="sk-test")
    headers = provider._build_headers()
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["Content-Type"] == "application/json"


def test_deepseek_provider_no_api_key_omits_auth_header() -> None:
    provider = DeepSeekProvider(api_key=None)
    headers = provider._build_headers()
    assert "Authorization" not in headers


def test_deepseek_provider_is_in_provider_enum() -> None:
    assert Provider.DEEPSEEK.value == "deepseek"
    assert Provider("deepseek") == Provider.DEEPSEEK
