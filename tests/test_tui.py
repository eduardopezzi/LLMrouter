from __future__ import annotations

import pytest

from llmrouter.config import Settings
from llmrouter.tui import LLMrouterTUI


@pytest.mark.asyncio
async def test_tui_reorders_and_toggles_catalog_models(tmp_path) -> None:
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n"
        "  - name: model-a\n"
        "    provider: ollama\n"
        "    tier: 1\n"
        "    priority: 1\n"
        "  - name: model-b\n"
        "    provider: zai\n"
        "    tier: 2\n"
        "    enabled: false\n"
        "    priority: 2\n",
        encoding="utf-8",
    )

    app = LLMrouterTUI(Settings(), models_file=models_file)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._order == ["model-a", "model-b"]

        await pilot.press("a")
        await pilot.press("e")
        await pilot.press("s")
        await pilot.pause()

    body = models_file.read_text(encoding="utf-8")
    assert "name: model-b" in body
    assert "name: model-a" in body
    assert body.count("enabled: false") == 2
    assert app._order == ["model-b", "model-a"]
