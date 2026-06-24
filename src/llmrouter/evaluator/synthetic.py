"""Synthetic prompt generation using the local evaluator model."""

from __future__ import annotations

import json

from llmrouter.evaluator.judge import QualityJudge


class SyntheticDataGenerator:
    """Generate prompts for router calibration."""

    def __init__(self, judge: QualityJudge) -> None:
        self._judge = judge

    async def generate_prompts(self, complexity: str, count: int = 10) -> list[str]:
        """Generate prompts at a requested complexity level."""
        return await self._generate(
            "Generate diverse user prompts for an LLM router calibration set.",
            complexity,
            count,
        )

    async def generate_adversarial_prompts(self, count: int = 5) -> list[str]:
        """Generate prompts likely to confuse a simple router."""
        return await self._generate(
            "Generate adversarial prompts that look simple but require careful routing.",
            "adversarial",
            count,
        )

    async def _generate(self, instruction: str, complexity: str, count: int) -> list[str]:
        data = await self._judge._chat_json(
            "Return only JSON with a prompts array of strings.",
            f"{instruction}\nComplexity: {complexity}\nCount: {count}",
        )
        prompts = data.get("prompts", [])
        if isinstance(prompts, str):
            try:
                prompts = json.loads(prompts)
            except json.JSONDecodeError:
                prompts = [prompts]
        if not isinstance(prompts, list):
            return []
        return [str(prompt) for prompt in prompts[:count]]
