"""Prompt complexity scorer.

Evaluates the complexity of a prompt using a weighted rule-based system.
The score (0.0–1.0) determines which model tier is appropriate.

The scorer is designed to be easily replaceable — a future ML-based scorer
(sentence-transformers + classifier) can implement the same interface.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from llmrouter.core.types import Tier

# ---------------------------------------------------------------------------
# Signal heuristics
# ---------------------------------------------------------------------------

_COMPLEXITY_KEYWORDS = frozenset(
    [
        "analyze",
        "architect",
        "compare",
        "complex",
        "comprehensive",
        "debug",
        "design",
        "detailed",
        "diagnose",
        "evaluate",
        "explain",
        "implement",
        "investigate",
        "optimize",
        "reason",
        "refactor",
        "research",
        "review",
        "summarize",
        "synthesis",
        "translate",
        "troubleshoot",
        "analise",
        "analisar",
        "arquitetar",
        "comparar",
        "complexo",
        "depurar",
        "diagnosticar",
        "implementar",
        "investigar",
        "otimizar",
        "pesquisar",
        "refatorar",
        "revisar",
    ]
)

_TASK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "security_audit",
        (
            r"\b(?:security audit|auditoria de seguran[cç]a|owasp|cve|"
            r"vulnerabilit\w*|vulnerabil\w*)\b",
            r"\b(?:injection|inje[cç][aã]o|xss|csrf|secret leakage|vazamento de segredo)\b",
        ),
    ),
    (
        "architecture",
        (
            r"\b(?:system design|software architecture|arquitetura de software|"
            r"arquitetura do sistema)\b",
            r"\b(?:desenhe|projete|design|architect)\b.*\b(?:sistema|system|servi[cç]o|service)\b",
        ),
    ),
    (
        "migration",
        (
            r"\b(?:migrat\w*|migra[cç]\w*|upgrade|atualiz\w*)\b.*\b(?:framework|vers[aã]o|version|api|banco|database)\b",
        ),
    ),
    (
        "test_generation",
        (
            r"\b(?:generate|write|create|gerar|criar|escrever)\b.*"
            r"\b(?:tests?|testes?|pytest|unit tests?|e2e)\b",
            r"\b(?:test coverage|cobertura de testes?|fixtures?|mocks?)\b",
        ),
    ),
    (
        "refactoring",
        (r"\b(?:refactor\w*|refator\w*|clean code|c[oó]digo legado|legacy code)\b",),
    ),
    (
        "review",
        (
            r"\b(?:code review|review (?:this|the)|revis(?:e|ar|[aã]o)|pull request|\bpr\b)\b",
        ),
    ),
    (
        "fix",
        (
            r"\b(?:debug|fix|corrij\w*|corrig\w*|depur\w*|bug|erro|error|"
            r"exception|stack trace|crash)\b",
        ),
    ),
    (
        "documentation",
        (
            r"\b(?:document\w*|documenta[cç]\w*|readme|docstring|swagger|openapi)\b",
        ),
    ),
    (
        "summarization",
        (r"\b(?:summari[sz]\w*|resum\w*|s[ií]ntese|tl;?dr)\b",),
    ),
)

_TASK_COMPLEXITY_FLOORS = {
    "security_audit": 0.67,
    "architecture": 0.52,
    "migration": 0.45,
    "test_generation": 0.40,
    "refactoring": 0.40,
    "review": 0.36,
    "fix": 0.36,
}

_ACTION_RE = re.compile(
    r"\b(?:analyze|architect|compare|debug|design|diagnose|evaluate|implement|investigate|"
    r"optimize|refactor|review|test|analis\w*|arquitet\w*|compar\w*|depur\w*|desenh\w*|"
    r"diagnostic\w*|implement(?:ar|e|a|amos|em|ando)|investig\w*|otimiz\w*|"
    r"refator\w*|revis\w*|test\w*)\b",
    re.IGNORECASE,
)

_CODE_PATTERNS = [
    r"```[\s\S]*?```",
    r"\bdef\s+\w+",
    r"\bclass\s+\w+",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import",
    r"\bfunction\s+\w+",
    r"\bconst\s+\w+",
    r"\binterface\s+\w+",
    r"\bpublic\s+(static\s+)?(void|int|String)",
    r"\b#include\s*<",
    r"\bfn\s+\w+",
    r"\bfunc\s+\w+",
    r"\bpackage\s+\w+",
    r"\bSELECT\s+.+\s+FROM",
    r"\bCREATE\s+TABLE",
    r"\bdocker",
    r"\bconsole\.log\b",
    r"\bprint\s*\(",
    r"\bprintf\s*\(",
    r"\bSystem\.out\.print",
]

_MATH_PATTERNS = [
    r"\bcalculate\b",
    r"\bsolve\b",
    r"\bequation\b",
    r"\bintegral\b",
    r"\bderivative\b",
    r"\bmatrix\b",
    r"\bprobability\b",
    r"\btheorem\b",
    r"\bproof\b",
    r"\balgorithm\b",
    r"\bcomple(x|city)\b",
    r"\boptimi[sz]ation\b",
    r"\bstatisti(cs|cal)\b",
    r"\bregression\b",
    r"\$\$.+?\$\$",
    r"\\[a-zA-Z]+\{",
    r"\b\d+\s*[+\-*/^]\s*\d+",
]


@dataclass(frozen=True)
class ScorerWeights:
    """Weights for each scoring signal. Should sum to ~1.0."""

    length: float = 0.15
    code_detection: float = 0.25
    complexity_keywords: float = 0.20
    math_detection: float = 0.20
    language_complexity: float = 0.20


@dataclass(frozen=True)
class ScoringResult:
    """The output of a scoring operation.

    Attributes:
        score: Overall complexity score (0.0–1.0).
        tier: Recommended tier based on the score.
        signals: Individual signal scores for debugging/logging.
    """

    score: float
    tier: Tier
    signals: dict[str, Any]


def _score_to_tier(score: float) -> Tier:
    """Map a complexity score to a model tier."""
    if score < 0.33:
        return Tier.T1
    if score < 0.66:
        return Tier.T2
    return Tier.T3


class PromptScorer:
    """Rule-based prompt complexity scorer.

    Computes a weighted score from multiple signals:
    - Length of the prompt (longer → more complex)
    - Code detection (code blocks/patterns → complex)
    - Complexity keywords (reasoning-related words)
    - Math detection (mathematical content)
    - Language complexity (vocabulary diversity, sentence length)

    The scorer is stateless and safe to share across tasks.
    """

    def __init__(
        self,
        weights: ScorerWeights | None = None,
        *,
        simple_threshold: float = 0.33,
        complex_threshold: float = 0.66,
    ) -> None:
        self._weights = weights or ScorerWeights()
        if not 0.0 <= simple_threshold < complex_threshold <= 1.0:
            raise ValueError("complexity thresholds must satisfy 0 <= simple < complex <= 1")
        self._simple_threshold = simple_threshold
        self._complex_threshold = complex_threshold
        self._code_re = [re.compile(p, re.IGNORECASE) for p in _CODE_PATTERNS]
        self._math_re = [re.compile(p, re.IGNORECASE) for p in _MATH_PATTERNS]
        self._complexity_kw = _COMPLEXITY_KEYWORDS
        self._task_patterns = [
            (task, [re.compile(pattern, re.IGNORECASE) for pattern in patterns])
            for task, patterns in _TASK_PATTERNS
        ]

    def score(self, prompt: str) -> ScoringResult:
        """Score the complexity of a prompt.

        Args:
            prompt: The concatenated prompt text (all messages).

        Returns:
            A :class:`ScoringResult` with the overall score and per-signal breakdown.
        """
        if not prompt or not prompt.strip():
            return ScoringResult(score=0.0, tier=Tier.T1, signals={})

        numeric_signals: dict[str, float] = {
            "length": self._score_length(prompt),
            "code_detection": self._score_code(prompt),
            "complexity_keywords": self._score_keywords(prompt),
            "math_detection": self._score_math(prompt),
            "language_complexity": self._score_language(prompt),
        }

        weight_map = {
            "length": self._weights.length,
            "code_detection": self._weights.code_detection,
            "complexity_keywords": self._weights.complexity_keywords,
            "math_detection": self._weights.math_detection,
            "language_complexity": self._weights.language_complexity,
        }

        total = sum(
            numeric_signals.get(name, 0.0) * weight for name, weight in weight_map.items()
        )

        task_type, task_hits = self._detect_task_type(prompt)
        complexity_floor = self._complexity_floor(prompt, task_type, numeric_signals)

        score = min(max(total, complexity_floor, 0.0), 1.0)
        tier = self._score_to_tier(score)
        complexity_level = (
            "simple"
            if tier == Tier.T1
            else "moderate"
            if tier == Tier.T2
            else "complex"
        )
        signals: dict[str, Any] = {
            **numeric_signals,
            "task_type": task_type,
            "task_confidence": min(task_hits / 2.0, 1.0),
            "complexity_floor": complexity_floor,
            "complexity_level": complexity_level,
        }
        return ScoringResult(score=score, tier=tier, signals=signals)

    def _score_to_tier(self, score: float) -> Tier:
        if score < self._simple_threshold:
            return Tier.T1
        if score < self._complex_threshold:
            return Tier.T2
        return Tier.T3

    def _detect_task_type(self, prompt: str) -> tuple[str, int]:
        ranked: list[tuple[int, int, str]] = []
        for priority, (task, patterns) in enumerate(self._task_patterns):
            hits = sum(1 for pattern in patterns if pattern.search(prompt))
            if hits:
                ranked.append((hits, -priority, task))
        if not ranked:
            return "general", 0
        hits, _, task = max(ranked)
        return task, hits

    @staticmethod
    def _complexity_floor(
        prompt: str,
        task_type: str,
        signals: dict[str, float],
    ) -> float:
        floor = _TASK_COMPLEXITY_FLOORS.get(task_type, 0.0)
        prompt_length = len(prompt)
        if prompt_length >= 12_000:
            floor = max(floor, 0.67)
        elif prompt_length >= 4_000:
            floor = max(floor, 0.36)
        action_count = len({match.group(0).lower() for match in _ACTION_RE.finditer(prompt)})
        if action_count >= 3:
            floor = max(floor, 0.67)
        elif signals["code_detection"] >= 0.67 and prompt_length >= 1_500:
            floor = max(floor, 0.67)
        return floor

    # ------------------------------------------------------------------
    # Individual signal scorers (each returns 0.0–1.0)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_length(prompt: str) -> float:
        """Score based on prompt length (logarithmic scale)."""
        length = len(prompt)
        if length == 0:
            return 0.0
        return min(math.log10(length + 1) / math.log10(5000), 1.0)

    def _score_code(self, prompt: str) -> float:
        """Detect code-related content. Returns 0.0–1.0."""
        matches = sum(1 for pattern in self._code_re if pattern.search(prompt))
        if matches == 0:
            return 0.0
        return min(matches / 3.0, 1.0)

    def _score_keywords(self, prompt: str) -> float:
        """Score based on complexity-related keyword density."""
        words = set(re.findall(r"\w+", prompt.lower(), flags=re.UNICODE))
        if not words:
            return 0.0
        hits = words & self._complexity_kw
        return min(len(hits) / 3.0, 1.0)

    def _score_math(self, prompt: str) -> float:
        """Detect mathematical content. Returns 0.0–1.0."""
        matches = sum(1 for pattern in self._math_re if pattern.search(prompt))
        if matches == 0:
            return 0.0
        return min(matches / 3.0, 1.0)

    @staticmethod
    def _score_language(prompt: str) -> float:
        """Score language complexity via vocabulary diversity and sentence length."""
        words = prompt.split()
        if not words:
            return 0.0

        ttr = len({w.lower() for w in words}) / len(words)

        sentences = [s for s in re.split(r"[.!?]+", prompt) if s.strip()]
        avg_sentence_len = (
            sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        )

        ttr_score = min(ttr / 0.6, 1.0)
        len_score = min(avg_sentence_len / 25.0, 1.0)

        return (ttr_score + len_score) / 2.0
