"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (defaults for gpt-4o-mini)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with OpenAI support and offline heuristic fallback."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = model or settings.openai_model
        self._client: Any = None
        if self.api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Could not initialize OpenAI client: {e}")
                self._client = None

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["gpt-4o-mini"])
        return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""
        if self._client is not None:
            return self._call_openai(system_prompt, user_prompt)
        return self._call_offline_mock(system_prompt, user_prompt)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            input_tokens = (
                response.usage.prompt_tokens
                if response.usage
                else len(system_prompt + user_prompt) // 4
            )
            output_tokens = (
                response.usage.completion_tokens if response.usage else len(content) // 4
            )
            cost = self._estimate_cost(input_tokens, output_tokens)
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        except Exception as e:
            logger.warning(f"OpenAI completion failed: {e}. Falling back if necessary.")
            raise

    def _call_offline_mock(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic offline mock completion for testing without API key."""
        sys_lower = system_prompt.lower()

        if "analyst" in sys_lower:
            content = (
                "### Analytical Evaluation & Insights\n"
                "1. **Core Architectural Insights**: Structured graph indexing combined with "
                "vector embeddings enhances multi-hop reasoning over unstructured corpora.\n"
                "2. **Comparative Trade-offs**: Traditional RAG offers lower indexing latency, "
                "whereas GraphRAG / fine-tuning provides superior contextual accuracy.\n"
                "3. **Source Reliability**: Sources demonstrate high credibility and consistency."
            )
        elif "writer" in sys_lower:
            snippet_title = user_prompt[:50].replace("\n", " ")
            content = (
                "# Research Synthesis & Executive Summary\n\n"
                "## Overview\n"
                f"Based on technical analysis regarding '{snippet_title}...', "
                "architectures demonstrate trade-offs in precision and cost.\n\n"
                "## Key Findings\n"
                "- **Graph-Enhanced Retrieval**: Knowledge graphs enable multi-hop reasoning [1].\n"
                "- **Accuracy vs Overhead**: RAG grounds responses in dynamic knowledge [2].\n"
                "- **Best Practices**: Modular agent coordination achieves faithfulness [3].\n\n"
                "## References\n"
                "[1] GraphRAG / RAG Architecture & Benchmarks (https://example.com/graphrag-sota)\n"
                "[2] Survey on Retrieval-Augmented Generation (https://example.com/rag-survey)\n"
                "[3] Best Practices for LLM Production Systems (https://example.com/llm-prod-guide)"
            )
        elif "critic" in sys_lower:
            content = (
                "### Review & Fact-Checking Report\n"
                "- **Factuality**: Claims are consistent with provided source snippets.\n"
                "- **Citation Quality**: Valid citations detected and mapped to sources.\n"
                "- **Clarity**: High quality structured sections for technical audience."
            )
        else:
            content = (
                f"### Baseline Research Summary: {user_prompt}\n\n"
                "Modern AI systems rely on structured knowledge retrieval. "
                "RAG grounds language models in external documents to prevent hallucinations, "
                "while GraphRAG further incorporates entity graphs for multi-hop synthesis. "
                "Fine-tuning adapts style, whereas RAG dynamically ingests up-to-date facts."
            )

        input_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        output_tokens = max(1, len(content) // 4)
        cost = self._estimate_cost(input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
