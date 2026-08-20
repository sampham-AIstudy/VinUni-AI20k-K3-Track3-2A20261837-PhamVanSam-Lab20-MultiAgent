"""Benchmark suite for single-agent vs multi-agent comparison."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate the ratio of retrieved sources referenced in the final answer."""
    if not state.sources or not state.final_answer:
        return 0.0

    answer = state.final_answer.lower()
    cited_count = 0

    for i, source in enumerate(state.sources):
        # Check citation markers like [1], [2], URL, or title keywords
        marker = f"[{i + 1}]"
        title_snippet = source.title.lower()[:25]
        url = (source.url or "").lower()

        if marker in state.final_answer or title_snippet in answer or (url and url in answer):
            cited_count += 1

    return min(1.0, cited_count / len(state.sources))


def compute_quality_score(state: ResearchState) -> float:
    """Calculate a 0-10 quality rubric score based on structure, citations, and content depth."""
    if not state.final_answer:
        return 0.0

    score = 4.0  # Base score for non-empty response

    answer = state.final_answer
    # 1. Structure check (+2 for headings/sections)
    if "#" in answer and ("overview" in answer.lower() or "summary" in answer.lower()):
        score += 2.0
    elif "#" in answer:
        score += 1.0

    # 2. Citations check (+2 for citations / references)
    if bool(re.search(r"\[\d+\]", answer)) or "references" in answer.lower():
        score += 2.0

    # 3. Source grounding (+1.0 if sources exist and were used)
    if state.sources and len(state.sources) > 0:
        score += 1.0

    # 4. Analysis depth (+1.0 if analytical steps occurred)
    if state.analysis_notes:
        score += 1.0

    # Deduct points for errors
    if state.errors:
        score = max(0.0, score - (1.0 * len(state.errors)))

    return min(10.0, score)


def compute_total_cost(state: ResearchState) -> float:
    """Sum estimated token costs across all agent operations."""
    total_cost = 0.0
    for res in state.agent_results:
        cost = res.metadata.get("cost_usd")
        if cost and isinstance(cost, (int, float)):
            total_cost += float(cost)
    return total_cost


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute runner, measure wall-clock latency, cost, quality, and citations."""
    started = perf_counter()
    failed = False
    try:
        state = runner(query)
    except Exception as exc:
        failed = True
        state = ResearchState(
            request={"query": query},  # type: ignore[arg-type]
            errors=[f"runner_exception: {exc}"],
        )

    latency = perf_counter() - started
    cost = compute_total_cost(state)
    coverage = compute_citation_coverage(state)
    quality = compute_quality_score(state)
    failure_rate = 1.0 if failed or (not state.final_answer and state.errors) else 0.0

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=failure_rate,
        notes=f"Iteration count: {state.iteration}; Sources: {len(state.sources)}",
    )
    return state, metrics
