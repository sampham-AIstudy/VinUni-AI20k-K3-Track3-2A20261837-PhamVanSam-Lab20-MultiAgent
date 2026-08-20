"""Unit tests for benchmarking and evaluation metrics."""

from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_quality_score,
    run_benchmark,
)


def test_citation_coverage_calculation() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Test query"),
        sources=[
            SourceDocument(title="Doc Alpha", url="https://example.com/alpha", snippet="..."),
            SourceDocument(title="Doc Beta", url="https://example.com/beta", snippet="..."),
        ],
        final_answer="According to [1] Doc Alpha and also [2] Doc Beta, everything is verified.",
    )
    coverage = compute_citation_coverage(state)
    assert coverage == 1.0


def test_quality_score_calculation() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Test query"),
        sources=[SourceDocument(title="Doc 1", snippet="...")],
        analysis_notes="Detailed analysis notes",
        final_answer="# Research Summary\nKey findings [1] and overview of concepts.",
    )
    score = compute_quality_score(state)
    assert score >= 8.0


def test_run_benchmark_execution() -> None:
    def sample_runner(q: str) -> ResearchState:
        return ResearchState(
            request=ResearchQuery(query=q),
            final_answer="# Summary\n[1] Reference",
            sources=[SourceDocument(title="Ref 1", snippet="...")],
        )

    state, metrics = run_benchmark("test_run", "Research query sample", sample_runner)
    assert metrics.run_name == "test_run"
    assert metrics.latency_seconds >= 0.0
    assert metrics.quality_score is not None
    assert metrics.citation_coverage == 1.0
