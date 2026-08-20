"""Unit tests for LangGraph MultiAgentWorkflow."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_end_to_end_run() -> None:
    workflow = MultiAgentWorkflow()
    state = ResearchState(
        request=ResearchQuery(query="GraphRAG architecture and tradeoffs", max_sources=3)
    )

    result = workflow.run(state)
    assert result.final_answer is not None
    assert len(result.sources) > 0
    assert result.analysis_notes is not None
    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert result.iteration == 4
    assert len(result.trace) >= 4
