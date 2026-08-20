"""Comprehensive unit tests for all agents in the multi-agent research lab."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routing_full_cycle() -> None:
    supervisor = SupervisorAgent(max_iterations=6)
    query = ResearchQuery(query="Explain LangGraph architecture")
    state = ResearchState(request=query)

    # 1. Initially no sources -> routes to researcher
    assert supervisor.decide_next_route(state) == "researcher"
    state = supervisor.run(state)
    assert state.route_history == ["researcher"]
    assert state.iteration == 1

    # 2. Add sources -> routes to analyst
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    assert supervisor.decide_next_route(state) == "analyst"
    state = supervisor.run(state)
    assert state.route_history == ["researcher", "analyst"]
    assert state.iteration == 2

    # 3. Add analysis notes -> routes to writer
    state.analysis_notes = "Analysis findings..."
    assert supervisor.decide_next_route(state) == "writer"
    state = supervisor.run(state)
    assert state.route_history == ["researcher", "analyst", "writer"]
    assert state.iteration == 3

    # 4. Add final answer -> routes to done
    state.final_answer = "# Final Report\n[1] Doc 1"
    assert supervisor.decide_next_route(state) == "done"
    state = supervisor.run(state)
    assert state.route_history == ["researcher", "analyst", "writer", "done"]


def test_supervisor_max_iterations_guardrail() -> None:
    supervisor = SupervisorAgent(max_iterations=3)
    state = ResearchState(request=ResearchQuery(query="Test query"), iteration=3)
    assert supervisor.decide_next_route(state) == "done"


def test_supervisor_error_fallbacks() -> None:
    supervisor = SupervisorAgent(max_iterations=6)
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.errors.append("researcher_failed")
    assert supervisor.decide_next_route(state) == "writer"


def test_researcher_agent_execution() -> None:
    researcher = ResearcherAgent()
    state = ResearchState(request=ResearchQuery(query="GraphRAG overview", max_sources=2))
    updated_state = researcher.run(state)

    assert len(updated_state.sources) <= 2
    assert updated_state.research_notes is not None
    assert any(res.agent.value == "researcher" for res in updated_state.agent_results)
    assert any(t["name"] == "researcher.done" for t in updated_state.trace)


def test_analyst_agent_execution() -> None:
    analyst = AnalystAgent()
    state = ResearchState(
        request=ResearchQuery(query="Test topic"),
        sources=[SourceDocument(title="S1", snippet="C1")],
        research_notes="[1] S1: C1",
    )
    updated_state = analyst.run(state)

    assert updated_state.analysis_notes is not None
    assert len(updated_state.analysis_notes) > 0
    assert any(res.agent.value == "analyst" for res in updated_state.agent_results)


def test_writer_agent_execution() -> None:
    writer = WriterAgent()
    state = ResearchState(
        request=ResearchQuery(query="Enterprise RAG"),
        sources=[SourceDocument(title="S1", url="https://example.com/s1", snippet="C1")],
        analysis_notes="Key findings on RAG.",
    )
    updated_state = writer.run(state)

    assert updated_state.final_answer is not None
    assert len(updated_state.final_answer) > 0
    assert any(res.agent.value == "writer" for res in updated_state.agent_results)


def test_critic_agent_execution() -> None:
    critic = CriticAgent()
    state = ResearchState(
        request=ResearchQuery(query="Enterprise RAG"),
        final_answer="# Summary\n[1] Reference",
    )
    updated_state = critic.run(state)
    assert any(res.agent.value == "critic" for res in updated_state.agent_results)
