"""Supervisor / router for multi-agent workflow."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None) -> None:
        settings = get_settings()
        self.max_iterations = (
            max_iterations if max_iterations is not None else settings.max_iterations
        )

    def decide_next_route(self, state: ResearchState) -> str:
        """Determine next agent or stop condition based on current state."""
        if state.iteration >= self.max_iterations:
            return "done"

        if state.final_answer is not None and state.final_answer.strip():
            return "done"

        if not state.sources:
            if "researcher_failed" in state.errors:
                return "writer"  # Fallback to direct writing if research permanently failed
            return "researcher"

        if not state.analysis_notes:
            if "analyst_failed" in state.errors:
                return "writer"  # Fallback directly to writer if analysis failed
            return "analyst"

        if not state.final_answer:
            return "writer"

        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Evaluate state, determine next route, and record the decision."""
        next_route = self.decide_next_route(state)
        state.record_route(next_route)
        summary = f"Routed to '{next_route}' (iter {state.iteration}/{self.max_iterations})"
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=summary,
                metadata={"next_route": next_route, "iteration": state.iteration},
            )
        )
        state.add_trace_event(
            "supervisor.route",
            {"next_route": next_route, "iteration": state.iteration},
        )
        return state
