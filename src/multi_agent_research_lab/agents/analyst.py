"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes` from research notes."""
        if not state.sources and not state.research_notes:
            state.errors.append("analyst_warning: no sources available to analyze")
            state.analysis_notes = "No sources available for analysis."
            return state

        system_prompt = (
            "You are an expert technical analyst. Your responsibility is to analyze "
            "the provided research notes, compare technical perspectives, evaluate reliability "
            "of evidence, and extract actionable structured insights."
        )

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research Notes:\n{state.research_notes or 'None'}\n\n"
            "Please provide a structured technical breakdown with core claims and tradeoffs."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "analyst.done",
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        except Exception as e:
            state.errors.append(f"analyst_error: {e}")
            state.analysis_notes = f"Analysis error encountered: {e}"

        return state
