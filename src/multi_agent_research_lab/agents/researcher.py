"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        try:
            docs = self.search_client.search(
                query=state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources = docs
            if docs:
                state.research_notes = "\n".join(
                    f"[{i + 1}] {doc.title} ({doc.url}): {doc.snippet}"
                    for i, doc in enumerate(docs)
                )
            else:
                state.research_notes = "No external documents retrieved."
        except Exception as e:
            state.errors.append(f"researcher_error: {e}")
            state.research_notes = "Search retrieval failed."

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes or "",
                metadata={"num_sources": len(state.sources)},
            )
        )
        state.add_trace_event("researcher.done", {"num_sources": len(state.sources)})
        return state
