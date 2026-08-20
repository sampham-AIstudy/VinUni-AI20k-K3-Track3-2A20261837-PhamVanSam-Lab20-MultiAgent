"""LangGraph workflow for Multi-Agent Research Assistant."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph with LangGraph."""

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self.supervisor = supervisor or SupervisorAgent()
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.settings = get_settings()

    def build(self) -> Any:
        """Create and compile a LangGraph graph with nodes, edges, and conditional routing."""
        builder = StateGraph(ResearchState)

        # 1. Define agent nodes
        def supervisor_node(state: ResearchState) -> ResearchState:
            return self.supervisor.run(state)

        def researcher_node(state: ResearchState) -> ResearchState:
            return self.researcher.run(state)

        def analyst_node(state: ResearchState) -> ResearchState:
            return self.analyst.run(state)

        def writer_node(state: ResearchState) -> ResearchState:
            return self.writer.run(state)

        builder.add_node("supervisor", supervisor_node)
        builder.add_node("researcher", researcher_node)
        builder.add_node("analyst", analyst_node)
        builder.add_node("writer", writer_node)

        # 2. Define routing function
        def router(state: ResearchState) -> str:
            if not state.route_history:
                return END
            last_route = state.route_history[-1]
            if last_route == "researcher":
                return "researcher"
            if last_route == "analyst":
                return "analyst"
            if last_route == "writer":
                return "writer"
            return END

        # 3. Connect nodes and conditional edges
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            router,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the final updated state."""
        app = self.build()
        result = app.invoke(state)
        if isinstance(result, ResearchState):
            return result
        if isinstance(result, dict):
            return ResearchState.model_validate(result)
        return state
