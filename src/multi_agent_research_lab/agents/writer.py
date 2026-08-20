"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` with synthesized report and citations."""
        system_prompt = (
            "You are a professional technical research writer. Your task is to produce a "
            "coherent and structured report based on research and analysis notes. "
            "You MUST format your output in clean Markdown with clear headings and include "
            "numbered inline citations like [1], [2] referencing sources in References."
        )

        sources_formatted = "\n".join(
            f"[{i + 1}] {doc.title} ({doc.url or 'N/A'})\nSnippet: {doc.snippet}"
            for i, doc in enumerate(state.sources)
        )

        analysis_ctx = state.analysis_notes or state.research_notes or "Direct synthesis required."

        user_prompt = (
            f"Research Question: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Available Sources:\n{sources_formatted or 'No external sources available.'}\n\n"
            f"Analysis Notes:\n{analysis_ctx}\n\n"
            "Please write the final report including Overview, Key Findings, and References."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "writer.done",
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        except Exception as e:
            state.errors.append(f"writer_error: {e}")
            state.final_answer = f"Error generating final report: {e}"

        return state
