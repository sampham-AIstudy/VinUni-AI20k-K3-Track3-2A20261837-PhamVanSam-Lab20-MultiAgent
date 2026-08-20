"""Critic agent implementation for fact-checking and validation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Fact-checking, citation validation, and quality assessment agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer, verify citation references, and append findings."""
        if not state.final_answer:
            state.errors.append("critic_warning: no final answer to evaluate")
            return state

        system_prompt = (
            "You are an AI research critic. Your job is to verify factuality, ensure claims "
            "are grounded in provided sources, verify that numbered citations are present, "
            "and check for hallucinations or unsupported assertions."
        )

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Sources:\n{state.research_notes or 'None'}\n\n"
            f"Final Answer:\n{state.final_answer}\n\n"
            "Evaluate the quality and accuracy of the answer."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "critic.done",
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        except Exception as e:
            state.errors.append(f"critic_error: {e}")

        return state
