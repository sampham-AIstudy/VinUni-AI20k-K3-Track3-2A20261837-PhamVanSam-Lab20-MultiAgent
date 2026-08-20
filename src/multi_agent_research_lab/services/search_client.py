"""Search client abstraction for ResearcherAgent."""

import json
import logging
import ssl
import urllib.request

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client supporting Tavily and domain-aware mock search."""

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self.api_key:
            try:
                return self._search_tavily(query, max_results)
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}. Falling back to mock search.")
        return self._search_mock(query, max_results)

    def _search_tavily(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Call Tavily Search API via HTTPS with SSL context."""
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "MultiAgentResearchLab/1.0",
            },
            method="POST",
        )

        try:
            import certifi

            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ssl_context = ssl.create_default_context()

        with urllib.request.urlopen(req, timeout=10, context=ssl_context) as response:
            res_data = json.loads(response.read().decode("utf-8"))

        results: list[SourceDocument] = []
        for item in res_data.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled Document"),
                    url=item.get("url", "https://tavily.com"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score", 1.0)},
                )
            )
        return results[:max_results]

    def _search_mock(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Deterministic keyword-aware mock search results."""
        q_lower = query.lower()

        if "graphrag" in q_lower or "graph" in q_lower:
            mock_docs = [
                SourceDocument(
                    title="From Local to Global: A Graph RAG Approach",
                    url="https://arxiv.org/abs/2404.16130",
                    snippet=(
                        "GraphRAG combines LLM knowledge graphs with community detection "
                        "to generate hierarchical summaries for broad dataset queries."
                    ),
                    metadata={"source": "arXiv:2404.16130", "year": 2024},
                ),
                SourceDocument(
                    title="Graph Retrieval-Augmented Generation: A Survey",
                    url="https://arxiv.org/abs/2408.08921",
                    snippet=(
                        "Survey on graph indexing, entity linking, and multi-hop reasoning "
                        "pathways that outperform vector-only retrieval."
                    ),
                    metadata={"source": "arXiv:2408.08921", "year": 2024},
                ),
                SourceDocument(
                    title="Microsoft GraphRAG Production Best Practices",
                    url="https://github.com/microsoft/graphrag",
                    snippet=(
                        "Modular pipeline for hierarchical text extraction, graph construction, "
                        "Leiden clustering, and prompt tuning."
                    ),
                    metadata={"source": "GitHub", "year": 2024},
                ),
            ]
        elif "fine-tuning" in q_lower or "finetune" in q_lower or "rag" in q_lower:
            mock_docs = [
                SourceDocument(
                    title="RAG vs Fine-tuning: Architectural Trade-offs",
                    url="https://example.com/rag-vs-finetuning",
                    snippet=(
                        "RAG excels at dynamic knowledge grounding with instant updates, "
                        "whereas fine-tuning is optimal for format and style adaptation."
                    ),
                    metadata={"category": "Enterprise Guide", "year": 2024},
                ),
                SourceDocument(
                    title="Retrieval-Augmented Generation for NLP Tasks",
                    url="https://arxiv.org/abs/2005.11401",
                    snippet=(
                        "Benchmark showing how non-parametric memory retrieval significantly "
                        "reduces factual hallucination in generation tasks."
                    ),
                    metadata={"source": "NeurIPS", "year": 2020},
                ),
                SourceDocument(
                    title="Parameter-Efficient Fine-Tuning with LoRA",
                    url="https://arxiv.org/abs/2106.09685",
                    snippet=(
                        "Low-rank adaptation enables cost-effective model specialization "
                        "without full weight retuning."
                    ),
                    metadata={"source": "ICLR", "year": 2022},
                ),
            ]
        elif "multi-agent" in q_lower or "supervisor" in q_lower or "langgraph" in q_lower:
            mock_docs = [
                SourceDocument(
                    title="LangGraph: Multi-Agent Workflows with Cyclic State",
                    url="https://langchain-ai.github.io/langgraph/",
                    snippet=(
                        "Framework for stateful multi-actor LLM applications with "
                        "human-in-the-loop, time-travel, and supervisor routing."
                    ),
                    metadata={"category": "Framework Docs", "year": 2024},
                ),
                SourceDocument(
                    title="Communicative Agents for Software Development",
                    url="https://arxiv.org/abs/2307.07924",
                    snippet=(
                        "Role specialization (programmer, reviewer, tester) mitigates context "
                        "dilution and improves multi-step execution."
                    ),
                    metadata={"source": "ACL", "year": 2023},
                ),
                SourceDocument(
                    title="Routing and Guardrails in Multi-Agent Systems",
                    url="https://example.com/multi-agent-guardrails",
                    snippet=(
                        "Production patterns including iteration limits, circuit breakers, "
                        "and strict shared state schema validation."
                    ),
                    metadata={"category": "Engineering Guide", "year": 2024},
                ),
            ]
        else:
            mock_docs = [
                SourceDocument(
                    title=f"Technical Research: {query[:40]}",
                    url="https://example.com/technical-overview",
                    snippet=(
                        f"Literature review and findings concerning '{query}'. "
                        "Highlights architectures, trade-offs, and benchmarks."
                    ),
                    metadata={"category": "Research Overview"},
                ),
                SourceDocument(
                    title="Modern LLM Architectures and Evaluation",
                    url="https://example.com/llm-eval-frameworks",
                    snippet=(
                        "State-of-the-art evaluation metrics comparing latency, cost, "
                        "groundedness, and task completion."
                    ),
                    metadata={"category": "Benchmark Report"},
                ),
                SourceDocument(
                    title="Best Practices for Production AI Agents",
                    url="https://example.com/production-ai-agents",
                    snippet=(
                        "Guidelines on state management, observability, citations, "
                        "and fallback recovery mechanisms."
                    ),
                    metadata={"category": "Best Practices"},
                ),
            ]

        return mock_docs[:max_results]
