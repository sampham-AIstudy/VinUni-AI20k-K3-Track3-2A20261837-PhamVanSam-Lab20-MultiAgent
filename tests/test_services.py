"""Unit tests for LLMClient and SearchClient."""

from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_llm_client_offline_mock() -> None:
    client = LLMClient(api_key="")
    resp = client.complete("You are an analyst", "Compare RAG and Fine-tuning")
    assert resp.content
    assert resp.input_tokens is not None
    assert resp.output_tokens is not None
    assert resp.cost_usd is not None


def test_search_client_offline_mock() -> None:
    client = SearchClient(api_key="")
    docs = client.search("GraphRAG state of the art", max_results=3)
    assert len(docs) == 3
    assert all(doc.title for doc in docs)
    assert all(doc.snippet for doc in docs)
