"""Search client abstraction for ResearcherAgent supporting live API and offline corpus."""

import json
import logging
import ssl
import urllib.request
from pathlib import Path

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client supporting Tavily and offline benchmark corpus."""

    def __init__(
        self,
        api_key: str | None = None,
        corpus_dir: str | Path = "data/topics",
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.tavily_api_key
        self.corpus_dir = Path(corpus_dir)
        self._corpus_cache: list[dict] | None = None

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self.api_key:
            try:
                return self._search_tavily(query, max_results)
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}. Falling back to offline corpus.")

        # Try searching offline benchmark corpus first
        corpus_results = self._search_corpus(query, max_results)
        if corpus_results:
            return corpus_results

        return self._search_mock_fallback(query, max_results)

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

    def _load_corpus(self) -> list[dict]:
        """Load and cache JSON topics from data/topics directory."""
        if self._corpus_cache is not None:
            return self._corpus_cache

        topics: list[dict] = []
        if self.corpus_dir.exists() and self.corpus_dir.is_dir():
            for json_path in sorted(self.corpus_dir.glob("*.json")):
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    topics.append(data)
                except Exception as exc:
                    logger.warning("Could not parse %s: %exc", json_path, exc)

        self._corpus_cache = topics
        return topics

    def _search_corpus(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search across offline benchmark corpus topics, articles, and sources."""
        topics = self._load_corpus()
        if not topics:
            return []

        q_terms = set(query.lower().split())
        scored_docs: list[tuple[float, SourceDocument]] = []

        for topic_obj in topics:
            topic_info = topic_obj.get("topic", {})
            topic_name = topic_info.get("name", "").lower()
            topic_tags = [t.lower() for t in topic_info.get("tags", [])]
            kb = topic_obj.get("knowledge_base", {})

            # Topic match bonus
            topic_score = sum(2.0 for term in q_terms if term in topic_name)
            topic_score += sum(1.5 for term in q_terms if any(term in tag for tag in topic_tags))

            # 1. Inspect source_documents in knowledge base
            for src in kb.get("source_documents", []):
                title = src.get("title", "")
                snippet = src.get("summary", src.get("snippet", ""))
                src_id = src.get("source_id", "SRC")
                url = src.get("url", f"corpus://{src_id}")
                is_synthetic = src.get("is_synthetic", False)

                content_lower = (title + " " + snippet).lower()
                doc_score = topic_score + sum(1.0 for term in q_terms if term in content_lower)

                if doc_score > 0:
                    scored_docs.append(
                        (
                            doc_score,
                            SourceDocument(
                                title=f"[{src_id}] {title}",
                                url=url,
                                snippet=snippet,
                                metadata={
                                    "source_id": src_id,
                                    "is_synthetic": is_synthetic,
                                    "topic_id": topic_obj.get("benchmark_metadata", {}).get(
                                        "topic_id"
                                    ),
                                },
                            ),
                        )
                    )

            # 2. Inspect knowledge_articles
            for art in kb.get("knowledge_articles", []):
                title = art.get("title", "")
                content = art.get("content", "")
                art_id = art.get("article_id", "ART")
                content_lower = (title + " " + content[:400]).lower()
                art_score = topic_score + sum(1.0 for term in q_terms if term in content_lower)

                if art_score > 0:
                    scored_docs.append(
                        (
                            art_score,
                            SourceDocument(
                                title=f"[{art_id}] {title}",
                                url=f"corpus://{art_id}",
                                snippet=content[:280] + "...",
                                metadata={
                                    "article_id": art_id,
                                    "topic_id": topic_obj.get("benchmark_metadata", {}).get(
                                        "topic_id"
                                    ),
                                },
                            ),
                        )
                    )

        # Sort by relevance score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:max_results]]

    def _search_mock_fallback(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Built-in heuristic fallback if data/topics is not present."""
        return [
            SourceDocument(
                title=f"Technical Research Overview: {query[:40]}",
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
        ][:max_results]
