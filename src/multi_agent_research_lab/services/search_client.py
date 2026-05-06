"""Search client abstraction for ResearcherAgent."""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import List, Optional

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger("search")


class SearchClient:
    """Provider-agnostic search client interface."""

    def search(self, query: str, max_results: int = 5) -> List[SourceDocument]:
        """Search for documents relevant to a query."""
        raise NotImplementedError


class TavilySearchClient(SearchClient):
    """Real Tavily-based search client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        if not self.api_key:
            raise AgentExecutionError(
                "TAVILY_API_KEY not set. Set env var or pass api_key param."
            )
        self.base_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 5) -> List[SourceDocument]:
        """Search Tavily API via urllib.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of SourceDocument with title, url, snippet
        """
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }

        try:
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = []
            for item in data.get("results", []):
                doc = SourceDocument(
                    title=item.get("title", ""),
                    url=item.get("url"),
                    snippet=item.get("content", "")[:500],
                    metadata={
                        "score": item.get("score"),
                        "published_date": item.get("published_date"),
                    },
                )
                results.append(doc)

            logger.info(f"Tavily search: query={query}, results={len(results)}")
            return results

        except (urllib.error.URLError, json.JSONDecodeError) as e:
            raise AgentExecutionError(f"Search failed: {e}") from e


class MockSearchClient(SearchClient):
    """Deterministic mock search client for offline testing."""

    def search(self, query: str, max_results: int = 5) -> List[SourceDocument]:
        """Return deterministic mock search results.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of 3-max_results deterministic SourceDocument
        """
        num_results = min(max_results, max(3, max_results))

        results = []
        for i in range(1, num_results + 1):
            doc = SourceDocument(
                title=f"{query} — source #{i}",
                url=f"https://mock.local/{query.replace(' ', '_')}/{i}",
                snippet=f"Deterministic mock snippet about {query} #{i}. "
                f"This snippet contains relevant information on the topic.",
                metadata={"mock": True, "source_index": i},
            )
            results.append(doc)

        return results


def get_search_client(api_key: Optional[str] = None) -> SearchClient:
    """Factory to get real or mock search client based on API key availability."""
    if api_key or os.getenv("TAVILY_API_KEY"):
        return TavilySearchClient(api_key=api_key)
    return MockSearchClient()
