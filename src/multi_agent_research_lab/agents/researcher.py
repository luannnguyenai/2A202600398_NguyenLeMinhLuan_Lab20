"""Researcher agent skeleton."""

import json
import logging
from typing import List, Optional

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import (
    LLMClient,
    get_llm_client,
)
from multi_agent_research_lab.services.search_client import SearchClient, get_search_client

logger = logging.getLogger("researcher")

SYSTEM_PROMPT = """You are a research assistant. Your job is to generate 1-3 sub-queries that will help answer the main query comprehensively.
Output ONLY valid JSON in this format: {"queries": ["sub-query 1", "sub-query 2", "sub-query 3"]}
Keep sub-queries concise and focused."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        search: Optional[SearchClient] = None,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.search = search or get_search_client()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.sources and state.research_notes through search + summarization."""
        with trace_span("researcher"):
            try:
                # Step 1: Query expansion
                sub_queries = self._expand_queries(state.request.query, state)

                # Step 2: Search
                sources_set = {}  # url → SourceDocument, for deduplication
                for query in sub_queries:
                    try:
                        results = self.search.search(
                            query, max_results=state.request.max_sources
                        )
                        for doc in results:
                            if doc.url and doc.url not in sources_set:
                                sources_set[doc.url] = doc
                    except Exception as e:
                        state.add_error(self.name, f"Search failed for '{query}': {e}")

                sources = list(sources_set.values())[: state.request.max_sources]
                if not sources:
                    raise ValidationError(
                        f"No sources found for query: {state.request.query}"
                    )

                state.sources = sources

                # Step 3: Summarize with citations
                notes = self._summarize_sources(state.request.query, sources, state)
                state.research_notes = notes

                # Track usage and result
                state.add_agent_result(
                    AgentResult(
                        agent=AgentName.RESEARCHER,
                        content=notes,
                        metadata={
                            "sub_queries": sub_queries,
                            "n_sources": len(sources),
                        },
                    )
                )

                logger.info(
                    f"Researcher: expanded {len(sub_queries)} queries, "
                    f"found {len(sources)} sources"
                )

                return state

            except Exception as e:
                n = state.bump_retry(self.name)
                state.add_error(self.name, str(e))
                if n >= 2:
                    raise AgentExecutionError(f"ResearcherAgent failed: {e}") from e
                return state

    def _expand_queries(self, query: str, state: ResearchState) -> List[str]:
        """Call LLM to expand main query into sub-queries.

        Args:
            query: Original query to expand
            state: ResearchState to accumulate usage on

        Returns:
            List of 1-3 sub-queries, or fallback to original query.
        """
        user_prompt = f'Generate sub-queries for: "{query}"'

        try:
            resp = self.llm.complete(SYSTEM_PROMPT, user_prompt, response_format="json")
            state.accumulate_usage(resp)

            try:
                parsed = json.loads(resp.content)
                queries = parsed.get("queries", [query])
                return queries if queries else [query]
            except json.JSONDecodeError:
                return [query]
        except Exception:
            return [query]

    def _summarize_sources(self, query: str, sources: list, state: ResearchState) -> str:
        """Summarize sources with inline citations [#i].

        Args:
            query: Original research query
            sources: List of SourceDocument
            state: ResearchState to accumulate usage on

        Returns:
            Markdown summary with citations
        """
        # Build source references
        source_refs = "\n".join(
            [f"[#{i+1}] {doc.title} ({doc.url}): {doc.snippet}"
             for i, doc in enumerate(sources)]
        )

        user_prompt = f"""Based on these sources, write a 200-400 word summary of "{query}".
Use inline citations like [#1], [#2], etc. to reference sources.

Sources:
{source_refs}

Write concisely and include key findings."""

        system = """You are a research summarizer. Write concise summaries with proper citations.
Use [#1], [#2], etc. to cite sources inline. Write 200-400 words."""

        resp = self.llm.complete(system, user_prompt)
        state.accumulate_usage(resp)

        return resp.content
