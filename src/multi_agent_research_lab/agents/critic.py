"""Optional critic agent skeleton for bonus work."""

import logging
import re
from typing import Set

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger("critic")


class CriticAgent(BaseAgent):
    """Optional fact-checking and citation coverage agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Check citation coverage and append findings if needed."""
        with trace_span("critic"):
            # Check prerequisites
            if not state.final_answer or not state.sources:
                return state

            # Calculate citation coverage
            citation_coverage = self._check_citation_coverage(
                state.final_answer, state.sources
            )

            # Append note if coverage is low
            if citation_coverage < 0.6:
                note = (
                    f"\n\n> **Critic note:** Citation coverage is {citation_coverage:.0%}. "
                    f"Consider adding more citations to strengthen the answer."
                )
                state.final_answer += note

            # Track result
            state.add_agent_result(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=f"Citation coverage: {citation_coverage:.0%}",
                    metadata={"citation_coverage": citation_coverage},
                )
            )

            logger.info(f"Critic: citation coverage {citation_coverage:.0%}")
            return state

    def _check_citation_coverage(self, answer: str, sources: list) -> float:
        """Calculate citation coverage ratio.

        Coverage = unique cited source indices / number of non-Sources headings

        Args:
            answer: Final answer markdown
            sources: List of source documents

        Returns:
            Coverage ratio 0.0-1.0
        """
        # Find all citations [#i]
        citations: Set[int] = set()
        for match in re.finditer(r"\[#(\d+)\]", answer):
            try:
                citations.add(int(match.group(1)))
            except ValueError:
                pass

        # Count headings (excluding Sources section)
        # Match ## or ### at start of line
        headings = re.findall(
            r"^#{2,3} (?!Sources)", answer, re.MULTILINE
        )
        n_headings = max(1, len(headings))

        # Coverage = cited sources / headings
        coverage = len(citations) / n_headings
        return min(1.0, coverage)
