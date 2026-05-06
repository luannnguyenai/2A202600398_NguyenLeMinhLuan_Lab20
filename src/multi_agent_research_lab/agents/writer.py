"""Writer agent skeleton."""

import logging
from typing import Optional

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client

logger = logging.getLogger("writer")

SYSTEM_PROMPT = """You are an expert writer. Synthesize comprehensive research into a well-structured answer.
Requirements:
- Write approximately 500 words in Markdown format
- Use at least 2 level 2 or 3 headings (## or ###)
- Include inline citations [#1], [#2], etc. based on sources provided
- End with "### Sources" section that lists "[#i] {title} — {url}" for each source
- Be clear and authoritative"""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or get_llm_client()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize research, analysis, and sources into final answer."""
        with trace_span("writer"):
            try:
                # Validate prerequisites
                if not state.analysis_notes:
                    raise ValidationError("Analysis notes required for writing")
                if not state.sources:
                    raise ValidationError("Sources required for writing")

                # Write answer
                answer = self._write_answer(state)

                # Validate output
                self._validate_answer(answer)

                state.final_answer = answer

                # Track result
                state.add_agent_result(
                    AgentResult(
                        agent=AgentName.WRITER,
                        content=answer,
                        metadata={
                            "word_count": len(answer.split()),
                            "has_sources_section": "### Sources" in answer,
                        },
                    )
                )

                logger.info(f"Writer: generated {len(answer.split())} word answer")
                return state

            except Exception as e:
                n = state.bump_retry(self.name)
                state.add_error(self.name, str(e))
                if n >= 2:
                    raise AgentExecutionError(f"WriterAgent failed: {e}") from e
                return state

    def _write_answer(self, state: ResearchState) -> str:
        """Call LLM to synthesize final answer.

        Args:
            state: ResearchState with analysis, research, and sources

        Returns:
            Markdown response with citations
        """
        # Build source reference block
        source_refs = "\n".join(
            [f"[#{i+1}] {doc.title}: {doc.snippet[:100]}... ({doc.url})"
             for i, doc in enumerate(state.sources)]
        )

        user_prompt = f"""Write a comprehensive answer to this query: "{state.request.query}"

For audience: {state.request.audience}

Based on this research:
{state.research_notes}

And this analysis:
{state.analysis_notes}

Using these sources (cite by [#1], [#2], etc.):
{source_refs}

Write ~500 words in Markdown with headings and proper citations."""

        resp = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        state.accumulate_usage(resp)

        return resp.content

    def _validate_answer(self, answer: str) -> None:
        """Validate answer meets quality standards.

        Raises:
            ValidationError if validation fails
        """
        word_count = len(answer.split())
        has_sources = "### Sources" in answer

        if word_count < 200:
            raise ValidationError(
                f"Answer too short: {word_count} words (need >=200)"
            )
        if not has_sources:
            raise ValidationError(
                "Answer missing '### Sources' section"
            )
