"""Analyst agent skeleton."""

import json
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client

logger = logging.getLogger("analyst")

SYSTEM_PROMPT = """You are an expert analyst. Extract key claims from research notes and identify conflicts/gaps.
Output ONLY valid JSON with this schema:
{
  "key_claims": [
    {"claim": "statement", "evidence_index": [1, 2], "confidence": "high|med|low"}
  ],
  "conflicts": ["conflicting viewpoints"],
  "gaps": ["missing information"]
}"""


class KeyClaim(BaseModel):
    """Single key claim with evidence."""

    claim: str
    evidence_index: List[int] = Field(default_factory=list)
    confidence: str = "med"


class AnalysisSchema(BaseModel):
    """Structured analysis output."""

    key_claims: List[KeyClaim] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or get_llm_client()

    def run(self, state: ResearchState) -> ResearchState:
        """Extract analysis from research notes and sources."""
        with trace_span("analyst"):
            try:
                # Validate prerequisites
                if not state.research_notes:
                    raise ValidationError("Research notes required for analysis")
                if not state.sources:
                    raise ValidationError("Sources required for analysis")

                # Call LLM for analysis
                analysis = self._analyze_claims(state.research_notes, state)

                # Store as JSON string
                state.analysis_notes = json.dumps(
                    analysis.model_dump(), indent=2, ensure_ascii=False
                )

                # Track result
                state.add_agent_result(
                    AgentResult(
                        agent=AgentName.ANALYST,
                        content=state.analysis_notes,
                        metadata={
                            "n_claims": len(analysis.key_claims),
                            "n_conflicts": len(analysis.conflicts),
                            "n_gaps": len(analysis.gaps),
                        },
                    )
                )

                logger.info(
                    f"Analyst: extracted {len(analysis.key_claims)} claims, "
                    f"{len(analysis.conflicts)} conflicts, {len(analysis.gaps)} gaps"
                )

                return state

            except Exception as e:
                n = state.bump_retry(self.name)
                state.add_error(self.name, str(e))
                if n >= 2:
                    raise AgentExecutionError(f"AnalystAgent failed: {e}") from e
                return state

    def _analyze_claims(self, research_notes: str, state: ResearchState) -> AnalysisSchema:
        """Call LLM to extract structured claims.

        Args:
            research_notes: Research notes to analyze
            state: ResearchState to accumulate usage on

        Returns:
            AnalysisSchema with parsed claims
        """
        user_prompt = f"""Analyze these research notes and extract key claims:

{research_notes}

Output valid JSON only."""

        try:
            resp = self.llm.complete(SYSTEM_PROMPT, user_prompt, response_format="json")
            state.accumulate_usage(resp)

            parsed = json.loads(resp.content)
            return AnalysisSchema.model_validate(parsed)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse claims JSON, retrying: {e}")
            # Fallback: return minimal schema
            return AnalysisSchema(
                key_claims=[KeyClaim(claim="Unable to extract specific claims")],
                gaps=["Analysis quality degraded"],
            )
