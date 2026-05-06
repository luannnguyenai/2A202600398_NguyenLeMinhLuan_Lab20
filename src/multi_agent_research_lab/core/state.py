"""Shared state for the multi-agent workflow.

Students should extend this file when adding new agents, outputs, or evaluation metrics.
"""

from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import AgentResult, ResearchQuery, SourceDocument


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None

    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # Extended tracking fields
    token_usage: dict[str, int] = Field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}
    )
    estimated_cost_usd: float = 0.0
    retries: dict[str, int] = Field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None
    status: Literal["pending", "running", "done", "failed"] = "pending"

    def record_route(self, route: str) -> None:
        self.route_history.append(route)
        self.iteration += 1

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def add_agent_result(self, result: AgentResult) -> None:
        """Append an agent result to the list."""
        self.agent_results.append(result)

    def add_error(self, agent: str, message: str) -> None:
        """Log an error with agent name prefix."""
        self.errors.append(f"[{agent}] {message}")

    def accumulate_usage(self, llm_response: Any) -> None:
        """Accumulate token usage and cost from LLM response."""
        self.token_usage["input_tokens"] += llm_response.input_tokens or 0
        self.token_usage["output_tokens"] += llm_response.output_tokens or 0
        self.estimated_cost_usd += llm_response.cost_usd or 0.0

    def bump_retry(self, agent: str) -> int:
        """Increment retry count for agent, return current count."""
        self.retries[agent] = self.retries.get(agent, 0) + 1
        return self.retries[agent]

    def elapsed_seconds(self) -> float | None:
        """Return elapsed time in seconds, or None if not started."""
        if self.start_time is None:
            return None
        end = self.end_time or perf_counter()
        return end - self.start_time
