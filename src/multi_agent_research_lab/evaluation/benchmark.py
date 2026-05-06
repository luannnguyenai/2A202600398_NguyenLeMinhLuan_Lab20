"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from time import perf_counter
from typing import Callable, Set

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, quality, cost, and other metrics for a runner.

    Args:
        run_name: Name of the run (e.g., "baseline" or "multi_agent")
        query: Research query
        runner: Callable that takes query and returns ResearchState

    Returns:
        Tuple of (state, metrics)
    """
    started = perf_counter()
    failure = False
    error_message = None

    try:
        state = runner(query)
    except Exception as e:
        state = ResearchState.__new__(ResearchState)
        state.request = type('Q', (), {'query': query})()  # type: ignore
        state.errors = [str(e)]
        state.final_answer = None
        state.sources = []
        state.route_history = []
        state.token_usage = {"input_tokens": 0, "output_tokens": 0}
        state.estimated_cost_usd = 0.0
        failure = True
        error_message = str(e)[:500]

    latency = perf_counter() - started

    # Calculate citation coverage
    answer = state.final_answer or ""
    n_sources = len(state.sources)

    citations: Set[int] = set()
    for match in re.finditer(r"\[#(\d+)\]", answer):
        try:
            citations.add(int(match.group(1)))
        except ValueError:
            pass

    citation_coverage = len(citations) / max(1, n_sources) if n_sources > 0 else 0.0

    # Quality score heuristic
    word_count = len(answer.split())
    has_headings = bool(re.search(r"^#{2,3} ", answer, re.MULTILINE))

    quality_score = max(
        0,
        min(
            10,
            citation_coverage * 6
            + (1 if word_count >= 400 else 0)
            + (1 if not failure else 0)
            + (2 if has_headings else 0),
        ),
    )

    # Build metrics
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.estimated_cost_usd,
        quality_score=quality_score,
        notes=f"len={word_count} sources={n_sources}",
        input_tokens=state.token_usage.get("input_tokens", 0),
        output_tokens=state.token_usage.get("output_tokens", 0),
        citation_coverage=citation_coverage,
        failure=failure,
        route_history=state.route_history,
        error_message=error_message,
    )

    return state, metrics
