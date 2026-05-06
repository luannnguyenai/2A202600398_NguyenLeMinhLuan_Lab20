"""Test benchmark evaluation module."""

import os

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report


def test_run_benchmark_baseline() -> None:
    """Test benchmark with a simple baseline runner."""
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("TAVILY_API_KEY", None)

    def simple_runner(query: str) -> ResearchState:
        request = ResearchQuery(query=query)
        state = ResearchState(request=request)
        state.final_answer = f"Mock answer to: {query}" * 50  # ~500 words
        state.sources = [
            type("Doc", (), {"title": "Source 1", "url": "http://example.com/1"})()
            for _ in range(3)
        ]
        return state

    state, metrics = run_benchmark("test_run", "Sample query", simple_runner)

    assert metrics.run_name == "test_run"
    assert metrics.latency_seconds > 0
    assert metrics.failure is False
    assert 0 <= metrics.citation_coverage <= 1.0
    assert 0 <= metrics.quality_score <= 10


def test_run_benchmark_with_failure() -> None:
    """Test benchmark gracefully handles runner failure."""

    def failing_runner(query: str) -> ResearchState:
        raise RuntimeError("Simulated failure")

    state, metrics = run_benchmark("failing_test", "Query", failing_runner)

    assert metrics.failure is True
    assert metrics.error_message is not None
    assert "Simulated failure" in metrics.error_message


def test_render_markdown_report() -> None:
    """Test report rendering with metric data."""
    from multi_agent_research_lab.core.schemas import BenchmarkMetrics

    metrics = [
        BenchmarkMetrics(
            run_name="baseline",
            latency_seconds=1.0,
            estimated_cost_usd=0.001,
            quality_score=7.0,
            notes="test run",
            input_tokens=100,
            output_tokens=200,
            citation_coverage=0.8,
            failure=False,
            route_history=["done"],
        ),
        BenchmarkMetrics(
            run_name="multi_agent",
            latency_seconds=3.5,
            estimated_cost_usd=0.005,
            quality_score=8.5,
            notes="test run",
            input_tokens=300,
            output_tokens=600,
            citation_coverage=0.9,
            failure=False,
            route_history=["researcher", "analyst", "writer", "done"],
        ),
    ]

    report = render_markdown_report(metrics)

    # Check structure
    assert "# Benchmark Report" in report
    assert "## Metrics Summary" in report
    assert "## Failure Modes & Fixes" in report
    assert "## When to Use Multi-Agent" in report

    # Check content
    assert "baseline" in report
    assert "multi_agent" in report
    assert "7.0" in report  # quality score
    assert "8.5" in report
    assert "citation" in report.lower()
