"""Test multi-agent workflow offline (with Mock clients)."""

import os

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_offline_complete() -> None:
    """Test that workflow runs to completion with Mock clients (no API keys)."""
    # Ensure no API keys are set
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("TAVILY_API_KEY", None)

    # Create request
    request = ResearchQuery(query="What is LangGraph?")
    state = ResearchState(request=request)

    # Run workflow
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    # Verify completion
    assert result.status == "done"
    assert result.final_answer is not None
    assert len(result.final_answer) > 50  # Non-trivial answer
    assert "researcher" in result.route_history
    assert "writer" in result.route_history
    assert len(result.sources) > 0
    assert result.research_notes is not None


def test_workflow_offline_tracing() -> None:
    """Test that workflow generates proper trace events."""
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("TAVILY_API_KEY", None)

    request = ResearchQuery(query="Multi-agent systems")
    state = ResearchState(request=request)

    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    # Check trace
    assert len(result.trace) > 0
    trace_names = [e["name"] for e in result.trace]
    assert "route" in trace_names  # Supervisor routes recorded


def test_workflow_offline_accumulates_usage() -> None:
    """Test that token usage is accumulated during workflow."""
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("TAVILY_API_KEY", None)

    request = ResearchQuery(query="AI research")
    state = ResearchState(request=request)

    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    # Check token accounting (Mock LLMClient estimates tokens)
    assert result.token_usage["input_tokens"] > 0
    assert result.token_usage["output_tokens"] > 0
    assert result.estimated_cost_usd >= 0.0
