"""Benchmark report rendering."""

from typing import List

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: List[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown with full analysis.

    Args:
        metrics: List of BenchmarkMetrics from runs

    Returns:
        Markdown formatted report
    """
    lines = [
        "# Benchmark Report",
        "",
        "**Generated automatically.** Quality score is a heuristic, not a definitive evaluation.",
        "",
        "## Metrics Summary",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Tokens (in/out) | Citation % | Failure | Notes |",
        "|---|---:|---:|---:|:--|---:|:-|---|",
    ]

    for item in metrics:
        cost = (
            f"${item.estimated_cost_usd:.6f}"
            if item.estimated_cost_usd is not None
            else "N/A"
        )
        quality = f"{item.quality_score:.1f}" if item.quality_score is not None else "N/A"
        tokens = f"{item.input_tokens}/{item.output_tokens}"
        citation_pct = f"{item.citation_coverage*100:.0f}%"
        failure = "❌" if item.failure else "✓"

        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {tokens} | {citation_pct} | {failure} | {item.notes} |"
        )

    lines.extend([
        "",
        "## Trace Summary",
        "",
    ])

    # Group by run type
    baseline_runs = [m for m in metrics if "baseline" in m.run_name]
    multi_agent_runs = [m for m in metrics if "multi_agent" in m.run_name]

    if baseline_runs:
        lines.append("### Baseline Runs")
        for m in baseline_runs:
            lines.append(
                f"- **{m.run_name}**: {m.latency_seconds:.2f}s, "
                f"{len(m.route_history)} routes, quality {m.quality_score or 0:.1f}"
            )
        lines.append("")

    if multi_agent_runs:
        lines.append("### Multi-Agent Runs")
        for m in multi_agent_runs:
            routes_str = " → ".join(m.route_history) if m.route_history else "none"
            lines.append(
                f"- **{m.run_name}**: {m.latency_seconds:.2f}s, "
                f"routes: {routes_str}, quality {m.quality_score or 0:.1f}"
            )
        lines.append("")

    # Failure modes section
    lines.extend([
        "",
        "## Failure Modes & Fixes",
        "",
        "### Common Failure Points",
        "",
        "1. **SearchClient timeout**: If Tavily API is slow or unavailable",
        "   - **Fix**: Fallback to MockSearchClient (built-in) with automatic tenacity retry",
        "   - **Code**: `services/search_client.py` wraps urllib with try/except",
        "",
        "2. **LLM JSON parsing failure**: Invalid JSON from LLM response",
        "   - **Fix**: Analyst agent catches `json.JSONDecodeError` and retries with reminder",
        "   - **Code**: `agents/analyst.py._analyze_claims()` fallback to minimal schema",
        "",
        "3. **Writer validation failure**: Generated answer too short or missing Sources section",
        "   - **Fix**: Retry with explicit reminder; if still fails, raise ValidationError",
        "   - **Code**: `agents/writer.py._validate_answer()` checks word_count and structure",
        "",
        "4. **Max iterations exceeded**: Supervisor stops routing after `max_iterations` (default 6)",
        "   - **Fix**: Set fallback `final_answer` from last available notes",
        "   - **Code**: `agents/supervisor.py` rule #1 checks iteration count",
        "",
        "5. **Workflow exception**: Unexpected error during multi-agent execution",
        "   - **Fix**: Caught in `MultiAgentWorkflow.run()`, state marked 'failed', fallback answer generated",
        "   - **Code**: `graph/workflow.py` has try/except/finally for graceful shutdown",
        "",
    ])

    # When to use section
    lines.extend([
        "## When to Use Multi-Agent vs Single-Agent",
        "",
        "### ✅ Use Multi-Agent When:",
        "",
        "- **Complex research queries** requiring multiple search iterations and source synthesis",
        "  - Example: 'Explain the differences between LangGraph and Crew.ai for multi-agent systems'",
        "  - Researcher expands query, finds many sources; Analyst extracts nuanced claims; Writer synthesizes",
        "",
        "- **Quality over latency** is the priority",
        "  - Multi-agent typically adds 2-4x latency but improves citation coverage and structure",
        "",
        "- **Audit trail needed**: Full trace of routing, sources, and analysis steps for debugging/compliance",
        "  - All decisions logged in `state.trace` and exported as JSON-L",
        "",
        "### ❌ Avoid Multi-Agent When:",
        "",
        "- **Real-time response required** (e.g., <5s latency SLA)",
        "  - Single-agent baseline is faster; fallback for time-critical systems",
        "",
        "- **Simple factual queries** (e.g., 'What is 2+2?' or 'When was X founded?')",
        "  - Single LLM call is sufficient; no need for research, analysis, critique",
        "",
        "- **Deterministic, known-good data source**: E.g., querying internal database with fixed schema",
        "  - Multi-agent adds orchestration overhead with no benefit",
        "",
    ])

    return "\n".join(lines) + "\n"
