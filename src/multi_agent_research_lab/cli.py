"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import export_trace_jsonl
from multi_agent_research_lab.services.llm_client import get_llm_client
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()

BASELINE_SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's query comprehensively in about 500 words.
If you reference sources or specific claims, be clear about your reasoning.
Structure your answer with clear sections if appropriate."""


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline (one LLM call)."""
    _init()
    settings = get_settings()

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    # Get LLM and call once
    llm = get_llm_client()
    resp = llm.complete(BASELINE_SYSTEM_PROMPT, query)
    state.final_answer = resp.content
    state.accumulate_usage(resp)

    # Show metrics
    elapsed = state.elapsed_seconds() or 0.0
    table = Table(title="Baseline Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Query", query[:60])
    table.add_row("Latency (s)", f"{elapsed:.2f}")
    table.add_row("Input tokens", str(resp.input_tokens))
    table.add_row("Output tokens", str(resp.output_tokens))
    table.add_row("Estimated cost (USD)", f"${resp.cost_usd:.6f}")
    table.add_row("Answer length (words)", str(len(state.final_answer.split())))

    console.print(table)
    console.print(Panel.fit(state.final_answer, title="Answer"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    # Show summary
    table = Table(title="Multi-Agent Workflow Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Status", result.status)
    table.add_row("Iterations", str(result.iteration))
    table.add_row("Route history", " → ".join(result.route_history))
    table.add_row("Sources collected", str(len(result.sources)))
    table.add_row("Latency (s)", f"{result.elapsed_seconds() or 0:.2f}")
    table.add_row("Cost (USD)", f"${result.estimated_cost_usd:.6f}")
    table.add_row("Errors", str(len(result.errors)))

    console.print(table)

    if result.final_answer:
        console.print(Panel.fit(result.final_answer, title="Final Answer"))


@app.command()
def benchmark(
    config: Annotated[
        str, typer.Option("--config", "-c", help="Config file path")
    ] = "configs/lab_default.yaml",
    output: Annotated[str, typer.Option("--output", "-o", help="Output markdown file")] = "reports/benchmark_report.md",
) -> None:
    """Run benchmark: baseline vs multi-agent on multiple queries."""
    _init()
    import yaml

    # Load config
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        raise typer.Exit(code=1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    queries = cfg.get("benchmark", {}).get("queries", [])
    if not queries:
        console.print("[red]No queries in config[/red]")
        raise typer.Exit(code=1)

    console.print(f"Running benchmark on {len(queries)} queries...")

    # Benchmark runners
    def baseline_runner(q: str) -> ResearchState:
        request = ResearchQuery(query=q)
        state = ResearchState(request=request)
        llm = get_llm_client()
        resp = llm.complete(BASELINE_SYSTEM_PROMPT, q)
        state.final_answer = resp.content
        state.accumulate_usage(resp)
        return state

    def multi_agent_runner(q: str) -> ResearchState:
        request = ResearchQuery(query=q)
        state = ResearchState(request=request)
        workflow = MultiAgentWorkflow()
        return workflow.run(state)

    # Run benchmarks
    metrics_list = []
    trace_state = None

    for i, query in enumerate(queries):
        console.print(f"\n[cyan]Query {i+1}/{len(queries)}: {query[:60]}...[/cyan]")

        # Baseline
        console.print("  Running baseline...")
        _, metrics_b = run_benchmark("baseline", query, baseline_runner)
        metrics_list.append(metrics_b)

        # Multi-agent
        console.print("  Running multi-agent...")
        state_ma, metrics_m = run_benchmark("multi_agent", query, multi_agent_runner)
        metrics_list.append(metrics_m)

        # Save trace from first multi-agent run
        if trace_state is None:
            trace_state = state_ma

    # Render report
    report_md = render_markdown_report(metrics_list)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md)

    console.print(f"\n[green]Report saved to {output_path}[/green]")

    # Export trace
    if trace_state:
        trace_path = output_path.parent / "trace_example.json"
        export_trace_jsonl(trace_state, trace_path)
        console.print(f"[green]Trace exported to {trace_path}[/green]")

    # Show summary table
    table = Table(title="Benchmark Results")
    table.add_column("Run", style="cyan")
    table.add_column("Query", style="magenta")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Citation%", justify="right")

    for m in metrics_list:
        table.add_row(
            m.run_name[:12],
            m.notes.split()[0] if m.notes else "?",
            f"{m.latency_seconds:.2f}",
            f"${m.estimated_cost_usd or 0:.6f}",
            f"{m.quality_score or 0:.1f}",
            f"{m.citation_coverage*100:.0f}%",
        )

    console.print(table)


@app.command()
def trace(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run multi-agent and show detailed trace."""
    _init()

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    # Show route history
    console.print("[cyan]Route history:[/cyan]")
    for i, route in enumerate(result.route_history):
        console.print(f"  {i+1}. {route}")

    # Show trace events
    console.print("\n[cyan]Trace events:[/cyan]")
    for event in result.trace:
        console.print(f"  {event['name']}: ", end="")
        pprint(event["payload"], max_length=80)

    # Show errors if any
    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for err in result.errors:
            console.print(f"  - {err}")


if __name__ == "__main__":
    app()
