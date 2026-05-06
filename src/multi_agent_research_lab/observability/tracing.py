"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger("trace")

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Minimal span context used by the skeleton.

    TODO(student): Replace or augment with LangSmith/Langfuse provider spans.
    """

    started = perf_counter()
    span: Dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
    }
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


def record_event(state: Any, name: str, payload: Dict[str, Any]) -> None:
    """Record a trace event in the state.

    Args:
        state: ResearchState instance
        name: Event name
        payload: Event data
    """
    state.add_trace_event(name, payload)


def export_trace_jsonl(state: Any, path: Path) -> None:
    """Export trace events to JSON-L file.

    Args:
        state: ResearchState with trace events
        path: Output file path
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # Write each trace event as a JSON line
        for event in state.trace:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # Write summary metadata as final line
        summary = {
            "name": "summary",
            "payload": {
                "run_status": state.status,
                "total_latency_seconds": state.elapsed_seconds(),
                "total_cost_usd": state.estimated_cost_usd,
                "final_iteration": state.iteration,
                "route_history": state.route_history,
            },
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    logger.info(f"Exported trace to {path}")


def traced(name: str) -> Callable[[F], F]:
    """Decorator to mark functions as traced (optional LangSmith integration).

    Args:
        name: Span name for tracing

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Try to use LangSmith if available
            try:
                from langsmith import traceable

                traced_func = traceable(name=name)(func)
                return traced_func(*args, **kwargs)
            except ImportError:
                # Fallback: just call the function normally
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
