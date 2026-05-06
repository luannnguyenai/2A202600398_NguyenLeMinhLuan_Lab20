"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        """No dependencies needed for routing logic."""
        pass

    def run(self, state: ResearchState) -> ResearchState:
        """Deterministic routing based on 6 rules.

        Rules (priority order):
        1. If iteration >= max_iterations → route "done"; set fallback answer if missing
        2. Elif research_notes is None → route "researcher"
        3. Elif analysis_notes is None → route "analyst"
        4. Elif final_answer is None → route "writer"
        5. Elif "critic" not in route_history and final_answer exists and app_env != "fast" → route "critic"
        6. Else → route "done"
        """
        settings = get_settings()

        # Rule 1: Max iterations check
        if state.iteration >= settings.max_iterations:
            route = "done"
            if state.final_answer is None:
                state.final_answer = (
                    f"[Fallback after max iterations] "
                    f"{state.research_notes or state.analysis_notes or state.request.query}"
                )
        # Rule 2: Need research
        elif state.research_notes is None:
            route = "researcher"
        # Rule 3: Need analysis
        elif state.analysis_notes is None:
            route = "analyst"
        # Rule 4: Need writing
        elif state.final_answer is None:
            route = "writer"
        # Rule 5: Critic (optional, based on env)
        elif (
            "critic" not in state.route_history
            and state.final_answer
            and settings.app_env != "fast"
        ):
            route = "critic"
        # Rule 6: Done
        else:
            route = "done"

        state.record_route(route)
        state.add_trace_event("route", {"next": route, "iteration": state.iteration})

        return state
