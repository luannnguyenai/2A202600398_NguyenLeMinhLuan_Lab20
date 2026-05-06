"""LangGraph workflow skeleton."""

import logging
from time import perf_counter
from typing import Any, Optional

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger("workflow")

# Try to import LangGraph
try:
    from langgraph.graph import StateGraph, END

    USE_LANGGRAPH = True
except ImportError:
    USE_LANGGRAPH = False
    logger.info("LangGraph not available, using manual workflow loop")


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        """Initialize agents once."""
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

        self._agents = {
            "researcher": self.researcher,
            "analyst": self.analyst,
            "writer": self.writer,
            "critic": self.critic,
        }

        self.graph: Optional[Any] = None

    def build(self) -> Optional[Any]:
        """Create a LangGraph graph if available.

        Returns:
            Compiled graph or None if LangGraph unavailable
        """
        if not USE_LANGGRAPH:
            return None

        from langgraph.graph import StateGraph, END

        graph = StateGraph(ResearchState)

        # Add nodes
        graph.add_node("supervisor", lambda state: self.supervisor.run(state))
        graph.add_node("researcher", lambda state: self.researcher.run(state))
        graph.add_node("analyst", lambda state: self.analyst.run(state))
        graph.add_node("writer", lambda state: self.writer.run(state))
        graph.add_node("critic", lambda state: self.critic.run(state))

        # Set entry point
        graph.set_entry_point("supervisor")

        # Conditional routing from supervisor to workers or end
        def route_after_supervisor(state: ResearchState) -> str:
            last_route = state.route_history[-1] if state.route_history else "done"
            if last_route == "done":
                return END
            return last_route

        graph.add_conditional_edges("supervisor", route_after_supervisor)

        # Workers route back to supervisor
        for worker in ["researcher", "analyst", "writer", "critic"]:
            graph.add_edge(worker, "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return final state.

        Args:
            state: Initial ResearchState with request

        Returns:
            Final ResearchState with all fields populated
        """
        settings = get_settings()
        state.start_time = perf_counter()
        state.status = "running"

        try:
            if USE_LANGGRAPH and self.graph is None:
                self.graph = self.build()

            if USE_LANGGRAPH and self.graph:
                # Use LangGraph
                logger.info("Running workflow with LangGraph")
                config = {"recursion_limit": settings.max_iterations * 3}
                result = self.graph.invoke(state, config=config)
                state = (
                    ResearchState.model_validate(result)
                    if isinstance(result, dict)
                    else result
                )
            else:
                # Fallback: manual loop
                logger.info("Running workflow with manual loop")
                max_steps = settings.max_iterations * 3

                for step in range(max_steps):
                    # Get next route from supervisor
                    state = self.supervisor.run(state)

                    # Check if done
                    if state.route_history[-1] == "done":
                        break

                    # Run the worker agent
                    agent_name = state.route_history[-1]
                    if agent_name in self._agents:
                        agent = self._agents[agent_name]
                        state = agent.run(state)
                    else:
                        logger.warning(f"Unknown agent: {agent_name}")
                        break

            state.status = "done"

        except Exception as e:
            logger.exception(f"Workflow error: {e}")
            state.errors.append(f"workflow: {e}")
            state.status = "failed"

            # Set fallback answer if missing
            if not state.final_answer:
                state.final_answer = (
                    f"[Fallback after workflow error] {e}\n\n"
                    f"Partial results:\n"
                    f"- Research: {state.research_notes[:100] if state.research_notes else 'N/A'}\n"
                    f"- Analysis: {state.analysis_notes[:100] if state.analysis_notes else 'N/A'}"
                )

        finally:
            state.end_time = perf_counter()

        return state
