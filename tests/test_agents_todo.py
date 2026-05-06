import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_no_notes() -> None:
    """SupervisorAgent should route to researcher when no research notes exist."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    agent = SupervisorAgent()
    result = agent.run(state)

    assert result.route_history[-1] == "researcher"
    assert result.iteration == 1


def test_supervisor_routes_to_done_when_max_iter() -> None:
    """SupervisorAgent should route to done when max_iterations reached."""
    settings = get_settings()
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.iteration = settings.max_iterations

    agent = SupervisorAgent()
    result = agent.run(state)

    assert result.route_history[-1] == "done"
    assert result.final_answer is not None
