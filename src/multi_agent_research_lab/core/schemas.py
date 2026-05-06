"""Public schemas exchanged between CLI, agents, and evaluators."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"


class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=5)
    max_sources: int = Field(default=5, ge=1, le=20)
    audience: str = "technical learners"


class AgentResult(BaseModel):
    agent: AgentName
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    title: str
    url: Optional[str] = None
    snippet: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetrics(BaseModel):
    run_name: str
    latency_seconds: float
    estimated_cost_usd: Optional[float] = None
    quality_score: Optional[float] = Field(default=None, ge=0, le=10)
    notes: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    citation_coverage: float = 0.0
    failure: bool = False
    route_history: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
