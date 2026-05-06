# Multi-Agent Research System - Design Document

## Problem

Answering complex research questions requires multiple steps: searching diverse sources, synthesizing information, identifying claims and evidence, and producing a well-cited summary. A single LLM call lacks the ability to:

- Expand vague queries into focused sub-queries for targeted search
- Cross-reference multiple sources and identify conflicts/gaps
- Structure claims with evidence attribution
- Produce formatted, properly-cited answers with quality assurance

The task: build a system that takes a research query and produces a comprehensive, well-cited answer (500+ words, 60%+ citation coverage, structured analysis).

## Why multi-agent?

**Single-agent limitations:**
- One LLM call can hallucinate or miss sources
- No intermediate validation or structured fact-checking
- No separation of concerns (research ≠ analysis ≠ writing)
- Difficult to debug or swap components (e.g., use cheaper model for research, premium model for writing)

**Multi-agent advantages:**
- **Role clarity**: Each agent has one job, making debugging and monitoring easier
- **Resilience**: Failure in one agent (e.g., search timeout) doesn't stop the pipeline; retry logic per agent
- **Quality gates**: Each stage validates output (Critic checks citations, Writer validates structure)
- **Cost optimization**: Use cheaper LLM for research/analysis, premium for final writing
- **Auditability**: Full trace of routing, sources, and decisions for compliance

## Agent Roles

| Agent | Responsibility | Input | Output | Failure Mode |
|---|---|---|---|---|
| **Supervisor** | Deterministic routing based on state | State with current notes | Next route (researcher/analyst/writer/critic/done) | Never fails; enforces max_iterations |
| **Researcher** | Query expansion + multi-source search + note synthesis | Query, max_sources | List of sources; research_notes (200-400w with [#i] citations) | SearchClient timeout → MockSearchClient; LLM error → retry 2x then raise |
| **Analyst** | Extract key claims, identify conflicts, flag gaps | research_notes + sources | analysis_notes (JSON: claims, conflicts, gaps) | Invalid JSON → retry 1x with reminder; fallback to minimal schema |
| **Writer** | Synthesize analysis into 500-word markdown answer | analysis_notes + research_notes + sources | final_answer (markdown with headings, inline [#i], ### Sources block) | Too short (<200w) or missing Sources section → retry 1x with reminder |
| **Critic** (optional) | Check citation coverage; append quality notes | final_answer + sources | (modified final_answer with notes if coverage <60%) | None; read-only analysis, never raises |

## Shared State (`ResearchState`)

| Field | Type | Purpose | Why needed |
|---|---|---|---|
| `request` | ResearchQuery | Original query + max_sources + audience | Define the task, enforce constraints |
| `iteration` | int | Current step count | Enforce max_iterations, track progress |
| `route_history` | list[str] | Agent names in order executed | Debug workflow, detect loops, generate trace |
| `sources` | list[SourceDocument] | Deduplicated search results | Researcher outputs, Writer cites them |
| `research_notes` | str \| None | Summarized findings with citations | Analyst reads, Writer uses for synthesis |
| `analysis_notes` | str \| None | Structured claims (JSON) | Writer builds on; Critic audits structure |
| `final_answer` | str \| None | Final markdown response | Output to user |
| `agent_results` | list[AgentResult] | Metadata per agent (content preview, token count) | Benchmarking, attribution |
| `trace` | list[dict] | Event log (agent runs, routing decisions) | Export as JSON-L for debugging |
| `errors` | list[str] | Formatted "[agent] message" error log | Diagnose failures |
| `token_usage` | dict | {input_tokens, output_tokens} cumulative | Cost estimation, benchmarking |
| `estimated_cost_usd` | float | Sum of all LLM calls | Budget tracking |
| `retries` | dict | {agent_name: count} | Detect flaky agents |
| `start_time` | float \| None | perf_counter() at workflow start | Latency measurement |
| `end_time` | float \| None | perf_counter() at workflow end | Latency measurement |
| `status` | str | pending/running/done/failed | Indicate workflow completion state |

## Routing Policy

```
Supervisor logic (deterministic, no LLM):

1. iteration >= max_iterations?
   → route = "done" (set fallback final_answer if missing)
2. research_notes is None?
   → route = "researcher"
3. analysis_notes is None?
   → route = "analyst"
4. final_answer is None?
   → route = "writer"
5. "critic" not in route_history AND final_answer exists AND app_env != "fast"?
   → route = "critic"
6. Else?
   → route = "done"

Graph topology:
┌─────────┐
│Supervisor│  (entry point)
└────┬────┘
     │
     ├─→ Researcher → (back to Supervisor)
     │
     ├─→ Analyst → (back to Supervisor)
     │
     ├─→ Writer → (back to Supervisor)
     │
     ├─→ Critic → (back to Supervisor)
     │
     └─→ END (when route="done")
```

**Conditional edges**: Supervisor's output determines next node via route_history[-1].

## Guardrails

- **Max iterations**: `settings.max_iterations = 6` (default).  
  Supervisor enforces: if `iteration >= max_iterations`, route = "done" and set fallback answer.

- **Timeout**: `settings.timeout_seconds = 60` (per LLM/search call).  
  LLMClient and SearchClient apply this via urllib/OpenAI timeout param.

- **Retry strategy**: `tenacity` with exponential backoff.
  - Stop: 3 attempts max
  - Wait: exponential(multiplier=1, min=1, max=8 seconds)
  - Applies to LLMClient API calls (handles rate limits, timeouts, connection errors)
  - Per-agent retry: each agent can retry internally up to 2 times before raising

- **Fallback answer**: If max_iterations reached with no final_answer, Supervisor sets:  
  `"[Fallback after max iterations] " + (research_notes or analysis_notes or request.query)`

- **Validation gates**:
  - **Researcher**: sources list non-empty
  - **Analyst**: research_notes truthy, sources non-empty; JSON parse validation with pydantic
  - **Writer**: analysis_notes truthy, sources non-empty; word_count >= 200, "### Sources" present
  - **Critic**: citation_coverage calculation; append note if <60%

- **Error handling**: Each agent catches exceptions, logs to state.errors, bumps retry count. If retry >= 2, raises AgentExecutionError.

## Benchmark Plan

**Test queries** (in `configs/lab_default.yaml`):
1. "What are the key differences between LangGraph and Crew.ai for multi-agent systems?"
2. "Explain the concept of state management in LLM applications."
3. "What are the latest best practices for LLM observability and tracing?"

**Metrics per run**:
- `latency_seconds`: Wall-clock time
- `estimated_cost_usd`: Sum of token costs (GPT-4o-mini: input $0.15/1M, output $0.60/1M)
- `quality_score` (0–10): Heuristic = citation_coverage×6 + (1 if len≥400w) + (1 if no error) + (2 if has headings)
- `citation_coverage` (0–1): Unique [#i] indices in answer / max(1, num_sources)
- `token_usage` (in/out): Cumulative LLM token counts
- `failure`: bool, true if runner threw exception
- `route_history`: list of agents executed (e.g., ["researcher", "analyst", "writer", "done"])

**Expected outcomes**:
- **Baseline**: 0.5–2s latency, ~$0.001 cost, quality ~6–7, 0 sources cited (no search)
- **Multi-agent**: 2–8s latency, ~$0.005–$0.02 cost, quality ~8–9, citation 70%+, 3–5 sources
- **Trade-off**: Multi-agent is 2–4× slower but +1–2 quality points and higher citation coverage, better for final publishing

**Report sections**:
1. **Metrics table**: latency, cost, quality, tokens, citation%, failure flag
2. **Trace summary**: route_history per run
3. **Failure modes & fixes**: SearchClient timeout, JSON parse error, validation failure, max_iter, workflow exception
4. **When to use multi-agent**: Best for complex queries, worst for simple/time-critical queries
