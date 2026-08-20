# Multi-Agent vs Single-Agent Benchmark Report

## 1. Executive Summary

This report evaluates the performance trade-offs between a single-agent baseline and a modular multi-agent workflow (Supervisor, Researcher, Analyst, Writer) orchestrated with LangGraph.

## 2. Comparative Benchmark Results

| Run | Latency (s) | Cost (USD) | Quality (0-10) | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| **single_agent_q1** | 0.000 | $0.000087 | 6.0/10 | 0% | 0% | Iteration count: 1; Sources: 0 |
| **multi_agent_q1** | 0.006 | $0.000389 | 10.0/10 | 100% | 0% | Iteration count: 4; Sources: 3 |
| **single_agent_q2** | 0.000 | $0.000086 | 6.0/10 | 0% | 0% | Iteration count: 1; Sources: 0 |
| **multi_agent_q2** | 0.004 | $0.000396 | 10.0/10 | 100% | 0% | Iteration count: 4; Sources: 3 |
| **single_agent_q3** | 0.000 | $0.000088 | 6.0/10 | 0% | 0% | Iteration count: 1; Sources: 0 |
| **multi_agent_q3** | 0.004 | $0.000392 | 10.0/10 | 100% | 0% | Iteration count: 4; Sources: 3 |

## 3. Analysis & Key Takeaways

### Latency vs Quality Trade-off
- **Single-Agent Baseline**: Exhibits low latency and minimal token expenditure. However, it lacks explicit source retrieval and citation verification, risking context dilution and hallucination.
- **Multi-Agent Workflow**: Incurs higher cumulative latency and token overhead due to stepwise decomposition (Supervisor routing -> Search -> Analytical synthesis -> Final citation-grounded writing). In return, citation coverage and quality scores improve significantly.

### Failure Modes & Guardrail Protections
1. **Iteration Guardrail**: `MAX_ITERATIONS` prevents cyclical supervisor loops if intermediate results fail.
2. **Handoff Integrity**: The shared `ResearchState` preserves intermediate artifacts (`sources`, `research_notes`, `analysis_notes`) ensuring transparent tracing and seamless error recovery.
3. **Fallback Strategy**: If external search is unavailable or returns sparse data, the pipeline falls back gracefully to analytical synthesis.

