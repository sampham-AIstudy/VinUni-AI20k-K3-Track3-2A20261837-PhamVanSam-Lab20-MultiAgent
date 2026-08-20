"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing, trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing()


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_baseline_query(query_text: str) -> ResearchState:
    """Single-agent baseline runner: single LLM call without search or multi-step coordination."""
    request = ResearchQuery(query=query_text)
    state = ResearchState(request=request)
    llm = LLMClient()
    system_prompt = (
        "You are a helpful research assistant. Answer the user question comprehensively, "
        "providing a structured technical overview."
    )
    user_prompt = f"Please research and answer: {query_text}"

    with trace_span("baseline.complete"):
        resp = llm.complete(system_prompt, user_prompt)
        state.final_answer = resp.content
        state.iteration = 1
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
        )
        state.add_trace_event("baseline.done", {"cost_usd": resp.cost_usd})

    return state


def run_multi_agent_query(query_text: str) -> ResearchState:
    """Multi-agent workflow runner: Supervisor -> Researcher -> Analyst -> Writer."""
    request = ResearchQuery(query=query_text)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    with trace_span("multi_agent.workflow"):
        return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run single-agent baseline."""
    _init()
    started = perf_counter()
    state = run_baseline_query(query)
    duration = perf_counter() - started

    console.print(Panel(state.final_answer or "", title="Single-Agent Baseline", style="cyan"))
    console.print(f"[dim]Latency: {duration:.3f}s | Iterations: {state.iteration}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    started = perf_counter()
    state = run_multi_agent_query(query)
    duration = perf_counter() - started

    # Display Route History
    routes_display = " ➔ ".join(state.route_history)
    console.print(
        Panel(f"[bold green]{routes_display}[/bold green]", title="Routes", style="green")
    )

    # Display Sources
    if state.sources:
        sources_text = "\n".join(
            f"• [{i+1}] {s.title} ({s.url})" for i, s in enumerate(state.sources)
        )
        console.print(Panel(sources_text, title=f"Sources ({len(state.sources)})", style="blue"))

    # Display Final Answer
    console.print(
        Panel(
            state.final_answer or "No answer produced.",
            title="Multi-Agent Synthesis",
            style="bold white",
        )
    )
    console.print(f"[dim]Completed in {duration:.3f}s across {state.iteration} iterations.[/dim]")


@app.command()
def benchmark(
    queries: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Optional queries to benchmark"),
    ] = None,
    output_file: Annotated[
        str,
        typer.Option("--output", "-o", help="Report output file path"),
    ] = "reports/benchmark_report.md",
) -> None:
    """Run comparative benchmark between single-agent and multi-agent systems."""
    _init()
    benchmark_queries = queries or [
        "Research GraphRAG state-of-the-art architectures and trade-offs",
        "Compare RAG vs Fine-tuning for enterprise domain adaptation",
        "Explain multi-agent supervisor coordination patterns and guardrails",
    ]

    all_metrics: list[BenchmarkMetrics] = []

    console.print("[bold blue]Starting Benchmark Evaluation...[/bold blue]")

    for idx, q in enumerate(benchmark_queries, 1):
        console.print(f"\n[bold]Evaluating Query {idx}/{len(benchmark_queries)}:[/bold] {q}")

        # 1. Run Baseline
        _, m_baseline = run_benchmark(f"single_agent_q{idx}", q, run_baseline_query)
        all_metrics.append(m_baseline)

        # 2. Run Multi-Agent
        _, m_multi = run_benchmark(f"multi_agent_q{idx}", q, run_multi_agent_query)
        all_metrics.append(m_multi)

    # Render report
    report_content = render_markdown_report(all_metrics)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_content, encoding="utf-8")

    console.print(f"\n[bold green]✓ Benchmark complete![/bold green] Saved: [u]{out_path}[/u]")

    # Print summary table
    table = Table(title="Benchmark Comparison Summary")
    table.add_column("Run", style="cyan")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Citation Cov.", justify="right")

    for m in all_metrics:
        cost = f"${m.estimated_cost_usd:.6f}" if m.estimated_cost_usd is not None else "N/A"
        quality = f"{m.quality_score:.1f}/10" if m.quality_score is not None else "N/A"
        citation = f"{m.citation_coverage:.0%}" if m.citation_coverage is not None else "N/A"
        table.add_row(m.run_name, f"{m.latency_seconds:.3f}", cost, quality, citation)

    console.print(table)


if __name__ == "__main__":
    app()
