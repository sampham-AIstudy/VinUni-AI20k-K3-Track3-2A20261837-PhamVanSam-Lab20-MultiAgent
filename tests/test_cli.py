"""Unit tests for typer CLI commands."""

from typer.testing import CliRunner

from multi_agent_research_lab.cli import app

runner = CliRunner()


def test_cli_baseline_command() -> None:
    result = runner.invoke(app, ["baseline", "--query", "Research GraphRAG state of the art"])
    assert result.exit_code == 0
    assert "Single-Agent Baseline" in result.stdout or "Baseline" in result.stdout


def test_cli_multi_agent_command() -> None:
    result = runner.invoke(app, ["multi-agent", "--query", "Research GraphRAG state of the art"])
    assert result.exit_code == 0
    assert "Execution Route History" in result.stdout or "researcher" in result.stdout


def test_cli_invalid_query_error() -> None:
    result = runner.invoke(app, ["baseline", "--query", "hi"])
    assert result.exit_code != 0
