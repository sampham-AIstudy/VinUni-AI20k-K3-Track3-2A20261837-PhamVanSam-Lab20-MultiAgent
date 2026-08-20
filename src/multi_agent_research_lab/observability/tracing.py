"""Tracing hooks and span context for observability.

Supports LangSmith integration when configured, as well as lightweight in-memory spans.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


def setup_tracing() -> None:
    """Configure LangSmith or tracing providers if keys are available in environment."""
    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled for project: %s", settings.langsmith_project)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for tracing execution spans."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": 0.0,
        "error": None,
    }
    try:
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
