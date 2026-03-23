"""Kelet SDK - OpenTelemetry integration for AI observability.

Standalone usage:

    import kelet

    kelet.configure()  # Creates TracerProvider, instruments pydantic-ai

    with kelet.agentic_session(session_id="session-123"):
        result = await agent.run(...)
        await kelet.signal(
            kind=kelet.SignalKind.FEEDBACK,
            source=kelet.SignalSource.HUMAN,
            score=1.0,
        )

With additional processors (e.g., logfire):

    import kelet

    kelet.configure(additional_span_processors=[logfire_processor])

Or use logfire as primary, kelet as secondary:

    import logfire
    import kelet

    logfire.configure(...)
    logfire.instrument_pydantic_ai()
    kelet.configure()  # Adds Kelet processor to logfire's provider
"""

from ._configure import configure, create_kelet_processor, shutdown
from ._context import (
    agent,
    agentic_session,
    get_agent_name,
    get_metadata_kwargs,
    get_session_id,
    get_trace_id,
    get_user_id,
)
from ._signal import signal
from .models import SignalKind, SignalSource

__all__ = [
    "configure",
    "create_kelet_processor",
    "shutdown",
    "signal",
    "agent",
    "agentic_session",
    "get_agent_name",
    "get_metadata_kwargs",
    "get_session_id",
    "get_trace_id",
    "get_user_id",
    "SignalKind",
    "SignalSource",
]
