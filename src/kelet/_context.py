"""Context helpers for session and trace management."""

from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import AsyncGenerator, Generator, Optional

from opentelemetry import trace

# Context variables for session/user (accessible without baggage propagation)
_session_id_var: ContextVar[Optional[str]] = ContextVar(
    "kelet_session_id", default=None
)
_user_id_var: ContextVar[Optional[str]] = ContextVar("kelet_user_id", default=None)

# Semantic convention attribute keys (for span attributes)
SESSION_ID_ATTR = "gen_ai.conversation.id"
USER_ID_ATTR = "user.id"


def get_session_id() -> Optional[str]:
    """Get session_id from context (set by agentic_session).

    Returns:
        Session ID if set in context, None otherwise.
    """
    return _session_id_var.get()


def get_trace_id() -> Optional[str]:
    """Get trace_id from current span context.

    Returns:
        Trace ID as hex string if in an active span, None otherwise.
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_user_id() -> Optional[str]:
    """Get user_id from context (set by agentic_session).

    Returns:
        User ID if set in context, None otherwise.
    """
    return _user_id_var.get()


@contextmanager
def agentic_session(
    *, session_id: str, user_id: Optional[str] = None
) -> Generator[None, None, None]:
    """Context manager to define an agentic session.

    All spans and signals within this context will automatically use
    the provided session_id and user_id. Session/user IDs are also added
    as attributes to the current span if one exists.

    Args:
        session_id: Conversation/session identifier (gen_ai.conversation.id)
        user_id: Optional user identifier (user.id)

    Example:
        with kelet.agentic_session(session_id="session-123", user_id="user-456"):
            result = await agent.run(...)
            await kelet.signal(source=..., vote=...)  # auto-uses session_id
    """
    session_token = _session_id_var.set(session_id)
    user_token = _user_id_var.set(user_id) if user_id else None

    # Add as span attributes for trace visibility
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(SESSION_ID_ATTR, session_id)
        if user_id:
            span.set_attribute(USER_ID_ATTR, user_id)

    try:
        yield
    finally:
        _session_id_var.reset(session_token)
        if user_token is not None:
            _user_id_var.reset(user_token)


@asynccontextmanager
async def agentic_session_async(
    *, session_id: str, user_id: Optional[str] = None
) -> AsyncGenerator[None, None]:
    """Async context manager to define an agentic session.

    Identical to agentic_session() but for async with syntax.

    Args:
        session_id: Conversation/session identifier (gen_ai.conversation.id)
        user_id: Optional user identifier (user.id)

    Example:
        async with kelet.agentic_session_async(session_id="session-123", user_id="user-456"):
            result = await agent.run(...)
            await kelet.signal(source=..., vote=...)
    """
    with agentic_session(session_id=session_id, user_id=user_id):
        yield
