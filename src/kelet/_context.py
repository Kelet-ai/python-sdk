"""Context helpers for session and trace management."""

import asyncio
from contextvars import ContextVar
from functools import wraps
from typing import Optional

from opentelemetry import baggage, context as otel_context, trace

# Context variables for session/user (accessible without baggage propagation)
_session_id_var: ContextVar[Optional[str]] = ContextVar(
    "kelet_session_id", default=None
)
_user_id_var: ContextVar[Optional[str]] = ContextVar("kelet_user_id", default=None)
_agent_name_var: ContextVar[Optional[str]] = ContextVar("kelet_agent_name", default=None)
_project_override_var: ContextVar[Optional[str]] = ContextVar("kelet_project", default=None)

# Semantic convention attribute keys (for span attributes)
SESSION_ID_ATTR = "gen_ai.conversation.id"
USER_ID_ATTR = "user.id"
AGENT_NAME_ATTR = "gen_ai.agent.name"


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


def get_agent_name() -> Optional[str]:
    """Get agent_name from context (set by agent()).

    Returns:
        Agent name if set in context, None otherwise.
    """
    return _agent_name_var.get()


class _AgenticSessionContext:
    def __init__(self, *, session_id: str, user_id: Optional[str] = None, project: Optional[str] = None, **kwargs: object):
        self._session_id: str = session_id
        self._user_id: Optional[str] = user_id
        self._project: Optional[str] = project
        self._kwargs: dict[str, object] = kwargs
        self._tokens: list = []
        self._baggage_token: Optional[object] = None

    def _enter(self) -> None:
        self._tokens = [
            _session_id_var.set(self._session_id),
            _user_id_var.set(self._user_id),
            _project_override_var.set(self._project),
        ]
        ctx = otel_context.get_current()
        ctx = baggage.set_baggage("kelet.session_id", self._session_id, context=ctx)
        if self._user_id:
            ctx = baggage.set_baggage("kelet.user_id", self._user_id, context=ctx)
        if self._project:
            ctx = baggage.set_baggage("kelet.project", self._project, context=ctx)
        self._baggage_token = otel_context.attach(ctx)
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute(SESSION_ID_ATTR, self._session_id)
            if self._user_id:
                span.set_attribute(USER_ID_ATTR, self._user_id)
            for k, v in self._kwargs.items():
                span.set_attribute(f"metadata.{k}", v if isinstance(v, (str, bool, int, float)) else str(v))

    def _exit(self) -> None:
        if self._baggage_token is not None:
            otel_context.detach(self._baggage_token)
        for token in reversed(self._tokens):
            token.var.reset(token)

    def __enter__(self) -> "_AgenticSessionContext":
        self._enter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._exit()

    async def __aenter__(self) -> "_AgenticSessionContext":
        self._enter()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._exit()

    def __call__(self, fn):  # type: ignore[override]
        if asyncio.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                async with _AgenticSessionContext(
                    session_id=self._session_id,
                    user_id=self._user_id,
                    project=self._project,
                    **self._kwargs,
                ):
                    return await fn(*args, **kwargs)
            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            with _AgenticSessionContext(
                session_id=self._session_id,
                user_id=self._user_id,
                project=self._project,
                **self._kwargs,
            ):
                return fn(*args, **kwargs)
        return sync_wrapper


def agentic_session(*, session_id: str, user_id: Optional[str] = None, project: Optional[str] = None, **kwargs: object) -> _AgenticSessionContext:
    """Context manager / decorator that sets agentic session context.

    Supports:
    - ``with agentic_session(...)``
    - ``async with agentic_session(...)``
    - ``@agentic_session(...)`` on sync functions
    - ``@agentic_session(...)`` on async functions

    Args:
        session_id: Conversation/session identifier (gen_ai.conversation.id)
        user_id: Optional user identifier (user.id)
        project: Optional project override (overrides global kelet.project for this session)
        **kwargs: Additional metadata stamped as metadata.{key} span attributes
    """
    return _AgenticSessionContext(session_id=session_id, user_id=user_id, project=project, **kwargs)


class _AgentContext:
    def __init__(self, *, name: str):
        self._name = name
        self._span = None
        self._agent_token = None
        self._otel_token = None

    def _start(self) -> None:
        tracer = trace.get_tracer("kelet")
        attrs: dict = {"gen_ai.operation.name": "invoke_agent", AGENT_NAME_ATTR: self._name}
        if sid := _session_id_var.get():
            attrs[SESSION_ID_ATTR] = sid
        if uid := _user_id_var.get():
            attrs[USER_ID_ATTR] = uid
        self._span = tracer.start_span(f"agent {self._name}", attributes=attrs)
        self._agent_token = _agent_name_var.set(self._name)
        ctx = trace.set_span_in_context(self._span)
        self._otel_token = otel_context.attach(ctx)

    def _stop(self, *exc: object) -> None:
        try:
            if self._otel_token is not None:
                otel_context.detach(self._otel_token)
        finally:
            try:
                if self._span is not None:
                    self._span.end()
            finally:
                if self._agent_token is not None:
                    _agent_name_var.reset(self._agent_token)

    def __enter__(self) -> "_AgentContext":
        self._start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop(*exc)

    async def __aenter__(self) -> "_AgentContext":
        self._start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop(*exc)

    def __call__(self, fn):  # type: ignore[override]
        if asyncio.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                async with _AgentContext(name=self._name):
                    return await fn(*args, **kwargs)
            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            with _AgentContext(name=self._name):
                return fn(*args, **kwargs)
        return sync_wrapper


def agent(*, name: str) -> _AgentContext:
    """Context manager / decorator that creates a gen_ai agent span.

    Creates an OTEL span with gen_ai.agent.name and invoke_agent operation.
    All LLM calls inside will be children of this span.

    Supports:
    - ``with agent(name=...)``
    - ``async with agent(name=...)``
    - ``@agent(name=...)`` on sync functions
    - ``@agent(name=...)`` on async functions
    """
    return _AgentContext(name=name)
