"""Context helpers for session and trace management."""

import asyncio
import logging
import sys
from contextvars import ContextVar
from functools import wraps
from typing import Optional

from opentelemetry import baggage, context as otel_context, trace

_logger = logging.getLogger(__name__)

# Hard timeout for _drain_background_logging_tasks. Bounded so a misbehaving
# integration can never hang user code at session-exit.
_DRAIN_TIMEOUT_SECONDS = 5.0
# Polling granularity while draining.
_DRAIN_POLL_INTERVAL = 0.01
# Consecutive idle polls required before declaring the task graph quiet.
# 5 × 10ms = 50ms — enough for LiteLLM's 4-hop callback chain
# (helper → worker init → worker loop → callback) to resolve without
# making fast tests slow.
_DRAIN_QUIET_ITERATIONS = 5
# LiteLLM's long-running worker-loop task awaits queue.get() forever.
# We exempt it from "pending work" detection or the drain would never idle.
# This is the ONLY private LiteLLM attribute we peek at — everything else
# we need is visible via stdlib asyncio APIs.
_LITELLM_WORKER_MODULE = "litellm.litellm_core_utils.logging_worker"


def _litellm_worker_loop_task() -> Optional[asyncio.Task]:
    """Return LiteLLM's long-lived queue-polling task, or None if not active.

    Coupled to LiteLLM internals by design: the task exists for the lifetime
    of the process and awaits ``queue.get()`` indefinitely, so
    ``asyncio.all_tasks()`` alone cannot distinguish it from a real
    in-flight callback. Upstream issue filed to expose a public drain API;
    remove this probe when that lands.
    """
    mod = sys.modules.get(_LITELLM_WORKER_MODULE)
    if mod is None:
        return None
    worker = getattr(mod, "GLOBAL_LOGGING_WORKER", None)
    if worker is None:
        _logger.debug(
            "litellm logging_worker module loaded but GLOBAL_LOGGING_WORKER missing; "
            "skipping worker-task exemption (drain may be unreliable)"
        )
        return None
    task = getattr(worker, "_worker_task", None)
    if task is None:
        _logger.debug(
            "litellm GLOBAL_LOGGING_WORKER found but _worker_task missing; "
            "skipping worker-task exemption"
        )
        return None
    return task


async def _drain_background_logging_tasks(
    baseline: Optional[set[asyncio.Task]] = None,
) -> None:
    """Drain background SDK-instrumentation callback tasks before aexit returns.

    Some LLM SDK integrations (notably LiteLLM) dispatch OTEL callbacks via
    ``asyncio.create_task(...)`` — fire-and-forget. The spans representing
    the actual request (``litellm_request``, ``raw_gen_ai_request``) are
    produced inside those callbacks.

    For short scenarios (single completion, extended-thinking), the callback
    can still be pending when the user's async entry point returns. At that
    point ``asyncio.run()`` cancels all pending tasks during loop teardown
    and the spans are never created — so the BatchSpanProcessor has nothing
    to flush on shutdown.

    We call this from ``_AgenticSessionContext.__aexit__`` BEFORE ``_exit``
    runs, while we still have a live event loop. The drain waits until the
    task graph is idle for a short quiet period, or a 5s hard timeout —
    whichever comes first. No hang.

    Strategy: only wait for tasks that appeared on the loop AFTER the
    session's ``__aenter__`` (the ``baseline`` snapshot). Unrelated
    user-owned background tasks — websockets, schedulers, long-lived
    workers — existed before the session and stay exempt, so they don't
    extend session exit. On top of that we always exempt our own task and
    the LiteLLM long-lived worker-loop task (which may have been spawned
    lazily inside the session on first use).

    ``baseline=None`` falls back to treating every task as in-scope; used
    by the standalone tests that exercise the drain without a session.
    """
    current_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _DRAIN_TIMEOUT_SECONDS
    quiet = 0
    baseline = baseline if baseline is not None else set()

    while loop.time() < deadline:
        await asyncio.sleep(_DRAIN_POLL_INTERVAL)

        exempt = {current_task, _litellm_worker_loop_task()} | baseline
        other_pending = any(
            t not in exempt and not t.done() for t in asyncio.all_tasks(loop=loop)
        )

        if other_pending:
            quiet = 0
            continue

        quiet += 1
        if quiet >= _DRAIN_QUIET_ITERATIONS:
            return

    _logger.debug(
        "drain_background_logging_tasks hit %.1fs timeout with tasks still pending; "
        "some spans may be lost",
        _DRAIN_TIMEOUT_SECONDS,
    )


# Context variables for session/user (accessible without baggage propagation)
_session_id_var: ContextVar[Optional[str]] = ContextVar(
    "kelet_session_id", default=None
)
_user_id_var: ContextVar[Optional[str]] = ContextVar("kelet_user_id", default=None)
_agent_name_var: ContextVar[Optional[str]] = ContextVar(
    "kelet_agent_name", default=None
)
_project_override_var: ContextVar[Optional[str]] = ContextVar(
    "kelet_project", default=None
)
_metadata_kwargs_var: ContextVar[Optional[dict[str, object]]] = ContextVar(
    "kelet_metadata", default=None
)

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


def get_metadata_kwargs() -> dict[str, object]:
    """Get metadata kwargs from context (set by agentic_session).

    Returns:
        Metadata kwargs dict if set in context, empty dict otherwise.
    """
    return _metadata_kwargs_var.get() or {}


class _AgenticSessionContext:
    def __init__(
        self,
        *,
        session_id: str,
        user_id: Optional[str] = None,
        project: Optional[str] = None,
        **kwargs: object,
    ):
        self._session_id: str = session_id
        self._user_id: Optional[str] = user_id
        self._project: Optional[str] = project
        self._kwargs: dict[str, object] = kwargs
        self._tokens: list = []
        self._baggage_token: Optional[object] = None
        self._aenter_tasks: Optional[set[asyncio.Task]] = None

    def _enter(self) -> None:
        # Nesting: inherit outer values unless explicitly provided
        outer_user_id = _user_id_var.get()
        outer_project = _project_override_var.get()
        outer_kwargs = _metadata_kwargs_var.get() or {}

        effective_user_id = (
            self._user_id if self._user_id is not None else outer_user_id
        )
        effective_project = (
            self._project if self._project is not None else outer_project
        )
        merged_kwargs = (
            {**outer_kwargs, **self._kwargs} if outer_kwargs or self._kwargs else None
        )

        self._tokens = [
            _session_id_var.set(self._session_id),
            _user_id_var.set(effective_user_id),
            _project_override_var.set(effective_project),
            _metadata_kwargs_var.set(merged_kwargs),
        ]
        ctx = otel_context.get_current()
        ctx = baggage.set_baggage("kelet.session_id", self._session_id, context=ctx)
        if effective_user_id is not None:
            ctx = baggage.set_baggage("kelet.user_id", effective_user_id, context=ctx)
        else:
            ctx = baggage.remove_baggage("kelet.user_id", context=ctx)
        if effective_project is not None:
            ctx = baggage.set_baggage("kelet.project", effective_project, context=ctx)
        else:
            ctx = baggage.remove_baggage("kelet.project", context=ctx)
        # Per-key baggage for metadata kwargs
        if merged_kwargs:
            for k, v in merged_kwargs.items():
                ctx = baggage.set_baggage(
                    f"kelet.metadata.{k}",
                    str(v) if not isinstance(v, str) else v,
                    context=ctx,
                )
        self._baggage_token = otel_context.attach(ctx)
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute(SESSION_ID_ATTR, self._session_id)
            if effective_user_id is not None:
                span.set_attribute(USER_ID_ATTR, effective_user_id)
            if merged_kwargs:
                for k, v in merged_kwargs.items():
                    span.set_attribute(
                        f"metadata.{k}",
                        v if isinstance(v, (str, bool, int, float)) else str(v),
                    )

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
        # Snapshot live tasks so the aexit-drain only waits for tasks
        # that appeared DURING the session body. Pre-existing user tasks
        # (websockets, schedulers, long-lived workers) are exempt from
        # the drain window — they should not extend session exit.
        try:
            self._aenter_tasks = set(asyncio.all_tasks())
        except RuntimeError:
            self._aenter_tasks = None
        return self

    async def __aexit__(self, *exc: object) -> None:
        # Drain background LLM-callback queues (e.g. LiteLLM's
        # GLOBAL_LOGGING_WORKER) while we still have a live event loop.
        # Doing this in aexit — rather than atexit — is what keeps
        # single-completion scenarios from losing their request spans when
        # asyncio.run() tears down the loop.
        try:
            try:
                await _drain_background_logging_tasks(baseline=self._aenter_tasks)
            except Exception:
                # Swallow drain errors so session cleanup still runs; re-raise
                # BaseException (e.g. CancelledError) after _exit() via finally.
                _logger.debug("drain_background_logging_tasks failed", exc_info=True)
        finally:
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


def agentic_session(
    *,
    session_id: str,
    user_id: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs: object,
) -> _AgenticSessionContext:
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
    return _AgenticSessionContext(
        session_id=session_id, user_id=user_id, project=project, **kwargs
    )


class _AgentContext:
    def __init__(self, *, name: str):
        self._name = name
        self._span = None
        self._agent_token = None
        self._otel_token = None

    def _start(self) -> None:
        tracer = trace.get_tracer("kelet")
        attrs: dict = {
            "gen_ai.operation.name": "invoke_agent",
            AGENT_NAME_ATTR: self._name,
        }
        if (sid := _session_id_var.get()) is not None:
            attrs[SESSION_ID_ATTR] = sid
        if (uid := _user_id_var.get()) is not None:
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
