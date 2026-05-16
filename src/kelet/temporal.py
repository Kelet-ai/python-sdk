"""Kelet plugin for Temporal: propagates the agentic-session through Temporal
headers across ``start_workflow → workflow → child_workflow → activity``.

Adapted from the production interceptor at
``workflows/src/tenant/interceptor.py`` in the Kelet monorepo. Drops the
tenant-specific concerns (org/project/S3 URI/system-workflow blacklist) and
generalises the SystemWorkflows allowlist into an ``auto_session`` knob.

Composes with Temporal's own ``OpenTelemetryPlugin``: by default Kelet
bundles it (so users get linked OTel traces for free); detect-and-skip if a
prior plugin has already registered an OTel interceptor; opt out via
``KeletPlugin(include_otel_plugin=False)``.

## Plugin ordering

If you already use ``temporalio.contrib.opentelemetry.OpenTelemetryPlugin`` (or
``TracingInterceptor``), register it **before** ``KeletPlugin``::

    plugins=[OpenTelemetryPlugin(), KeletPlugin()]   # ✅

Kelet detects the existing OTel interceptor and skips its bundled OTel to avoid
duplicate spans. If you register them in the opposite order, set
``KeletPlugin(include_otel_plugin=False)`` explicitly::

    plugins=[KeletPlugin(include_otel_plugin=False), OpenTelemetryPlugin()]   # ✅
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, AsyncIterator, Mapping, Optional, Type, Union

# All `temporalio` imports are required at module import time. The package
# is an optional dependency of `kelet` (extra: `temporal`); `kelet/__init__.py`
# guards `from .temporal import ...` with a try/except so users who don't
# install the extra never trigger this module's imports.
import temporalio.client
import temporalio.worker
from temporalio import activity, workflow
from temporalio.contrib.opentelemetry import (
    OpenTelemetryInterceptor,
    OpenTelemetryPlugin,
    TracingInterceptor,
)
from temporalio.converter import PayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.worker import (
    ContinueAsNewInput,
    ExecuteActivityInput,
    ExecuteWorkflowInput,
    HandleQueryInput,
    HandleSignalInput,
    HandleUpdateInput,
    SignalChildWorkflowInput,
    SignalExternalWorkflowInput,
    StartActivityInput,
    StartChildWorkflowInput,
    StartLocalActivityInput,
    WorkflowInterceptorClassInput,
    WorkflowRunner,
)
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

import kelet
from kelet._context import (
    _metadata_kwargs_var,
    _session_id_var,
    _user_id_var,
)
from kelet._signal import _kelet_signal_activity

logger = logging.getLogger("kelet")

SESSION_HEADER = "x-kelet-session-id"
USER_HEADER = "x-kelet-user-id"
METADATA_HEADER = "x-kelet-metadata"

# Key under which we stash the per-instance auto_session resolver in the
# workflow's ``unsafe_extern_functions`` map. Same mechanism Temporal's own
# ``TracingInterceptor`` uses to plumb data into the sandbox without a
# module-level global (see contrib/opentelemetry/_interceptor.py).
_AUTO_SESSION_EXTERN_KEY = "__kelet_auto_session_resolver"

AutoSession = Union[bool, Callable[[Any], Optional[str]]]


# ── helpers ──────────────────────────────────────────────────────────────────


def _derive_session_id(workflow_id: str) -> str:
    """Default fallback when ``auto_session=True``.

    For workflow IDs that follow the ``{prefix}/session/{id}`` convention
    (used internally by Kelet's monorepo), return the ``{id}`` segment so
    the session ID is human-readable. Otherwise fall back to the full
    workflow ID — still correlatable, just less pretty.
    """
    parts = workflow_id.split("/")
    for i, p in enumerate(parts):
        if p == "session" and i + 1 < len(parts):
            return parts[i + 1]
    return workflow_id


def _inject(headers: Mapping[str, Any]) -> Mapping[str, Any]:
    """Stamp current contextvars (session/user/metadata) into outbound headers.

    Returns the input mapping unchanged if no session is set, so non-Kelet
    contexts pay zero copy cost on every outbound call.
    """
    sess = _session_id_var.get()
    if sess is None:
        return headers
    out = dict(headers)
    out[SESSION_HEADER] = PayloadConverter.default.to_payloads([sess])[0]
    user = _user_id_var.get()
    if user is not None:
        out[USER_HEADER] = PayloadConverter.default.to_payloads([user])[0]
    meta = _metadata_kwargs_var.get()
    if meta:
        out[METADATA_HEADER] = PayloadConverter.default.to_payloads([meta])[0]
    return out


def _extract(
    headers: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    """Reconstruct (session_id, user_id, metadata) from inbound headers."""
    sp = headers.get(SESSION_HEADER)
    if sp is None:
        return (None, None, None)
    sess = PayloadConverter.default.from_payload(sp, type_hint=str)
    up = headers.get(USER_HEADER)
    user = PayloadConverter.default.from_payload(up, type_hint=str) if up else None
    mp = headers.get(METADATA_HEADER)
    meta = PayloadConverter.default.from_payload(mp, type_hint=dict) if mp else None
    return (sess, user, meta)


def _resolve_workflow_session(
    headers: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    """Workflow-side resolution: header > auto_session > none.

    The ``auto_session`` resolver is plumbed into the workflow via
    ``unsafe_extern_functions`` so per-plugin-instance config doesn't leak
    across other ``KeletPlugin`` instances in the same process.
    """
    sess, user, meta = _extract(headers)
    if sess is not None:
        return (sess, user, meta)
    resolver = workflow.extern_functions().get(_AUTO_SESSION_EXTERN_KEY)
    if resolver is None:
        return (None, None, None)
    derived = resolver(workflow.info())
    if not derived:
        return (None, None, None)
    return (derived, None, None)


# ── client outbound ──────────────────────────────────────────────────────────


class _ClientOutbound(temporalio.client.OutboundInterceptor):
    """Stamps Kelet headers on outbound client calls."""

    async def start_workflow(self, input: temporalio.client.StartWorkflowInput):
        return await super().start_workflow(
            replace(input, headers=_inject(input.headers))
        )

    async def signal_workflow(
        self, input: temporalio.client.SignalWorkflowInput
    ) -> None:
        await super().signal_workflow(replace(input, headers=_inject(input.headers)))

    async def query_workflow(self, input: temporalio.client.QueryWorkflowInput):
        return await super().query_workflow(
            replace(input, headers=_inject(input.headers))
        )


# ── workflow outbound ────────────────────────────────────────────────────────


class _WorkflowOutbound(temporalio.worker.WorkflowOutboundInterceptor):
    """Stamps Kelet headers on outbound calls made *from* workflow code."""

    def start_activity(self, input: StartActivityInput):
        return super().start_activity(replace(input, headers=_inject(input.headers)))

    def start_local_activity(self, input: StartLocalActivityInput):
        return super().start_local_activity(
            replace(input, headers=_inject(input.headers))
        )

    async def start_child_workflow(self, input: StartChildWorkflowInput):
        return await super().start_child_workflow(
            replace(input, headers=_inject(input.headers))
        )

    async def signal_child_workflow(self, input: SignalChildWorkflowInput) -> None:
        await super().signal_child_workflow(
            replace(input, headers=_inject(input.headers))
        )

    async def signal_external_workflow(
        self, input: SignalExternalWorkflowInput
    ) -> None:
        await super().signal_external_workflow(
            replace(input, headers=_inject(input.headers))
        )

    def continue_as_new(self, input: ContinueAsNewInput):
        return super().continue_as_new(replace(input, headers=_inject(input.headers)))


# ── workflow inbound (lite mode) ─────────────────────────────────────────────
# Only sets contextvars — sandbox-safe. No OTEL attach, no drain.


class _WorkflowInbound(temporalio.worker.WorkflowInboundInterceptor):
    def init(self, outbound: temporalio.worker.WorkflowOutboundInterceptor) -> None:
        super().init(_WorkflowOutbound(outbound))

    async def execute_workflow(self, input: ExecuteWorkflowInput):
        sess, user, meta = _resolve_workflow_session(input.headers)
        if sess is None:
            return await super().execute_workflow(input)
        tokens = [
            _session_id_var.set(sess),
            _user_id_var.set(user),
            _metadata_kwargs_var.set(meta),
        ]
        try:
            return await super().execute_workflow(input)
        finally:
            for t in reversed(tokens):
                t.var.reset(t)

    async def handle_signal(self, input: HandleSignalInput) -> None:
        sess, *_ = _extract(input.headers)
        if sess is None:
            await super().handle_signal(input)
            return
        tok = _session_id_var.set(sess)
        try:
            await super().handle_signal(input)
        finally:
            _session_id_var.reset(tok)

    async def handle_query(self, input: HandleQueryInput):
        sess, *_ = _extract(input.headers)
        if sess is None:
            return await super().handle_query(input)
        tok = _session_id_var.set(sess)
        try:
            return await super().handle_query(input)
        finally:
            _session_id_var.reset(tok)

    def handle_update_validator(self, input: HandleUpdateInput) -> None:
        sess, *_ = _extract(input.headers)
        if sess is None:
            super().handle_update_validator(input)
            return
        tok = _session_id_var.set(sess)
        try:
            super().handle_update_validator(input)
        finally:
            _session_id_var.reset(tok)

    async def handle_update_handler(self, input: HandleUpdateInput):
        sess, *_ = _extract(input.headers)
        if sess is None:
            return await super().handle_update_handler(input)
        tok = _session_id_var.set(sess)
        try:
            return await super().handle_update_handler(input)
        finally:
            _session_id_var.reset(tok)


# ── activity inbound (full mode) ─────────────────────────────────────────────
# Opens kelet.agentic_session(...) — OTEL attach + drain are safe here.


class _ActivityInbound(temporalio.worker.ActivityInboundInterceptor):
    def __init__(
        self,
        next: temporalio.worker.ActivityInboundInterceptor,
        auto_session: AutoSession,
    ) -> None:
        super().__init__(next)
        self._auto_session = auto_session

    async def execute_activity(self, input: ExecuteActivityInput):
        sess, user, meta = _extract(input.headers)
        if sess is None:
            sess = self._derive()
        if sess is None:
            return await super().execute_activity(input)
        async with kelet.agentic_session(session_id=sess, user_id=user, **(meta or {})):
            return await super().execute_activity(input)

    def _derive(self) -> Optional[str]:
        if self._auto_session is False:
            return None
        info = activity.info()
        if self._auto_session is True:
            wf_id = info.workflow_id
            return _derive_session_id(wf_id) if wf_id else None
        return self._auto_session(info)


# ── interceptor + plugin facade ──────────────────────────────────────────────


class KeletInterceptor(temporalio.client.Interceptor, temporalio.worker.Interceptor):
    """Standalone interceptor — use this if you want manual control instead
    of ``KeletPlugin``.

    Args:
        auto_session: When no Kelet session header is present on an inbound
            workflow / activity, decide whether to derive one.

            - ``False`` (default): no auto-derivation; user code can still set a
              session manually via ``kelet.agentic_session()``.
            - ``True``: derive from the workflow ID. If the ID matches
              ``.../session/{id}`` returns just ``{id}``; otherwise returns the
              whole workflow ID.
            - ``Callable[[WorkflowInfo | ActivityInfo], str | None]``: invoked
              on the inbound side with the relevant Temporal Info; return a
              session ID string or ``None`` to skip. **Must be deterministic**
              with respect to its argument — non-deterministic resolvers cause
              workflow replay failures.
    """

    def __init__(self, *, auto_session: AutoSession = False):
        self._auto_session = auto_session

    def intercept_client(
        self, next: temporalio.client.OutboundInterceptor
    ) -> temporalio.client.OutboundInterceptor:
        return _ClientOutbound(next)

    def intercept_activity(
        self, next: temporalio.worker.ActivityInboundInterceptor
    ) -> temporalio.worker.ActivityInboundInterceptor:
        return _ActivityInbound(next, self._auto_session)

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> Type[temporalio.worker.WorkflowInboundInterceptor]:
        # Plumb per-instance config through unsafe_extern_functions so
        # multiple KeletPlugin instances in one process don't clobber each
        # other (Issue 1 in plan: avoids a module-level global).
        resolver = self._build_resolver()
        if resolver is not None:
            input.unsafe_extern_functions[_AUTO_SESSION_EXTERN_KEY] = resolver
        return _WorkflowInbound

    def _build_resolver(self) -> Optional[Callable[[Any], Optional[str]]]:
        """Return a callable ``(Info) -> str | None``, or ``None`` to disable.

        ``True``/callable inputs are normalised here so the workflow side just
        invokes ``resolver(info)`` without a type-narrowing dance.
        """
        if self._auto_session is False:
            return None
        if self._auto_session is True:
            return lambda info: _derive_session_id(info.workflow_id)
        return self._auto_session


def _has_existing_otel(config: Any) -> bool:
    """Detect whether an earlier plugin already registered an OTel interceptor.

    Checks for both the new ``OpenTelemetryInterceptor`` and the legacy
    ``TracingInterceptor``. Used to skip our bundled OTel when the user has
    one of their own registered before us (the common ``[OTel, Kelet]``
    ordering — see module docstring).
    """
    existing = config.get("interceptors", []) or []
    return any(
        isinstance(i, (OpenTelemetryInterceptor, TracingInterceptor)) for i in existing
    )


class KeletPlugin(temporalio.client.Plugin, temporalio.worker.Plugin):
    """Recommended install for Temporal users.

    Bundles ``KeletInterceptor`` (session propagation), composes Temporal's
    ``OpenTelemetryPlugin`` (so users get linked OTel traces for free), adds
    ``kelet`` to the workflow sandbox passthrough modules, and flushes Kelet's
    ``BatchSpanProcessor`` on worker shutdown.

    Register on the client; the worker inherits automatically::

        client = await Client.connect("localhost:7233", plugins=[KeletPlugin()])
        worker = Worker(client, ..., activities=[my_activity])

    Args:
        auto_session: Forwarded to ``KeletInterceptor``. See its docstring.
        include_otel_plugin: When ``True`` (default), bundle Temporal's
            ``OpenTelemetryPlugin``. Set to ``False`` if you've already
            configured OTel yourself; set to ``False`` and register your own
            OTel plugin **after** ``KeletPlugin`` if you want manual control.
            See the module docstring for plugin-ordering recommendations.
    """

    def __init__(
        self,
        *,
        auto_session: AutoSession = False,
        include_otel_plugin: bool = True,
    ):
        self._kelet_interceptor = KeletInterceptor(auto_session=auto_session)
        self._otel_plugin: Optional[OpenTelemetryPlugin] = (
            OpenTelemetryPlugin() if include_otel_plugin else None
        )
        self._include_otel_plugin = include_otel_plugin
        # Tracks whether we suppressed our bundled OTel because an earlier
        # plugin already registered one. Set in configure_client; consumed by
        # configure_worker / configure_replayer / run_worker so all phases
        # stay consistent.
        self._otel_was_skipped: bool = False

        @asynccontextmanager
        async def run_context() -> AsyncIterator[None]:
            try:
                yield
            finally:
                # Bound the flush so a slow/unreachable Kelet API can't hang
                # worker shutdown. Mirrors LangSmithPlugin's flush-on-exit.
                from kelet._configure import shutdown

                try:
                    await asyncio.wait_for(asyncio.to_thread(shutdown), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "kelet.shutdown() exceeded 10s flush timeout; "
                        "some spans may be lost"
                    )
                except Exception:
                    pass

        def workflow_runner(runner: Optional[WorkflowRunner]) -> WorkflowRunner:
            if runner is None:
                # SimplePlugin treats ``workflow_runner=None`` as "leave the
                # existing runner alone", so its callable form is only invoked
                # when there's a runner to wrap. Defensive: return a fresh
                # SandboxedWorkflowRunner if Temporal ever calls us with None.
                runner = SandboxedWorkflowRunner()
            if isinstance(runner, SandboxedWorkflowRunner):
                # Allow workflow code to ``import kelet`` cleanly. We do NOT
                # add ``opentelemetry`` here — the bundled OpenTelemetryPlugin
                # handles its own passthrough — to avoid double-application
                # if a user disables include_otel_plugin and adds OTel later.
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules("kelet"),
                )
            return runner

        self._kelet_simple_plugin = SimplePlugin(
            "kelet.KeletPlugin",
            interceptors=[self._kelet_interceptor],
            activities=[_kelet_signal_activity],
            workflow_runner=workflow_runner,
            run_context=lambda: run_context(),
        )

    def name(self) -> str:
        return "kelet.KeletPlugin"

    def configure_client(
        self, config: temporalio.client.ClientConfig
    ) -> temporalio.client.ClientConfig:
        # Issue 17 (17A): only bundle OTel if no other plugin registered it
        # first. Catches the common [OTel, Kelet] ordering. For [Kelet, OTel]
        # we can't detect (user's plugin runs after ours); the module
        # docstring recommends users either reorder or pass
        # include_otel_plugin=False explicitly.
        if self._otel_plugin is not None:
            if _has_existing_otel(config):
                self._otel_was_skipped = True
                logger.info(
                    "KeletPlugin: detected existing OTel interceptor; skipping "
                    "bundled OpenTelemetryPlugin. Set include_otel_plugin=False "
                    "to silence this."
                )
            else:
                config = self._otel_plugin.configure_client(config)

        config = self._kelet_simple_plugin.configure_client(config)

        # Issue 4 (4A): warn if the user explicitly disabled the bundled OTel
        # and there's no OTel interceptor registered to fill the gap.
        if not self._include_otel_plugin and not _has_existing_otel(config):
            logger.warning(
                "KeletPlugin: include_otel_plugin=False but no OTel interceptor "
                "found on client. Workflow and activity spans will not be linked "
                "into one trace. Either set include_otel_plugin=True or register "
                "your own OTel interceptor (e.g. OpenTelemetryPlugin) before "
                "KeletPlugin."
            )
        return config

    async def connect_service_client(
        self,
        config: Any,
        next: Callable[[Any], Any],
    ) -> Any:
        return await next(config)

    def configure_worker(
        self, config: temporalio.worker.WorkerConfig
    ) -> temporalio.worker.WorkerConfig:
        if self._otel_plugin is not None and not self._otel_was_skipped:
            config = self._otel_plugin.configure_worker(config)
        return self._kelet_simple_plugin.configure_worker(config)

    def configure_replayer(
        self, config: temporalio.worker.ReplayerConfig
    ) -> temporalio.worker.ReplayerConfig:
        if self._otel_plugin is not None and not self._otel_was_skipped:
            config = self._otel_plugin.configure_replayer(config)
        return self._kelet_simple_plugin.configure_replayer(config)

    async def run_worker(
        self,
        worker: temporalio.worker.Worker,
        next: Callable[[temporalio.worker.Worker], Any],
    ) -> None:
        if self._otel_plugin is not None and not self._otel_was_skipped:
            return await self._otel_plugin.run_worker(
                worker,
                lambda w: self._kelet_simple_plugin.run_worker(w, next),
            )
        return await self._kelet_simple_plugin.run_worker(worker, next)

    @asynccontextmanager
    async def run_replayer(
        self,
        replayer: Any,
        histories: Any,
        next: Callable[..., Any],
    ):
        if self._otel_plugin is not None and not self._otel_was_skipped:
            async with self._otel_plugin.run_replayer(
                replayer,
                histories,
                lambda r, h: self._kelet_simple_plugin.run_replayer(r, h, next),
            ) as result:
                yield result
        else:
            async with self._kelet_simple_plugin.run_replayer(
                replayer, histories, next
            ) as result:
                yield result


__all__ = [
    "KeletInterceptor",
    "KeletPlugin",
    "SESSION_HEADER",
    "USER_HEADER",
    "METADATA_HEADER",
]
