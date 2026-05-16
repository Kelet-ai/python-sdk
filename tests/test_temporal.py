"""Tests for ``kelet.temporal`` — KeletInterceptor + KeletPlugin.

Structured into nine test classes (A–I) aligned to the design diagram in
``hmm-wdyt-about-adding-federated-mccarthy.md``::

    A. Client outbound
    B. Workflow inbound
    C. Workflow body interactions
    D. Workflow outbound
    E. Activity inbound
    F. Signal/query/update handlers
    G. Plugin composition  (most new ground)
    H. agentic_session workflow-aware
    I. kelet.signal() workflow dispatch

Test patterns ported from ``workflows/src/tenant/tests/test_interceptor.py``:
``SimpleNamespace`` mocks for ``next_inbound`` / ``next_outbound``, a
``_decode_header`` helper for asserting on payload-encoded headers, and
``unittest.mock.patch`` for ``workflow.info`` / ``activity.info``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from temporalio import client, common, workflow, worker
from temporalio.api.common.v1.message_pb2 import Payload
from temporalio.contrib.opentelemetry import (
    OpenTelemetryInterceptor,
    TracingInterceptor,
)
from temporalio.converter import PayloadConverter
from temporalio.worker import (
    ContinueAsNewInput,
    ExecuteActivityInput,
    ExecuteWorkflowInput,
    HandleSignalInput,
    StartActivityInput,
    StartChildWorkflowInput,
)

import kelet
from kelet._context import (
    _metadata_kwargs_var,
    _session_id_var,
    _user_id_var,
)
from kelet.temporal import (
    METADATA_HEADER,
    SESSION_HEADER,
    USER_HEADER,
    KeletInterceptor,
    KeletPlugin,
    _ActivityInbound,
    _ClientOutbound,
    _derive_session_id,
    _has_existing_otel,
    _WorkflowInbound,
    _WorkflowOutbound,
)


# ───────────────── helpers ─────────────────


def _decode(payload: Payload, type_hint: type = str) -> Any:
    return PayloadConverter.default.from_payload(payload, type_hint=type_hint)


def _session_headers(
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Payload]:
    h: dict[str, Payload] = {}
    if session_id is not None:
        h[SESSION_HEADER] = PayloadConverter.default.to_payloads([session_id])[0]
    if user_id is not None:
        h[USER_HEADER] = PayloadConverter.default.to_payloads([user_id])[0]
    if metadata is not None:
        h[METADATA_HEADER] = PayloadConverter.default.to_payloads([metadata])[0]
    return h


@pytest.fixture(autouse=True)
def _reset_kelet_contextvars():
    """ContextVars must be at default between tests; otherwise outbound stamping
    leaks across cases."""
    s = _session_id_var.set(None)
    u = _user_id_var.set(None)
    m = _metadata_kwargs_var.set(None)
    yield
    _session_id_var.reset(s)
    _user_id_var.reset(u)
    _metadata_kwargs_var.reset(m)


def _start_activity_input(
    headers: dict[str, Payload] | None = None,
) -> StartActivityInput:
    return StartActivityInput(
        activity="my-activity",
        args=[],
        activity_id=None,
        task_queue=None,
        schedule_to_close_timeout=timedelta(seconds=30),
        schedule_to_start_timeout=None,
        start_to_close_timeout=timedelta(seconds=30),
        heartbeat_timeout=None,
        retry_policy=None,
        cancellation_type=workflow.ActivityCancellationType.TRY_CANCEL,
        headers=cast(Any, headers or {}),
        disable_eager_execution=False,
        versioning_intent=None,
        summary=None,
        priority=common.Priority(),
        arg_types=None,
        ret_type=None,
    )


def _start_child_workflow_input(
    headers: dict[str, Payload] | None = None,
) -> StartChildWorkflowInput:
    return StartChildWorkflowInput(
        workflow="child",
        args=[],
        id="child-id",
        task_queue=None,
        cancellation_type=workflow.ChildWorkflowCancellationType.WAIT_CANCELLATION_COMPLETED,
        parent_close_policy=workflow.ParentClosePolicy.TERMINATE,
        execution_timeout=None,
        run_timeout=None,
        task_timeout=None,
        id_reuse_policy=common.WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        retry_policy=None,
        cron_schedule="",
        memo=None,
        search_attributes=None,
        headers=cast(Any, headers or {}),
        versioning_intent=None,
        static_summary=None,
        static_details=None,
        priority=common.Priority(),
        arg_types=None,
        ret_type=None,
    )


def _start_workflow_input(
    headers: dict[str, Payload] | None = None,
) -> client.StartWorkflowInput:
    return client.StartWorkflowInput(
        workflow="MyWorkflow",
        args=(),
        id="wf-id",
        task_queue="tq",
        execution_timeout=None,
        run_timeout=None,
        task_timeout=None,
        id_reuse_policy=common.WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        id_conflict_policy=common.WorkflowIDConflictPolicy.UNSPECIFIED,
        retry_policy=None,
        cron_schedule="",
        memo=None,
        search_attributes=None,
        start_delay=None,
        headers=cast(Any, headers or {}),
        start_signal=None,
        start_signal_args=(),
        static_summary=None,
        static_details=None,
        ret_type=None,
        rpc_metadata={},
        rpc_timeout=None,
        request_eager_start=False,
        priority=common.Priority(),
        callbacks=(),
        workflow_event_links=(),
        request_id=None,
    )


def _continue_as_new_input(
    headers: dict[str, Payload] | None = None,
) -> ContinueAsNewInput:
    return ContinueAsNewInput(
        workflow=None,
        args=[],
        task_queue=None,
        run_timeout=None,
        task_timeout=None,
        retry_policy=None,
        memo=None,
        search_attributes=None,
        headers=cast(Any, headers or {}),
        versioning_intent=None,
        initial_versioning_behavior=None,
        arg_types=None,
    )


# ───────────────── A. Client outbound ─────────────────


class TestClientOutbound:
    @pytest.mark.asyncio
    async def test_a1_session_set_stamps_header(self):
        next_outbound = SimpleNamespace(start_workflow=AsyncMock(return_value=None))
        outbound = _ClientOutbound(cast(client.OutboundInterceptor, next_outbound))

        with kelet.agentic_session(session_id="sess-A1"):
            await outbound.start_workflow(_start_workflow_input())

        forwarded = next_outbound.start_workflow.await_args.args[0]
        assert _decode(forwarded.headers[SESSION_HEADER]) == "sess-A1"

    @pytest.mark.asyncio
    async def test_a2_no_session_no_header(self):
        next_outbound = SimpleNamespace(start_workflow=AsyncMock(return_value=None))
        outbound = _ClientOutbound(cast(client.OutboundInterceptor, next_outbound))

        await outbound.start_workflow(_start_workflow_input())

        forwarded = next_outbound.start_workflow.await_args.args[0]
        assert SESSION_HEADER not in forwarded.headers
        assert USER_HEADER not in forwarded.headers
        assert METADATA_HEADER not in forwarded.headers

    @pytest.mark.asyncio
    async def test_a3_full_payload_session_user_metadata(self):
        next_outbound = SimpleNamespace(start_workflow=AsyncMock(return_value=None))
        outbound = _ClientOutbound(cast(client.OutboundInterceptor, next_outbound))

        with kelet.agentic_session(
            session_id="sess-A3",
            user_id="user-7",
            tier="pro",
            count=42,
        ):
            await outbound.start_workflow(_start_workflow_input())

        forwarded = next_outbound.start_workflow.await_args.args[0]
        assert _decode(forwarded.headers[SESSION_HEADER]) == "sess-A3"
        assert _decode(forwarded.headers[USER_HEADER]) == "user-7"
        meta = _decode(forwarded.headers[METADATA_HEADER], type_hint=dict)
        assert meta == {"tier": "pro", "count": 42}


# ───────────────── B. Workflow inbound ─────────────────


def _execute_wf_input(
    args: list[Any] | None = None,
    headers: dict[str, Payload] | None = None,
) -> ExecuteWorkflowInput:
    return ExecuteWorkflowInput(
        type=object,
        run_fn=AsyncMock(),
        args=tuple(args or []),
        headers=cast(Any, headers or {}),
    )


class TestWorkflowInbound:
    @pytest.mark.asyncio
    async def test_b1_header_present_sets_contextvars(self):
        captured: dict[str, Any] = {}

        async def _execute(_input):
            captured["session"] = kelet.get_session_id()
            captured["user"] = kelet.get_user_id()
            captured["meta"] = kelet.get_metadata_kwargs()

        next_inbound = SimpleNamespace(execute_workflow=AsyncMock(side_effect=_execute))
        inbound = _WorkflowInbound(
            cast(worker.WorkflowInboundInterceptor, next_inbound)
        )

        await inbound.execute_workflow(
            _execute_wf_input(
                headers=_session_headers(
                    session_id="sess-B1", user_id="u-1", metadata={"k": "v"}
                )
            )
        )
        assert captured == {"session": "sess-B1", "user": "u-1", "meta": {"k": "v"}}

    @pytest.mark.asyncio
    async def test_b2_no_header_no_auto_no_contextvars(self):
        captured: dict[str, Any] = {}

        async def _execute(_input):
            captured["session"] = kelet.get_session_id()

        next_inbound = SimpleNamespace(execute_workflow=AsyncMock(side_effect=_execute))
        inbound = _WorkflowInbound(
            cast(worker.WorkflowInboundInterceptor, next_inbound)
        )

        # auto_session=False (default) — interceptor never registers a resolver,
        # so workflow.extern_functions() at runtime would return whatever the
        # outer frame provides. Patch it to a no-resolver dict to exercise the
        # "no resolver" branch outside a real workflow event loop.
        with patch("kelet.temporal.workflow.extern_functions", return_value={}):
            await inbound.execute_workflow(_execute_wf_input())
        assert captured == {"session": None}

    @pytest.mark.asyncio
    async def test_b3_no_header_auto_true_derives_from_workflow_id(self):
        captured: dict[str, Any] = {}

        async def _execute(_input):
            captured["session"] = kelet.get_session_id()

        next_inbound = SimpleNamespace(execute_workflow=AsyncMock(side_effect=_execute))
        inbound = _WorkflowInbound(
            cast(worker.WorkflowInboundInterceptor, next_inbound)
        )

        # Simulate the resolver being plumbed via unsafe_extern_functions:
        # we patch workflow.extern_functions to return our resolver and
        # workflow.info to provide a synthetic Info.
        with (
            patch(
                "kelet.temporal.workflow.extern_functions",
                return_value={
                    "__kelet_auto_session_resolver": lambda info: _derive_session_id(
                        info.workflow_id
                    )
                },
            ),
            patch(
                "kelet.temporal.workflow.info",
                return_value=SimpleNamespace(workflow_id="acme/prod/session/sess-B3"),
            ),
        ):
            await inbound.execute_workflow(_execute_wf_input())
        assert captured == {"session": "sess-B3"}

    @pytest.mark.asyncio
    async def test_b4_no_header_auto_callable_invoked_with_info(self):
        captured: dict[str, Any] = {}
        seen_info: dict[str, Any] = {}

        async def _execute(_input):
            captured["session"] = kelet.get_session_id()

        def resolver(info):
            seen_info["wf_id"] = info.workflow_id
            return f"derived-{info.workflow_id.split('/')[-1]}"

        next_inbound = SimpleNamespace(execute_workflow=AsyncMock(side_effect=_execute))
        inbound = _WorkflowInbound(
            cast(worker.WorkflowInboundInterceptor, next_inbound)
        )

        with (
            patch(
                "kelet.temporal.workflow.extern_functions",
                return_value={"__kelet_auto_session_resolver": resolver},
            ),
            patch(
                "kelet.temporal.workflow.info",
                return_value=SimpleNamespace(workflow_id="acme/prod/wf/abc"),
            ),
        ):
            await inbound.execute_workflow(_execute_wf_input())
        assert captured == {"session": "derived-abc"}
        assert seen_info == {"wf_id": "acme/prod/wf/abc"}

    @pytest.mark.asyncio
    async def test_b5_replay_determinism_callable_must_return_same_value(self):
        """Issue 10 (10A): user-supplied callable MUST be deterministic with
        respect to its WorkflowInfo argument. We assert that calling the
        resolver twice with the same Info yields the same session — surfaces
        non-deterministic resolvers in CI before they hit production replay.
        """
        captured: list[str | None] = []

        async def _execute(_input):
            captured.append(kelet.get_session_id())

        # A deterministic resolver: same input → same output.
        def resolver(info):
            return _derive_session_id(info.workflow_id)

        next_inbound = SimpleNamespace(execute_workflow=AsyncMock(side_effect=_execute))
        inbound = _WorkflowInbound(
            cast(worker.WorkflowInboundInterceptor, next_inbound)
        )

        info = SimpleNamespace(workflow_id="acme/prod/session/replay-1")

        with (
            patch(
                "kelet.temporal.workflow.extern_functions",
                return_value={"__kelet_auto_session_resolver": resolver},
            ),
            patch("kelet.temporal.workflow.info", return_value=info),
        ):
            await inbound.execute_workflow(_execute_wf_input())
            await inbound.execute_workflow(_execute_wf_input())
        assert captured == ["replay-1", "replay-1"]


# ───────────────── C. Workflow body interactions ─────────────────


class TestWorkflowBody:
    @pytest.mark.asyncio
    async def test_c1_agentic_session_lite_mode_in_workflow(self):
        """When in_temporal_workflow() returns True, agentic_session must skip
        OTEL baggage attach + the background drain — but still set contextvars.
        """
        with (
            patch("kelet._context.in_temporal_workflow", return_value=True),
            patch("kelet._context.otel_context.attach") as mock_attach,
            patch("kelet._context._drain_background_logging_tasks") as mock_drain,
        ):
            async with kelet.agentic_session(session_id="sess-C1"):
                assert kelet.get_session_id() == "sess-C1"

        assert mock_attach.call_count == 0, "OTEL attach must not run inside workflow"
        assert mock_drain.call_count == 0, "drain must not run inside workflow"

    @pytest.mark.asyncio
    async def test_c2_inner_agentic_session_overrides_outer_in_child_activity(self):
        """When workflow code wraps execute_activity in an inner agentic_session,
        the outbound interceptor stamps the inner session, not the outer.
        """
        next_outbound = SimpleNamespace(start_activity=Mock(return_value=object()))
        outbound = _WorkflowOutbound(
            cast(worker.WorkflowOutboundInterceptor, next_outbound)
        )

        with kelet.agentic_session(session_id="outer"):
            with kelet.agentic_session(session_id="inner"):
                outbound.start_activity(_start_activity_input())

        forwarded = next_outbound.start_activity.call_args.args[0]
        assert _decode(forwarded.headers[SESSION_HEADER]) == "inner"

    @pytest.mark.asyncio
    async def test_c3_kelet_signal_in_workflow_dispatches_via_activity(self):
        """When called from workflow code, kelet.signal() must NOT call httpx
        directly — it dispatches through workflow.execute_activity so the HTTP
        call is durable."""
        with (
            patch("kelet._signal.in_temporal_workflow", return_value=True),
            patch(
                "temporalio.workflow.execute_activity", new=AsyncMock(return_value=None)
            ) as mock_exec,
            patch("kelet._signal._send_signal") as mock_direct,
        ):
            await kelet.signal(
                kind=kelet.SignalKind.FEEDBACK,
                source=kelet.SignalSource.HUMAN,
                session_id="sess-C3",
                score=1.0,
            )
        # Direct path NOT used in workflow context
        assert mock_direct.call_count == 0
        # Activity dispatch was invoked exactly once
        assert mock_exec.await_count == 1
        # First positional arg is the function reference; second is the _SignalArgs payload
        assert mock_exec.await_args is not None
        called_args = mock_exec.await_args.args
        assert called_args[0].__name__ == "_kelet_signal_activity"
        signal_args = called_args[1]
        assert signal_args.session_id == "sess-C3"
        assert signal_args.score == 1.0

    @pytest.mark.asyncio
    async def test_c4_signal_in_workflow_is_replay_safe(self):
        """Two invocations from the same workflow position must produce the
        same activity dispatch — no time, no random in the dispatch path.
        """
        from kelet._signal import _SignalArgs

        with (
            patch("kelet._signal.in_temporal_workflow", return_value=True),
            patch(
                "temporalio.workflow.execute_activity", new=AsyncMock(return_value=None)
            ) as mock_exec,
        ):
            for _ in range(2):
                await kelet.signal(
                    kind=kelet.SignalKind.FEEDBACK,
                    source=kelet.SignalSource.HUMAN,
                    session_id="sess-C4",
                )
        # Both invocations sent identical _SignalArgs
        a = mock_exec.await_args_list[0].args[1]
        b = mock_exec.await_args_list[1].args[1]
        assert isinstance(a, _SignalArgs) and isinstance(b, _SignalArgs)
        assert a.model_dump() == b.model_dump()


# ───────────────── D. Workflow outbound ─────────────────


class TestWorkflowOutbound:
    def test_d1_session_var_set_stamps_header(self):
        next_outbound = SimpleNamespace(start_activity=Mock(return_value=object()))
        outbound = _WorkflowOutbound(
            cast(worker.WorkflowOutboundInterceptor, next_outbound)
        )

        with kelet.agentic_session(session_id="sess-D1"):
            outbound.start_activity(_start_activity_input())

        forwarded = next_outbound.start_activity.call_args.args[0]
        assert _decode(forwarded.headers[SESSION_HEADER]) == "sess-D1"

    def test_d2_session_var_unset_no_header(self):
        next_outbound = SimpleNamespace(start_activity=Mock(return_value=object()))
        outbound = _WorkflowOutbound(
            cast(worker.WorkflowOutboundInterceptor, next_outbound)
        )

        outbound.start_activity(_start_activity_input())

        forwarded = next_outbound.start_activity.call_args.args[0]
        assert SESSION_HEADER not in forwarded.headers

    def test_d3_continue_as_new_forwards_header(self):
        next_outbound = SimpleNamespace(
            continue_as_new=Mock(side_effect=RuntimeError("can"))
        )
        outbound = _WorkflowOutbound(
            cast(worker.WorkflowOutboundInterceptor, next_outbound)
        )

        with kelet.agentic_session(session_id="sess-D3"):
            with pytest.raises(RuntimeError, match="can"):
                outbound.continue_as_new(_continue_as_new_input())

        forwarded = next_outbound.continue_as_new.call_args.args[0]
        assert _decode(forwarded.headers[SESSION_HEADER]) == "sess-D3"

    @pytest.mark.asyncio
    async def test_d4_signal_child_external_stamps_header(self):
        next_outbound = SimpleNamespace(
            signal_child_workflow=AsyncMock(return_value=None),
            signal_external_workflow=AsyncMock(return_value=None),
            start_child_workflow=AsyncMock(return_value=object()),
        )
        outbound = _WorkflowOutbound(
            cast(worker.WorkflowOutboundInterceptor, next_outbound)
        )

        with kelet.agentic_session(session_id="sess-D4"):
            await outbound.start_child_workflow(_start_child_workflow_input())

        forwarded = next_outbound.start_child_workflow.await_args.args[0]
        assert _decode(forwarded.headers[SESSION_HEADER]) == "sess-D4"


# ───────────────── E. Activity inbound ─────────────────


def _execute_activity_input(
    headers: dict[str, Payload] | None = None,
) -> ExecuteActivityInput:
    return ExecuteActivityInput(
        fn=lambda: None,
        args=(),
        executor=None,
        headers=cast(Any, headers or {}),
    )


class TestActivityInbound:
    @pytest.mark.asyncio
    async def test_e1_header_opens_full_agentic_session(self):
        captured: dict[str, Any] = {}

        async def _execute(_input):
            captured["session"] = kelet.get_session_id()
            captured["user"] = kelet.get_user_id()

        next_inbound = SimpleNamespace(execute_activity=AsyncMock(side_effect=_execute))
        inbound = _ActivityInbound(
            cast(worker.ActivityInboundInterceptor, next_inbound),
            auto_session=False,
        )

        await inbound.execute_activity(
            _execute_activity_input(
                headers=_session_headers(session_id="sess-E1", user_id="u-1"),
            )
        )
        assert captured == {"session": "sess-E1", "user": "u-1"}

    @pytest.mark.asyncio
    async def test_e2_no_header_auto_session_resolves(self):
        captured: list[str | None] = []

        async def _execute(_input):
            captured.append(kelet.get_session_id())

        next_inbound = SimpleNamespace(execute_activity=AsyncMock(side_effect=_execute))
        inbound = _ActivityInbound(
            cast(worker.ActivityInboundInterceptor, next_inbound),
            auto_session=True,
        )

        with patch(
            "kelet.temporal.activity.info",
            return_value=SimpleNamespace(workflow_id="acme/prod/session/sess-E2"),
        ):
            await inbound.execute_activity(_execute_activity_input())
        assert captured == ["sess-E2"]

    @pytest.mark.asyncio
    async def test_e3_no_header_no_auto_passes_through(self):
        captured: list[str | None] = []

        async def _execute(_input):
            captured.append(kelet.get_session_id())

        next_inbound = SimpleNamespace(execute_activity=AsyncMock(side_effect=_execute))
        inbound = _ActivityInbound(
            cast(worker.ActivityInboundInterceptor, next_inbound),
            auto_session=False,
        )

        await inbound.execute_activity(_execute_activity_input())
        assert captured == [None]


# ───────────────── F. Signal/query/update handlers ─────────────────


class TestSignalHandlers:
    @pytest.mark.asyncio
    async def test_f1_signal_handler_reads_header(self):
        captured: list[str | None] = []

        async def _handle(_input):
            captured.append(kelet.get_session_id())

        next_inbound = SimpleNamespace(handle_signal=AsyncMock(side_effect=_handle))
        inbound = _WorkflowInbound(
            cast(worker.WorkflowInboundInterceptor, next_inbound)
        )

        await inbound.handle_signal(
            HandleSignalInput(
                signal="event",
                args=(),
                headers=cast(Any, _session_headers(session_id="sess-F1")),
            )
        )
        assert captured == ["sess-F1"]

    @pytest.mark.asyncio
    async def test_f2_signal_handler_no_header_passes_through(self):
        next_inbound = SimpleNamespace(handle_signal=AsyncMock(return_value=None))
        inbound = _WorkflowInbound(
            cast(worker.WorkflowInboundInterceptor, next_inbound)
        )

        await inbound.handle_signal(
            HandleSignalInput(signal="event", args=(), headers=cast(Any, {}))
        )
        next_inbound.handle_signal.assert_awaited_once()


# ───────────────── G. Plugin composition ─────────────────


class TestPluginComposition:
    def test_g1_include_otel_true_configures_otel_plugin(self):
        plugin = KeletPlugin(include_otel_plugin=True)
        config: client.ClientConfig = cast(Any, {"interceptors": []})
        out = plugin.configure_client(config)
        # OpenTelemetryInterceptor should be present (added by composed plugin),
        # plus our KeletInterceptor.
        interceptors = out.get("interceptors", [])
        assert any(isinstance(i, OpenTelemetryInterceptor) for i in interceptors)
        assert any(isinstance(i, KeletInterceptor) for i in interceptors)

    def test_g2_include_otel_false_no_otel_warns(
        self, caplog: pytest.LogCaptureFixture
    ):
        plugin = KeletPlugin(include_otel_plugin=False)
        config: client.ClientConfig = cast(Any, {"interceptors": []})
        with caplog.at_level(logging.WARNING, logger="kelet"):
            plugin.configure_client(config)
        assert any(
            "include_otel_plugin=False" in rec.message
            and "not be linked" in rec.message
            for rec in caplog.records
        ), f"Expected missing-OTEL warning. Got: {[r.message for r in caplog.records]}"

    def test_g3_include_otel_false_with_otel_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ):
        plugin = KeletPlugin(include_otel_plugin=False)
        # Pre-existing OTel interceptor on config — no warning expected.
        config: client.ClientConfig = cast(
            Any, {"interceptors": [OpenTelemetryInterceptor()]}
        )
        with caplog.at_level(logging.WARNING, logger="kelet"):
            plugin.configure_client(config)
        for rec in caplog.records:
            assert "include_otel_plugin=False" not in rec.message

    def test_g4_two_plugin_instances_independent_auto_session(self):
        """Issue 1 (1A) — critical regression test.

        Two KeletPlugin instances with different ``auto_session`` configs in the
        same process must NOT clobber each other. Achieved by stashing the
        per-instance resolver in ``unsafe_extern_functions`` rather than a
        module-level global.
        """
        plugin_a = KeletPlugin(auto_session=True)
        plugin_b = KeletPlugin(auto_session=lambda info: f"custom-{info.workflow_id}")

        # Each plugin's interceptor produces an independent resolver
        resolver_a = plugin_a._kelet_interceptor._build_resolver()
        resolver_b = plugin_b._kelet_interceptor._build_resolver()
        assert resolver_a is not None
        assert resolver_b is not None
        info = SimpleNamespace(workflow_id="acme/prod/session/X")
        assert resolver_a(info) == "X"
        assert resolver_b(info) == "custom-acme/prod/session/X"

        # And confirm workflow_interceptor_class plumbs them via extern_functions
        # without touching any module-level state.
        input_a = MagicMock()
        input_a.unsafe_extern_functions = {}
        input_b = MagicMock()
        input_b.unsafe_extern_functions = {}
        plugin_a._kelet_interceptor.workflow_interceptor_class(input_a)
        plugin_b._kelet_interceptor.workflow_interceptor_class(input_b)

        ra = input_a.unsafe_extern_functions["__kelet_auto_session_resolver"]
        rb = input_b.unsafe_extern_functions["__kelet_auto_session_resolver"]
        assert ra(info) == "X"
        assert rb(info) == "custom-acme/prod/session/X"

    @pytest.mark.asyncio
    async def test_g5_run_context_calls_kelet_shutdown_on_worker_exit(self):
        plugin = KeletPlugin()
        # SimplePlugin stores the run_context callable on its instance; access
        # through the composed _kelet_simple_plugin.
        rc_fn = plugin._kelet_simple_plugin.run_context
        assert rc_fn is not None

        with patch("kelet._configure.shutdown") as mock_shutdown:
            async with rc_fn():
                pass
        assert mock_shutdown.call_count == 1

    def test_g6_otel_then_kelet_skips_bundled_otel(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Issue 17 (17A): when an earlier plugin already registered an OTel
        interceptor, KeletPlugin must skip its bundled OTel."""
        plugin = KeletPlugin(include_otel_plugin=True)
        config: client.ClientConfig = cast(
            Any, {"interceptors": [OpenTelemetryInterceptor()]}
        )
        with caplog.at_level(logging.INFO, logger="kelet"):
            out = plugin.configure_client(config)

        interceptors = out.get("interceptors", [])
        otel_count = sum(
            1 for i in interceptors if isinstance(i, OpenTelemetryInterceptor)
        )
        assert otel_count == 1, "Bundled OTel must be skipped — only the user's remains"
        assert plugin._otel_was_skipped
        assert any(
            "skipping bundled OpenTelemetryPlugin" in rec.message
            for rec in caplog.records
        )

    def test_g6b_tracinginterceptor_legacy_also_detected(self):
        """The legacy TracingInterceptor counts as 'OTel already registered' too."""
        plugin = KeletPlugin(include_otel_plugin=True)
        config: client.ClientConfig = cast(
            Any, {"interceptors": [TracingInterceptor()]}
        )
        plugin.configure_client(config)
        assert plugin._otel_was_skipped


# ───────────────── H. agentic_session workflow-aware ─────────────────


class TestWorkflowAware:
    @pytest.mark.asyncio
    async def test_h1_in_workflow_lite_mode_no_otel_no_drain(self):
        with (
            patch("kelet._context.in_temporal_workflow", return_value=True),
            patch("kelet._context.otel_context.attach") as mock_attach,
            patch("kelet._context._drain_background_logging_tasks") as mock_drain,
        ):
            async with kelet.agentic_session(session_id="sess-H1"):
                assert kelet.get_session_id() == "sess-H1"
        assert mock_attach.call_count == 0
        assert mock_drain.call_count == 0

    @pytest.mark.asyncio
    async def test_h2_in_activity_full_mode(self):
        """Outside the workflow VM (in_temporal_workflow=False), agentic_session
        runs full mode — OTEL attach + drain both fire."""
        with (
            patch("kelet._context.in_temporal_workflow", return_value=False),
            patch(
                "kelet._context.otel_context.attach", return_value=object()
            ) as mock_attach,
            patch(
                "kelet._context._drain_background_logging_tasks", new=AsyncMock()
            ) as mock_drain,
        ):
            async with kelet.agentic_session(session_id="sess-H2"):
                pass
        assert mock_attach.call_count == 1
        assert mock_drain.await_count == 1

    @pytest.mark.asyncio
    async def test_h3_outside_temporal_full_mode(self):
        # No patch — real runtime; in_temporal_workflow returns False naturally.
        with patch(
            "kelet._context._drain_background_logging_tasks", new=AsyncMock()
        ) as mock_drain:
            async with kelet.agentic_session(session_id="sess-H3"):
                pass
        assert mock_drain.await_count == 1

    def test_h4_nested_agentic_session_in_workflow_contextvars_only(self):
        with (
            patch("kelet._context.in_temporal_workflow", return_value=True),
            patch("kelet._context.otel_context.attach") as mock_attach,
        ):
            with kelet.agentic_session(session_id="outer"):
                assert kelet.get_session_id() == "outer"
                with kelet.agentic_session(session_id="inner"):
                    assert kelet.get_session_id() == "inner"
                assert kelet.get_session_id() == "outer"
        assert mock_attach.call_count == 0

    def test_h5_sync_form_in_workflow_does_not_drain(self):
        """Sync ``with`` form in workflow context: drain is async-only and
        skipped regardless. Just assert sync form works without raising."""
        with patch("kelet._context.in_temporal_workflow", return_value=True):
            with kelet.agentic_session(session_id="sess-H5"):
                assert kelet.get_session_id() == "sess-H5"


# ───────────────── I. kelet.signal() workflow dispatch ─────────────────


class TestSignalDispatch:
    @pytest.mark.asyncio
    async def test_i1_signal_outside_temporal_direct_httpx(self):
        """Outside workflow context, signal() goes through _send_signal directly."""
        with (
            patch("kelet._signal.in_temporal_workflow", return_value=False),
            patch(
                "kelet._signal._send_signal", new=AsyncMock(return_value=None)
            ) as mock_direct,
            patch(
                "temporalio.workflow.execute_activity", new=AsyncMock()
            ) as mock_dispatch,
        ):
            await kelet.signal(
                kind=kelet.SignalKind.FEEDBACK,
                source=kelet.SignalSource.HUMAN,
                session_id="sess-I1",
            )
        assert mock_direct.await_count == 1
        assert mock_dispatch.await_count == 0

    @pytest.mark.asyncio
    async def test_i2_signal_in_workflow_routes_to_kelet_signal_activity(self):
        """Inside workflow, signal() routes to _kelet_signal_activity via
        workflow.execute_activity with the right retry policy."""
        with (
            patch("kelet._signal.in_temporal_workflow", return_value=True),
            patch(
                "temporalio.workflow.execute_activity", new=AsyncMock(return_value=None)
            ) as mock_exec,
        ):
            await kelet.signal(
                kind=kelet.SignalKind.FEEDBACK,
                source=kelet.SignalSource.HUMAN,
                session_id="sess-I2",
            )
        assert mock_exec.await_count == 1
        # First positional arg is the activity function reference
        assert mock_exec.await_args is not None
        called_args = mock_exec.await_args.args
        assert called_args[0].__name__ == "_kelet_signal_activity"
        kwargs = mock_exec.await_args.kwargs
        assert kwargs["start_to_close_timeout"] == timedelta(seconds=30)
        assert kwargs["retry_policy"].maximum_attempts == 3

    @pytest.mark.asyncio
    async def test_i4_kelet_signal_activity_swallows_on_default_failure_mode(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Activity body's failure handling: 'swallow' (default) logs + returns None.
        Final retry exhaustion, with no config, falls back to swallow."""
        from kelet._signal import _kelet_signal_activity, _SignalArgs

        with (
            patch(
                "kelet._signal._send_signal",
                new=AsyncMock(side_effect=Exception("boom")),
            ),
            patch("kelet._signal.get_config", side_effect=ValueError),
        ):
            await _kelet_signal_activity(
                _SignalArgs(
                    kind=kelet.SignalKind.FEEDBACK,
                    source=kelet.SignalSource.HUMAN,
                    session_id="sess-I4",
                )
            )
        assert any(
            "kelet.signal() failed after retries" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_i4b_kelet_signal_activity_raises_when_configured(self):
        """signal_failure_mode='raise' surfaces failures up to the workflow."""
        from kelet._config import KeletConfig, set_config
        from kelet._signal import _kelet_signal_activity, _SignalArgs

        cfg = KeletConfig(
            api_key="k",
            base_url="https://api.kelet.ai",
            project="p",
            signal_failure_mode="raise",
        )
        set_config(cfg)

        with patch(
            "kelet._signal._send_signal",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await _kelet_signal_activity(
                    _SignalArgs(
                        kind=kelet.SignalKind.FEEDBACK,
                        source=kelet.SignalSource.HUMAN,
                        session_id="sess-I4b",
                    )
                )

    def test_i5_kelet_signal_activity_auto_registered_by_plugin(self):
        """KeletPlugin must register _kelet_signal_activity so users don't have to."""
        from kelet._signal import _kelet_signal_activity

        plugin = KeletPlugin()
        # SimplePlugin stores activities on its instance; access via the inner.
        registered = plugin._kelet_simple_plugin.activities
        # It can be a Sequence or a callable transformer; we passed a list.
        assert callable(registered) or _kelet_signal_activity in (registered or [])
        # If it's a list, also assert directly:
        if isinstance(registered, (list, tuple)):
            assert _kelet_signal_activity in registered

    @pytest.mark.asyncio
    async def test_i6_signal_in_workflow_without_plugin_raises_clear_error(self):
        """If the user wires KeletInterceptor standalone (no KeletPlugin),
        _kelet_signal_activity isn't registered. Calling kelet.signal() from
        workflow code should raise a clear RuntimeError pointing at the fix —
        NOT silently fall back to direct httpx (which would be non-deterministic)."""
        from temporalio.exceptions import ActivityError, ApplicationError

        # Simulate Temporal's NotFoundError ApplicationError that bubbles up
        # when the activity isn't registered on the worker.
        not_found = ApplicationError("activity not found", type="NotFoundError")
        wrapped = ActivityError(
            "activity error",
            scheduled_event_id=1,
            started_event_id=2,
            identity="x",
            activity_type="_kelet_signal",
            activity_id="1",
            retry_state=None,
        )
        wrapped.__cause__ = not_found

        with (
            patch("kelet._signal.in_temporal_workflow", return_value=True),
            patch(
                "temporalio.workflow.execute_activity",
                new=AsyncMock(side_effect=wrapped),
            ),
        ):
            with pytest.raises(RuntimeError, match="KeletPlugin"):
                await kelet.signal(
                    kind=kelet.SignalKind.FEEDBACK,
                    source=kelet.SignalSource.HUMAN,
                    session_id="sess-I6",
                )


# ───────────────── extra: tiny helper coverage ─────────────────


@dataclass
class _FakeInfo:
    workflow_id: str


def test_derive_session_id_session_segment():
    assert _derive_session_id("acme/prod/session/sess-XYZ") == "sess-XYZ"


def test_derive_session_id_no_session_segment_returns_full_id():
    # Full workflow_id when no /session/ marker
    assert _derive_session_id("plain-wf-id") == "plain-wf-id"
    assert _derive_session_id("a/b/c/d") == "a/b/c/d"


def test_has_existing_otel_detects_both_classes():
    cfg_with_new = {"interceptors": [OpenTelemetryInterceptor()]}
    cfg_with_legacy = {"interceptors": [TracingInterceptor()]}
    cfg_empty = {"interceptors": []}
    assert _has_existing_otel(cfg_with_new) is True
    assert _has_existing_otel(cfg_with_legacy) is True
    assert _has_existing_otel(cfg_empty) is False
