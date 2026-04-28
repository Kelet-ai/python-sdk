"""Integration-ish tests for ``ClaudeAgentSDKInstrumentor``.

These tests run a full loop: build a stub ``claude_agent_sdk`` module,
install the instrumentor, drive a fake stream through the wrapped entry
point, and assert that the OTLP log records the server would accept land
in the in-memory exporter.

The goal is to catch issues the unit-level ``test_reasoning_observer.py``
can't — in particular, the scope-name contract (the Kelet server filters
``/api/logs`` on ``startswith("com.anthropic.claude_code")``) and the
integration-scoped LoggerProvider plumbing (must not clobber the global
provider).
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest
from opentelemetry._logs import (
    NoOpLoggerProvider,
    get_logger_provider,
    set_logger_provider,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)

from kelet._integrations.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
    _reasoning_observer,
)
from kelet._integrations.claude_agent_sdk._reasoning_observer import (
    REASONING_EVENT_NAME,
    REASONING_SCOPE_NAME,
)


class _ThinkingBlock:
    def __init__(self, thinking: str, signature: str = "") -> None:
        self.thinking = thinking
        self.signature = signature


class _AssistantMessage:
    def __init__(
        self,
        content: list[Any],
        message_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.content = content
        if message_id is not None:
            self.message_id = message_id
        if session_id is not None:
            self.session_id = session_id


# ---------------------------------------------------------------------------
# Fake claude_agent_sdk module — installed into sys.modules so wrapt can
# patch ``claude_agent_sdk.query`` and ``claude_agent_sdk.client.ClaudeSDKClient``.
# ---------------------------------------------------------------------------


_STREAM: list[Any] = []


async def _fake_query(*_args: Any, **_kwargs: Any):
    for item in _STREAM:
        yield item


class _FakeClaudeSDKClient:
    async def receive_messages(self):
        for item in _STREAM:
            yield item

    async def receive_response(self):
        for item in _STREAM:
            yield item


@pytest.fixture
def fake_sdk_module():
    """Install a minimal ``claude_agent_sdk`` package in ``sys.modules``.

    The instrumentor patches three names:
    * ``claude_agent_sdk.query``
    * ``claude_agent_sdk.client.ClaudeSDKClient.receive_messages``
    * ``claude_agent_sdk.client.ClaudeSDKClient.receive_response``

    All three need to exist on the fake modules for ``wrap_function_wrapper``
    to succeed. The fixture tears everything down so tests remain isolated.
    """
    pkg = types.ModuleType("claude_agent_sdk")
    pkg.query = _fake_query  # type: ignore[attr-defined]

    client_mod = types.ModuleType("claude_agent_sdk.client")
    client_mod.ClaudeSDKClient = _FakeClaudeSDKClient  # type: ignore[attr-defined]

    sys.modules["claude_agent_sdk"] = pkg
    sys.modules["claude_agent_sdk.client"] = client_mod
    try:
        yield pkg, client_mod
    finally:
        sys.modules.pop("claude_agent_sdk", None)
        sys.modules.pop("claude_agent_sdk.client", None)


@pytest.fixture
def in_memory_provider():
    """Install a fresh LoggerProvider that captures emitted records."""
    exporter = InMemoryLogExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))

    # Also swap the global provider so tests can assert it isn't clobbered
    # when the instrumentor uses its own dedicated provider.
    original_global = get_logger_provider()
    set_logger_provider(NoOpLoggerProvider())
    try:
        yield provider, exporter
    finally:
        provider.shutdown()
        # Reset the module-local override in case the test forgot to uninstrument.
        _reasoning_observer.reset_logger()
        # Restore what we found — OTel SDK silently ignores re-sets, but we
        # try anyway so other tests in the session see the intended state.
        try:
            set_logger_provider(original_global)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instrument_emits_under_cc_scope_name(
    fake_sdk_module: Any, in_memory_provider: Any
) -> None:
    """The emitted log record's instrumentation_scope.name must start with
    the CC prefix so the Kelet server's /api/logs filter lets it through."""
    provider, exporter = in_memory_provider
    _STREAM.clear()
    _STREAM.append(
        _AssistantMessage(
            content=[_ThinkingBlock("hello", signature="sig-1")],
            message_id="msg_abc",
            session_id="sess_xyz",
        )
    )

    inst = ClaudeAgentSDKInstrumentor()
    # skip_dep_check: the fake module has no version metadata, so OTel's
    # ``instrumentation_dependencies`` matcher would reject it. The real
    # package has proper versioning in prod.
    inst.instrument(logger_provider=provider, skip_dep_check=True)
    try:
        import claude_agent_sdk

        async def run() -> None:
            async for _ in claude_agent_sdk.query():
                pass

        asyncio.run(run())
        provider.force_flush()

        records = exporter.get_finished_logs()
        assert len(records) == 1, f"Expected 1 record, got {len(records)}"
        data = records[0]
        # Pick up both names: LogData wraps a LogRecord + InstrumentationScope
        scope_name = (
            getattr(data.instrumentation_scope, "name", None)
            or getattr(data, "name", None)
            or ""
        )
        assert scope_name == REASONING_SCOPE_NAME
        assert scope_name.startswith("com.anthropic.claude_code")
        assert data.log_record.body == REASONING_EVENT_NAME
        attrs = data.log_record.attributes or {}
        assert attrs.get("reasoning.text") == "hello"
        assert attrs.get("reasoning.message_id") == "msg_abc"
        assert attrs.get("session.id") == "sess_xyz"
    finally:
        inst.uninstrument()


def test_instrument_does_not_clobber_global_logger_provider(
    fake_sdk_module: Any, in_memory_provider: Any
) -> None:
    """Installing the instrumentor with an explicit ``logger_provider`` must
    not replace whatever's registered as the OTel global provider."""
    provider, _exporter = in_memory_provider
    global_before = get_logger_provider()

    inst = ClaudeAgentSDKInstrumentor()
    inst.instrument(logger_provider=provider, skip_dep_check=True)
    try:
        assert get_logger_provider() is global_before, (
            "Instrumentor replaced the global LoggerProvider — host app "
            "pipelines (Datadog/Sentry/etc.) would be clobbered."
        )
    finally:
        inst.uninstrument()


def test_uninstrument_restores_original_callables(fake_sdk_module: Any) -> None:
    """Symmetry: wrap + unwrap restores the original module attributes."""
    import claude_agent_sdk
    from claude_agent_sdk import client as _client_mod

    original_query = claude_agent_sdk.query
    original_receive = _client_mod.ClaudeSDKClient.receive_messages
    original_response = _client_mod.ClaudeSDKClient.receive_response

    inst = ClaudeAgentSDKInstrumentor()
    inst.instrument(skip_dep_check=True)
    # Sanity: instrument swapped the attributes.
    assert claude_agent_sdk.query is not original_query
    inst.uninstrument()

    assert claude_agent_sdk.query is original_query
    assert _client_mod.ClaudeSDKClient.receive_messages is original_receive
    assert _client_mod.ClaudeSDKClient.receive_response is original_response


def test_uninstrument_resets_module_logger(
    fake_sdk_module: Any, in_memory_provider: Any
) -> None:
    """After uninstrument the module-local logger is re-resolved against
    the global provider, so the provider-specific logger from the
    integration-scoped provider is no longer the active one."""
    provider, exporter = in_memory_provider

    inst = ClaudeAgentSDKInstrumentor()
    inst.instrument(logger_provider=provider, skip_dep_check=True)
    scoped_logger = _reasoning_observer._LOGGER
    # Emit while scoped — record should land in our exporter.
    _reasoning_observer._LOGGER.emit(
        body="test-while-scoped", attributes={}, event_name="kelet.reasoning"
    )
    provider.force_flush()
    scoped_count_before = len(exporter.get_finished_logs())

    inst.uninstrument()
    assert _reasoning_observer._LOGGER is not scoped_logger
    # Emit again (now via the global NoOp) — exporter count shouldn't change.
    _reasoning_observer._LOGGER.emit(
        body="test-post-uninstrument", attributes={}, event_name="kelet.reasoning"
    )
    provider.force_flush()
    assert len(exporter.get_finished_logs()) == scoped_count_before


def test_instrument_without_logger_provider_still_wraps(
    fake_sdk_module: Any,
) -> None:
    """When no LoggerProvider is supplied and Kelet isn't configured, the
    instrumentor still wraps streams — emissions just go to the global
    (no-op) provider and silently drop."""
    inst = ClaudeAgentSDKInstrumentor()
    inst.instrument()  # no logger_provider, no KELET_API_KEY in env for CI
    try:
        import claude_agent_sdk

        _STREAM.clear()
        _STREAM.append(
            _AssistantMessage(
                content=[_ThinkingBlock("silent")], message_id="m"
            )
        )

        async def run() -> list[Any]:
            items: list[Any] = []
            async for item in claude_agent_sdk.query():
                items.append(item)
            return items

        items = asyncio.run(run())
        # Users still iterate normally — the observer is transparent.
        assert len(items) == 1
    finally:
        inst.uninstrument()
