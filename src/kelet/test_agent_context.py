"""Tests for agent() context manager and agentic_session() decorator support."""

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from kelet._context import (
    agent,
    agentic_session,
    get_agent_name,
    get_session_id,
    get_user_id,
    AGENT_NAME_ATTR,
    SESSION_ID_ATTR,
    USER_ID_ATTR,
)
from kelet._configure import _KeletSpanProcessor


# ---------------------------------------------------------------------------
# Module-level TracerProvider setup
#
# otel_trace.set_tracer_provider() can only be called once per process (any
# subsequent calls are silently ignored with a warning).  We therefore set up
# TWO providers at module import time:
#   • _plain_provider  – backed by SimpleSpanProcessor (no kelet stamping)
#   • _kelet_provider  – backed by _KeletSpanProcessor wrapping a
#                        SimpleSpanProcessor
#
# Each test fixture clears + returns the appropriate InMemorySpanExporter so
# every test starts with an empty span list.
# ---------------------------------------------------------------------------

_plain_exporter = InMemorySpanExporter()
_kelet_exporter = InMemorySpanExporter()

# Build the providers.
_plain_provider = TracerProvider()
_plain_provider.add_span_processor(SimpleSpanProcessor(_plain_exporter))

_kelet_provider = TracerProvider()
_kelet_provider.add_span_processor(
    _KeletSpanProcessor(SimpleSpanProcessor(_kelet_exporter), project="test")
)

# Install the plain provider as the global one.  Tests that need the kelet
# provider use it directly via otel_trace.set_tracer_provider() – but because
# that call is a no-op after the first call we instead swap providers by
# temporarily replacing the global provider inside the fixture using the
# internal _TRACER_PROVIDER global.  The cleanest portable approach that
# works across otel versions is to set it once (plain) and for kelet tests
# supply the provider explicitly to the tracer instead.
otel_trace.set_tracer_provider(_plain_provider)


@pytest.fixture(autouse=True)
def _reset_exporters():
    """Clear both exporters before every test so spans don't bleed across tests."""
    _plain_exporter.clear()
    _kelet_exporter.clear()
    yield
    _plain_exporter.clear()
    _kelet_exporter.clear()


@pytest.fixture
def span_exporter():
    """Return the plain in-memory exporter (global provider)."""
    return _plain_exporter


@pytest.fixture
def kelet_span_exporter():
    """Return the kelet in-memory exporter (kelet provider)."""
    return _kelet_exporter


def _kelet_tracer(name: str = "test"):
    """Return a tracer backed by the kelet provider."""
    return _kelet_provider.get_tracer(name)


# ---------------------------------------------------------------------------
# Patch agent() so that kelet-provider tests use _kelet_provider's tracer.
#
# agent() internally calls trace.get_tracer("kelet") which resolves against
# the global (plain) provider.  For tests that need attribute-stamping we
# monkeypatch _context.trace so it returns a tracer from _kelet_provider.
# ---------------------------------------------------------------------------

import kelet._context as _ctx_module  # noqa: E402


@pytest.fixture
def use_kelet_provider(monkeypatch):
    """Redirect _context.trace.get_tracer to use the kelet provider."""
    import opentelemetry.trace as _otel  # noqa: F401

    class _PatchedTrace:
        @staticmethod
        def get_tracer(name, *args, **kwargs):
            return _kelet_provider.get_tracer(name, *args, **kwargs)

        # Proxy everything else through the real module.
        def __getattr__(self, item):
            return getattr(_otel, item)

    monkeypatch.setattr(_ctx_module, "trace", _PatchedTrace())
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentSpan:
    def test_agent_creates_span_with_correct_attributes(self, span_exporter):
        with agent(name="support-bot"):
            pass

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "agent support-bot"
        assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert span.attributes[AGENT_NAME_ATTR] == "support-bot"

    def test_agent_span_is_parent_of_child_spans(self, span_exporter):
        tracer = otel_trace.get_tracer("test")
        with agent(name="my-agent"):
            child = tracer.start_span("llm-call")
            child.end()

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        agent_span = next(s for s in spans if s.name == "agent my-agent")
        child_span = next(s for s in spans if s.name == "llm-call")
        assert child_span.parent.span_id == agent_span.context.span_id

    def test_agent_name_stamped_on_child_spans_via_processor(
        self, kelet_span_exporter, use_kelet_provider
    ):
        tracer = _kelet_tracer("test")
        with agent(name="classifier"):
            child = tracer.start_span("llm-call")
            child.end()

        spans = kelet_span_exporter.get_finished_spans()
        child_span = next(s for s in spans if s.name == "llm-call")
        assert child_span.attributes[AGENT_NAME_ATTR] == "classifier"

    def test_agent_inherits_session_id_from_agentic_session(self, span_exporter):
        with agentic_session(session_id="sess-123"):
            with agent(name="responder"):
                pass

        spans = span_exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "agent responder")
        assert agent_span.attributes[SESSION_ID_ATTR] == "sess-123"

    def test_agent_sets_get_agent_name(self):
        assert get_agent_name() is None
        with agent(name="my-bot"):
            assert get_agent_name() == "my-bot"
        assert get_agent_name() is None

    def test_agent_decorator_sync(self, span_exporter):
        @agent(name="sync-bot")
        def run_sync():
            assert get_agent_name() == "sync-bot"
            return "done"

        result = run_sync()
        assert result == "done"
        spans = span_exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "agent sync-bot")
        assert agent_span.attributes[AGENT_NAME_ATTR] == "sync-bot"

    @pytest.mark.asyncio
    async def test_agent_decorator_async(self, span_exporter):
        @agent(name="async-bot")
        async def run_async():
            assert get_agent_name() == "async-bot"
            return "async-done"

        result = await run_async()
        assert result == "async-done"
        spans = span_exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "agent async-bot")
        assert agent_span.attributes[AGENT_NAME_ATTR] == "async-bot"

    def test_multi_agent_in_one_session(self, span_exporter):
        with agentic_session(session_id="sess-multi"):
            with agent(name="classifier"):
                pass
            with agent(name="responder"):
                pass

        spans = span_exporter.get_finished_spans()
        classifier_span = next(s for s in spans if s.name == "agent classifier")
        responder_span = next(s for s in spans if s.name == "agent responder")

        assert classifier_span.attributes[AGENT_NAME_ATTR] == "classifier"
        assert responder_span.attributes[AGENT_NAME_ATTR] == "responder"
        assert classifier_span.attributes[SESSION_ID_ATTR] == "sess-multi"
        assert responder_span.attributes[SESSION_ID_ATTR] == "sess-multi"
        # No cross-contamination
        assert get_agent_name() is None

    def test_no_agent_name_leak_after_exit(self):
        with agent(name="temp-agent"):
            assert get_agent_name() == "temp-agent"
        assert get_agent_name() is None

    def test_agent_span_ends_on_sync_exception(self, span_exporter):
        """agent() span ends properly even when body raises."""
        with pytest.raises(ValueError):
            with agent(name="failing-bot"):
                raise ValueError("oops")

        spans = span_exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "agent failing-bot")
        assert agent_span is not None
        assert get_agent_name() is None  # context cleaned up

    @pytest.mark.asyncio
    async def test_agent_span_ends_on_async_exception(self, span_exporter):
        """agent() span ends properly when async body raises."""
        with pytest.raises(ValueError):
            async with agent(name="async-failing-bot"):
                raise ValueError("async oops")

        spans = span_exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "agent async-failing-bot")
        assert agent_span is not None
        assert get_agent_name() is None  # context cleaned up

    def test_agent_decorator_sync_exception_cleans_up(self):
        """@agent() decorator cleans up context even when function raises."""

        @agent(name="decorator-failing-bot")
        def failing():
            raise RuntimeError("decorator fail")

        with pytest.raises(RuntimeError):
            failing()

        assert get_agent_name() is None

    @pytest.mark.asyncio
    async def test_agent_async_context_manager(self, span_exporter):
        async with agent(name="async-cm-bot"):
            assert get_agent_name() == "async-cm-bot"

        assert get_agent_name() is None
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "agent async-cm-bot"
        assert spans[0].attributes[AGENT_NAME_ATTR] == "async-cm-bot"


class TestAgenticSessionDualStyle:
    def test_agentic_session_context_manager_sync(self):
        with agentic_session(session_id="sess-sync"):
            assert get_session_id() == "sess-sync"
        assert get_session_id() is None

    @pytest.mark.asyncio
    async def test_agentic_session_context_manager_async(self):
        async with agentic_session(session_id="sess-async"):
            assert get_session_id() == "sess-async"
        assert get_session_id() is None

    def test_agentic_session_decorator_sync(self):
        @agentic_session(session_id="sess-dec-sync")
        def handle():
            assert get_session_id() == "sess-dec-sync"
            return "ok"

        result = handle()
        assert result == "ok"
        assert get_session_id() is None

    @pytest.mark.asyncio
    async def test_agentic_session_decorator_async(self):
        @agentic_session(session_id="sess-dec-async")
        async def handle():
            assert get_session_id() == "sess-dec-async"
            return "async-ok"

        result = await handle()
        assert result == "async-ok"
        assert get_session_id() is None

    def test_agentic_session_user_id(self):
        with agentic_session(session_id="sess-user", user_id="user-42"):
            assert get_session_id() == "sess-user"
            assert get_user_id() == "user-42"
        assert get_session_id() is None

    def test_agentic_session_no_leak_after_exit(self):
        assert get_session_id() is None
        with agentic_session(session_id="sess-temp"):
            pass
        assert get_session_id() is None

    def test_agentic_session_stamps_current_span_attributes(self, span_exporter):
        """agentic_session sets session_id on the current span if one is recording."""
        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("outer"):
            with agentic_session(session_id="sess-stamp"):
                pass  # _enter() sets attribute on the outer span

        spans = span_exporter.get_finished_spans()
        outer = next(s for s in spans if s.name == "outer")
        assert outer.attributes[SESSION_ID_ATTR] == "sess-stamp"

    def test_agentic_session_kelet_processor_stamps_child_spans(
        self, kelet_span_exporter, use_kelet_provider
    ):
        """KeletSpanProcessor stamps session_id on every span started inside the session."""
        tracer = _kelet_tracer("test")
        with agentic_session(session_id="sess-kelet"):
            child = tracer.start_span("inner-call")
            child.end()

        spans = kelet_span_exporter.get_finished_spans()
        inner = next(s for s in spans if s.name == "inner-call")
        assert inner.attributes[SESSION_ID_ATTR] == "sess-kelet"

    def test_nested_session_no_user_id_inherits_outer(self):
        """Inner session without user_id inherits the outer user_id."""
        with agentic_session(session_id="outer", user_id="alice"):
            assert get_user_id() == "alice"
            with agentic_session(session_id="inner"):  # no user_id — inherits outer
                assert get_user_id() == "alice"  # inherited from outer
            assert get_user_id() == "alice"  # outer restored
