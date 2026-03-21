"""Integration tests: agentic_session propagates attributes to child spans via _KeletSpanProcessor."""

import pytest
from typing import Sequence
from opentelemetry import baggage as otel_baggage, context as otel_context, trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from kelet._context import agentic_session, SESSION_ID_ATTR, USER_ID_ATTR
from kelet._configure import _KeletSpanProcessor


class _CollectingExporter(SpanExporter):
    """Simple exporter that collects spans in a list for assertions."""

    def __init__(self):
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


@pytest.fixture
def collector():
    return _CollectingExporter()


@pytest.fixture
def tracer(collector):
    """Tracer wired through _KeletSpanProcessor -> SimpleSpanProcessor -> collector."""
    provider = TracerProvider()
    inner = SimpleSpanProcessor(collector)
    processor = _KeletSpanProcessor(inner, project="test-project")
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    yield provider.get_tracer("test")
    provider.shutdown()


def _attrs(span):
    """Extract attributes dict from a finished span."""
    return dict(span.attributes) if span.attributes else {}


# --- Propagation tests ---


def test_child_span_gets_session_attributes(tracer, collector):
    """Child spans created inside agentic_session get session/user attributes."""
    with tracer.start_as_current_span("parent"):
        with agentic_session(session_id="sess-1", user_id="user-1"):
            with tracer.start_as_current_span("child"):
                pass  # child span started while ContextVars are set

    spans = {s.name: s for s in collector.spans}
    child = spans["child"]
    assert _attrs(child)[SESSION_ID_ATTR] == "sess-1"
    assert _attrs(child)[USER_ID_ATTR] == "user-1"


def test_deeply_nested_spans_inherit(tracer, collector):
    """Attributes propagate through multiple nesting levels."""
    with tracer.start_as_current_span("root"):
        with agentic_session(session_id="deep-sess", user_id="deep-user"):
            with tracer.start_as_current_span("mid"):
                with tracer.start_as_current_span("leaf"):
                    pass

    spans = {s.name: s for s in collector.spans}
    for name in ("mid", "leaf"):
        assert _attrs(spans[name])[SESSION_ID_ATTR] == "deep-sess"
        assert _attrs(spans[name])[USER_ID_ATTR] == "deep-user"


def test_span_outside_session_has_no_session_attrs(tracer, collector):
    """Spans created outside agentic_session don't get session/user attributes."""
    with tracer.start_as_current_span("outside"):
        pass

    spans = {s.name: s for s in collector.spans}
    attrs = _attrs(spans["outside"])
    assert SESSION_ID_ATTR not in attrs
    assert USER_ID_ATTR not in attrs


def test_only_session_id_when_no_user_id(tracer, collector):
    """When user_id is omitted, only session_id propagates."""
    with tracer.start_as_current_span("parent"):
        with agentic_session(session_id="sess-only"):
            with tracer.start_as_current_span("child"):
                pass

    spans = {s.name: s for s in collector.spans}
    child_attrs = _attrs(spans["child"])
    assert child_attrs[SESSION_ID_ATTR] == "sess-only"
    assert USER_ID_ATTR not in child_attrs


def test_kelet_project_always_set(tracer, collector):
    """kelet.project attribute is set on every span regardless of session."""
    with tracer.start_as_current_span("any-span"):
        pass

    assert len(collector.spans) == 1
    assert _attrs(collector.spans[0])["kelet.project"] == "test-project"


def test_session_attrs_dont_leak_after_exit(tracer, collector):
    """After agentic_session exits, new spans don't get session attributes."""
    with tracer.start_as_current_span("inside-parent"):
        with agentic_session(session_id="temp-sess", user_id="temp-user"):
            with tracer.start_as_current_span("inside-child"):
                pass

    # Now outside the session
    with tracer.start_as_current_span("after-session"):
        pass

    spans = {s.name: s for s in collector.spans}
    inside_attrs = _attrs(spans["inside-child"])
    assert inside_attrs[SESSION_ID_ATTR] == "temp-sess"

    after_attrs = _attrs(spans["after-session"])
    assert SESSION_ID_ATTR not in after_attrs
    assert USER_ID_ATTR not in after_attrs


def test_child_span_gets_agent_name(tracer, collector):
    """agent_name propagates to child spans via processor."""
    from kelet._context import AGENT_NAME_ATTR, agent
    with tracer.start_as_current_span("parent"):
        with agentic_session(session_id="sess-1", user_id="user-1"):
            with agent(name="support-bot"):
                with tracer.start_as_current_span("child"):
                    pass

    spans = {s.name: s for s in collector.spans}
    child_attrs = _attrs(spans["child"])
    assert child_attrs[AGENT_NAME_ATTR] == "support-bot"


def test_agent_child_span_has_agent_name_and_session(tracer, collector):
    """agent() stamps child spans with agent name and session via processor."""
    from kelet._context import AGENT_NAME_ATTR, agent
    with agentic_session(session_id="sess-2"):
        with agent(name="my-agent"):
            with tracer.start_as_current_span("child-in-agent"):
                pass

    spans = {s.name: s for s in collector.spans}
    child_attrs = _attrs(spans["child-in-agent"])
    assert child_attrs[AGENT_NAME_ATTR] == "my-agent"
    assert child_attrs[SESSION_ID_ATTR] == "sess-2"


def test_agent_name_without_user_id(tracer, collector):
    """agent(name=...) + session_id works fine without user_id."""
    from kelet._context import AGENT_NAME_ATTR, agent
    with tracer.start_as_current_span("parent"):
        with agentic_session(session_id="sess-3"):
            with agent(name="no-user-agent"):
                with tracer.start_as_current_span("child"):
                    pass

    spans = {s.name: s for s in collector.spans}
    child_attrs = _attrs(spans["child"])
    assert child_attrs[SESSION_ID_ATTR] == "sess-3"
    assert child_attrs[AGENT_NAME_ATTR] == "no-user-agent"
    assert USER_ID_ATTR not in child_attrs


def test_agent_name_doesnt_leak_after_exit(tracer, collector):
    """After agent() exits, new spans don't get AGENT_NAME_ATTR."""
    from kelet._context import AGENT_NAME_ATTR, agent
    with tracer.start_as_current_span("inside-parent"):
        with agentic_session(session_id="sess-4"):
            with agent(name="temp-agent"):
                with tracer.start_as_current_span("inside-child"):
                    pass

    with tracer.start_as_current_span("after-session"):
        pass

    spans = {s.name: s for s in collector.spans}
    assert _attrs(spans["inside-child"])[AGENT_NAME_ATTR] == "temp-agent"
    assert AGENT_NAME_ATTR not in _attrs(spans["after-session"])


def test_span_outside_session_has_no_agent_name(tracer, collector):
    """Spans outside agentic_session have no AGENT_NAME_ATTR."""
    from kelet._context import AGENT_NAME_ATTR
    with tracer.start_as_current_span("outside"):
        pass

    spans = {s.name: s for s in collector.spans}
    assert AGENT_NAME_ATTR not in _attrs(spans["outside"])


# --- Project override tests ---


def test_project_override_in_agentic_session(tracer, collector):
    """agentic_session with project= overrides kelet.project on spans."""
    with agentic_session(session_id="sess-X", project="override-project"):
        with tracer.start_as_current_span("child"):
            pass

    spans = {s.name: s for s in collector.spans}
    assert _attrs(spans["child"])["kelet.project"] == "override-project"


def test_project_global_used_when_no_override(tracer, collector):
    """agentic_session without project= uses the global project."""
    with agentic_session(session_id="sess-X"):
        with tracer.start_as_current_span("child"):
            pass

    spans = {s.name: s for s in collector.spans}
    assert _attrs(spans["child"])["kelet.project"] == "test-project"


def test_baggage_propagates_session_user_project(tracer, collector):
    """Simulated cross-process: baggage set manually stamps attributes without agentic_session."""
    ctx = otel_context.get_current()
    ctx = otel_baggage.set_baggage("kelet.session_id", "baggage-sess", context=ctx)
    ctx = otel_baggage.set_baggage("kelet.user_id", "baggage-user", context=ctx)
    ctx = otel_baggage.set_baggage("kelet.project", "baggage-project", context=ctx)
    token = otel_context.attach(ctx)
    try:
        with tracer.start_as_current_span("downstream"):
            pass
    finally:
        otel_context.detach(token)

    spans = {s.name: s for s in collector.spans}
    attrs = _attrs(spans["downstream"])
    assert attrs[SESSION_ID_ATTR] == "baggage-sess"
    assert attrs[USER_ID_ATTR] == "baggage-user"
    assert attrs["kelet.project"] == "baggage-project"


def test_project_override_doesnt_leak_after_exit(tracer, collector):
    """project override is cleaned up after agentic_session exits."""
    with agentic_session(session_id="s1", project="temp-project"):
        with tracer.start_as_current_span("inside"):
            pass
    with tracer.start_as_current_span("outside"):
        pass

    spans = {s.name: s for s in collector.spans}
    assert _attrs(spans["inside"])["kelet.project"] == "temp-project"
    assert _attrs(spans["outside"])["kelet.project"] == "test-project"


def test_inner_session_without_user_id_does_not_inherit_outer_via_baggage(tracer, collector):
    """Inner agentic_session without user_id should not inherit outer user_id from baggage."""
    with agentic_session(session_id="outer", user_id="outer-user"):
        with agentic_session(session_id="inner"):  # no user_id
            with tracer.start_as_current_span("inner-span"):
                pass

    spans = {s.name: s for s in collector.spans}
    attrs = _attrs(spans["inner-span"])
    assert attrs.get(USER_ID_ATTR) is None


def test_nested_sessions_restore_outer_project(tracer, collector):
    """Nested agentic_session with different projects restores outer project on exit."""
    with agentic_session(session_id="outer", project="outer-project"):
        with tracer.start_as_current_span("outer-span"):
            pass
        with agentic_session(session_id="inner", project="inner-project"):
            with tracer.start_as_current_span("inner-span"):
                pass
        with tracer.start_as_current_span("after-inner"):
            pass

    spans = {s.name: s for s in collector.spans}
    assert _attrs(spans["outer-span"])["kelet.project"] == "outer-project"
    assert _attrs(spans["inner-span"])["kelet.project"] == "inner-project"
    assert _attrs(spans["after-inner"])["kelet.project"] == "outer-project"


def test_inner_session_clears_outer_user_id_from_baggage(tracer, collector):
    """Inner agentic_session without user_id explicitly removes outer user_id from baggage.

    Cross-process scenario: the inner session must clear kelet.user_id from the
    baggage it propagates downstream, not merely suppress it in-process.
    We simulate this by reading baggage directly inside the inner session context.
    """
    outer_baggage_user: list[str | None] = []
    inner_baggage_user: list[str | None] = []

    with agentic_session(session_id="outer", user_id="outer-user"):
        outer_baggage_user.append(
            otel_baggage.get_baggage("kelet.user_id", context=otel_context.get_current())
        )
        with agentic_session(session_id="inner"):  # no user_id
            inner_baggage_user.append(
                otel_baggage.get_baggage("kelet.user_id", context=otel_context.get_current())
            )
            with tracer.start_as_current_span("inner-span"):
                pass

    # Outer session should have user_id in baggage
    assert outer_baggage_user[0] == "outer-user"
    # Inner session must have cleared it from baggage (cross-process safety)
    assert inner_baggage_user[0] is None


def test_inner_session_clears_outer_project_from_baggage(tracer, collector):
    """Inner agentic_session without project explicitly removes outer project from baggage."""
    outer_baggage_proj: list[str | None] = []
    inner_baggage_proj: list[str | None] = []

    with agentic_session(session_id="outer", project="outer-project"):
        outer_baggage_proj.append(
            otel_baggage.get_baggage("kelet.project", context=otel_context.get_current())
        )
        with agentic_session(session_id="inner"):  # no project
            inner_baggage_proj.append(
                otel_baggage.get_baggage("kelet.project", context=otel_context.get_current())
            )
            with tracer.start_as_current_span("inner-span"):
                pass

    assert outer_baggage_proj[0] == "outer-project"
    assert inner_baggage_proj[0] is None
