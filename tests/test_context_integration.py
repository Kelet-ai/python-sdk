"""Integration tests: agentic_session propagates attributes to child spans via _KeletSpanProcessor."""

import pytest
from typing import Sequence
from opentelemetry import trace
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
