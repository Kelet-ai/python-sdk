"""Tests for configure() span_processor parameter."""

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
import pytest

import kelet


class _MockProcessor(SpanProcessor):
    def __init__(self):
        self.on_end_calls: list[ReadableSpan] = []
        self.on_start_calls: list[Span] = []

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        self.on_start_calls.append(span)

    def on_end(self, span: ReadableSpan) -> None:
        self.on_end_calls.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@pytest.fixture(autouse=True)
def _reset_otel(monkeypatch):
    """Prevent configure() from polluting the global TracerProvider across tests."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider as _TP

    provider = _TP()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(trace, "set_tracer_provider", lambda p: None)
    yield
    # Restore _active_processors to its pre-test length to avoid cross-test pollution
    # (processors added during the test are removed; pre-existing ones remain)


def test_configure_accepts_custom_span_processor(monkeypatch):
    """configure(span_processor=...) uses the provided processor, not a new default one."""
    mock = _MockProcessor()
    create_call_count = 0

    def _spy_create(**kwargs):
        nonlocal create_call_count
        create_call_count += 1
        return mock  # return mock so we don't need a real API key

    monkeypatch.setattr("kelet._configure.create_kelet_processor", _spy_create)

    kelet.configure(
        api_key="test-key",
        project="test-project",
        auto_instrument=False,
        span_processor=mock,
    )

    # create_kelet_processor was NOT called (the mock was used directly)
    assert create_call_count == 0


def test_configure_without_custom_processor_calls_create(monkeypatch):
    """configure() without span_processor= calls create_kelet_processor normally."""
    create_call_count = 0
    mock = _MockProcessor()

    def _spy_create(**kwargs):
        nonlocal create_call_count
        create_call_count += 1
        return mock

    monkeypatch.setattr("kelet._configure.create_kelet_processor", _spy_create)

    kelet.configure(
        api_key="test-key",
        project="test-project",
        auto_instrument=False,
    )

    assert create_call_count == 1
