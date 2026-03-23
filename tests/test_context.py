"""Tests for context helpers (ContextVar-based session/user management)."""

from unittest.mock import MagicMock, patch

from opentelemetry import baggage as otel_baggage, context as otel_context

from kelet._context import (
    _metadata_kwargs_var,
    agentic_session,
    get_metadata_kwargs,
    get_session_id,
    get_user_id,
    get_trace_id,
)


def test_get_session_id_default_none():
    """Returns None when no session is active."""
    assert get_session_id() is None


def test_get_user_id_default_none():
    """Returns None when no user is set."""
    assert get_user_id() is None


def test_agentic_session_sets_session_id():
    """agentic_session sets session_id in ContextVar."""
    with agentic_session(session_id="sess-1"):
        assert get_session_id() == "sess-1"
    assert get_session_id() is None


def test_agentic_session_sets_user_id():
    """agentic_session sets user_id in ContextVar."""
    with agentic_session(session_id="sess-1", user_id="user-1"):
        assert get_user_id() == "user-1"
    assert get_user_id() is None


def test_agentic_session_without_user_id():
    """user_id is optional and remains None when not provided."""
    with agentic_session(session_id="sess-1"):
        assert get_session_id() == "sess-1"
        assert get_user_id() is None


def test_nested_agentic_sessions():
    """Inner session overrides outer, outer restores on exit."""
    with agentic_session(session_id="outer", user_id="user-outer"):
        assert get_session_id() == "outer"
        with agentic_session(session_id="inner", user_id="user-inner"):
            assert get_session_id() == "inner"
            assert get_user_id() == "user-inner"
        assert get_session_id() == "outer"
        assert get_user_id() == "user-outer"


def test_get_trace_id_none_without_span():
    """Returns None when no span is active."""
    assert get_trace_id() is None


def test_agentic_session_kwargs_stamped():
    """Extra kwargs are stamped as metadata.{key} on the current span."""
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True

    with patch("kelet._context.trace") as mock_trace:
        mock_trace.get_current_span.return_value = mock_span
        with agentic_session(session_id="s", foo="bar"):
            pass

    mock_span.set_attribute.assert_any_call("metadata.foo", "bar")


def test_agentic_session_dollar_key():
    """Dollar-prefixed kwargs are stamped correctly as metadata.$key."""
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True

    with patch("kelet._context.trace") as mock_trace:
        mock_trace.get_current_span.return_value = mock_span
        with agentic_session(session_id="s", **{"$reserved": True}):  # type: ignore[arg-type]
            pass

    mock_span.set_attribute.assert_any_call("metadata.$reserved", True)


def test_metadata_kwargs_set_and_reset():
    """_metadata_kwargs_var is set during enter and reset after exit."""
    assert _metadata_kwargs_var.get() is None
    with agentic_session(session_id="s", foo="bar"):
        assert _metadata_kwargs_var.get() == {"foo": "bar"}
    assert _metadata_kwargs_var.get() is None


def test_nested_session_inherits_user_id():
    """Inner session without user_id inherits outer's user_id."""
    with agentic_session(session_id="outer", user_id="outer-user"):
        assert get_user_id() == "outer-user"
        with agentic_session(session_id="inner"):
            assert get_user_id() == "outer-user"
        assert get_user_id() == "outer-user"
    assert get_user_id() is None


def test_nested_session_inherits_project():
    """Inner session without project inherits outer's project."""
    from kelet._context import _project_override_var

    with agentic_session(session_id="outer", project="outer-proj"):
        assert _project_override_var.get() == "outer-proj"
        with agentic_session(session_id="inner"):
            assert _project_override_var.get() == "outer-proj"
        assert _project_override_var.get() == "outer-proj"
    assert _project_override_var.get() is None


def test_nested_session_merges_kwargs():
    """{**outer, **inner} merge with inner precedence."""
    with agentic_session(session_id="outer", a="1", b="2"):
        assert get_metadata_kwargs() == {"a": "1", "b": "2"}
        with agentic_session(session_id="inner", b="override", c="3"):
            assert get_metadata_kwargs() == {"a": "1", "b": "override", "c": "3"}
        assert get_metadata_kwargs() == {"a": "1", "b": "2"}
    assert get_metadata_kwargs() == {}


def test_metadata_kwargs_baggage_set():
    """Per-key baggage entries are set for kwargs."""
    with agentic_session(session_id="s", my_key="my_val"):
        ctx = otel_context.get_current()
        val = otel_baggage.get_baggage("kelet.metadata.my_key", context=ctx)
        assert val == "my_val"
