"""Tests for context helpers (ContextVar-based session/user management)."""

from unittest.mock import MagicMock, patch

from kelet._context import (
    agentic_session,
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
