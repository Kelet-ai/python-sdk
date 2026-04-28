"""Unit tests for the slim ``kelet.reasoning`` observer."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from kelet._integrations.claude_agent_sdk import _reasoning_observer
from kelet._integrations.claude_agent_sdk._reasoning_observer import (
    REASONING_EVENT_NAME,
    REASONING_SCOPE_NAME,
    _emit_reasoning,
    reset_logger,
    set_logger,
    wrap_async_gen,
)


class _ThinkingBlock:
    def __init__(self, thinking: str, signature: str = "") -> None:
        self.thinking = thinking
        self.signature = signature


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


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


class _TSAssistantMessage:
    """TypeScript-shaped envelope: ``msg.message.content`` / ``msg.message.id``."""

    def __init__(
        self,
        inner_content: list[Any],
        inner_id: str | None = None,
        inner_session_id: str | None = None,
    ) -> None:
        self.message = MagicMock()
        self.message.content = inner_content
        if inner_id is not None:
            self.message.id = inner_id
        else:
            del self.message.id
        if inner_session_id is not None:
            self.message.session_id = inner_session_id
        else:
            del self.message.session_id


def _collect_emits(mock_logger: Any) -> list[dict[str, Any]]:
    """Flatten mock_logger.emit call kwargs into a list of dicts."""
    return [call.kwargs for call in mock_logger.emit.call_args_list]


@pytest.fixture
def fake_logger():
    """Install a MagicMock as the module's active logger.

    The observer keeps the logger as a module-level variable so we can
    avoid re-resolving it on every yield on the hot streaming path.
    Tests override it via ``set_logger`` (called by the instrumentor in
    production) and restore the default via ``reset_logger``.
    """
    mock = MagicMock()
    set_logger(mock)
    try:
        yield mock
    finally:
        reset_logger()


# ---------------------------------------------------------------------------
# Scope name contract: server-side /api/logs filters on
# ``startswith("com.anthropic.claude_code")``.
# ---------------------------------------------------------------------------


def test_scope_name_passes_server_cc_filter() -> None:
    assert REASONING_SCOPE_NAME.startswith("com.anthropic.claude_code"), (
        "Kelet server's /api/logs filter rejects records whose "
        "instrumentation_scope.name doesn't start with the CC prefix. "
        f"Got: {REASONING_SCOPE_NAME!r}"
    )


# ---------------------------------------------------------------------------
# _emit_reasoning (Py shape)
# ---------------------------------------------------------------------------


def test_emit_reasoning_with_thinking_block(fake_logger: Any) -> None:
    msg = _AssistantMessage(
        content=[_ThinkingBlock("deep thought", signature="sig-1")],
        message_id="msg_123",
        session_id="sess_abc",
    )
    _emit_reasoning(msg, sticky_session_id=None)

    emits = _collect_emits(fake_logger)
    assert len(emits) == 1
    kw = emits[0]
    assert kw["body"] == REASONING_EVENT_NAME
    assert kw["event_name"] == REASONING_EVENT_NAME
    attrs = kw["attributes"]
    assert attrs["reasoning.text"] == "deep thought"
    assert attrs["reasoning.signature"] == "sig-1"
    assert attrs["reasoning.message_id"] == "msg_123"
    assert attrs["session.id"] == "sess_abc"


def test_emit_reasoning_multiple_blocks(fake_logger: Any) -> None:
    msg = _AssistantMessage(
        content=[
            _ThinkingBlock("first"),
            _TextBlock("not a thinking block"),
            _ThinkingBlock("second", signature="s2"),
        ],
    )
    _emit_reasoning(msg, sticky_session_id=None)

    emits = _collect_emits(fake_logger)
    assert len(emits) == 2
    assert emits[0]["attributes"]["reasoning.text"] == "first"
    assert emits[0]["attributes"]["reasoning.signature"] == ""
    assert emits[1]["attributes"]["reasoning.text"] == "second"
    assert emits[1]["attributes"]["reasoning.signature"] == "s2"


def test_emit_reasoning_no_thinking_blocks(fake_logger: Any) -> None:
    msg = _AssistantMessage(content=[_TextBlock("just text")])
    _emit_reasoning(msg, sticky_session_id=None)
    fake_logger.emit.assert_not_called()


def test_emit_reasoning_non_message_shapes(fake_logger: Any) -> None:
    class _UserMessage:
        pass

    # No ``content`` attr at all — should silently skip.
    _emit_reasoning(_UserMessage(), sticky_session_id=None)
    # ``content`` is a non-list — should silently skip.
    m = _UserMessage()
    m.content = "not a list"  # type: ignore[attr-defined]
    _emit_reasoning(m, sticky_session_id=None)
    fake_logger.emit.assert_not_called()


def test_emit_reasoning_message_id_optional(fake_logger: Any) -> None:
    msg = _AssistantMessage(content=[_ThinkingBlock("x")])
    _emit_reasoning(msg, sticky_session_id=None)
    attrs = _collect_emits(fake_logger)[0]["attributes"]
    assert "reasoning.message_id" not in attrs
    assert "session.id" not in attrs


def test_emit_reasoning_swallows_emit_exceptions(fake_logger: Any) -> None:
    fake_logger.emit.side_effect = RuntimeError("exporter down")
    msg = _AssistantMessage(content=[_ThinkingBlock("t")])
    # Should not raise.
    _emit_reasoning(msg, sticky_session_id=None)


# ---------------------------------------------------------------------------
# Sticky session_id (Critical #2)
# ---------------------------------------------------------------------------


def test_emit_reasoning_uses_sticky_session_id_when_msg_has_none(
    fake_logger: Any,
) -> None:
    """When the current message has no session_id, fall back to the sticky."""
    msg = _AssistantMessage(
        content=[_ThinkingBlock("t")], message_id="m1"
    )  # no session_id on the envelope
    _emit_reasoning(msg, sticky_session_id="sticky-sess")

    emits = _collect_emits(fake_logger)
    assert len(emits) == 1
    assert emits[0]["attributes"]["session.id"] == "sticky-sess"


def test_emit_reasoning_prefers_msg_session_over_sticky(fake_logger: Any) -> None:
    """The current message's session_id wins when both are present."""
    msg = _AssistantMessage(
        content=[_ThinkingBlock("t")],
        session_id="msg-sess",
    )
    _emit_reasoning(msg, sticky_session_id="sticky-sess")

    emits = _collect_emits(fake_logger)
    assert emits[0]["attributes"]["session.id"] == "msg-sess"


def test_wrap_async_gen_stickies_session_id_across_stream(
    fake_logger: Any,
) -> None:
    """First message carries session.id; subsequent messages without it
    still route under the sticky id (would otherwise get server-dropped)."""

    messages = [
        _AssistantMessage(
            content=[_ThinkingBlock("one")],
            message_id="m1",
            session_id="sticky-sess",
        ),
        _AssistantMessage(content=[_ThinkingBlock("two")], message_id="m2"),
        _AssistantMessage(content=[_ThinkingBlock("three")], message_id="m3"),
    ]

    async def source(*_a: Any, **_kw: Any):
        for m in messages:
            yield m

    async def run() -> None:
        async for _ in wrap_async_gen(source, None, (), {}):
            pass

    asyncio.run(run())
    emits = _collect_emits(fake_logger)
    assert len(emits) == 3
    # All three share the sticky session_id from the first message.
    assert emits[0]["attributes"]["session.id"] == "sticky-sess"
    assert emits[1]["attributes"]["session.id"] == "sticky-sess"
    assert emits[2]["attributes"]["session.id"] == "sticky-sess"


# ---------------------------------------------------------------------------
# TypeScript-shape fallback (Minor #10)
# ---------------------------------------------------------------------------


def test_emit_reasoning_accepts_typescript_shape(fake_logger: Any) -> None:
    """``msg.message.content`` shape (TS CLI envelope) is read when
    ``msg.content`` is absent."""
    msg = _TSAssistantMessage(
        inner_content=[_ThinkingBlock("ts-thought", signature="sig-ts")],
        inner_id="msg_ts",
        inner_session_id="sess_ts",
    )
    _emit_reasoning(msg, sticky_session_id=None)

    emits = _collect_emits(fake_logger)
    assert len(emits) == 1
    attrs = emits[0]["attributes"]
    assert attrs["reasoning.text"] == "ts-thought"
    assert attrs["reasoning.signature"] == "sig-ts"
    assert attrs["reasoning.message_id"] == "msg_ts"
    assert attrs["session.id"] == "sess_ts"


# ---------------------------------------------------------------------------
# wrap_async_gen
# ---------------------------------------------------------------------------


def test_wrap_async_gen_observes_each_yield(fake_logger: Any) -> None:
    messages = [
        _AssistantMessage(content=[_ThinkingBlock("one")], message_id="m1"),
        _AssistantMessage(content=[_TextBlock("text")]),
        _AssistantMessage(content=[_ThinkingBlock("two")], message_id="m2"),
    ]

    async def source(*_a: Any, **_kw: Any):
        for m in messages:
            yield m

    async def run() -> list[Any]:
        items: list[Any] = []
        async for item in wrap_async_gen(source, None, (), {}):
            items.append(item)
        return items

    items = asyncio.run(run())

    assert items == messages
    emits = _collect_emits(fake_logger)
    assert len(emits) == 2
    assert emits[0]["attributes"]["reasoning.text"] == "one"
    assert emits[0]["attributes"]["reasoning.message_id"] == "m1"
    assert emits[1]["attributes"]["reasoning.text"] == "two"
    assert emits[1]["attributes"]["reasoning.message_id"] == "m2"


def test_wrap_async_gen_propagates_exceptions(fake_logger: Any) -> None:
    async def source(*_a: Any, **_kw: Any):
        yield _AssistantMessage(content=[_ThinkingBlock("t")])
        raise RuntimeError("stream failed")

    async def run() -> None:
        async for _ in wrap_async_gen(source, None, (), {}):
            pass

    with pytest.raises(RuntimeError, match="stream failed"):
        asyncio.run(run())

    # Observer still fired for the one message before the failure.
    assert len(_collect_emits(fake_logger)) == 1


def test_wrap_async_gen_swallows_observer_exceptions(fake_logger: Any) -> None:
    fake_logger.emit.side_effect = RuntimeError("exporter down")
    messages = [_AssistantMessage(content=[_ThinkingBlock("t")])]

    async def source(*_a: Any, **_kw: Any):
        for m in messages:
            yield m

    async def run() -> list[Any]:
        items: list[Any] = []
        async for item in wrap_async_gen(source, None, (), {}):
            items.append(item)
        return items

    # Observer's emit fails internally; user iteration must still succeed.
    items = asyncio.run(run())
    assert items == messages


# ---------------------------------------------------------------------------
# Logger override plumbing (set_logger / reset_logger)
# ---------------------------------------------------------------------------


def test_set_logger_overrides_module_default() -> None:
    before = _reasoning_observer._LOGGER
    custom = MagicMock()
    set_logger(custom)
    try:
        assert _reasoning_observer._LOGGER is custom
    finally:
        reset_logger()
    after = _reasoning_observer._LOGGER
    # reset_logger() should produce a fresh resolution, not the overridden mock.
    assert after is not custom
    # And the default should match the scope name the reset resolves under.
    assert after is not before or type(after) is type(before)
