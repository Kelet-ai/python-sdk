"""Unit tests for the slim ``kelet.reasoning`` observer."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from kelet._integrations.claude_agent_sdk._reasoning_observer import (
    REASONING_EVENT_NAME,
    _emit_reasoning,
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


def _collect_emits(mock_logger: Any) -> list[dict[str, Any]]:
    """Flatten mock_logger.emit call kwargs into a list of dicts."""
    return [call.kwargs for call in mock_logger.emit.call_args_list]


@pytest.fixture
def fake_logger():
    """Patch ``get_logger`` to return a MagicMock so we can assert ``emit`` calls."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    with patch(
        "kelet._integrations.claude_agent_sdk._reasoning_observer.get_logger",
        return_value=mock,
    ):
        yield mock


def test_emit_reasoning_with_thinking_block(fake_logger: Any) -> None:
    msg = _AssistantMessage(
        content=[_ThinkingBlock("deep thought", signature="sig-1")],
        message_id="msg_123",
        session_id="sess_abc",
    )
    _emit_reasoning(msg)

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
    assert attrs["event.name"] == REASONING_EVENT_NAME


def test_emit_reasoning_multiple_blocks(fake_logger: Any) -> None:
    msg = _AssistantMessage(
        content=[
            _ThinkingBlock("first"),
            _TextBlock("not a thinking block"),
            _ThinkingBlock("second", signature="s2"),
        ],
    )
    _emit_reasoning(msg)

    emits = _collect_emits(fake_logger)
    assert len(emits) == 2
    assert emits[0]["attributes"]["reasoning.text"] == "first"
    assert emits[0]["attributes"]["reasoning.signature"] == ""
    assert emits[1]["attributes"]["reasoning.text"] == "second"
    assert emits[1]["attributes"]["reasoning.signature"] == "s2"


def test_emit_reasoning_no_thinking_blocks(fake_logger: Any) -> None:
    msg = _AssistantMessage(content=[_TextBlock("just text")])
    _emit_reasoning(msg)
    fake_logger.emit.assert_not_called()


def test_emit_reasoning_non_message_shapes(fake_logger: Any) -> None:
    class _UserMessage:
        pass

    # No ``content`` attr at all — should silently skip.
    _emit_reasoning(_UserMessage())
    # ``content`` is a non-list — should silently skip.
    m = _UserMessage()
    m.content = "not a list"  # type: ignore[attr-defined]
    _emit_reasoning(m)
    fake_logger.emit.assert_not_called()


def test_emit_reasoning_message_id_optional(fake_logger: Any) -> None:
    msg = _AssistantMessage(content=[_ThinkingBlock("x")])
    _emit_reasoning(msg)
    attrs = _collect_emits(fake_logger)[0]["attributes"]
    assert "reasoning.message_id" not in attrs
    assert "session.id" not in attrs


def test_emit_reasoning_swallows_emit_exceptions(fake_logger: Any) -> None:
    fake_logger.emit.side_effect = RuntimeError("exporter down")
    msg = _AssistantMessage(content=[_ThinkingBlock("t")])
    # Should not raise.
    _emit_reasoning(msg)


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
