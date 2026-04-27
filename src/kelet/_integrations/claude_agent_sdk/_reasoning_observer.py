"""Observe ``claude-agent-sdk`` streams and emit ``kelet.reasoning`` log records.

Claude Code's native OTLP pipeline redacts reasoning text in its
``api_response_body`` log event (``"thinking":"<REDACTED>"``). The only way
to recover the full thinking content is to observe the SDK's in-process
message stream, which surfaces ``ThinkingBlock`` entries intact.

This module wraps three async-iterator entry points on ``claude-agent-sdk``:

* ``claude_agent_sdk.query`` — module-level async generator.
* ``ClaudeSDKClient.receive_messages`` — instance method, async generator.
* ``ClaudeSDKClient.receive_response`` — convenience wrapper over the above.

For each yielded ``AssistantMessage`` we scan ``content[]`` for any block
with a string ``thinking`` attribute and emit one ``kelet.reasoning`` log
record per block via the global OTel logger provider. The extraction side
matches these records to the Claude Code LLM interaction by
``reasoning.message_id`` (``AssistantMessage.message_id`` from the SDK).

Unlike the previous span-event-based observer, we do NOT open a parent
span — Claude Code already emits its own ``claude_code.interaction`` span
and the log records carry the message id that links them back.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable

from opentelemetry._logs import get_logger

logger = logging.getLogger(__name__)

REASONING_EVENT_NAME = "kelet.reasoning"


def _emit_reasoning(msg: Any) -> None:
    """Emit one ``kelet.reasoning`` log record per ``ThinkingBlock`` in ``msg``.

    Duck-typed: we accept anything with a ``content`` list whose entries have
    a string ``thinking`` attribute. ``message_id`` is optional (per the SDK
    contract) but included as ``reasoning.message_id`` when present so
    extraction can correlate back to the LLM request.
    """
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return

    message_id = getattr(msg, "message_id", None)
    session_id = getattr(msg, "session_id", None)

    otel_logger = get_logger(__name__)
    for block in content:
        thinking = getattr(block, "thinking", None)
        if not isinstance(thinking, str):
            continue
        signature = getattr(block, "signature", "") or ""
        attributes: dict[str, Any] = {
            "reasoning.text": thinking,
            "reasoning.signature": signature,
            "event.name": REASONING_EVENT_NAME,
        }
        if message_id:
            attributes["reasoning.message_id"] = message_id
        if session_id:
            attributes["session.id"] = session_id
        try:
            otel_logger.emit(
                body=REASONING_EVENT_NAME,
                attributes=attributes,
                event_name=REASONING_EVENT_NAME,
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "failed to emit kelet.reasoning log record", exc_info=True
            )


def wrap_async_gen(
    wrapped: Callable[..., AsyncIterator[Any]],
    _instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AsyncIterator[Any]:
    """Wrap an async-generator factory to observe yielded ``AssistantMessage``s.

    ``wrapt.wrap_function_wrapper`` calls this as
    ``wrap_async_gen(wrapped, instance, args, kwargs)``. The underlying SDK
    function is an ``async def`` + ``yield`` (async-generator function), so
    we return an async generator directly.

    We observe each yielded item BEFORE forwarding it so a caller exception
    mid-yield doesn't cost us the event. Observer failures are caught and
    logged but never propagate into user iteration.
    """

    async def _generator() -> AsyncIterator[Any]:
        async for item in wrapped(*args, **kwargs):
            try:
                _emit_reasoning(item)
            except Exception:  # pragma: no cover - defensive
                logger.debug("reasoning observer raised", exc_info=True)
            yield item

    return _generator()


__all__ = ["REASONING_EVENT_NAME", "wrap_async_gen"]
