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
record per block. The extraction side matches these records to the Claude
Code LLM interaction by ``reasoning.message_id``
(``AssistantMessage.message_id`` from the SDK).

Instrumentation scope
---------------------
The OTel SDK stamps each log record's ``instrumentation_scope.name`` from
the ``get_logger(name)`` argument. The Kelet ingestion endpoint accepts
scopes prefixed with ``com.anthropic.claude_code`` — anything else is
warn-and-dropped. We therefore emit under
``com.anthropic.claude_code.kelet_reasoning`` so the records reach the
CC ingestion workflow.

Session id stickiness
---------------------
The CC workflow routes by ``session.id`` on each log record. Early
``AssistantMessage``s can arrive before the SDK populates ``session_id``
on the envelope, which would land those records in the server's
``missing``-counter and drop them. We cache the last-seen session id
per stream so subsequent messages inherit it if their own is absent.

Unlike the previous span-event-based observer, we do NOT open a parent
span — Claude Code already emits its own ``claude_code.interaction`` span
and the log records carry the message id / session id that link them back.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, Optional

from opentelemetry._logs import Logger, LoggerProvider, get_logger

logger = logging.getLogger(__name__)

REASONING_EVENT_NAME = "kelet.reasoning"

# Scope name the Kelet server accepts on /api/logs. Must start with
# ``com.anthropic.claude_code`` so the server's ``_is_cc_request`` filter
# lets the record through to the CC workflow. Changing this string is a
# breaking change on the ingestion contract.
REASONING_SCOPE_NAME = "com.anthropic.claude_code.kelet_reasoning"


# Module-level logger — the OTel SDK caches them internally, but resolving
# the logger once at import time still saves a per-message dict lookup
# and a lock on the hot streaming path.
#
# NOTE: if the instrumentor installs a *dedicated* LoggerProvider
# (to avoid clobbering the host app's global provider), it also overrides
# the module-local ``_LOGGER`` via ``set_logger(...)`` below so emissions
# route through the scoped provider rather than the global one.
_LOGGER: Logger = get_logger(REASONING_SCOPE_NAME)


def set_logger(new_logger: Logger) -> None:
    """Override the module-level logger used for emissions.

    Called by ``ClaudeAgentSDKInstrumentor._instrument`` when it provisions
    a dedicated ``LoggerProvider`` for the Kelet integration. Scoping
    emissions to an integration-owned provider keeps them off the host
    application's global logging pipeline.
    """
    global _LOGGER
    _LOGGER = new_logger


def reset_logger() -> None:
    """Reset the module-level logger to the global provider's default.

    Called by ``_uninstrument`` to drop references to the dedicated
    provider when the instrumentor is torn down.
    """
    global _LOGGER
    _LOGGER = get_logger(REASONING_SCOPE_NAME)


def _iter_thinking_blocks(msg: Any) -> Optional[list[Any]]:
    """Return a list of content blocks to scan, or None if the message
    doesn't look like an assistant envelope we can parse.

    Accepts both the native Python SDK shape (``msg.content``) and the
    TypeScript CLI shape (``msg.message.content``) — the contract doc
    advertises both, and the shared fixtures exercise both during
    captures.
    """
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        return content
    inner = getattr(msg, "message", None)
    if inner is not None:
        inner_content = getattr(inner, "content", None)
        if isinstance(inner_content, list):
            return inner_content
    return None


def _extract_message_id(msg: Any) -> Optional[str]:
    """Pull ``message_id`` from either the Py or TS-shaped envelope."""
    mid = getattr(msg, "message_id", None)
    if mid:
        return str(mid)
    inner = getattr(msg, "message", None)
    if inner is not None:
        mid = getattr(inner, "id", None) or getattr(inner, "message_id", None)
        if mid:
            return str(mid)
    return None


def _extract_session_id(msg: Any) -> Optional[str]:
    """Pull ``session_id`` from either the Py or TS-shaped envelope."""
    sid = getattr(msg, "session_id", None)
    if sid:
        return str(sid)
    inner = getattr(msg, "message", None)
    if inner is not None:
        sid = getattr(inner, "session_id", None)
        if sid:
            return str(sid)
    return None


def _emit_reasoning(msg: Any, sticky_session_id: Optional[str]) -> Optional[str]:
    """Emit one ``kelet.reasoning`` log record per ``ThinkingBlock`` in ``msg``.

    Returns the session id to remember for subsequent messages in the
    same stream. A returned value of ``None`` means the input didn't
    carry a session id — callers should keep whatever they had.

    Duck-typed: anything with a ``content`` list whose entries have a
    string ``thinking`` attribute is fair game. ``message_id`` is
    optional but attached when present so extraction can correlate back
    to the exact LLM request. ``session.id`` is required for server-side
    routing; we fall back to ``sticky_session_id`` if the current
    message doesn't carry one so early-stream records aren't dropped.
    """
    blocks = _iter_thinking_blocks(msg)
    if blocks is None:
        return None

    message_id = _extract_message_id(msg)
    session_id = _extract_session_id(msg) or sticky_session_id

    for block in blocks:
        thinking = getattr(block, "thinking", None)
        if not isinstance(thinking, str):
            continue
        signature = getattr(block, "signature", "") or ""
        attributes: dict[str, Any] = {
            "reasoning.text": thinking,
            "reasoning.signature": signature,
        }
        if message_id:
            attributes["reasoning.message_id"] = message_id
        if session_id:
            attributes["session.id"] = session_id
        try:
            # ``event_name`` stamps the OTLP LogRecord.event_name field.
            # ``body`` carries the canonical discriminator the Kelet
            # workflow matches on via ``_cc_log_event_body``.
            _LOGGER.emit(
                body=REASONING_EVENT_NAME,
                attributes=attributes,
                event_name=REASONING_EVENT_NAME,
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "failed to emit kelet.reasoning log record", exc_info=True
            )

    return _extract_session_id(msg)


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

    Each wrapper gets its own sticky session_id closure so concurrent
    streams don't poison each other. We observe each yielded item BEFORE
    forwarding it so a caller exception mid-yield doesn't cost us the
    event. Observer failures are caught and logged but never propagate
    into user iteration.
    """

    async def _generator() -> AsyncIterator[Any]:
        sticky_session_id: Optional[str] = None
        async for item in wrapped(*args, **kwargs):
            try:
                updated = _emit_reasoning(item, sticky_session_id)
                if updated:
                    sticky_session_id = updated
            except Exception:  # pragma: no cover - defensive
                logger.debug("reasoning observer raised", exc_info=True)
            yield item

    return _generator()


__all__ = [
    "REASONING_EVENT_NAME",
    "REASONING_SCOPE_NAME",
    "wrap_async_gen",
    "set_logger",
    "reset_logger",
]


# ---------------------------------------------------------------------------
# Dedicated LoggerProvider support — used by the instrumentor to keep
# ``kelet.reasoning`` off the host app's global logging pipeline.
# ---------------------------------------------------------------------------


def logger_from_provider(provider: LoggerProvider) -> Logger:
    """Resolve the module's scope name against a specific LoggerProvider.

    The instrumentor calls this + ``set_logger`` to route Kelet emissions
    through its integration-owned provider rather than the global one.
    """
    return provider.get_logger(REASONING_SCOPE_NAME)
