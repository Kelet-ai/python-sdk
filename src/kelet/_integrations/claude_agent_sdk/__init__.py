"""Claude Agent SDK integration for Kelet.

Installs a ``ThinkingBlock`` observer on ``claude-agent-sdk`` that emits
``kelet.reasoning`` OTLP log records. Wraps ``claude_agent_sdk.query``,
``ClaudeSDKClient.receive_messages``, and ``ClaudeSDKClient.receive_response``
via ``wrapt.wrap_function_wrapper``.

All other observability (interaction spans, tool spans, hooks, skills,
compaction events) flows natively from Claude Code when the caller sets
``CLAUDE_CODE_ENABLE_TELEMETRY=1`` + ``OTEL_EXPORTER_OTLP_*`` env vars on
their process before invoking the SDK. See
``docs/claude-agent-sdk-contract.md`` for the full env-var recipe.

Kelet's job is minimal: observe the in-process message stream for
``ThinkingBlock`` entries (which Claude Code redacts in its own OTLP
payloads) and emit them as parallel ``kelet.reasoning`` log records.
"""

from ._instrumentor import ClaudeAgentSDKInstrumentor

__all__ = ["ClaudeAgentSDKInstrumentor"]
