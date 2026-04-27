"""Lean ``BaseInstrumentor`` that installs the reasoning observer only.

Claude Code v2.1.119+ natively emits OTLP traces, logs, and metrics when
``CLAUDE_CODE_ENABLE_TELEMETRY=1`` + ``OTEL_EXPORTER_OTLP_*`` env vars are
set — which users set themselves via the contract doc's recipe. The only
piece of observability we still need to add is capturing reasoning text,
since Claude Code redacts ``thinking`` in its own OTLP payloads.

We wrap three stream entry points on ``claude-agent-sdk``:

* ``claude_agent_sdk.query`` — module-level async generator.
* ``ClaudeSDKClient.receive_messages`` — instance async generator.
* ``ClaudeSDKClient.receive_response`` — convenience async generator.

Each wrapped iterator observes yielded ``AssistantMessage``s for
``ThinkingBlock`` entries and emits a ``kelet.reasoning`` log record per
block. No transport patches, no env injection, no parent span.
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from ._reasoning_observer import wrap_async_gen

logger = logging.getLogger(__name__)

_SDK_MODULE = "claude_agent_sdk"
_CLIENT_MODULE = "claude_agent_sdk.client"
_CLIENT_CLASS = "ClaudeSDKClient"

# (module, attribute_path, wrapper). One source of truth for instrument/
# uninstrument symmetry.
_PATCHES: tuple[tuple[str, str, Any], ...] = (
    (_SDK_MODULE, "query", wrap_async_gen),
    (_CLIENT_MODULE, f"{_CLIENT_CLASS}.receive_messages", wrap_async_gen),
    (_CLIENT_MODULE, f"{_CLIENT_CLASS}.receive_response", wrap_async_gen),
)

_instruments = ("claude-agent-sdk >= 0.1.0",)


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):
    """Install the ``kelet.reasoning`` observer on ``claude-agent-sdk``.

    Usage::

        from kelet._integrations.claude_agent_sdk import ClaudeAgentSDKInstrumentor
        ClaudeAgentSDKInstrumentor().instrument()

    Kelet's ``configure()`` calls this automatically when the package is
    installed; manual invocation is only needed in custom setups.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        for module, attr, wrapper in _PATCHES:
            try:
                wrap_function_wrapper(module, attr, wrapper)
            except (ModuleNotFoundError, AttributeError) as exc:
                logger.debug(
                    "claude-agent-sdk instrumentation skipped for %s.%s: %s",
                    module,
                    attr,
                    exc,
                    exc_info=False,
                )

    def _uninstrument(self, **kwargs: Any) -> None:
        for module, attr, _wrapper in _PATCHES:
            target, _, method = attr.rpartition(".")
            dotted = f"{module}.{target}" if target else module
            try:
                unwrap(dotted, method if target else attr)
            except (ModuleNotFoundError, AttributeError):
                pass
