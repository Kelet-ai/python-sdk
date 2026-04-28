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

LoggerProvider ownership
------------------------
This instrumentor owns the ``LoggerProvider`` that exports Kelet
reasoning log records, rather than installing one globally in
``configure()``. That keeps Kelet's logging pipeline off the host app's
global ``opentelemetry._logs`` slot — a host that wires its own
``LoggerProvider`` (Datadog, Grafana, Sentry, etc.) isn't clobbered.
The Kelet observer explicitly resolves its logger from this
instrumentor-scoped provider via ``set_logger``.
"""

from __future__ import annotations

import logging
from typing import Any, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.sdk._logs import LoggerProvider, LogRecordProcessor
from wrapt import wrap_function_wrapper

from . import _reasoning_observer
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

# Version floor matches the one documented in docs/claude-agent-sdk.md and
# required in pyproject.toml. Older releases don't expose the streaming
# message shapes the observer duck-types on.
_instruments = ("claude-agent-sdk >= 0.1.45",)


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):
    """Install the ``kelet.reasoning`` observer on ``claude-agent-sdk``.

    Usage::

        from kelet._integrations.claude_agent_sdk import ClaudeAgentSDKInstrumentor
        ClaudeAgentSDKInstrumentor().instrument()

    Kelet's ``configure()`` calls this automatically when the package is
    installed; manual invocation is only needed in custom setups.

    ``instrument(logger_provider=...)`` accepts an optional pre-built
    ``LoggerProvider`` — useful when the caller wants to share one
    across integrations or preload processors. When omitted, the
    instrumentor builds its own provider using the same Kelet endpoint +
    auth headers as the trace exporter; see ``_build_default_provider``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._own_provider: Optional[LoggerProvider] = None
        self._own_processor: Optional[LogRecordProcessor] = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        # Install an integration-scoped LoggerProvider and route the
        # reasoning observer through it. Callers can inject one via
        # ``logger_provider=``; otherwise we build a default that exports
        # OTLP logs to Kelet using the currently-configured API key +
        # project.
        provider: Optional[LoggerProvider] = kwargs.get("logger_provider")
        if provider is None:
            provider = self._build_default_provider()

        if provider is not None:
            self._own_provider = provider
            _reasoning_observer.set_logger(
                _reasoning_observer.logger_from_provider(provider)
            )
        else:
            # Fallback: keep the module's default (resolves against the
            # global provider). Emissions will still flow if the host app
            # installed a LoggerProvider upstream, otherwise they're
            # collected by the no-op provider and dropped.
            logger.debug(
                "ClaudeAgentSDKInstrumentor: no LoggerProvider available "
                "(Kelet config missing?) — reasoning emissions will fall "
                "through to the global provider"
            )

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
        import importlib
        import sys as _sys

        for module, attr, _wrapper in _PATCHES:
            target, _, method = attr.rpartition(".")
            if target:
                dotted = f"{module}.{target}"
                attr_name = method
            else:
                # Module-level attribute (e.g. ``query``). ``wrapt.unwrap``
                # requires a dotted import path OR a module object. Import
                # the module object directly because ``unwrap(str, attr)``
                # insists on a class-path form.
                dotted = module
                attr_name = attr
            try:
                if "." in dotted:
                    unwrap(dotted, attr_name)
                else:
                    mod_obj = _sys.modules.get(dotted)
                    if mod_obj is None:
                        mod_obj = importlib.import_module(dotted)
                    unwrap(mod_obj, attr_name)
            except (ImportError, ModuleNotFoundError, AttributeError) as exc:
                logger.debug(
                    "claude-agent-sdk uninstrumentation skipped for %s.%s: %s",
                    dotted,
                    attr_name,
                    exc,
                    exc_info=False,
                )

        # Drop the module-local reference to our provider so downstream
        # emissions fall back to the global provider's default logger.
        _reasoning_observer.reset_logger()

        if self._own_processor is not None:
            try:
                self._own_processor.shutdown()
            except Exception:  # pragma: no cover - defensive
                logger.debug(
                    "claude-agent-sdk: failed to shutdown log processor",
                    exc_info=True,
                )
            self._own_processor = None
        self._own_provider = None

    def _build_default_provider(self) -> Optional[LoggerProvider]:
        """Build a LoggerProvider that exports OTLP logs to Kelet.

        Returns ``None`` if the Kelet SDK hasn't been configured yet
        (no API key / project) — in that case the caller will log a
        debug message and the observer falls back to the global
        provider. Importing ``_resolve_config`` is done locally to keep
        the import cycle out of module load (the instrumentor is
        discovered via entry points).
        """
        try:
            from kelet._config import get_config
            from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                OTLPLogExporter,
            )
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        except Exception:  # pragma: no cover - import error is catastrophic
            logger.debug(
                "ClaudeAgentSDKInstrumentor: failed to import log-exporter deps",
                exc_info=True,
            )
            return None

        # ``get_config()`` raises ValueError when KELET_API_KEY/PROJECT
        # aren't set and ``configure()`` hasn't been called — that's the
        # "user turned on claude-agent-sdk instrumentation without
        # configuring Kelet" path. Swallow and fall back to the global
        # provider so the integration can still no-op safely.
        try:
            cfg = get_config()
        except ValueError:
            logger.debug(
                "ClaudeAgentSDKInstrumentor: Kelet not configured "
                "(KELET_API_KEY / KELET_PROJECT missing) — reasoning emissions "
                "will fall through to the global LoggerProvider"
            )
            return None
        if cfg is None or not cfg.api_key or not cfg.project:
            return None

        exporter = OTLPLogExporter(
            endpoint=f"{cfg.base_url}/api/logs",
            headers={
                "Authorization": cfg.api_key,
                "X-Kelet-Project": cfg.project,
            },
        )
        processor = BatchLogRecordProcessor(exporter)
        provider = LoggerProvider()
        provider.add_log_record_processor(processor)
        self._own_processor = processor
        return provider
