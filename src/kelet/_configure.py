"""Configuration function for Kelet SDK."""

import atexit
import os
from typing import NamedTuple, Optional, Sequence

from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, ReadableSpan, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import ProxyTracerProvider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ._config import KeletConfig, set_config
from ._context import _session_id_var, _user_id_var, _agent_name_var, SESSION_ID_ATTR, USER_ID_ATTR, AGENT_NAME_ATTR

# Track processors for shutdown
_active_processors: list[SpanProcessor] = []


class _ResolvedConfig(NamedTuple):
    """Resolved configuration values."""

    api_key: str
    base_url: str
    project: str


def _resolve_config(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    base_url: Optional[str] = None,
) -> _ResolvedConfig:
    """Resolve configuration from parameters or environment variables.

    Args:
        api_key: API key (default: KELET_API_KEY env var)
        project: Project name (default: KELET_PROJECT env var or "default")
        base_url: API base URL (default: KELET_API_URL env var or "https://api.kelet.ai")

    Returns:
        Resolved configuration values.

    Raises:
        ValueError: If KELET_API_KEY is not provided.
    """
    resolved_api_key = api_key or os.environ.get("KELET_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "KELET_API_KEY required. Pass api_key parameter or set KELET_API_KEY env var."
        )
    base_url = base_url or os.environ.get("KELET_API_URL", "https://api.kelet.ai")
    base_url = base_url.rstrip("/").removesuffix("/api")

    return _ResolvedConfig(
        api_key=resolved_api_key,
        base_url=base_url,
        project=project or os.environ.get("KELET_PROJECT", "default"),
    )


class _KeletSpanProcessor(SpanProcessor):
    """SpanProcessor that adds kelet.project attribute and delegates to wrapped processor."""

    def __init__(self, wrapped: SpanProcessor, project: str) -> None:
        self._wrapped = wrapped
        self._project = project

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        span.set_attribute("kelet.project", self._project)

        # Propagate session/user attributes from agentic_session context
        session_id = _session_id_var.get()
        if session_id is not None:
            span.set_attribute(SESSION_ID_ATTR, session_id)
        user_id = _user_id_var.get()
        if user_id is not None:
            span.set_attribute(USER_ID_ATTR, user_id)
        agent_name = _agent_name_var.get()
        if agent_name is not None:
            span.set_attribute(AGENT_NAME_ATTR, agent_name)

        self._wrapped.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        self._wrapped.on_end(span)

    def shutdown(self) -> None:
        self._wrapped.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._wrapped.force_flush(timeout_millis)


def _shutdown_processors() -> None:
    """Shutdown all active processors. Called via atexit."""
    for processor in _active_processors:
        try:
            processor.shutdown()
        except Exception:
            pass  # Best effort shutdown


# Register shutdown hook
atexit.register(_shutdown_processors)


def create_kelet_processor(
    *,
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    base_url: Optional[str] = None,
) -> SpanProcessor:
    """Create a Kelet SpanProcessor for manual TracerProvider setup.

    Use this when integrating with other OTEL-based tools (logfire, etc.)
    that manage their own TracerProvider.

    Args:
        api_key: API key (default: KELET_API_KEY env var)
        project: Project name (default: KELET_PROJECT env var or "default")
        base_url: API base URL (default: KELET_API_URL env var or "https://api.kelet.ai")

    Returns:
        A SpanProcessor that exports spans to Kelet.

    Raises:
        ValueError: If KELET_API_KEY is not provided.

    Example:
        import logfire
        import kelet

        logfire.configure(
            additional_span_processors=[kelet.create_kelet_processor()]
        )
        logfire.instrument_pydantic_ai()
        kelet.configure()  # Sets up config for signal() API
    """
    cfg = _resolve_config(api_key, project, base_url)

    # Create OTLP exporter to send traces to Kelet
    exporter = OTLPSpanExporter(
        endpoint=f"{cfg.base_url}/api/traces",
        headers={
            "Authorization": cfg.api_key,
            "X-Kelet-Project": cfg.project,
        },
    )
    batch_processor = BatchSpanProcessor(exporter)

    # Wrap with processor that adds kelet.project attribute to every span
    processor = _KeletSpanProcessor(batch_processor, cfg.project)

    # Track for shutdown
    _active_processors.append(processor)

    return processor


def shutdown() -> None:
    """Explicitly shutdown Kelet SDK and flush pending spans.

    This is called automatically via atexit, but can be called manually
    to ensure spans are flushed before process exit.
    """
    _shutdown_processors()


def configure(
    *,
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    base_url: Optional[str] = None,
    auto_instrument: bool = True,
    additional_span_processors: Optional[Sequence[SpanProcessor]] = None,
) -> None:
    """Configure Kelet SDK.

    If a TracerProvider already exists (e.g., from logfire), adds Kelet's
    processor to it. Otherwise, creates a new TracerProvider.

    Args:
        api_key: API key (default: KELET_API_KEY env var)
        project: Project name (default: KELET_PROJECT env var or "default")
        base_url: API base URL (default: KELET_API_URL env var or "https://api.kelet.ai")
        auto_instrument: Auto-instrument pydantic-ai (default: True). Only applies
                        when creating a new TracerProvider.
        additional_span_processors: Extra SpanProcessors to add (e.g., logfire's processor).
                                   Only applies when creating a new TracerProvider.

    Raises:
        ValueError: If KELET_API_KEY is not provided.
        RuntimeError: If existing TracerProvider doesn't support add_span_processor.

    Environment variables:
        KELET_API_KEY: API key
        KELET_PROJECT: Project name
        KELET_API_URL: Base URL

    Example:
        import kelet
        kelet.configure()
    """
    cfg = _resolve_config(api_key, project, base_url)

    # Store config for signal() and other API calls
    config = KeletConfig(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        project=cfg.project,
    )
    set_config(config)

    kelet_processor = create_kelet_processor(
        api_key=cfg.api_key,
        project=cfg.project,
        base_url=cfg.base_url,
    )

    # Check if a TracerProvider already exists (not just the default proxy)
    existing_provider = trace.get_tracer_provider()
    is_noop = isinstance(existing_provider, ProxyTracerProvider)

    if not is_noop:
        # Existing provider found - try to add our processor
        if not hasattr(existing_provider, "add_span_processor"):
            raise RuntimeError(
                f"Existing TracerProvider ({type(existing_provider).__name__}) "
                "does not support add_span_processor. "
                "Add kelet.create_kelet_processor() to it manually."
            )
        existing_provider.add_span_processor(kelet_processor)  # type: ignore[union-attr]
    else:
        # No existing provider - create our own
        set_global_textmap(
            CompositePropagator(
                [
                    TraceContextTextMapPropagator(),
                    W3CBaggagePropagator(),
                ]
            )
        )

        provider = TracerProvider()
        provider.add_span_processor(kelet_processor)

        # Add any additional processors
        if additional_span_processors:
            for processor in additional_span_processors:
                provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)

        # Auto-instrument frameworks (only when creating our own provider)
        if auto_instrument:
            _auto_instrument_frameworks()


def _auto_instrument_frameworks() -> None:
    """Detect and instrument supported frameworks.

    Currently supports:
    - pydantic-ai: Auto-instruments all Agent instances
    - Anthropic (openinference)
    - OpenAI (openinference)
    - LangChain (openinference, also covers LangGraph)
    """
    try:
        from pydantic_ai import Agent  # pyright: ignore [reportMissingImports]

        Agent.instrument_all()
    except ImportError:
        pass  # pydantic-ai not installed

    # Anthropic (openinference)
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().instrument()
    except ImportError:
        pass

    # OpenAI (openinference)
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
    except ImportError:
        pass

    # LangChain (also covers LangGraph — no dedicated langgraph package exists)
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
    except ImportError:
        pass
