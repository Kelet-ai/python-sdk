"""Signal submission for Kelet SDK."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from ._config import get_config
from ._context import get_session_id, get_trace_id
from ._temporal_detect import in_temporal_workflow
from .models import Signal, SignalKind, SignalSource

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 0.5  # seconds
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Warn at most once per process when signal() is called without configuration.
_warned_unconfigured = False


def _warn_unconfigured_once() -> None:
    global _warned_unconfigured
    if not _warned_unconfigured:
        _warned_unconfigured = True
        logger.warning(
            "kelet.signal() called before configure() and KELET_API_KEY/KELET_PROJECT "
            "not set — dropping signal. Call kelet.configure() to enable."
        )


def _reset_warn_state() -> None:
    """Reset the warn-once flag. For testing only."""
    global _warned_unconfigured
    _warned_unconfigured = False


class _SignalArgs(BaseModel):
    """Arguments for ``_kelet_signal_activity``.

    Bundled into one Pydantic model so the activity has a single positional
    arg. Fields mirror the public ``signal()`` API but use already-resolved
    session_id / trace_id (the workflow side does the resolution while it
    still has access to context vars).
    """

    kind: SignalKind
    source: SignalSource
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    trigger_name: Optional[str] = None
    score: Optional[float] = None
    value: Optional[Any] = None
    confidence: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    timestamp: Optional[str] = None  # ISO-formatted; serialized in caller
    raise_on_failure: bool = False


async def signal(
    kind: SignalKind,
    source: SignalSource,
    *,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    trigger_name: Optional[str] = None,
    score: Optional[float] = None,
    value: Optional[Any] = None,
    confidence: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
    raise_on_failure: bool = False,
) -> None:
    """Submit a signal for an AI session.

    Signals provide observations about AI responses, enabling continuous improvement.
    They can be linked to sessions via session_id or trace_id.

    Includes retry logic with exponential backoff for transient failures.

    When called from inside a Temporal workflow, ``signal()`` dispatches via a
    Temporal activity (auto-registered by ``KeletPlugin``) so the HTTP call is
    durable and retried by Temporal — required because httpx is non-deterministic
    and can't be invoked from workflow code directly. Final failure behavior in
    that path is controlled by ``kelet.configure(signal_failure_mode=...)``.

    Args:
        kind: Signal kind (feedback, edit, event, metric, arbitrary)
        source: Signal source (human, label, synthetic)
        session_id: Session identifier (optional, auto-detected from context)
        trace_id: Trace identifier (optional, auto-detected from span)
        trigger_name: Name of the trigger (e.g., "thumbs_down", "user_copy")
        score: Score value (0.0 to 1.0)
        value: Text content (feedback text, diff, reasoning, etc.)
        confidence: Confidence level (0.0 to 1.0)
        metadata: Additional metadata
        timestamp: Event timestamp
        raise_on_failure: Re-raise request/transport failures after retries.
                          Defaults to False, which logs and returns.

    Raises:
        ValueError: If neither session_id nor trace_id can be determined
        ValueError: If score or confidence is outside 0-1 range
        httpx.HTTPStatusError: If an HTTP error occurs after retries and
                               raise_on_failure is True
        httpx.ConnectError: If a connection error occurs after retries and
                            raise_on_failure is True
        httpx.TimeoutException: If a timeout occurs after retries and
                                raise_on_failure is True

    Example:
        with kelet.agentic_session(session_id="session-123"):
            await kelet.signal(
                kind=kelet.SignalKind.FEEDBACK,
                source=kelet.SignalSource.HUMAN,
                score=1.0,
            )

        await kelet.signal(
            kind=kelet.SignalKind.METRIC,
            source=kelet.SignalSource.SYNTHETIC,
            trace_id="abc123",
            score=0.85,
            trigger_name="accuracy",
        )
    """
    # Resolve identifier with priority. Done on the caller side so that
    # workflow context vars (set by the Kelet temporal interceptor's lite
    # mode) propagate into the activity dispatch.
    resolved_session_id = session_id or get_session_id()
    resolved_trace_id = trace_id or (
        get_trace_id() if not resolved_session_id else None
    )

    if not resolved_session_id and not resolved_trace_id:
        raise ValueError(
            "Either session_id or trace_id required. "
            "Use agentic_session() context, or pass explicitly."
        )

    resolved_timestamp: Optional[str] = (
        timestamp.isoformat() if timestamp is not None else None
    )

    # Workflow-context dispatch: route through the registered Temporal activity
    # so the HTTP call is non-deterministic-safe and retried by Temporal.
    if in_temporal_workflow():
        await _dispatch_via_activity(
            _SignalArgs(
                kind=kind,
                source=source,
                session_id=resolved_session_id,
                trace_id=resolved_trace_id,
                trigger_name=trigger_name,
                score=score,
                value=value,
                confidence=confidence,
                metadata=metadata,
                timestamp=resolved_timestamp,
                raise_on_failure=raise_on_failure,
            )
        )
        return

    await _send_signal(
        kind=kind,
        source=source,
        session_id=resolved_session_id,
        trace_id=resolved_trace_id,
        trigger_name=trigger_name,
        score=score,
        value=value,
        confidence=confidence,
        metadata=metadata,
        timestamp=resolved_timestamp,
        raise_on_failure=raise_on_failure,
    )


_KELET_PLUGIN_INSTALLED_MSG = (
    "kelet.signal() called from workflow code, but the _kelet_signal activity "
    "is not registered on this worker. Use `KeletPlugin` (not the standalone "
    "`KeletInterceptor`) so the activity is auto-registered, or call "
    "`kelet.signal()` only from activity / orchestrator code where direct "
    "HTTP is safe."
)


async def _dispatch_via_activity(args: _SignalArgs) -> None:
    """Dispatch signal through ``_kelet_signal_activity`` from inside a workflow.

    Imported lazily so that ``signal()`` callers outside Temporal don't pay
    the import cost (and so this module stays importable when ``temporalio``
    is not installed).

    If ``_kelet_signal_activity`` is not registered on the worker (the user
    wired ``KeletInterceptor`` standalone instead of ``KeletPlugin``), Temporal
    raises ``ActivityError`` wrapping an ``ApplicationError`` with a
    NotFoundError type. We surface that as a clear Kelet-side ``RuntimeError``
    pointing users at the fix — silently falling back to direct httpx would
    corrupt workflow replay determinism, so fail loudly instead.
    """
    from temporalio import workflow  # type: ignore[reportMissingImports]
    from temporalio.common import RetryPolicy  # type: ignore[reportMissingImports]
    from temporalio.exceptions import ActivityError, ApplicationError  # type: ignore[reportMissingImports]

    try:
        await workflow.execute_activity(
            _kelet_signal_activity,
            args,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
    except ActivityError as e:
        cause = e.__cause__
        if isinstance(cause, ApplicationError) and cause.type == "NotFoundError":
            raise RuntimeError(_KELET_PLUGIN_INSTALLED_MSG) from e
        raise


async def _kelet_signal_activity(args: _SignalArgs) -> None:
    """Activity-side signal sender. Registered by ``KeletPlugin``.

    On final failure (after Temporal's retries exhaust), behavior is governed
    by ``kelet.configure(signal_failure_mode=...)``:
      - ``"swallow"`` (default): log warning, return None.
      - ``"raise"``: re-raise so Temporal records ``ApplicationError``.

    Defined here (not via ``@activity.defn``) so this module stays importable
    without ``temporalio`` installed; ``KeletPlugin`` wraps the function with
    ``@activity.defn(name="_kelet_signal")`` at registration time.
    """
    try:
        await _send_signal(
            kind=args.kind,
            source=args.source,
            session_id=args.session_id,
            trace_id=args.trace_id,
            trigger_name=args.trigger_name,
            score=args.score,
            value=args.value,
            confidence=args.confidence,
            metadata=args.metadata,
            timestamp=args.timestamp,
            raise_on_failure=True,  # always raise inside the activity body
        )
    except Exception as e:
        try:
            cfg = get_config()
            mode = cfg.signal_failure_mode
        except ValueError:
            # No config — fall back to swallow so we don't fail user workflows
            # for telemetry that's already going nowhere.
            mode = "swallow"
        if mode == "raise":
            raise
        # default: swallow + log
        try:
            from temporalio import activity  # type: ignore[reportMissingImports]

            activity.logger.warning(
                "kelet.signal() failed after retries: %s", e, exc_info=True
            )
        except Exception:
            logger.warning("kelet.signal() failed after retries: %s", e, exc_info=True)


async def _send_signal(
    *,
    kind: SignalKind,
    source: SignalSource,
    session_id: Optional[str],
    trace_id: Optional[str],
    trigger_name: Optional[str],
    score: Optional[float],
    value: Optional[Any],
    confidence: Optional[float],
    metadata: Optional[dict[str, Any]],
    timestamp: Optional[str],
    raise_on_failure: bool,
) -> None:
    """Direct HTTP send. Reuses the pooled httpx client cached on KeletConfig
    so signals from inside a single activity invocation share connections.
    """
    try:
        config = get_config()
    except ValueError:
        # Missing KELET_API_KEY / KELET_PROJECT at the process level.
        # configure() would have already warned if it was called;
        # if not, warn once here so the silent drop is traceable.
        _warn_unconfigured_once()
        return
    client = await config.get_client()

    url = f"{config.base_url}/api/projects/{config.project}/signal"

    payload = Signal(
        kind=kind,
        source=source,
        session_id=session_id,
        trace_id=trace_id,
        trigger_name=trigger_name,
        score=score,
        value=value,
        confidence=confidence,
        metadata=metadata,
        timestamp=timestamp,
    )

    # Retry with exponential backoff
    last_error: Optional[Exception] = None
    attempt_count = 0
    for attempt in range(_MAX_RETRIES):
        attempt_count = attempt + 1
        try:
            response = await client.post(
                url, json=payload.model_dump(exclude_none=True)
            )
            response.raise_for_status()
            return
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in _RETRYABLE_STATUS_CODES:
                if raise_on_failure:
                    raise
                last_error = e
                break
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                wait_time = _RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    f"Signal request failed (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                wait_time = _RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    f"Signal request failed (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)

    # All retries exhausted
    if last_error:
        logger.warning(
            "Signal request failed after %s attempt(s): %s",
            attempt_count,
            last_error,
        )
        if raise_on_failure:
            raise last_error
