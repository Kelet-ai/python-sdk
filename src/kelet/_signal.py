"""Signal submission for Kelet SDK."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

import httpx

from ._config import get_config
from ._context import get_session_id, get_trace_id
from .models import Signal, SignalKind, SignalSource

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 0.5  # seconds
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


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
) -> None:
    """Submit a signal for an AI session.

    Signals provide observations about AI responses, enabling continuous improvement.
    They can be linked to sessions via session_id or trace_id.

    Includes retry logic with exponential backoff for transient failures.

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

    Raises:
        ValueError: If neither session_id nor trace_id can be determined
        ValueError: If score or confidence is outside 0-1 range
        httpx.HTTPStatusError: If the request fails after retries

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
    # Resolve identifier with priority
    resolved_session_id = session_id or get_session_id()
    resolved_trace_id = trace_id or (
        get_trace_id() if not resolved_session_id else None
    )

    if not resolved_session_id and not resolved_trace_id:
        raise ValueError(
            "Either session_id or trace_id required. "
            "Use agentic_session() context, or pass explicitly."
        )

    config = get_config()
    client = await config.get_client()

    url = f"{config.base_url}/api/projects/{config.project}/signal"

    # Serialize timestamp
    resolved_timestamp: Optional[str] = None
    if timestamp is not None:
        resolved_timestamp = timestamp.isoformat()

    payload = Signal(
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
    )

    # Retry with exponential backoff
    last_error: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.post(
                url, json=payload.model_dump(exclude_none=True)
            )
            response.raise_for_status()
            return
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in _RETRYABLE_STATUS_CODES:
                raise
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
        raise last_error
