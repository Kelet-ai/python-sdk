"""Unit tests for signal submission behavior."""

from unittest.mock import AsyncMock

import httpx
import pytest

from kelet._config import KeletConfig, set_config
from kelet._signal import signal
from kelet.models import SignalKind, SignalSource


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.kelet.ai/api/projects/test/signal")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"{status_code} error",
        request=request,
        response=response,
    )


@pytest.mark.asyncio
async def test_signal_warns_and_returns_on_retryable_failure_by_default(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Retryable request failures should warn and return by default."""
    client = AsyncMock()
    client.post.side_effect = _http_status_error(503)

    config = KeletConfig(
        api_key="test-key",
        base_url="https://api.kelet.ai",
        project="test",
    )
    config._http_client = client
    set_config(config)

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("kelet._signal.asyncio.sleep", _no_sleep)

    await signal(
        kind=SignalKind.FEEDBACK,
        source=SignalSource.HUMAN,
        session_id="session-123",
    )

    assert client.post.call_count == 3
    assert caplog.text.count("retrying in") == 2
    assert "Signal request failed after 3 attempt(s)" in caplog.text


@pytest.mark.asyncio
async def test_signal_warns_and_returns_on_non_retryable_failure_by_default(
    caplog: pytest.LogCaptureFixture,
):
    """Non-retryable request failures should warn and return by default."""
    client = AsyncMock()
    client.post.side_effect = _http_status_error(400)

    config = KeletConfig(
        api_key="test-key",
        base_url="https://api.kelet.ai",
        project="test",
    )
    config._http_client = client
    set_config(config)

    await signal(
        kind=SignalKind.FEEDBACK,
        source=SignalSource.HUMAN,
        session_id="session-123",
    )

    assert client.post.call_count == 1
    assert "retrying in" not in caplog.text
    assert "Signal request failed after 1 attempt(s)" in caplog.text


@pytest.mark.asyncio
async def test_signal_raises_on_failure_when_requested(
    monkeypatch: pytest.MonkeyPatch,
):
    """Request failures should still be raisable explicitly."""
    client = AsyncMock()
    client.post.side_effect = _http_status_error(503)

    config = KeletConfig(
        api_key="test-key",
        base_url="https://api.kelet.ai",
        project="test",
    )
    config._http_client = client
    set_config(config)

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("kelet._signal.asyncio.sleep", _no_sleep)

    with pytest.raises(httpx.HTTPStatusError, match="503 error"):
        await signal(
            kind=SignalKind.FEEDBACK,
            source=SignalSource.HUMAN,
            session_id="session-123",
            raise_on_failure=True,
        )


@pytest.mark.asyncio
async def test_signal_raises_immediately_on_non_retryable_failure_when_requested(
    caplog: pytest.LogCaptureFixture,
):
    """Non-retryable errors (4xx) raise immediately when raise_on_failure=True."""
    client = AsyncMock()
    client.post.side_effect = _http_status_error(400)

    config = KeletConfig(
        api_key="test-key",
        base_url="https://api.kelet.ai",
        project="test",
    )
    config._http_client = client
    set_config(config)

    with pytest.raises(httpx.HTTPStatusError, match="400 error"):
        await signal(
            kind=SignalKind.FEEDBACK,
            source=SignalSource.HUMAN,
            session_id="session-123",
            raise_on_failure=True,
        )

    assert client.post.call_count == 1
    assert "retrying in" not in caplog.text
