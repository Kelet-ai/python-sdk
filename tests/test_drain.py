"""Tests for _drain_background_logging_tasks.

Covers the LiteLLM-callback-race fix: drain must wait for pending tasks,
exempt the LiteLLM worker-loop task so it doesn't hang, be bounded by a
hard timeout, and no-op when LiteLLM isn't loaded.
"""

import asyncio
import sys
import time
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from kelet import _context
from kelet._context import _drain_background_logging_tasks, agentic_session


@pytest.fixture
def fast_drain(monkeypatch):
    """Shrink drain timeouts so tests stay snappy."""
    monkeypatch.setattr(_context, "_DRAIN_POLL_INTERVAL", 0.001)
    monkeypatch.setattr(_context, "_DRAIN_QUIET_ITERATIONS", 3)
    monkeypatch.setattr(_context, "_DRAIN_TIMEOUT_SECONDS", 1.0)


@pytest.mark.asyncio
async def test_drain_returns_quickly_when_no_pending_tasks(fast_drain):
    """With no other tasks in flight, the drain exits after the quiet period."""
    start = time.monotonic()
    await _drain_background_logging_tasks()
    elapsed = time.monotonic() - start
    # 3 × 1ms quiet + tiny overhead. Far below the 1s timeout.
    assert elapsed < 0.2, f"drain took {elapsed:.3f}s with no pending work"


@pytest.mark.asyncio
async def test_drain_waits_for_pending_task(fast_drain):
    """A background task in flight delays the drain until it finishes."""
    task_ran = asyncio.Event()

    async def background() -> None:
        await asyncio.sleep(0.05)
        task_ran.set()

    asyncio.create_task(background())
    await _drain_background_logging_tasks()
    assert task_ran.is_set(), "drain returned before background task completed"


@pytest.mark.asyncio
async def test_drain_timeout_does_not_hang(fast_drain):
    """If a task never finishes, drain bails out at the deadline, not sooner."""
    never_done = asyncio.Event()

    async def forever() -> None:
        await never_done.wait()

    hanger = asyncio.create_task(forever())
    try:
        start = time.monotonic()
        await _drain_background_logging_tasks()
        elapsed = time.monotonic() - start
        # _DRAIN_TIMEOUT_SECONDS=1.0 in the fixture. Allow slack for polling.
        assert 0.8 <= elapsed < 2.0, f"drain elapsed {elapsed:.3f}s, expected ~1.0s"
    finally:
        never_done.set()
        await hanger


@pytest.mark.asyncio
async def test_drain_exempts_litellm_worker_loop_task(fast_drain):
    """The long-lived LiteLLM worker-loop task must not keep drain spinning."""
    # Create a real pending task and stash it where _litellm_worker_loop_task
    # can find it — then assert drain returns promptly despite it being pending.
    never_done = asyncio.Event()

    async def worker_loop() -> None:
        await never_done.wait()

    worker_task = asyncio.create_task(worker_loop())
    try:
        fake_worker = MagicMock()
        fake_worker._worker_task = worker_task
        fake_mod = ModuleType(_context._LITELLM_WORKER_MODULE)
        fake_mod.GLOBAL_LOGGING_WORKER = fake_worker  # type: ignore[attr-defined]
        sys.modules[_context._LITELLM_WORKER_MODULE] = fake_mod
        try:
            start = time.monotonic()
            await _drain_background_logging_tasks()
            elapsed = time.monotonic() - start
            # Would be ~1.0s (timeout) without the exemption. With it, ~quiet period.
            assert elapsed < 0.2, (
                f"drain took {elapsed:.3f}s with exempted worker task; "
                "worker-loop exemption may be broken"
            )
        finally:
            sys.modules.pop(_context._LITELLM_WORKER_MODULE, None)
    finally:
        never_done.set()
        await worker_task


@pytest.mark.asyncio
async def test_drain_handles_missing_litellm_module(fast_drain):
    """When LiteLLM isn't imported, drain treats the worker as absent."""
    sys.modules.pop(_context._LITELLM_WORKER_MODULE, None)
    assert _context._litellm_worker_loop_task() is None
    await _drain_background_logging_tasks()  # must not raise


@pytest.mark.asyncio
async def test_drain_handles_worker_without_expected_attrs(fast_drain, caplog):
    """Future LiteLLM refactor removes _worker_task — drain degrades gracefully."""
    fake_mod = ModuleType(_context._LITELLM_WORKER_MODULE)
    fake_mod.GLOBAL_LOGGING_WORKER = MagicMock(
        spec=[]
    )  # no _worker_task  # type: ignore[attr-defined]
    sys.modules[_context._LITELLM_WORKER_MODULE] = fake_mod
    try:
        with caplog.at_level("DEBUG", logger="kelet._context"):
            assert _context._litellm_worker_loop_task() is None
            await _drain_background_logging_tasks()
        # A debug log must be emitted so the regression is observable.
        assert any("_worker_task missing" in rec.message for rec in caplog.records), (
            "expected debug log when worker _worker_task attribute is missing"
        )
    finally:
        sys.modules.pop(_context._LITELLM_WORKER_MODULE, None)


@pytest.mark.asyncio
async def test_agentic_session_async_exit_invokes_drain(fast_drain, monkeypatch):
    """``async with agentic_session(...)`` awaits the drain before _exit."""
    calls: list[str] = []
    real_drain = _context._drain_background_logging_tasks

    async def spy_drain() -> None:
        calls.append("drain")
        await real_drain()

    monkeypatch.setattr(_context, "_drain_background_logging_tasks", spy_drain)

    async with agentic_session(session_id="sess-drain-test"):
        calls.append("inside")

    assert calls == ["inside", "drain"], (
        f"drain must run after body, before exit. Got {calls}"
    )


@pytest.mark.asyncio
async def test_aexit_swallows_drain_exceptions(fast_drain, monkeypatch, caplog):
    """A broken drain must never break user control flow."""

    async def explode() -> None:
        raise RuntimeError("simulated drain failure")

    monkeypatch.setattr(_context, "_drain_background_logging_tasks", explode)

    with caplog.at_level("DEBUG", logger="kelet._context"):
        async with agentic_session(session_id="sess-explode"):
            pass  # must exit cleanly even though drain raised

    assert any(
        "drain_background_logging_tasks failed" in rec.message for rec in caplog.records
    ), "expected debug log when drain raises"
