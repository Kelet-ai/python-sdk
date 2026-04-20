"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def reset_kelet_config():
    """Reset module-level config singleton and warn-once flags before/after each test."""
    from kelet import _signal
    from kelet._config import reset_config

    reset_config()
    _signal._warned_unconfigured = False
    yield
    reset_config()
    _signal._warned_unconfigured = False
