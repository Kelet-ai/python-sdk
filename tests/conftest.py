"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def reset_kelet_config():
    """Reset module-level config singleton and warn-once flags before/after each test."""
    from kelet._config import reset_config
    from kelet._signal import _reset_warn_state

    reset_config()
    _reset_warn_state()
    yield
    reset_config()
    _reset_warn_state()
