"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def reset_kelet_config():
    """Reset module-level config singleton before and after every test."""
    from kelet._config import reset_config

    reset_config()
    yield
    reset_config()
