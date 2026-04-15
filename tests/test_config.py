"""Unit tests for configuration resolution."""

import pytest

from kelet._config import get_config
from kelet._configure import _resolve_config


def test_get_config_raises_when_project_missing(monkeypatch):
    """get_config() must raise when KELET_PROJECT is not set."""
    monkeypatch.setenv("KELET_API_KEY", "test-key")
    monkeypatch.delenv("KELET_PROJECT", raising=False)

    with pytest.raises(ValueError, match="KELET_PROJECT required"):
        get_config()


def test_get_config_succeeds_when_project_set(monkeypatch):
    """get_config() must succeed when both env vars are set."""
    monkeypatch.setenv("KELET_API_KEY", "test-key")
    monkeypatch.setenv("KELET_PROJECT", "my-project")

    config = get_config()
    assert config.project == "my-project"
    assert config.api_key == "test-key"


def test_resolve_config_raises_when_project_missing(monkeypatch):
    """_resolve_config() must raise when project not passed and KELET_PROJECT not set."""
    monkeypatch.delenv("KELET_PROJECT", raising=False)

    with pytest.raises(ValueError, match="KELET_PROJECT required"):
        _resolve_config(api_key="test-key")


def test_resolve_config_succeeds_with_explicit_project(monkeypatch):
    """_resolve_config() must succeed when project is passed explicitly."""
    monkeypatch.delenv("KELET_PROJECT", raising=False)

    config = _resolve_config(api_key="test-key", project="my-project")
    assert config.project == "my-project"
