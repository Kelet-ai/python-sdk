"""Unit tests for configuration resolution."""

import logging

import pytest

from kelet._config import get_config, is_configured
from kelet._configure import _resolve_config, configure


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


def test_configure_warns_and_noops_on_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """configure() with no args and no env must warn and install no-op."""
    monkeypatch.delenv("KELET_API_KEY", raising=False)
    monkeypatch.delenv("KELET_PROJECT", raising=False)

    with caplog.at_level(logging.WARNING, logger="kelet._configure"):
        configure()

    assert not is_configured()
    assert "Kelet telemetry disabled" in caplog.text
    assert "KELET_API_KEY required" in caplog.text


def test_configure_warns_and_noops_on_missing_project(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """configure() with api_key but no project must also warn and no-op."""
    monkeypatch.setenv("KELET_API_KEY", "test-key")
    monkeypatch.delenv("KELET_PROJECT", raising=False)

    with caplog.at_level(logging.WARNING, logger="kelet._configure"):
        configure()

    assert not is_configured()
    assert "Kelet telemetry disabled" in caplog.text
    assert "KELET_PROJECT required" in caplog.text


def test_configure_strict_raises_on_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    """configure(strict=True) must re-raise ValueError instead of disabling."""
    monkeypatch.delenv("KELET_API_KEY", raising=False)
    monkeypatch.delenv("KELET_PROJECT", raising=False)

    with pytest.raises(ValueError, match="KELET_API_KEY required"):
        configure(strict=True)


def test_configure_strict_raises_on_missing_project(monkeypatch: pytest.MonkeyPatch):
    """configure(strict=True) re-raises for missing project too."""
    monkeypatch.setenv("KELET_API_KEY", "test-key")
    monkeypatch.delenv("KELET_PROJECT", raising=False)

    with pytest.raises(ValueError, match="KELET_PROJECT required"):
        configure(strict=True)
