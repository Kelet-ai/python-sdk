"""Unit tests for _auto_instrument_frameworks using mocks."""

import os
import sys
from unittest.mock import patch, MagicMock

from kelet._configure import _auto_instrument_frameworks


def _make_instrumentor_mock():
    """Create a mock instrumentor class that tracks .instrument() calls."""
    inst_instance = MagicMock()
    inst_class = MagicMock(return_value=inst_instance)
    return inst_class, inst_instance


def test_anthropic_instrumented_when_installed():
    """AnthropicInstrumentor.instrument() is called when package is installed."""
    inst_class, inst_instance = _make_instrumentor_mock()
    mock_module = MagicMock()
    mock_module.AnthropicInstrumentor = inst_class

    with patch.dict(
        sys.modules, {"openinference.instrumentation.anthropic": mock_module}
    ):
        _auto_instrument_frameworks()

    inst_class.assert_called_once()
    inst_instance.instrument.assert_called_once()


def test_openai_instrumented_when_installed():
    """OpenAIInstrumentor.instrument() is called when package is installed."""
    inst_class, inst_instance = _make_instrumentor_mock()
    mock_module = MagicMock()
    mock_module.OpenAIInstrumentor = inst_class

    with patch.dict(sys.modules, {"openinference.instrumentation.openai": mock_module}):
        _auto_instrument_frameworks()

    inst_class.assert_called_once()
    inst_instance.instrument.assert_called_once()


def test_langchain_instrumented_when_installed():
    """LangChainInstrumentor.instrument() is called when package is installed."""
    inst_class, inst_instance = _make_instrumentor_mock()
    mock_module = MagicMock()
    mock_module.LangChainInstrumentor = inst_class

    with patch.dict(
        sys.modules, {"openinference.instrumentation.langchain": mock_module}
    ):
        _auto_instrument_frameworks()

    inst_class.assert_called_once()
    inst_instance.instrument.assert_called_once()


def test_anthropic_skipped_when_not_installed():
    """No exception when anthropic openinference package is absent."""
    # Ensure the module is not present
    with patch.dict(sys.modules, {"openinference.instrumentation.anthropic": None}):
        # Should not raise
        _auto_instrument_frameworks()


def test_partial_install_independent():
    """anthropic raises ImportError; openai + langchain stubs are still called."""
    openai_class, openai_inst = _make_instrumentor_mock()
    openai_module = MagicMock()
    openai_module.OpenAIInstrumentor = openai_class

    langchain_class, langchain_inst = _make_instrumentor_mock()
    langchain_module = MagicMock()
    langchain_module.LangChainInstrumentor = langchain_class

    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.anthropic": None,  # simulate missing
            "openinference.instrumentation.openai": openai_module,
            "openinference.instrumentation.langchain": langchain_module,
        },
    ):
        _auto_instrument_frameworks()

    openai_inst.instrument.assert_called_once()
    langchain_inst.instrument.assert_called_once()


# ---------------------------------------------------------------------------
# LiteLLM
# ---------------------------------------------------------------------------


class _FakeOpenTelemetry:
    """Stub that stands in for litellm.integrations.opentelemetry.OpenTelemetry."""


def _make_litellm_mock(existing_callbacks=None):
    """Return a mock litellm module and its integrations.opentelemetry sub-module."""
    litellm_mod = MagicMock()
    litellm_mod.callbacks = existing_callbacks if existing_callbacks is not None else []
    litellm_mod.success_callback = []
    litellm_mod.failure_callback = []
    litellm_mod._async_success_callback = []
    litellm_mod._async_failure_callback = []
    litellm_mod.service_callback = []

    otel_mod = MagicMock()
    otel_mod.OpenTelemetry = _FakeOpenTelemetry

    return litellm_mod, otel_mod


def test_litellm_otel_registered_when_installed():
    """'otel' is appended when litellm is installed and callbacks is empty."""
    litellm_mod, otel_mod = _make_litellm_mock([])

    with patch.dict(
        sys.modules,
        {"litellm": litellm_mod, "litellm.integrations.opentelemetry": otel_mod},
    ):
        _auto_instrument_frameworks()

    assert litellm_mod.callbacks == ["otel"]


def test_litellm_otel_not_duplicated_string():
    """'otel' is not added again if the string is already in callbacks."""
    litellm_mod, otel_mod = _make_litellm_mock(["otel"])

    with patch.dict(
        sys.modules,
        {"litellm": litellm_mod, "litellm.integrations.opentelemetry": otel_mod},
    ):
        _auto_instrument_frameworks()

    assert litellm_mod.callbacks.count("otel") == 1


def test_litellm_otel_not_duplicated_instance():
    """'otel' string is not added if an OpenTelemetry() instance is already present."""
    litellm_mod, otel_mod = _make_litellm_mock()
    litellm_mod.callbacks = [_FakeOpenTelemetry()]  # simulate user manually added an instance

    with patch.dict(
        sys.modules,
        {"litellm": litellm_mod, "litellm.integrations.opentelemetry": otel_mod},
    ):
        _auto_instrument_frameworks()

    # Should not have added "otel" string alongside the existing instance
    assert "otel" not in litellm_mod.callbacks


def test_litellm_existing_callbacks_preserved():
    """Non-otel callbacks are kept when 'otel' is appended."""
    litellm_mod, otel_mod = _make_litellm_mock(["some_handler"])

    with patch.dict(
        sys.modules,
        {"litellm": litellm_mod, "litellm.integrations.opentelemetry": otel_mod},
    ):
        _auto_instrument_frameworks()

    assert litellm_mod.callbacks == ["some_handler", "otel"]


def test_litellm_prefers_request_spans_when_env_unset(monkeypatch):
    """Auto-instrumentation enables nested LiteLLM request spans by default."""
    litellm_mod, otel_mod = _make_litellm_mock([])
    monkeypatch.delenv("USE_OTEL_LITELLM_REQUEST_SPAN", raising=False)

    with patch.dict(
        sys.modules,
        {"litellm": litellm_mod, "litellm.integrations.opentelemetry": otel_mod},
    ):
        _auto_instrument_frameworks()

    assert os.environ["USE_OTEL_LITELLM_REQUEST_SPAN"] == "true"


def test_litellm_existing_request_span_env_preserved(monkeypatch):
    """Auto-instrumentation does not override an explicit LiteLLM span preference."""
    litellm_mod, otel_mod = _make_litellm_mock([])
    monkeypatch.setenv("USE_OTEL_LITELLM_REQUEST_SPAN", "false")

    with patch.dict(
        sys.modules,
        {"litellm": litellm_mod, "litellm.integrations.opentelemetry": otel_mod},
    ):
        _auto_instrument_frameworks()

    assert os.environ["USE_OTEL_LITELLM_REQUEST_SPAN"] == "false"


def test_litellm_otel_not_duplicated_from_success_callback():
    """General callbacks stay unchanged if OTEL is already registered as a success callback."""
    litellm_mod, otel_mod = _make_litellm_mock(["some_handler"])
    litellm_mod.success_callback = ["otel"]

    with patch.dict(
        sys.modules,
        {"litellm": litellm_mod, "litellm.integrations.opentelemetry": otel_mod},
    ):
        _auto_instrument_frameworks()

    assert litellm_mod.callbacks == ["some_handler"]


def test_litellm_otel_not_duplicated_from_failure_callback_instance():
    """General callbacks stay unchanged if an OTEL instance is already registered elsewhere."""
    litellm_mod, otel_mod = _make_litellm_mock(["some_handler"])
    litellm_mod.failure_callback = [_FakeOpenTelemetry()]

    with patch.dict(
        sys.modules,
        {"litellm": litellm_mod, "litellm.integrations.opentelemetry": otel_mod},
    ):
        _auto_instrument_frameworks()

    assert litellm_mod.callbacks == ["some_handler"]


def test_litellm_skipped_when_not_installed():
    """No exception when litellm is not installed."""
    with patch.dict(sys.modules, {"litellm": None}):
        _auto_instrument_frameworks()  # must not raise


# ---------------------------------------------------------------------------
# Google ADK
# ---------------------------------------------------------------------------


def test_google_adk_openinference_instrumented_when_installed():
    """GoogleADKInstrumentor.instrument() is preferred when OpenInference is installed."""
    inst_class, inst_instance = _make_instrumentor_mock()
    mock_module = MagicMock()
    mock_module.GoogleADKInstrumentor = inst_class

    with patch.dict(
        sys.modules, {"openinference.instrumentation.google_adk": mock_module}
    ):
        _auto_instrument_frameworks()

    inst_class.assert_called_once()
    inst_instance.instrument.assert_called_once()


def test_google_adk_openinference_preferred_over_native_telemetry():
    """OpenInference ADK instrumentation wins when both OpenInference and native ADK telemetry are available."""
    inst_class, inst_instance = _make_instrumentor_mock()
    mock_module = MagicMock()
    mock_module.GoogleADKInstrumentor = inst_class
    adk_telemetry_mod = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.google_adk": mock_module,
            "google.adk.telemetry": adk_telemetry_mod,
        },
    ):
        _auto_instrument_frameworks()

    inst_class.assert_called_once()
    inst_instance.instrument.assert_called_once()


def test_google_adk_native_telemetry_imported_as_fallback():
    """Native google.adk.telemetry is used only when the OpenInference instrumentor is absent."""
    adk_telemetry_mod = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.google_adk": None,
            "google.adk.telemetry": adk_telemetry_mod,
        },
    ):
        _auto_instrument_frameworks()
        assert sys.modules.get("google.adk.telemetry") is adk_telemetry_mod


def test_litellm_and_google_adk_can_be_present_together():
    """LiteLLM OTEL registration still works when ADK OpenInference instrumentation is available in the same process."""
    litellm_mod, otel_mod = _make_litellm_mock([])
    adk_inst_class, adk_inst_instance = _make_instrumentor_mock()
    adk_module = MagicMock()
    adk_module.GoogleADKInstrumentor = adk_inst_class

    with patch.dict(
        sys.modules,
        {
            "litellm": litellm_mod,
            "litellm.integrations.opentelemetry": otel_mod,
            "openinference.instrumentation.google_adk": adk_module,
        },
    ):
        _auto_instrument_frameworks()

    assert litellm_mod.callbacks == ["otel"]
    adk_inst_class.assert_called_once()
    adk_inst_instance.instrument.assert_called_once()


def test_google_adk_skipped_when_not_installed():
    """No exception when google-adk is not installed."""
    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.google_adk": None,
            "google.adk.telemetry": None,
        },
    ):
        _auto_instrument_frameworks()  # must not raise
