"""Unit tests for _auto_instrument_frameworks using mocks."""

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
# LiteLLM (via openinference-instrumentation-litellm)
# ---------------------------------------------------------------------------


def test_litellm_instrumented_when_installed():
    """LiteLLMInstrumentor.instrument() is called when package is installed."""
    inst_class, inst_instance = _make_instrumentor_mock()
    mock_module = MagicMock()
    mock_module.LiteLLMInstrumentor = inst_class

    with patch.dict(
        sys.modules, {"openinference.instrumentation.litellm": mock_module}
    ):
        _auto_instrument_frameworks()

    inst_class.assert_called_once()
    inst_instance.instrument.assert_called_once()


def test_litellm_skipped_when_not_installed():
    """No exception when the openinference litellm instrumentor is absent."""
    with patch.dict(sys.modules, {"openinference.instrumentation.litellm": None}):
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
    """LiteLLM and ADK OpenInference instrumentations coexist in the same process."""
    litellm_class, litellm_inst = _make_instrumentor_mock()
    litellm_module = MagicMock()
    litellm_module.LiteLLMInstrumentor = litellm_class

    adk_class, adk_inst = _make_instrumentor_mock()
    adk_module = MagicMock()
    adk_module.GoogleADKInstrumentor = adk_class

    with patch.dict(
        sys.modules,
        {
            "openinference.instrumentation.litellm": litellm_module,
            "openinference.instrumentation.google_adk": adk_module,
        },
    ):
        _auto_instrument_frameworks()

    litellm_inst.instrument.assert_called_once()
    adk_inst.instrument.assert_called_once()


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
