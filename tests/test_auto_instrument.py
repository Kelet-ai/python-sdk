"""Unit tests for _auto_instrument_frameworks using mocks."""

from unittest.mock import patch, MagicMock
import sys

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

    with patch.dict(sys.modules, {"openinference.instrumentation.anthropic": mock_module}):
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

    with patch.dict(sys.modules, {"openinference.instrumentation.langchain": mock_module}):
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

    with patch.dict(sys.modules, {
        "openinference.instrumentation.anthropic": None,  # simulate missing
        "openinference.instrumentation.openai": openai_module,
        "openinference.instrumentation.langchain": langchain_module,
    }):
        _auto_instrument_frameworks()

    openai_inst.instrument.assert_called_once()
    langchain_inst.instrument.assert_called_once()
