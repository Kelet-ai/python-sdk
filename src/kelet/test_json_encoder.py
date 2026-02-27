"""Tests for AdvancedJsonEncoder."""

import json
from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch, MagicMock

import pytest

from kelet.json_encoder import AdvancedJsonEncoder


class SampleEnum(Enum):
    """Sample enum for serialization."""

    VALUE_A = "value_a"
    VALUE_B = "value_b"


@dataclass
class SampleDataclass:
    """Sample dataclass for serialization."""

    name: str
    value: int


def test_function():
    """Test function for callable serialization."""
    pass  # Test functions should not return values


class SampleObject:
    """Sample object with __repr__."""

    def __repr__(self):
        return "SampleObject(repr)"


class TestAdvancedJsonEncoder:
    """Test AdvancedJsonEncoder functionality."""

    def test_basic_json_types(self):
        """Test encoding of basic JSON types."""
        encoder = AdvancedJsonEncoder()
        data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        result = json.dumps(data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)
        assert parsed == data

    def test_enum_serialization(self):
        """Test enum serialization."""
        encoder = AdvancedJsonEncoder()
        data = {"enum_value": SampleEnum.VALUE_A}
        result = json.dumps(data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)
        assert parsed["enum_value"] == "value_a"

    def test_dataclass_serialization(self):
        """Test dataclass serialization."""
        encoder = AdvancedJsonEncoder()
        obj = SampleDataclass(name="test", value=123)
        data = {"dataclass": obj}
        result = json.dumps(data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)
        assert parsed["dataclass"] == {"name": "test", "value": 123}

    def test_callable_serialization_with_name(self):
        """Test callable serialization with function name."""
        encoder = AdvancedJsonEncoder()
        data = {"function": test_function}

        with patch("kelet.json_encoder.warning") as mock_warning:
            result = json.dumps(data, cls=AdvancedJsonEncoder)
            parsed = json.loads(result)

            # Should serialize to module.function_name format
            assert "test_json_encoder.test_function" in parsed["function"]
            # Should log warnings
            assert mock_warning.call_count >= 2

    def test_callable_serialization_fallback(self):
        """Test callable serialization fallback to repr."""
        encoder = AdvancedJsonEncoder()

        # Create a callable that will trigger exception handling
        class BadCallable:
            def __call__(self):
                return None

            # No __name__ or __module__ attributes to trigger fallback

        callable_obj = BadCallable()
        data = {"callable": callable_obj}

        with patch("kelet.json_encoder.warning") as mock_warning:
            result = json.dumps(data, cls=AdvancedJsonEncoder)
            parsed = json.loads(result)

            # Should fall back to repr since it has no __name__ or __module__
            assert "BadCallable" in parsed["callable"]
            assert mock_warning.called

    def test_none_type_serialization(self):
        """Test None type serialization."""
        encoder = AdvancedJsonEncoder()
        data = {"none_value": type(None)()}
        result = json.dumps(data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)
        assert parsed["none_value"] is None

    def test_repr_fallback(self):
        """Test fallback to __repr__ for unknown objects."""
        encoder = AdvancedJsonEncoder()
        obj = SampleObject()
        data = {"object": obj}
        result = json.dumps(data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)
        assert parsed["object"] == "SampleObject(repr)"

    def test_pydantic_serialization_when_available(self):
        """Test Pydantic model serialization when available."""
        from pydantic import BaseModel

        class SampleModel(BaseModel):
            name: str
            value: int

        obj = SampleModel(name="test", value=42)
        data = {"model": obj}
        result = json.dumps(data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)
        assert parsed["model"] == {"name": "test", "value": 42}

    def test_pydantic_serialization_when_unavailable(self):
        """Test behavior when Pydantic is not available."""
        encoder = AdvancedJsonEncoder()

        # Create a mock object that looks like a BaseModel but isn't
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "BaseModel"

        # Since HAVE_PYDANTIC is False, should fall back to repr
        with patch("kelet.json_encoder.HAVE_PYDANTIC", False):
            data = {"model": mock_model}
            result = json.dumps(data, cls=AdvancedJsonEncoder)
            parsed = json.loads(result)
            # Should serialize using __repr__ fallback
            assert isinstance(parsed["model"], str)

    def test_pandas_serialization_when_available(self):
        """Test pandas serialization when available."""
        pytest.importorskip("pandas")
        import pandas as pd

        series = pd.Series({"a": 1, "b": 2})
        data = {"series": series}
        result = json.dumps(data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)
        assert parsed["series"] == {"a": 1, "b": 2}

        df = pd.DataFrame([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        data = {"dataframe": df}
        result = json.dumps(data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)
        assert parsed["dataframe"] == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]

    def test_pandas_serialization_when_unavailable(self):
        """Test behavior when pandas is not available."""
        encoder = AdvancedJsonEncoder()

        # Create a mock object that looks like pandas but isn't
        mock_series = MagicMock()
        mock_df = MagicMock()

        with patch("kelet.json_encoder.HAVE_PANDAS", False):
            with patch("kelet.json_encoder.is_series", return_value=False):
                with patch("kelet.json_encoder.is_dataframe", return_value=False):
                    # Should fall back to repr since pandas helpers return False
                    data = {"series": mock_series, "dataframe": mock_df}
                    result = json.dumps(data, cls=AdvancedJsonEncoder)
                    parsed = json.loads(result)
                    assert isinstance(parsed["series"], str)
                    assert isinstance(parsed["dataframe"], str)

    def test_complex_nested_structure(self):
        """Test encoding of complex nested structures."""
        encoder = AdvancedJsonEncoder()

        nested_data = {
            "enum": SampleEnum.VALUE_B,
            "dataclass": SampleDataclass(name="nested", value=456),
            "list_with_objects": [
                SampleEnum.VALUE_A,
                SampleDataclass(name="in_list", value=789),
                {"nested_dict": SampleEnum.VALUE_B},
            ],
            "normal_data": {"string": "normal", "number": 42},
        }

        result = json.dumps(nested_data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)

        assert parsed["enum"] == "value_b"
        assert parsed["dataclass"] == {"name": "nested", "value": 456}
        assert parsed["list_with_objects"][0] == "value_a"
        assert parsed["list_with_objects"][1] == {"name": "in_list", "value": 789}
        assert parsed["list_with_objects"][2]["nested_dict"] == "value_b"
        assert parsed["normal_data"] == {"string": "normal", "number": 42}

    def test_unsupported_object_fallback(self):
        """Test fallback behavior for objects with __repr__."""
        encoder = AdvancedJsonEncoder()

        # Create an object that will use __repr__ fallback
        class CustomObject:
            def __repr__(self):
                return "CustomObject(test)"

        obj = CustomObject()
        data = {"custom": obj}

        # Should use __repr__ fallback
        result = json.dumps(data, cls=AdvancedJsonEncoder)
        parsed = json.loads(result)
        assert parsed["custom"] == "CustomObject(test)"

    def test_encoder_inheritance(self):
        """Test that AdvancedJsonEncoder properly inherits from JSONEncoder."""
        encoder = AdvancedJsonEncoder()
        assert isinstance(encoder, json.JSONEncoder)

        # Test that it can handle standard encoding
        standard_data = {"key": "value"}
        result = encoder.encode(standard_data)
        assert result == '{"key": "value"}'
