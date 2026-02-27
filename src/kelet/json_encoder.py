import dataclasses
import json
from enum import Enum
from logging import warning
from typing import (
    Callable,
    Protocol,
    runtime_checkable,
    ClassVar,
    Any,
    Union,
    Dict,
    List,
)
from dataclasses import Field as DataclassField

try:
    from pydantic import BaseModel

    HAVE_PYDANTIC = True
except ImportError:
    HAVE_PYDANTIC = False
    BaseModel = type(None)  # Create a dummy type that won't match anything

try:
    import pandas as pd

    def is_series(obj):  # pyright: ignore [reportRedeclaration]
        return isinstance(obj, pd.Series)

    def is_dataframe(obj):  # pyright: ignore [reportRedeclaration]
        return isinstance(obj, pd.DataFrame)

    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False

    def is_series(obj):
        return False

    def is_dataframe(obj):
        return False


@runtime_checkable
class DataclassInstance(Protocol):
    """Protocol for dataclass instances"""

    __dataclass_fields__: ClassVar[Dict[str, DataclassField[Any]]]


@runtime_checkable
class PydanticModel(Protocol):
    """Protocol for Pydantic model instances"""

    def model_dump(self, mode: str = "json") -> Dict[str, Any]: ...


@runtime_checkable
class PandasSeries(Protocol):
    """Protocol for Pandas Series"""

    def to_dict(self) -> Dict[Any, Any]: ...


@runtime_checkable
class PandasDataFrame(Protocol):
    """Protocol for Pandas DataFrame"""

    def to_dict(self, orient: str = "records") -> List[Dict[str, Any]]: ...


@runtime_checkable
class HasRepr(Protocol):
    """Protocol for objects with __repr__"""

    def __repr__(self) -> str: ...


# JSON primitive types
JsonPrimitive = Union[str, int, float, bool, None]

# Composite JSON types
JsonObject = Dict[str, Any]
JsonArray = List[Any]

# All types that AdvancedJsonEncoder can serialize
AdvancedJsonSerializable = Union[
    JsonPrimitive,
    JsonObject,
    JsonArray,
    DataclassInstance,
    PydanticModel,
    PandasSeries,
    PandasDataFrame,
    Enum,
    Callable[..., Any],
    HasRepr,  # Fallback for objects with __repr__
]


class AdvancedJsonEncoder(json.JSONEncoder):
    """JSON encoder that handles Pydantic models (if installed) and other special types."""

    # noinspection PyBroadException
    def default(self, o):
        if HAVE_PYDANTIC and isinstance(o, BaseModel):
            return o.model_dump(mode="json")  # type: ignore
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)  # type: ignore
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, Callable):
            warning("Serializing callable as a name for ")
            try:
                name = f"<{o.__module__}.{o.__name__}>"
                warning(f"Serializing callable as a name for {name}")
                return name
            except Exception:
                try:
                    name = f"<{o.__module__}.{o.__class__.__name__}>"
                    warning(f"Serializing callable as a name for {name}")
                    return name
                except Exception:
                    name = repr(o)
                    warning(f"Serializing callable as a name for {name}")
                    return name
        if isinstance(o, type(None)):
            return None
        if HAVE_PANDAS and is_series(o):
            return o.to_dict()
        if HAVE_PANDAS and is_dataframe(o):
            return o.to_dict(orient="records")
        if hasattr(o, "__repr__"):
            return repr(o)
        return super().default(o)
