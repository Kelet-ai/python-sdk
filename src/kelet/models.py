"""Signal models for Kelet SDK."""

import json
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, field_validator

from .json_encoder import AdvancedJsonEncoder


class InsensitiveEnum(str, Enum):
    """Case-insensitive string enum base class."""

    @classmethod
    def _missing_(cls, value: object) -> "InsensitiveEnum | None":
        if isinstance(value, str):
            value_upper = value.upper()
            for member in cls:
                if member.value.upper() == value_upper:
                    return member
        return None


class SignalKind(InsensitiveEnum):
    """Kind of signal — what type of observation this is."""

    FEEDBACK = "feedback"
    EDIT = "edit"
    EVENT = "event"
    METRIC = "metric"
    ARBITRARY = "arbitrary"


class SignalSource(InsensitiveEnum):
    """Source of the signal — who/what generated it."""

    HUMAN = "human"
    LABEL = "label"
    SYNTHETIC = "synthetic"


class Signal(BaseModel):
    """Signal model for API requests."""

    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    kind: SignalKind
    source: SignalSource
    trigger_name: Optional[str] = None
    score: Optional[float] = None
    value: Optional[Any] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 1):
            raise ValueError("score must be between 0 and 1 (inclusive)")
        return v

    @field_validator("value")
    @classmethod
    def serialize_value(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            return json.dumps(v, cls=AdvancedJsonEncoder)
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 1):
            raise ValueError("confidence must be between 0 and 1 (inclusive)")
        return v
