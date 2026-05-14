"""
``inference`` — single façade for prediction and continuous learning.

Callers should import from here only:

    from inference import InferenceEngine, InferenceConfig
    from inference import PatientInput, LearningRecord, PredictionResult
    from inference import (
        InferenceError, ConfigurationError, ValidationError,
        ModelError, ExplanationError,
    )
"""
from __future__ import annotations

from .config import InferenceConfig
from .engine import ExplainArg, InferenceEngine
from .errors import (
    ConfigurationError,
    ExplanationError,
    InferenceError,
    ModelError,
    ValidationError,
)
from .events import LearningStepEvent, LearningStream
from .schemas import (
    LearningAck,
    LearningRecord,
    PatientInput,
    PredictionResult,
    Treatment,
)
from .streaming import AsyncLearningSession, LearningSession
from .stub_client import StubClient

__all__ = [
    # Engine + config
    "InferenceEngine",
    "InferenceConfig",
    "ExplainArg",
    # Schemas
    "PatientInput",
    "LearningRecord",
    "PredictionResult",
    "LearningAck",
    "Treatment",
    # Errors
    "InferenceError",
    "ConfigurationError",
    "ValidationError",
    "ModelError",
    "ExplanationError",
    # Streaming
    "LearningSession",
    "AsyncLearningSession",
    # Rich per-step event stream
    "LearningStepEvent",
    "LearningStream",
    # Stub client (for deterministic tests/demos)
    "StubClient",
]
