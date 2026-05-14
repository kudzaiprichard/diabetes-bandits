"""
Inference-module exception hierarchy.

Each subclass maps to a distinct caller action — see
``.docs/inference_module_design.md §4``.
"""
from __future__ import annotations

from typing import Any, List, Optional


class InferenceError(Exception):
    """Base class for every error raised by ``inference``."""


class ConfigurationError(InferenceError):
    """
    The engine could not be set up: missing or corrupt artefact, invalid
    config value, or a mismatch between the loaded pipeline and the schema.
    """


class ValidationError(InferenceError):
    """
    Caller-supplied input failed schema or value-range validation.

    Wraps Pydantic's ``ValidationError`` so the caller receives a stable,
    framework-agnostic error list (``.errors()`` returns ``list[dict]`` with
    ``loc``/``msg``/``type``).
    """

    def __init__(
        self,
        message: str,
        errors: Optional[List[dict]] = None,
        source: Optional[Any] = None,
    ):
        super().__init__(message)
        self._errors = list(errors or [])
        self._source = source

    def errors(self) -> List[dict]:
        return list(self._errors)

    def to_dict(self) -> dict:
        return {"message": str(self), "errors": self.errors()}


class ModelError(InferenceError):
    """
    The underlying model or pipeline raised during inference — NaN output,
    shape mismatch, CUDA OOM, or an uninitialised posterior.
    """


class ExplanationError(InferenceError):
    """
    The LLM step failed (all retries exhausted, schema violation, jargon
    guard, or network failure). ``predict`` catches this by default and
    returns a result with ``explanation=None``; pass ``explain="require"``
    to propagate it.
    """
