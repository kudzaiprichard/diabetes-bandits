"""
Inference-engine configuration.

Three-layer loading, highest priority first:

    1. Environment variables — prefix ``BANDITS_``
    2. YAML file — ``BANDITS_CONFIG_FILE`` env or the ``file=`` arg
    3. Defaults baked into this module

``InferenceConfig.load()`` returns the merged config. ``from_env()`` on the
engine is a thin shortcut.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .errors import ConfigurationError

_ENV_PREFIX = "BANDITS_"


class InferenceConfig(BaseModel):
    """
    Immutable-ish runtime configuration for the inference engine.

    Only ``model_path`` and ``pipeline_path`` are strictly required at load
    time — everything else defaults to the values that the six-phase upgrade
    ships with.
    """

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, protected_namespaces=(),
    )

    # ── paths ─────────────────────────────────────────────────────────────
    model_path: Path = Path("models/neural_thompson.pt")
    pipeline_path: Path = Path("models/feature_pipeline.joblib")
    data_dir: Path = Path("data/")

    # ── model behaviour ───────────────────────────────────────────────────
    n_confidence_draws: int = Field(200, ge=1)
    attribution_enabled: bool = True

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_enabled: bool = False
    llm_provider: Literal["gemini", "stub", "none"] = "none"
    llm_api_key: Optional[SecretStr] = None
    llm_model_name: str = "gemini-2.0-flash"
    llm_max_retries: int = Field(2, ge=0)
    llm_temperature: float = Field(0.3, ge=0.0, le=2.0)

    # ── continuous learning ───────────────────────────────────────────────
    online_retraining: bool = True
    replay_buffer_size: int = Field(50_000, ge=1)
    retrain_every: int = Field(2_000, ge=1)
    min_buffer_for_retrain: int = Field(2_000, ge=1)
    minibatch_size: int = Field(1_024, ge=1)
    retrain_epochs: int = Field(1, ge=1)

    # ── drift ─────────────────────────────────────────────────────────────
    drift_enabled: bool = True
    drift_baseline_size: int = Field(2_000, ge=1)
    drift_window_size: int = Field(2_000, ge=1)
    drift_threshold_z: float = Field(3.0, gt=0.0)

    # ── runtime ───────────────────────────────────────────────────────────
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 42

    # ── checkpointing ─────────────────────────────────────────────────────
    checkpoint_dir: Optional[Path] = None
    checkpoint_every: Optional[int] = None

    @field_validator("model_path", "pipeline_path", "data_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Any:
        if v is None or isinstance(v, Path):
            return v
        return Path(str(v))

    # ── loading ──────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        file: Optional[str | Path] = None,
        **overrides: Any,
    ) -> "InferenceConfig":
        """
        Merge defaults ⟵ file ⟵ env ⟵ explicit overrides.

        Explicit kwargs win over env vars. File path precedence:
            1. ``file=`` argument
            2. ``BANDITS_CONFIG_FILE`` env var
            3. nothing (skip file layer)
        """
        merged: Dict[str, Any] = {}

        file_path: Optional[Path] = None
        if file is not None:
            file_path = Path(file)
        elif os.environ.get(f"{_ENV_PREFIX}CONFIG_FILE"):
            file_path = Path(os.environ[f"{_ENV_PREFIX}CONFIG_FILE"])

        if file_path is not None:
            if not file_path.exists():
                raise ConfigurationError(
                    f"Config file not found: {file_path}"
                )
            merged.update(_load_yaml(file_path))

        merged.update(_from_env())
        merged.update(overrides)

        try:
            return cls(**merged)
        except Exception as e:  # pydantic ValidationError + anything else
            raise ConfigurationError(f"Invalid configuration: {e}") from e

    def resolve_api_key(self) -> Optional[str]:
        """Return the plain-text API key, falling back to ``GEMINI_API_KEY``."""
        if self.llm_api_key is not None:
            return self.llm_api_key.get_secret_value()
        return os.environ.get("GEMINI_API_KEY")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ConfigurationError(
            "PyYAML is required to load config files; install pyyaml"
        ) from e
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Config file {path} must contain a mapping at the top level"
        )
    return data


def _from_env() -> Dict[str, Any]:
    """
    Read ``BANDITS_*`` env vars. Field names are lower-cased; the prefix is
    stripped. Booleans are parsed from {true,false,1,0,yes,no}; ints/floats
    are coerced.
    """
    out: Dict[str, Any] = {}
    field_names = set(InferenceConfig.model_fields.keys())
    for k, v in os.environ.items():
        if not k.startswith(_ENV_PREFIX):
            continue
        key = k[len(_ENV_PREFIX):].lower()
        if key == "config_file":  # handled separately
            continue
        if key not in field_names:
            continue
        out[key] = _coerce(v, key)
    return out


_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off"}


def _coerce(raw: str, key: str) -> Any:
    """Best-effort coercion of an env-var string to the field's Python type."""
    info = InferenceConfig.model_fields.get(key)
    if info is None:
        return raw
    annot = info.annotation
    ann_str = str(annot).lower()
    low = raw.strip().lower()
    if "bool" in ann_str:
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ConfigurationError(f"Invalid bool for BANDITS_{key.upper()}: {raw!r}")
    if "int" in ann_str and "float" not in ann_str:
        try:
            return int(raw)
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid int for BANDITS_{key.upper()}: {raw!r}"
            ) from e
    if "float" in ann_str:
        try:
            return float(raw)
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid float for BANDITS_{key.upper()}: {raw!r}"
            ) from e
    if "path" in ann_str:
        return Path(raw)
    return raw


__all__ = ["InferenceConfig"]
