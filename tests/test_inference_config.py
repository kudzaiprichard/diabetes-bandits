"""Config loading: defaults, YAML, env vars, explicit overrides."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference import ConfigurationError, InferenceConfig


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("BANDITS_"):
            monkeypatch.delenv(k, raising=False)
    yield


def test_defaults_are_sensible():
    cfg = InferenceConfig()
    assert cfg.model_path == Path("models/neural_thompson.pt")
    assert cfg.pipeline_path == Path("models/feature_pipeline.joblib")
    assert cfg.n_confidence_draws == 200
    assert cfg.llm_enabled is False
    assert cfg.llm_provider == "none"
    assert cfg.online_retraining is True
    assert cfg.drift_enabled is True


def test_load_without_file_or_env_returns_defaults():
    cfg = InferenceConfig.load()
    assert cfg.device == "auto"
    assert cfg.llm_enabled is False


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("BANDITS_LLM_ENABLED", "true")
    monkeypatch.setenv("BANDITS_LLM_PROVIDER", "stub")
    monkeypatch.setenv("BANDITS_N_CONFIDENCE_DRAWS", "50")
    monkeypatch.setenv("BANDITS_DRIFT_THRESHOLD_Z", "2.5")
    monkeypatch.setenv("BANDITS_MODEL_PATH", "somewhere/model.pt")

    cfg = InferenceConfig.load()
    assert cfg.llm_enabled is True
    assert cfg.llm_provider == "stub"
    assert cfg.n_confidence_draws == 50
    assert cfg.drift_threshold_z == 2.5
    assert cfg.model_path == Path("somewhere/model.pt")


def test_explicit_overrides_win_over_env(monkeypatch):
    monkeypatch.setenv("BANDITS_N_CONFIDENCE_DRAWS", "50")
    cfg = InferenceConfig.load(n_confidence_draws=123)
    assert cfg.n_confidence_draws == 123


def test_yaml_file_layer(tmp_path: Path):
    yaml_text = """
llm_enabled: true
llm_provider: stub
n_confidence_draws: 77
"""
    cfg_file = tmp_path / "inference.yaml"
    cfg_file.write_text(yaml_text, encoding="utf-8")
    cfg = InferenceConfig.load(file=cfg_file)
    assert cfg.llm_enabled is True
    assert cfg.llm_provider == "stub"
    assert cfg.n_confidence_draws == 77


def test_yaml_and_env_merge_with_env_priority(tmp_path: Path, monkeypatch):
    cfg_file = tmp_path / "inference.yaml"
    cfg_file.write_text("n_confidence_draws: 50\nllm_provider: stub\n", encoding="utf-8")
    monkeypatch.setenv("BANDITS_N_CONFIDENCE_DRAWS", "99")
    monkeypatch.setenv("BANDITS_CONFIG_FILE", str(cfg_file))
    cfg = InferenceConfig.load()
    assert cfg.n_confidence_draws == 99  # env wins over yaml
    assert cfg.llm_provider == "stub"    # yaml value retained


def test_missing_yaml_file_raises():
    with pytest.raises(ConfigurationError):
        InferenceConfig.load(file="/does/not/exist.yaml")


def test_invalid_bool_env_raises(monkeypatch):
    monkeypatch.setenv("BANDITS_LLM_ENABLED", "notabool")
    with pytest.raises(ConfigurationError):
        InferenceConfig.load()


def test_invalid_int_env_raises(monkeypatch):
    monkeypatch.setenv("BANDITS_N_CONFIDENCE_DRAWS", "abc")
    with pytest.raises(ConfigurationError):
        InferenceConfig.load()


def test_resolve_api_key_prefers_explicit_over_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from_env")
    cfg = InferenceConfig(llm_api_key="from_config")
    assert cfg.resolve_api_key() == "from_config"


def test_resolve_api_key_falls_back_to_gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from_env")
    cfg = InferenceConfig()
    assert cfg.resolve_api_key() == "from_env"
