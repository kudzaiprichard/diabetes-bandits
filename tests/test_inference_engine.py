"""End-to-end engine tests: prediction, batch, explain with stub, update, CSV."""
from __future__ import annotations

import asyncio
import csv

import numpy as np
import pytest

from inference import (
    ConfigurationError,
    InferenceConfig,
    InferenceEngine,
    PredictionResult,
    ValidationError,
)

from tests.inference_fixtures import build_tiny_artefacts, sample_patient


@pytest.fixture(scope="module")
def _artefacts(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("inference_artefacts")
    model_path, pipeline_path, pipeline, model = build_tiny_artefacts(tmp)
    return {
        "model_path": model_path,
        "pipeline_path": pipeline_path,
        "pipeline": pipeline,
        "model": model,
    }


@pytest.fixture
def engine(_artefacts):
    cfg = InferenceConfig(
        model_path=_artefacts["model_path"],
        pipeline_path=_artefacts["pipeline_path"],
        n_confidence_draws=32,
        online_retraining=True,
        replay_buffer_size=500,
        retrain_every=10,
        min_buffer_for_retrain=10,
        minibatch_size=16,
        drift_baseline_size=20,
        drift_window_size=20,
        drift_threshold_z=3.0,
        device="cpu",
    )
    return InferenceEngine.from_config(cfg)


@pytest.fixture
def engine_with_stub_llm(_artefacts):
    cfg = InferenceConfig(
        model_path=_artefacts["model_path"],
        pipeline_path=_artefacts["pipeline_path"],
        n_confidence_draws=32,
        llm_enabled=True,
        llm_provider="stub",
        online_retraining=False,
        drift_enabled=False,
        device="cpu",
    )
    return InferenceEngine.from_config(cfg)


# ── construction & introspection ─────────────────────────────────────────────

def test_engine_ready_and_snapshot(engine):
    assert engine.ready
    snap = engine.snapshot()
    assert snap["n_updates"] == 0
    assert "feature_names" in snap and len(snap["feature_names"]) > 0
    assert snap["llm_enabled"] is False


def test_from_config_missing_model_raises(tmp_path):
    cfg = InferenceConfig(
        model_path=tmp_path / "nope.pt",
        pipeline_path=tmp_path / "nope.joblib",
    )
    with pytest.raises(ConfigurationError):
        InferenceEngine.from_config(cfg)


# ── prediction ───────────────────────────────────────────────────────────────

def test_predict_returns_valid_result(engine):
    rng = np.random.RandomState(1)
    patient = sample_patient(rng)
    result = engine.predict(patient)
    assert isinstance(result, PredictionResult)
    assert result.accepted
    assert result.recommended in ["Metformin", "GLP-1", "SGLT-2", "DPP-4", "Insulin"]
    assert 0 <= result.confidence_pct <= 100
    assert result.safety_status in ["CLEAR", "WARNING", "CONTRAINDICATION_FOUND"]


def test_predict_raises_validation_error_on_bad_patient(engine):
    bad = {"age": 200, "bmi": 34, "hba1c_baseline": 8.0, "egfr": 85.0}
    with pytest.raises(ValidationError) as excinfo:
        engine.predict(bad)
    assert excinfo.value.errors()


def test_predict_with_stub_explain(engine_with_stub_llm):
    rng = np.random.RandomState(2)
    patient = sample_patient(rng)
    result = engine_with_stub_llm.predict(patient, explain=True)
    assert result.explanation is not None
    assert "recommendation_summary" in result.explanation


def test_predict_explain_require_propagates_failure(_artefacts):
    # Explain enabled but no provider configured → ConfigurationError on require
    cfg = InferenceConfig(
        model_path=_artefacts["model_path"],
        pipeline_path=_artefacts["pipeline_path"],
        n_confidence_draws=16,
        llm_enabled=False,  # disabled
        device="cpu",
    )
    eng = InferenceEngine.from_config(cfg)
    rng = np.random.RandomState(3)
    with pytest.raises(ConfigurationError):
        eng.predict(sample_patient(rng), explain="require")


def test_predict_explain_soft_fails_silently(_artefacts):
    cfg = InferenceConfig(
        model_path=_artefacts["model_path"],
        pipeline_path=_artefacts["pipeline_path"],
        n_confidence_draws=16,
        llm_enabled=False,
        device="cpu",
    )
    eng = InferenceEngine.from_config(cfg)
    rng = np.random.RandomState(4)
    result = eng.predict(sample_patient(rng), explain=True)
    assert result.accepted
    assert result.explanation is None


def test_predict_batch_mixed_rows(engine):
    rng = np.random.RandomState(5)
    good = sample_patient(rng)
    bad = {"age": 200, "bmi": 34}  # missing + out-of-range
    results = engine.predict_batch([good, bad])
    assert results[0].accepted is True
    assert results[1].accepted is False
    assert results[1].validation_errors


def test_predict_batch_accepts_dataframe(engine):
    import pandas as pd
    rng = np.random.RandomState(6)
    rows = [sample_patient(rng) for _ in range(3)]
    df = pd.DataFrame(rows)
    results = engine.predict_batch(df)
    assert len(results) == 3
    assert all(r.accepted for r in results)


# ── safety gate ──────────────────────────────────────────────────────────────

def test_safety_gate_fires_on_low_egfr(engine):
    """Contraindicated arms (Metformin < 30, SGLT-2 < 25) must be excluded
    from the final recommendation when eGFR is very low."""
    rng = np.random.RandomState(7)
    patient = sample_patient(rng)
    patient["egfr"] = 20.0
    result = engine.predict(patient)
    # Neither contraindicated arm may be the final recommendation
    assert result.recommended not in {"Metformin", "SGLT-2"}
    # Either the gate fired (override present) and its final_treatment agrees,
    # or the model didn't pick a contraindicated arm in the first place.
    if result.override is not None:
        assert result.override["original_treatment"] in {"Metformin", "SGLT-2"}
        assert result.override["final_treatment"] == result.recommended


# ── continuous learning ──────────────────────────────────────────────────────

def test_update_posts_ack(engine):
    rng = np.random.RandomState(8)
    patient = sample_patient(rng)
    rec = {"patient": patient, "action": 0, "reward": 1.0}
    ack = engine.update(rec)
    assert ack.accepted
    assert ack.posterior_updated
    assert ack.n_updates_so_far == 1


def test_update_rejects_bad_reward(engine):
    rng = np.random.RandomState(9)
    patient = sample_patient(rng)
    ack = engine.update({"patient": patient, "action": 0, "reward": 99.0})
    assert ack.accepted is False
    assert ack.validation_errors


def test_update_many_yields_acks(engine):
    rng = np.random.RandomState(10)
    records = [
        {"patient": sample_patient(rng), "action": i % 5, "reward": 1.0}
        for i in range(5)
    ]
    acks = list(engine.update_many(records))
    assert len(acks) == 5
    assert all(a.accepted for a in acks)


def test_backbone_retrain_flag_fires(engine):
    """With retrain_every=10 and min_buffer=10, the 10th update triggers it."""
    rng = np.random.RandomState(11)
    fired = False
    for i in range(15):
        rec = {"patient": sample_patient(rng), "action": i % 5, "reward": 1.0}
        ack = engine.update(rec)
        assert ack.accepted
        if ack.backbone_retrained:
            fired = True
    assert fired


def test_ingest_csv_end_to_end(engine, tmp_path):
    rng = np.random.RandomState(12)
    csv_path = tmp_path / "updates.csv"
    patients = [sample_patient(rng) for _ in range(4)]
    keys = list(patients[0].keys()) + ["action", "reward"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for i, p in enumerate(patients):
            row = dict(p)
            row["action"] = i % 5
            row["reward"] = 1.0
            writer.writerow(row)
    acks = list(engine.ingest_csv(csv_path))
    assert len(acks) == 4
    assert all(a.accepted for a in acks)


def test_ingest_csv_missing_columns_raises(engine, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("age,bmi\n60,30\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        list(engine.ingest_csv(bad))


# ── async surface ────────────────────────────────────────────────────────────

def test_apredict(engine):
    rng = np.random.RandomState(13)
    patient = sample_patient(rng)
    result = asyncio.run(engine.apredict(patient))
    assert result.accepted


def test_aupdate(engine):
    rng = np.random.RandomState(14)
    rec = {"patient": sample_patient(rng), "action": 1, "reward": 1.2}
    ack = asyncio.run(engine.aupdate(rec))
    assert ack.accepted
