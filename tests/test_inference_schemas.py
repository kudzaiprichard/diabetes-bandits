"""Pydantic boundary contracts — value ranges, normalisations, result shape."""
from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from inference import LearningRecord, PatientInput, PredictionResult


def _valid_patient_dict() -> dict:
    return {
        "age": 62, "bmi": 34.2, "hba1c_baseline": 8.9, "egfr": 85.0,
        "diabetes_duration": 6.0, "fasting_glucose": 180.0, "c_peptide": 1.4,
        "bp_systolic": 140.0, "ldl": 120.0, "hdl": 45.0,
        "triglycerides": 200.0, "alt": 30.0,
        "cvd": 1, "ckd": 0, "nafld": 1, "hypertension": 1,
    }


def test_patient_input_accepts_valid_dict():
    pi = PatientInput.model_validate(_valid_patient_dict())
    assert pi.age == 62
    assert pi.bmi == 34.2


def test_patient_input_rejects_out_of_range_egfr():
    bad = _valid_patient_dict()
    bad["egfr"] = -1.0
    with pytest.raises(PydanticValidationError):
        PatientInput.model_validate(bad)


def test_patient_input_rejects_impossible_age():
    bad = _valid_patient_dict()
    bad["age"] = 12
    with pytest.raises(PydanticValidationError):
        PatientInput.model_validate(bad)


def test_patient_input_safety_flags_default_zero():
    pi = PatientInput.model_validate(_valid_patient_dict())
    assert pi.medullary_thyroid_history == 0
    assert pi.pancreatitis_history == 0
    assert pi.type1_suspicion == 0


def test_patient_input_feature_dict_excludes_protected():
    d = _valid_patient_dict()
    d["gender"] = "F"
    d["ethnicity"] = "Hispanic"
    pi = PatientInput.model_validate(d)
    fd = pi.feature_dict()
    assert "gender" not in fd
    assert "ethnicity" not in fd
    assert "patient_id" not in fd
    assert fd["bmi"] == 34.2


def test_learning_record_nested_shape():
    rec = LearningRecord.model_validate({
        "patient": _valid_patient_dict(),
        "action": 1,
        "reward": 1.2,
    })
    assert rec.action == 1
    assert rec.treatment == "GLP-1"


def test_learning_record_flat_shape():
    data = dict(_valid_patient_dict())
    data["treatment"] = "SGLT-2"
    data["reward"] = 1.5
    rec = LearningRecord.model_validate(data)
    assert rec.treatment == "SGLT-2"
    assert rec.action == 2


def test_learning_record_rejects_missing_action_and_treatment():
    with pytest.raises(PydanticValidationError):
        LearningRecord.model_validate({
            "patient": _valid_patient_dict(),
            "reward": 1.2,
        })


def test_learning_record_rejects_reward_above_cap():
    with pytest.raises(PydanticValidationError):
        LearningRecord.model_validate({
            "patient": _valid_patient_dict(),
            "action": 0,
            "reward": 10.0,
        })


def test_learning_record_rejects_conflicting_action_treatment():
    with pytest.raises(PydanticValidationError):
        LearningRecord.model_validate({
            "patient": _valid_patient_dict(),
            "action": 0,
            "treatment": "GLP-1",
            "reward": 1.0,
        })


def test_prediction_result_rejected_sentinel_shape():
    r = PredictionResult.rejected(errors=[{"loc": ["egfr"], "msg": "x"}])
    assert r.accepted is False
    assert r.validation_errors[0]["loc"] == ["egfr"]
    assert r.recommended is None


def test_prediction_result_from_payload_flattens():
    payload = {
        "patient": {},
        "decision": {
            "recommended_treatment": "GLP-1",
            "recommended_idx": 1,
            "model_top_treatment": "GLP-1",
            "confidence_pct": 73,
            "confidence_label": "MODERATE",
            "win_rates": {"GLP-1": 0.73, "Metformin": 0.2},
            "posterior_means": {"GLP-1": 1.2, "Metformin": 0.8},
            "runner_up": "Metformin",
            "runner_up_win_rate": 0.2,
            "mean_gap": 0.4,
            "n_draws": 200,
            "override": None,
        },
        "safety": {
            "status": "CLEAR",
            "recommended_contraindications": [],
            "recommended_warnings": [],
            "excluded_treatments": {},
        },
    }
    r = PredictionResult.from_payload(payload, patient_id="p1")
    assert r.accepted is True
    assert r.recommended == "GLP-1"
    assert r.confidence_pct == 73
    assert r.safety_status == "CLEAR"
    assert r.patient_id == "p1"
