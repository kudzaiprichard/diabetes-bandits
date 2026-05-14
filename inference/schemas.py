"""
Pydantic schemas — the module boundary contract.

Four shapes cross the engine boundary:
    PatientInput     — single patient context (prediction input)
    LearningRecord   — patient + action + reward (continuous-learning input)
    PredictionResult — everything a caller gets back from ``predict``
    LearningAck      — per-update acknowledgement (what happened + when)

See ``.docs/inference_module_design.md §3`` for value ranges and justification.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._internal.constants import (
    IDX_TO_TREATMENT,
    N_TREATMENTS,
    REWARD_CAP_PP,
    TREATMENT_TO_IDX,
    TREATMENTS,
)


Treatment = Literal["Metformin", "GLP-1", "SGLT-2", "DPP-4", "Insulin"]


# ─────────────────────────────────────────────────────────────────────────────
# PatientInput
# ─────────────────────────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """
    Raw patient context at the engine boundary.

    Only the 16 clinical ``CONTEXT_FEATURES`` flow into the model. ``gender``,
    ``ethnicity``, and ``patient_id`` are accepted for audit/UX only — they are
    **never** passed to the feature pipeline (G-15 fairness posture).

    The four structured safety flags (``medullary_thyroid_history`` etc.)
    default to 0 — they are optional but, when supplied, drive the G-14 safety
    rules.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Core clinical features
    age: int = Field(..., ge=18, le=110)
    bmi: float = Field(..., ge=10.0, le=80.0)
    hba1c_baseline: float = Field(..., ge=4.0, le=20.0)
    egfr: float = Field(..., ge=0.0, le=200.0)
    diabetes_duration: float = Field(..., ge=0.0, le=80.0)
    fasting_glucose: float = Field(..., ge=40.0, le=600.0)
    c_peptide: float = Field(..., ge=0.0, le=10.0)
    bp_systolic: float = Field(..., ge=60.0, le=260.0)
    ldl: float = Field(..., ge=20.0, le=400.0)
    hdl: float = Field(..., ge=10.0, le=150.0)
    triglycerides: float = Field(..., ge=10.0, le=2000.0)
    alt: float = Field(..., ge=0.0, le=1000.0)

    cvd: Literal[0, 1]
    ckd: Literal[0, 1]
    nafld: Literal[0, 1]
    hypertension: Literal[0, 1]

    # G-14 structured safety flags (optional, default 0)
    medullary_thyroid_history: Literal[0, 1] = 0
    men2_history: Literal[0, 1] = 0
    pancreatitis_history: Literal[0, 1] = 0
    type1_suspicion: Literal[0, 1] = 0

    # Audit-only (NOT passed to the feature pipeline)
    gender: Optional[Literal["M", "F", "Other"]] = None
    ethnicity: Optional[str] = None
    patient_id: Optional[str] = None

    def feature_dict(self) -> Dict[str, Any]:
        """Return the subset of fields that enter the feature pipeline."""
        return {
            "age": self.age,
            "bmi": self.bmi,
            "hba1c_baseline": self.hba1c_baseline,
            "egfr": self.egfr,
            "diabetes_duration": self.diabetes_duration,
            "fasting_glucose": self.fasting_glucose,
            "c_peptide": self.c_peptide,
            "cvd": self.cvd,
            "ckd": self.ckd,
            "nafld": self.nafld,
            "hypertension": self.hypertension,
            "bp_systolic": self.bp_systolic,
            "ldl": self.ldl,
            "hdl": self.hdl,
            "triglycerides": self.triglycerides,
            "alt": self.alt,
        }

    def context_dict(self) -> Dict[str, Any]:
        """Full context dict including safety flags (for the safety gate)."""
        d = self.feature_dict()
        d["medullary_thyroid_history"] = self.medullary_thyroid_history
        d["men2_history"] = self.men2_history
        d["pancreatitis_history"] = self.pancreatitis_history
        d["type1_suspicion"] = self.type1_suspicion
        return d


# ─────────────────────────────────────────────────────────────────────────────
# LearningRecord
# ─────────────────────────────────────────────────────────────────────────────

class LearningRecord(BaseModel):
    """
    A single continuous-learning record. Either ``action`` (int 0–4) or
    ``treatment`` (string name) is required — the validator normalises to
    ``action`` so downstream code never has to care which was supplied.

    Supports two input shapes:

        # nested patient
        {"patient": {...}, "action": 1, "reward": 1.2}

        # flat (patient fields at top level)
        {"age": 62, "bmi": 34, ..., "treatment": "GLP-1", "reward": 1.2}
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    patient: PatientInput
    action: Optional[int] = Field(None, ge=0, le=N_TREATMENTS - 1)
    treatment: Optional[Treatment] = None
    reward: float = Field(..., ge=0.0, le=REWARD_CAP_PP + 0.5)
    observed_at: Optional[datetime] = None
    source: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_shape(cls, data: Any) -> Any:
        """Accept a flat dict by lifting patient fields into ``patient``."""
        if not isinstance(data, dict):
            return data
        if "patient" in data:
            return data
        patient_fields = set(PatientInput.model_fields.keys())
        nested = {k: data[k] for k in list(data.keys()) if k in patient_fields}
        rest = {k: v for k, v in data.items() if k not in patient_fields}
        rest["patient"] = nested
        return rest

    @model_validator(mode="after")
    def _normalise_action(self) -> "LearningRecord":
        if self.action is None and self.treatment is None:
            raise ValueError("either 'action' or 'treatment' must be supplied")
        if self.action is None and self.treatment is not None:
            self.action = TREATMENT_TO_IDX[self.treatment]
        elif self.treatment is None and self.action is not None:
            self.treatment = IDX_TO_TREATMENT[self.action]
        elif TREATMENT_TO_IDX[self.treatment] != self.action:
            raise ValueError(
                f"action={self.action} and treatment={self.treatment!r} disagree"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# PredictionResult
# ─────────────────────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """
    Flattened view of ``ExplainabilityExtractor.extract()`` plus an optional
    ``ClinicalExplanation`` from ``LLMExplainer``.

    For batch prediction, rows that fail schema validation return a sentinel
    result with ``accepted=False`` and the error list populated — the engine
    does not raise.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    accepted: bool = True
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list)

    patient_id: Optional[str] = None

    recommended: Optional[str] = None
    recommended_idx: Optional[int] = None
    model_top_treatment: Optional[str] = None

    confidence_pct: Optional[int] = None
    confidence_label: Optional[Literal["HIGH", "MODERATE", "LOW"]] = None

    posterior_means: Dict[str, float] = Field(default_factory=dict)
    win_rates: Dict[str, float] = Field(default_factory=dict)

    runner_up: Optional[str] = None
    runner_up_win_rate: Optional[float] = None
    mean_gap: Optional[float] = None

    safety_status: Optional[Literal["CLEAR", "WARNING", "CONTRAINDICATION_FOUND"]] = None
    safety_findings: List[Dict[str, Any]] = Field(default_factory=list)
    excluded_treatments: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    override: Optional[Dict[str, Any]] = None

    explanation: Optional[Dict[str, Any]] = None
    attribution: Optional[Dict[str, Any]] = None
    contrast: Optional[Dict[str, Any]] = None
    uncertainty_drivers: Optional[List[Dict[str, Any]]] = None

    model_version: Optional[str] = None
    pipeline_version: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        patient_id: Optional[str] = None,
        explanation: Optional[Dict[str, Any]] = None,
        model_version: Optional[str] = None,
        pipeline_version: Optional[str] = None,
    ) -> "PredictionResult":
        d = payload["decision"]
        s = payload["safety"]
        return cls(
            accepted=True,
            patient_id=patient_id,
            recommended=d["recommended_treatment"],
            recommended_idx=d.get("recommended_idx"),
            model_top_treatment=d.get("model_top_treatment"),
            confidence_pct=d["confidence_pct"],
            confidence_label=d["confidence_label"],
            posterior_means=dict(d["posterior_means"]),
            win_rates=dict(d["win_rates"]),
            runner_up=d["runner_up"],
            runner_up_win_rate=d["runner_up_win_rate"],
            mean_gap=d["mean_gap"],
            safety_status=s["status"],
            safety_findings=list(s.get("recommended_contraindications", []))
            + list(s.get("recommended_warnings", [])),
            excluded_treatments=dict(s.get("excluded_treatments", {})),
            override=d.get("override"),
            explanation=explanation,
            attribution=d.get("attribution"),
            contrast=d.get("contrast"),
            uncertainty_drivers=d.get("uncertainty_drivers"),
            model_version=model_version,
            pipeline_version=pipeline_version,
        )

    @classmethod
    def rejected(
        cls,
        errors: List[Dict[str, Any]],
        *,
        patient_id: Optional[str] = None,
    ) -> "PredictionResult":
        return cls(accepted=False, validation_errors=errors, patient_id=patient_id)


# ─────────────────────────────────────────────────────────────────────────────
# LearningAck
# ─────────────────────────────────────────────────────────────────────────────

class LearningAck(BaseModel):
    """Per-update acknowledgement returned by ``update`` / ``aupdate``."""

    model_config = ConfigDict(extra="allow")

    accepted: bool
    n_updates_so_far: int
    posterior_updated: bool = False
    backbone_retrained: bool = False
    drift_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list)
    patient_id: Optional[str] = None
    observed_at: Optional[datetime] = None


__all__ = [
    "PatientInput",
    "LearningRecord",
    "PredictionResult",
    "LearningAck",
    "Treatment",
    "TREATMENTS",
]
