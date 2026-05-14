"""Canonical treatment/context/reward constants used by the inference stack.

Vendored from src/data_generator.py so inference/ has no src/ dependency.
These values define the on-the-wire contract (treatment names, feature
order, reward cap). Changing them is a breaking API change.
"""
from __future__ import annotations

TREATMENTS = ["Metformin", "GLP-1", "SGLT-2", "DPP-4", "Insulin"]
N_TREATMENTS = len(TREATMENTS)
TREATMENT_TO_IDX = {t: i for i, t in enumerate(TREATMENTS)}
IDX_TO_TREATMENT = {i: t for i, t in enumerate(TREATMENTS)}

CONTEXT_FEATURES = [
    "age", "bmi", "hba1c_baseline", "egfr", "diabetes_duration",
    "fasting_glucose", "c_peptide", "cvd", "ckd", "nafld",
    "hypertension", "bp_systolic", "ldl", "hdl", "triglycerides", "alt",
]

REWARD_SCALE = 0.25
REWARD_CAP_PP = 3.0

__all__ = [
    "TREATMENTS", "N_TREATMENTS", "TREATMENT_TO_IDX", "IDX_TO_TREATMENT",
    "CONTEXT_FEATURES", "REWARD_SCALE", "REWARD_CAP_PP",
]
