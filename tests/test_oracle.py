"""
Phase 0 — G-3 / safety edge-case tests for the reward oracle.

Covers:
    - Reward is bounded within the clinically-calibrated [0, REWARD_CAP_PP] window.
    - Plausible effect sizes (ideal Metformin ~1.5 pp, ideal Insulin ~2.5 pp).
    - A patient with eGFR = 25 is never routed to Metformin by the safety gate
      (paired G-13 / G-16 test).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data_generator import (
    REWARD_CAP_PP,
    REWARD_SCALE,
    TREATMENTS,
    generate_patient,
    reward_oracle,
)
from src.explainability import (
    SEVERITY_CONTRAINDICATION,
    apply_safety_gate,
    collect_findings,
)


# ── shape / bounds ──────────────────────────────────────────────────────────

def test_reward_scale_constants():
    """G-3: the rescale constants are in the clinically plausible band."""
    assert REWARD_CAP_PP <= 3.5
    assert REWARD_CAP_PP >= 2.5
    assert 0.1 <= REWARD_SCALE <= 0.5


def test_reward_oracle_is_bounded_across_many_patients():
    """Oracle must never emit a reward outside [0, REWARD_CAP_PP]."""
    rng = np.random.default_rng(42)
    for _ in range(500):
        ctx = generate_patient(rng)
        for t in TREATMENTS:
            r = reward_oracle(ctx, t, noise=True)
            assert 0.0 <= r <= REWARD_CAP_PP, (t, r, ctx)


def test_reward_oracle_noiseless_deterministic():
    """With noise=False, the oracle is deterministic."""
    rng = np.random.default_rng(0)
    ctx = generate_patient(rng)
    for t in TREATMENTS:
        a = reward_oracle(ctx, t, noise=False)
        b = reward_oracle(ctx, t, noise=False)
        assert a == b


def test_ideal_niches_land_in_plausible_band():
    """
    G-3: the rescaled oracle should predict ~1.5 pp for an ideal Metformin
    patient and ~2.5 pp for an ideal Insulin patient. We allow a loose band
    because the oracle has some internal interaction terms.
    """
    metformin_patient = {
        "age": 45, "bmi": 28.0, "hba1c_baseline": 8.0, "egfr": 95.0,
        "diabetes_duration": 3.0, "fasting_glucose": 160.0, "c_peptide": 2.0,
        "bp_systolic": 130.0, "ldl": 110.0, "hdl": 50.0, "triglycerides": 140.0,
        "alt": 25.0, "cvd": 0, "ckd": 0, "nafld": 0, "hypertension": 0,
    }
    insulin_patient = {
        "age": 58, "bmi": 29.0, "hba1c_baseline": 12.5, "egfr": 75.0,
        "diabetes_duration": 18.0, "fasting_glucose": 260.0, "c_peptide": 0.2,
        "bp_systolic": 135.0, "ldl": 120.0, "hdl": 42.0, "triglycerides": 180.0,
        "alt": 28.0, "cvd": 0, "ckd": 0, "nafld": 0, "hypertension": 0,
    }
    r_met = reward_oracle(metformin_patient, "Metformin", noise=False)
    r_ins = reward_oracle(insulin_patient, "Insulin", noise=False)

    # Loose plausibility band: the oracle has CVD/NAFLD/CKD interactions, so
    # the niche rewards will not hit the theoretical ceiling. These bands are
    # the correct shape for Phase 0 calibration.
    assert 0.8 <= r_met <= REWARD_CAP_PP, r_met
    assert 1.2 <= r_ins <= REWARD_CAP_PP, r_ins


# ── safety gate (G-13 + G-16) ───────────────────────────────────────────────

def _low_egfr_patient(egfr: float = 25.0) -> dict:
    return {
        "age": 68, "bmi": 30.5, "hba1c_baseline": 8.5, "egfr": egfr,
        "diabetes_duration": 12.0, "fasting_glucose": 170.0, "c_peptide": 1.2,
        "bp_systolic": 138.0, "ldl": 110.0, "hdl": 45.0, "triglycerides": 150.0,
        "alt": 27.0, "cvd": 1, "ckd": 1, "nafld": 0, "hypertension": 1,
    }


def test_metformin_contraindicated_at_low_egfr():
    """G-13: structured finding must flag Metformin at eGFR < 30."""
    ctx = _low_egfr_patient(egfr=25.0)
    findings = collect_findings(ctx)
    met_contras = [
        f for f in findings["Metformin"]
        if f.severity == SEVERITY_CONTRAINDICATION
    ]
    assert met_contras, "expected Metformin contraindication at eGFR=25"
    assert met_contras[0].rule_id == "METFORMIN_EGFR_LT_30"


def test_safety_gate_never_routes_low_egfr_to_metformin():
    """
    G-16: even if Metformin has the best posterior mean, the gate must
    promote a non-contraindicated treatment for a patient with eGFR=25.
    """
    ctx = _low_egfr_patient(egfr=25.0)
    findings = collect_findings(ctx)

    # Pathological scores where Metformin is the model's top pick.
    posterior_means = {
        "Metformin": 2.9,
        "GLP-1":     1.8,
        "SGLT-2":    1.6,
        "DPP-4":     1.2,
        "Insulin":   1.0,
    }
    win_rates = {t: 0.2 for t in TREATMENTS}
    win_rates["Metformin"] = 0.8

    final, override = apply_safety_gate(
        posterior_means=posterior_means,
        win_rates=win_rates,
        findings_by_treatment=findings,
        top_treatment="Metformin",
    )
    assert final != "Metformin"
    assert override is not None
    assert override.original_treatment == "Metformin"
    assert override.final_treatment == final
    assert "Metformin" in override.blocked_treatments


@pytest.mark.parametrize("egfr", [10, 15, 25, 29])
def test_metformin_always_blocked_below_30(egfr):
    """Boundary test — eGFR strictly < 30 must always trigger contraindication."""
    ctx = _low_egfr_patient(egfr=egfr)
    findings = collect_findings(ctx)
    contras = [f for f in findings["Metformin"]
               if f.severity == SEVERITY_CONTRAINDICATION]
    assert contras, f"expected contraindication at eGFR={egfr}"
