"""
Explainability Extraction for NeuralThompson

Extracts the information needed to generate clinical explanations
via an LLM:

    Layer 1 — Patient context (human-readable features)
    Layer 2 — Model decision info (scores, confidence from posterior sampling)
    Layer 3 — Structured safety findings (dataclass-backed, G-13)
                + optional attribution & uncertainty drivers (Phase 2, G-7/8/9)
                + optional subgroup fairness report (G-15)
                + recommendation override if top-1 is contraindicated (G-16)

Usage:
    from src.explainability import ExplainabilityExtractor
    extractor = ExplainabilityExtractor(model)
    payload = extractor.extract(patient_context, x_features)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from src.data_generator import TREATMENTS, N_TREATMENTS, IDX_TO_TREATMENT


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED SAFETY FINDINGS (G-13)
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_CONTRAINDICATION = "contraindication"
SEVERITY_WARNING = "warning"


@dataclass
class SafetyFinding:
    """
    A single deterministic safety check result.

    G-13: structured representation so downstream clinical systems can
    enforce rules, audit them, and translate them — the LLM is not the
    authority on safety, this dataclass is.
    """
    treatment: str
    rule_id: str
    severity: str                  # "contraindication" or "warning"
    threshold: Optional[float]     # numeric threshold crossed, if applicable
    observed_value: Optional[float]
    feature: Optional[str]         # which patient feature drove the finding
    message: str                   # clinician-readable summary

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY RULES — return structured findings
# ─────────────────────────────────────────────────────────────────────────────

def _metformin_findings(ctx: Dict) -> List[SafetyFinding]:
    out: List[SafetyFinding] = []
    egfr = ctx.get("egfr", 999)
    bmi = ctx.get("bmi", 0.0)
    nafld = ctx.get("nafld", 0)

    if egfr < 30:
        out.append(SafetyFinding(
            treatment="Metformin",
            rule_id="METFORMIN_EGFR_LT_30",
            severity=SEVERITY_CONTRAINDICATION,
            threshold=30.0, observed_value=float(egfr), feature="egfr",
            message=(f"Metformin contraindicated: eGFR {egfr:.0f} < 30 mL/min/1.73m² "
                     f"(risk of lactic acidosis)."),
        ))
    elif 30 <= egfr < 45:
        out.append(SafetyFinding(
            treatment="Metformin",
            rule_id="METFORMIN_EGFR_30_45",
            severity=SEVERITY_WARNING,
            threshold=45.0, observed_value=float(egfr), feature="egfr",
            message=(f"Reduced efficacy: eGFR {egfr:.0f} is between 30–45 — "
                     f"dose reduction recommended, monitor renal function closely."),
        ))
    elif 45 <= egfr < 60:
        out.append(SafetyFinding(
            treatment="Metformin",
            rule_id="METFORMIN_EGFR_45_60",
            severity=SEVERITY_WARNING,
            threshold=60.0, observed_value=float(egfr), feature="egfr",
            message=(f"eGFR {egfr:.0f} approaching threshold — "
                     f"periodic renal function monitoring advisable."),
        ))
    if bmi > 35 and nafld == 1:
        out.append(SafetyFinding(
            treatment="Metformin",
            rule_id="METFORMIN_BMI_NAFLD",
            severity=SEVERITY_WARNING,
            threshold=35.0, observed_value=float(bmi), feature="bmi",
            message=(f"BMI {bmi:.1f} with NAFLD — Metformin is appropriate first-line "
                     f"but consider GLP-1 agonist for additional weight and hepatic benefits."),
        ))
    return out


def _glp1_findings(ctx: Dict) -> List[SafetyFinding]:
    out: List[SafetyFinding] = []
    age = ctx.get("age", 0)
    bmi = ctx.get("bmi", 0.0)
    egfr = ctx.get("egfr", 999)
    # G-14: structured flag for MTC/MEN2 history (we don't have the feature in
    # this synthetic cohort, but the rule is encoded so a downstream system can
    # supply it and trigger the contraindication path.)
    mtc_history = int(ctx.get("medullary_thyroid_history", 0))
    men2 = int(ctx.get("men2_history", 0))
    if mtc_history or men2:
        out.append(SafetyFinding(
            treatment="GLP-1",
            rule_id="GLP1_MTC_MEN2",
            severity=SEVERITY_CONTRAINDICATION,
            threshold=None, observed_value=float(max(mtc_history, men2)),
            feature="medullary_thyroid_history/men2_history",
            message=("GLP-1 contraindicated: personal or family history of "
                     "medullary thyroid carcinoma or MEN2."),
        ))
    if age > 75:
        out.append(SafetyFinding(
            treatment="GLP-1",
            rule_id="GLP1_AGE_75",
            severity=SEVERITY_WARNING,
            threshold=75.0, observed_value=float(age), feature="age",
            message=(f"Patient age {age} > 75 — GLP-1 tolerability may be reduced, "
                     f"start with lowest dose and titrate slowly."),
        ))
    if egfr < 30:
        out.append(SafetyFinding(
            treatment="GLP-1",
            rule_id="GLP1_EGFR_30",
            severity=SEVERITY_WARNING,
            threshold=30.0, observed_value=float(egfr), feature="egfr",
            message=(f"eGFR {egfr:.0f} < 30 — limited data on GLP-1 safety in "
                     f"severe renal impairment, use with caution."),
        ))
    elif 30 <= egfr < 45:
        out.append(SafetyFinding(
            treatment="GLP-1",
            rule_id="GLP1_EGFR_30_45",
            severity=SEVERITY_WARNING,
            threshold=45.0, observed_value=float(egfr), feature="egfr",
            message=(f"eGFR {egfr:.0f} between 30–45 — dose adjustment may be "
                     f"required, monitor renal function closely."),
        ))
    if bmi < 25:
        out.append(SafetyFinding(
            treatment="GLP-1",
            rule_id="GLP1_LOW_BMI",
            severity=SEVERITY_WARNING,
            threshold=25.0, observed_value=float(bmi), feature="bmi",
            message=(f"BMI {bmi:.1f} is in normal range — GLP-1 associated weight "
                     f"loss may not be desirable, monitor weight closely."),
        ))
    return out


def _sglt2_findings(ctx: Dict) -> List[SafetyFinding]:
    out: List[SafetyFinding] = []
    age = ctx.get("age", 0)
    egfr = ctx.get("egfr", 999)
    hypertension = ctx.get("hypertension", 0)

    if egfr < 25:
        out.append(SafetyFinding(
            treatment="SGLT-2",
            rule_id="SGLT2_EGFR_LT_25",
            severity=SEVERITY_CONTRAINDICATION,
            threshold=25.0, observed_value=float(egfr), feature="egfr",
            message=(f"SGLT-2 contraindicated: eGFR {egfr:.0f} < 25 mL/min/1.73m² "
                     f"(insufficient renal function for efficacy)."),
        ))
    elif 25 <= egfr < 45:
        out.append(SafetyFinding(
            treatment="SGLT-2",
            rule_id="SGLT2_EGFR_25_45",
            severity=SEVERITY_WARNING,
            threshold=45.0, observed_value=float(egfr), feature="egfr",
            message=(f"Reduced glycaemic efficacy: eGFR {egfr:.0f} is between 25–45 — "
                     f"cardiorenal benefits may still apply, monitor closely."),
        ))
    if age > 70:
        out.append(SafetyFinding(
            treatment="SGLT-2",
            rule_id="SGLT2_AGE_70",
            severity=SEVERITY_WARNING,
            threshold=70.0, observed_value=float(age), feature="age",
            message=(f"Patient age {age} > 70 — risk of volume depletion with SGLT-2, "
                     f"ensure adequate hydration and monitor orthostatic BP."),
        ))
    if hypertension and age > 65:
        out.append(SafetyFinding(
            treatment="SGLT-2",
            rule_id="SGLT2_HTN_AGE_65",
            severity=SEVERITY_WARNING,
            threshold=65.0, observed_value=float(age), feature="age",
            message=(f"Hypertensive patient aged {age} — SGLT-2 lowers BP; "
                     f"monitor for excessive hypotension, particularly on antihypertensives."),
        ))
    return out


def _dpp4_findings(ctx: Dict) -> List[SafetyFinding]:
    out: List[SafetyFinding] = []
    age = ctx.get("age", 0)
    egfr = ctx.get("egfr", 999)
    hba1c = ctx.get("hba1c_baseline", 0.0)
    pancreatitis = int(ctx.get("pancreatitis_history", 0))   # G-14

    if pancreatitis:
        out.append(SafetyFinding(
            treatment="DPP-4",
            rule_id="DPP4_PANCREATITIS",
            severity=SEVERITY_CONTRAINDICATION,
            threshold=None, observed_value=1.0,
            feature="pancreatitis_history",
            message=("DPP-4 contraindicated: history of pancreatitis."),
        ))
    if egfr < 45:
        out.append(SafetyFinding(
            treatment="DPP-4",
            rule_id="DPP4_EGFR_45",
            severity=SEVERITY_WARNING,
            threshold=45.0, observed_value=float(egfr), feature="egfr",
            message=(f"eGFR {egfr:.0f} < 45 — DPP-4 dose adjustment required "
                     f"(except linagliptin which is not renally cleared)."),
        ))
    if age < 45:
        out.append(SafetyFinding(
            treatment="DPP-4",
            rule_id="DPP4_AGE_LT_45",
            severity=SEVERITY_WARNING,
            threshold=45.0, observed_value=float(age), feature="age",
            message=(f"Patient age {age} < 45 — DPP-4 has modest efficacy, "
                     f"more potent options may be appropriate."),
        ))
    if hba1c > 9.5:
        out.append(SafetyFinding(
            treatment="DPP-4",
            rule_id="DPP4_HBA1C_95",
            severity=SEVERITY_WARNING,
            threshold=9.5, observed_value=float(hba1c), feature="hba1c_baseline",
            message=(f"HbA1c {hba1c:.1f}% > 9.5% — DPP-4 monotherapy may provide "
                     f"insufficient glycaemic control at this severity."),
        ))
    return out


def _insulin_findings(ctx: Dict) -> List[SafetyFinding]:
    out: List[SafetyFinding] = []
    age = ctx.get("age", 0)
    bmi = ctx.get("bmi", 0.0)
    egfr = ctx.get("egfr", 999)
    hba1c = ctx.get("hba1c_baseline", 0.0)
    cpep = ctx.get("c_peptide", 999)
    cvd = ctx.get("cvd", 0)
    ckd = ctx.get("ckd", 0)
    duration = ctx.get("diabetes_duration", 0)
    t1dm_flag = int(ctx.get("type1_suspicion", 0))   # G-14

    if t1dm_flag:
        out.append(SafetyFinding(
            treatment="Insulin",
            rule_id="INSULIN_TYPE1_SUSPICION",
            severity=SEVERITY_CONTRAINDICATION,
            threshold=None, observed_value=1.0, feature="type1_suspicion",
            message=("Clinical suspicion of misclassified type-1 diabetes — "
                     "insulin strategy must be owned by the diabetologist "
                     "before a bandit recommendation is acted upon."),
        ))
    if hba1c < 8.0 and cpep > 1.5:
        out.append(SafetyFinding(
            treatment="Insulin",
            rule_id="INSULIN_LOW_SEVERITY",
            severity=SEVERITY_WARNING,
            threshold=8.0, observed_value=float(hba1c), feature="hba1c_baseline",
            message=(f"HbA1c {hba1c:.1f}% with preserved C-peptide {cpep:.2f} — "
                     f"insulin may not be necessary, risk of hypoglycaemia and weight gain."),
        ))
    elif hba1c < 9.0 and cpep > 1.0:
        out.append(SafetyFinding(
            treatment="Insulin",
            rule_id="INSULIN_MODERATE_SEVERITY",
            severity=SEVERITY_WARNING,
            threshold=9.0, observed_value=float(hba1c), feature="hba1c_baseline",
            message=(f"HbA1c {hba1c:.1f}% with C-peptide {cpep:.2f} — consider "
                     f"whether oral agents could achieve adequate control first."),
        ))
    if bmi > 30:
        out.append(SafetyFinding(
            treatment="Insulin",
            rule_id="INSULIN_BMI_30",
            severity=SEVERITY_WARNING,
            threshold=30.0, observed_value=float(bmi), feature="bmi",
            message=(f"BMI {bmi:.1f} > 30 — insulin-associated weight gain may worsen "
                     f"metabolic profile, consider weight-neutral alternatives if severity permits."),
        ))
    if age > 70:
        out.append(SafetyFinding(
            treatment="Insulin",
            rule_id="INSULIN_AGE_70",
            severity=SEVERITY_WARNING,
            threshold=70.0, observed_value=float(age), feature="age",
            message=(f"Patient age {age} > 70 — increased hypoglycaemia risk, "
                     f"conservative dosing and less stringent HbA1c targets recommended."),
        ))
    if cvd == 1:
        out.append(SafetyFinding(
            treatment="Insulin",
            rule_id="INSULIN_CVD",
            severity=SEVERITY_WARNING,
            threshold=None, observed_value=1.0, feature="cvd",
            message=("Cardiovascular disease present — hypoglycaemia may raise "
                     "CV risk, close glucose monitoring essential during initiation."),
        ))
    if ckd == 1 or egfr < 60:
        out.append(SafetyFinding(
            treatment="Insulin",
            rule_id="INSULIN_RENAL",
            severity=SEVERITY_WARNING,
            threshold=60.0, observed_value=float(egfr), feature="egfr",
            message=(f"Renal impairment (eGFR {egfr:.0f}) — insulin clearance may be "
                     f"reduced, start with lower doses and titrate cautiously."),
        ))
    if duration > 15 and cpep < 0.5:
        out.append(SafetyFinding(
            treatment="Insulin",
            rule_id="INSULIN_LONG_DURATION",
            severity=SEVERITY_WARNING,
            threshold=15.0, observed_value=float(duration), feature="diabetes_duration",
            message=(f"Long-standing diabetes ({duration:.0f} years) with depleted "
                     f"C-peptide ({cpep:.2f}) — ensure robust hypoglycaemia education."),
        ))
    return out


_RULE_REGISTRY = {
    "Metformin": _metformin_findings,
    "GLP-1": _glp1_findings,
    "SGLT-2": _sglt2_findings,
    "DPP-4": _dpp4_findings,
    "Insulin": _insulin_findings,
}


def get_findings(context: Dict, treatment: str) -> List[SafetyFinding]:
    """G-13: the canonical, structured safety API."""
    fn = _RULE_REGISTRY.get(treatment)
    if fn is None:
        raise ValueError(f"Unknown treatment: {treatment}")
    return fn(context)


# ─────────────────────────────────────────────────────────────────────────────
# BACK-COMPAT STRING WRAPPERS — notebooks 10/older still rely on these
# ─────────────────────────────────────────────────────────────────────────────

def check_contraindications(context: Dict, treatment: str) -> List[str]:
    return [f.message for f in get_findings(context, treatment)
            if f.severity == SEVERITY_CONTRAINDICATION]


def check_warnings(context: Dict, treatment: str) -> List[str]:
    return [f.message for f in get_findings(context, treatment)
            if f.severity == SEVERITY_WARNING]


def run_safety_checks(context: Dict) -> Dict:
    """
    Back-compat dict shape: {"contraindications": {t: [str]},
                             "warnings": {t: [str]}}
    Prefer ``collect_findings`` for new code.
    """
    all_contras: Dict[str, List[str]] = {}
    all_warns: Dict[str, List[str]] = {}
    for t in TREATMENTS:
        findings = get_findings(context, t)
        cs = [f.message for f in findings if f.severity == SEVERITY_CONTRAINDICATION]
        ws = [f.message for f in findings if f.severity == SEVERITY_WARNING]
        if cs:
            all_contras[t] = cs
        if ws:
            all_warns[t] = ws
    return {"contraindications": all_contras, "warnings": all_warns}


def collect_findings(context: Dict) -> Dict[str, List[SafetyFinding]]:
    """Return the full structured-findings dict keyed by treatment."""
    return {t: get_findings(context, t) for t in TREATMENTS}


# ─────────────────────────────────────────────────────────────────────────────
# G-16: SAFETY GATE — override a contraindicated top-1 recommendation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecommendationOverride:
    """Describes a safety-driven override of the model's top-1 pick."""
    original_treatment: str
    final_treatment: str
    reason: str
    blocked_treatments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_treatment": self.original_treatment,
            "final_treatment": self.final_treatment,
            "reason": self.reason,
            "blocked_treatments": list(self.blocked_treatments),
        }


def apply_safety_gate(
    posterior_means: Dict[str, float],
    win_rates: Dict[str, float],
    findings_by_treatment: Dict[str, List[SafetyFinding]],
    top_treatment: str,
) -> Tuple[str, Optional[RecommendationOverride]]:
    """
    G-16: if the model's pick has any CONTRAINDICATION severity finding,
    fall through to the highest-posterior-mean non-contraindicated arm
    and record the override.
    """
    blocked = {
        t for t, fs in findings_by_treatment.items()
        if any(f.severity == SEVERITY_CONTRAINDICATION for f in fs)
    }
    if top_treatment not in blocked:
        return top_treatment, None

    # Rank remaining arms by posterior mean; fall back to win rate on ties.
    remaining = [t for t in TREATMENTS if t not in blocked]
    if not remaining:
        # Every arm is contraindicated: surface the fact, keep original pick.
        return top_treatment, RecommendationOverride(
            original_treatment=top_treatment,
            final_treatment=top_treatment,
            reason=("All treatments carry contraindications for this patient; "
                    "a specialist-level review is required before prescribing."),
            blocked_treatments=sorted(blocked),
        )
    remaining.sort(
        key=lambda t: (posterior_means.get(t, 0.0), win_rates.get(t, 0.0)),
        reverse=True,
    )
    new_top = remaining[0]
    reasons = "; ".join(
        f.message
        for f in findings_by_treatment[top_treatment]
        if f.severity == SEVERITY_CONTRAINDICATION
    )
    return new_top, RecommendationOverride(
        original_treatment=top_treatment,
        final_treatment=new_top,
        reason=reasons,
        blocked_treatments=sorted(blocked),
    )


# ─────────────────────────────────────────────────────────────────────────────
# G-15: FAIRNESS — default is no static paragraph in the payload
# ─────────────────────────────────────────────────────────────────────────────

CLINICAL_FEATURES = [
    "bmi", "hba1c_baseline", "egfr", "diabetes_duration",
    "fasting_glucose", "c_peptide", "bp_systolic", "ldl", "hdl",
    "triglycerides", "alt", "cvd", "ckd", "nafld", "hypertension",
]

DUAL_USE_FEATURES = {
    "age": ("Age is used as a clinical risk factor (renal function decline, "
            "frailty risk, drug tolerability) — not as a demographic discriminator."),
}

EXCLUDED_PROTECTED = ["gender", "ethnicity", "socioeconomic_status"]


@dataclass
class SubgroupRegret:
    subgroup: str
    n_patients: int
    regret: float
    accuracy: float


def build_fairness_report(
    subgroup_regrets: Optional[List[SubgroupRegret]] = None,
) -> Dict[str, Any]:
    """
    G-15: produce a fairness block only when we have a real subgroup-regret
    computation. The previous static paragraph is deleted — if no subgroup
    report is supplied, the payload simply omits the fairness block.
    """
    if not subgroup_regrets:
        return {
            "decision_features": CLINICAL_FEATURES,
            "dual_use_features": DUAL_USE_FEATURES,
            "excluded_protected_features": EXCLUDED_PROTECTED,
            "subgroup_regret": [],
            "attestation_computed": False,
        }

    return {
        "decision_features": CLINICAL_FEATURES,
        "dual_use_features": DUAL_USE_FEATURES,
        "excluded_protected_features": EXCLUDED_PROTECTED,
        "subgroup_regret": [asdict(sr) for sr in subgroup_regrets],
        "attestation_computed": True,
    }


# Keep a minimal, back-compat ``check_fairness`` that returns the non-
# authoritative view (no clinical claims) so notebook 10 continues to render.
def check_fairness() -> Dict:
    return build_fairness_report(subgroup_regrets=None)


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class ExplainabilityExtractor:
    """
    Extracts all information needed for LLM-based explanation
    from a NeuralThompson model and patient context.

    Confidence is computed directly from the model's posterior via repeated
    sampling; safety is run as structured findings (G-13); the top-1 pick
    passes through a safety gate (G-16); fairness is only reported if a
    subgroup-regret table is supplied (G-15); optional attribution and
    uncertainty drivers populate Phase 2 fields (G-7/8/9).
    """

    def __init__(
        self,
        model,
        n_confidence_draws: int = 200,
        subgroup_regrets: Optional[List[SubgroupRegret]] = None,
        attribution_engine: Optional[Any] = None,
    ):
        self.model = model
        self.n_confidence_draws = n_confidence_draws
        self.subgroup_regrets = subgroup_regrets
        # attribution_engine: anything exposing .explain(phi, mu, Ainv, ...) →
        # {"attribution": ..., "contrast": ..., "uncertainty_drivers": ...}
        self.attribution_engine = attribution_engine

        logger.info(
            f"ExplainabilityExtractor initialized — "
            f"confidence method: posterior sampling ({n_confidence_draws} draws), "
            f"attribution={'on' if attribution_engine else 'off'}, "
            f"fairness={'computed' if subgroup_regrets else 'omitted'}"
        )

    # ─────────────────────────────────────────────────────────────────
    # LAYER 1 — Patient Context
    # ─────────────────────────────────────────────────────────────────

    def extract_patient_context(self, context: Dict) -> Dict:
        binary_map = {0: "No", 1: "Yes"}
        return {
            "age": context["age"],
            "bmi": round(context["bmi"], 1),
            "hba1c_baseline": round(context["hba1c_baseline"], 1),
            "egfr": round(context["egfr"], 1),
            "diabetes_duration": round(context["diabetes_duration"], 1),
            "fasting_glucose": round(context["fasting_glucose"], 1),
            "c_peptide": round(context["c_peptide"], 2),
            "bp_systolic": round(context["bp_systolic"], 1),
            "ldl": round(context["ldl"], 1),
            "hdl": round(context["hdl"], 1),
            "triglycerides": round(context["triglycerides"], 1),
            "cvd": binary_map.get(context["cvd"], str(context["cvd"])),
            "ckd": binary_map.get(context["ckd"], str(context["ckd"])),
            "nafld": binary_map.get(context["nafld"], str(context["nafld"])),
            "hypertension": binary_map.get(context["hypertension"],
                                           str(context["hypertension"])),
        }

    # ─────────────────────────────────────────────────────────────────
    # LAYER 2 — Model Decision + Confidence (+ optional attribution)
    # ─────────────────────────────────────────────────────────────────

    def extract_model_decision(self, x: np.ndarray) -> Dict:
        confidence = self.model.compute_confidence(
            x, n_draws=self.n_confidence_draws
        )
        decision = {
            "recommended_treatment": confidence["recommended"],
            "recommended_idx": confidence["recommended_idx"],
            "confidence_pct": confidence["confidence_pct"],
            "confidence_label": confidence["confidence_label"],
            "win_rates": confidence["win_rates"],
            "posterior_means": confidence["posterior_means"],
            "runner_up": confidence["runner_up"],
            "runner_up_win_rate": confidence["runner_up_win_rate"],
            "mean_gap": confidence["mean_gap"],
            "n_draws": confidence["n_draws"],
        }

        if self.attribution_engine is not None:
            try:
                attr = self.attribution_engine.explain(
                    self.model, x,
                    top_treatment=confidence["recommended"],
                    runner_up=confidence["runner_up"],
                )
                decision["attribution"] = attr.get("attribution")
                decision["contrast"] = attr.get("contrast")
                decision["uncertainty_drivers"] = attr.get("uncertainty_drivers")
            except Exception as e:
                logger.warning(f"Attribution engine failed: {e}")
                decision["attribution"] = None
                decision["contrast"] = None
                decision["uncertainty_drivers"] = None

        return decision

    # ─────────────────────────────────────────────────────────────────
    # LAYER 3 — Structured safety + safety gate
    # ─────────────────────────────────────────────────────────────────

    def extract_safety(
        self,
        context: Dict,
        recommended: str,
        posterior_means: Dict[str, float],
        win_rates: Dict[str, float],
    ) -> Tuple[Dict[str, Any], Optional[RecommendationOverride]]:
        findings_by_t = collect_findings(context)
        final_treatment, override = apply_safety_gate(
            posterior_means=posterior_means,
            win_rates=win_rates,
            findings_by_treatment=findings_by_t,
            top_treatment=recommended,
        )

        rec_findings = findings_by_t.get(final_treatment, [])
        rec_contras = [f for f in rec_findings if f.severity == SEVERITY_CONTRAINDICATION]
        rec_warns = [f for f in rec_findings if f.severity == SEVERITY_WARNING]

        # Other-treatment findings for context in the LLM payload
        other_findings: Dict[str, List[SafetyFinding]] = {
            t: fs for t, fs in findings_by_t.items() if t != final_treatment and fs
        }

        if rec_contras:
            status = "CONTRAINDICATION_FOUND"
        elif rec_warns:
            status = "WARNING"
        else:
            status = "CLEAR"

        if status == "CONTRAINDICATION_FOUND":
            logger.warning(
                f"SAFETY ALERT: final recommendation {final_treatment} retains "
                f"contraindication after gate."
            )
        elif override is not None:
            logger.warning(
                f"SAFETY OVERRIDE: {override.original_treatment} → "
                f"{override.final_treatment} due to: {override.reason}"
            )
        elif status == "WARNING":
            logger.info(f"Safety warnings for {final_treatment}")
        else:
            logger.info(f"Safety check CLEAR for {final_treatment}")

        return {
            "status": status,
            "final_treatment": final_treatment,
            "recommended_contraindications": [f.to_dict() for f in rec_contras],
            "recommended_warnings": [f.to_dict() for f in rec_warns],
            "excluded_treatments": {
                t: [f.to_dict() for f in fs if f.severity == SEVERITY_CONTRAINDICATION]
                for t, fs in findings_by_t.items()
                if any(fi.severity == SEVERITY_CONTRAINDICATION for fi in fs)
            },
            "all_findings": {
                t: [f.to_dict() for f in fs]
                for t, fs in findings_by_t.items()
                if fs
            },
            "other_treatment_warnings": {
                t: [f.to_dict() for f in fs if f.severity == SEVERITY_WARNING]
                for t, fs in other_findings.items()
                if any(fi.severity == SEVERITY_WARNING for fi in fs)
            },
        }, override

    # ─────────────────────────────────────────────────────────────────
    # FULL EXTRACTION
    # ─────────────────────────────────────────────────────────────────

    def extract(self, context: Dict, x: np.ndarray) -> Dict:
        patient = self.extract_patient_context(context)
        decision = self.extract_model_decision(x)
        safety, override = self.extract_safety(
            context=context,
            recommended=decision["recommended_treatment"],
            posterior_means=decision["posterior_means"],
            win_rates=decision["win_rates"],
        )

        # G-16: if the gate promoted a different arm, the payload's
        # "recommended_treatment" is the final clinical choice. We keep the
        # model's original pick as ``model_top_treatment`` so the UI can still
        # explain what happened.
        final_treatment = safety["final_treatment"]
        if override is not None and final_treatment != decision["recommended_treatment"]:
            decision["model_top_treatment"] = decision["recommended_treatment"]
            decision["recommended_treatment"] = final_treatment
            decision["override"] = override.to_dict()
        else:
            decision["model_top_treatment"] = decision["recommended_treatment"]
            decision["override"] = None

        payload: Dict[str, Any] = {
            "patient": patient,
            "decision": decision,
            "safety": safety,
        }

        fairness = build_fairness_report(self.subgroup_regrets)
        if fairness["attestation_computed"]:
            payload["fairness"] = fairness
        # Otherwise omit fairness entirely (G-15: no static boilerplate)

        logger.info(
            f"Extracted payload: final={final_treatment}, "
            f"orig={decision['model_top_treatment']}, "
            f"confidence={decision['confidence_pct']}% ({decision['confidence_label']}), "
            f"safety={safety['status']}"
        )
        return payload


# ─────────────────────────────────────────────────────────────────────────────
# BACK-COMPAT: legacy helper used by notebook 10
# ─────────────────────────────────────────────────────────────────────────────

def get_safety_for_recommended(safety_all: Dict, recommended: str) -> Dict:
    rec_contras = safety_all["contraindications"].get(recommended, [])
    rec_warns = safety_all["warnings"].get(recommended, [])
    excluded = dict(safety_all["contraindications"])
    status = ("CONTRAINDICATION_FOUND" if rec_contras
              else "WARNING" if rec_warns
              else "CLEAR")
    return {
        "recommended_contraindications": rec_contras,
        "recommended_warnings": rec_warns,
        "excluded_treatments": excluded,
        "all_warnings": safety_all["warnings"],
        "status": status,
    }
