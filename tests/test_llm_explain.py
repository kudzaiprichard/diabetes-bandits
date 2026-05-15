"""
Phase 3 — guardrail tests for the LLM explanation pipeline.

Uses a stub LLM client so tests run without any network access or API key.
Covers:
    - The pluggable LLMClient interface.
    - Pydantic schema validation rejects responses missing required keys.
    - Implausible HbA1c claims (> REWARD_CAP_PP) are rejected.
    - ML-jargon leakage is rejected.
    - A happy-path stub response passes end-to-end.
"""
from __future__ import annotations

import json
import pytest

from src.data_generator import REWARD_CAP_PP
from src.llm_explain import (
    GEMINI_AVAILABLE,
    LLMClient,
    LLMExplainer,
    PYDANTIC_AVAILABLE,
    build_prompt,
    parse_llm_response,
)


pytestmark = pytest.mark.skipif(
    not PYDANTIC_AVAILABLE, reason="pydantic not installed"
)


class StubClient(LLMClient):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        self.calls += 1
        return self.responses.pop(0)


def _sample_payload() -> dict:
    return {
        "patient": {
            "age": 60, "bmi": 29.0, "hba1c_baseline": 8.5, "egfr": 75.0,
            "diabetes_duration": 8.0, "fasting_glucose": 170.0,
            "c_peptide": 1.5, "bp_systolic": 135.0, "ldl": 110.0,
            "hdl": 45.0, "triglycerides": 150.0, "cvd": "Yes",
            "ckd": "No", "nafld": "No", "hypertension": "Yes",
        },
        "decision": {
            "recommended_treatment": "SGLT-2",
            "recommended_idx": 2,
            "model_top_treatment": "SGLT-2",
            "override": None,
            "confidence_pct": 78,
            "confidence_label": "MODERATE",
            "win_rates": {"Metformin": 0.05, "GLP-1": 0.07,
                          "SGLT-2": 0.78, "DPP-4": 0.05, "Insulin": 0.05},
            "posterior_means": {"Metformin": 0.8, "GLP-1": 1.4,
                                "SGLT-2": 2.1, "DPP-4": 0.9, "Insulin": 0.5},
            "runner_up": "GLP-1",
            "runner_up_win_rate": 0.07,
            "mean_gap": 0.7,
            "n_draws": 200,
        },
        "safety": {
            "status": "CLEAR",
            "final_treatment": "SGLT-2",
            "recommended_contraindications": [],
            "recommended_warnings": [],
            "excluded_treatments": {},
            "all_findings": {},
            "other_treatment_warnings": {},
        },
    }


def _good_response() -> str:
    return json.dumps({
        "recommendation_summary": (
            "For this 60-year-old patient with cardiovascular disease and "
            "preserved renal function (eGFR 75), the model predicts SGLT-2 "
            "would achieve a 2.1 pp HbA1c reduction."
        ),
        "runner_up_analysis": (
            "GLP-1 was the next best alternative with a predicted 1.4 pp "
            "HbA1c reduction and 7% win rate."
        ),
        "confidence_statement": (
            "The model is moderately confident — SGLT-2 won 78% of 200 "
            "simulations."
        ),
        "safety_assessment": (
            "No contraindications were identified for SGLT-2 in this patient."
        ),
        "monitoring_note": (
            "Monitor eGFR every 3-6 months and watch for volume depletion."
        ),
        "disclaimer": (
            "This is an AI-assisted decision support tool. Final treatment "
            "decisions must be made by the treating physician."
        ),
    })


def test_build_prompt_does_not_exceed_cap():
    prompt = build_prompt(_sample_payload())
    import re
    vals = [float(m.group(1)) for m in re.finditer(r"(-?\d+\.\d+)\s*pp", prompt)]
    assert vals, "expected pp values in prompt"
    assert all(v <= REWARD_CAP_PP + 0.1 for v in vals)


def test_parse_llm_response_requires_all_keys():
    incomplete = json.dumps({"recommendation_summary": "x"})
    with pytest.raises(ValueError):
        parse_llm_response(incomplete)


def test_explainer_happy_path_with_stub():
    client = StubClient(responses=[_good_response()])
    explainer = LLMExplainer(client=client, max_retries=0)
    out = explainer.explain(_sample_payload())
    assert "SGLT-2" in out["recommendation_summary"]
    assert client.calls == 1


def test_explainer_rejects_implausible_effect_size():
    bad = json.loads(_good_response())
    bad["recommendation_summary"] = (
        "For this patient, SGLT-2 would achieve a 12.0 pp HbA1c reduction, "
        "which is far above the clinically plausible range."
    )
    client = StubClient(responses=[json.dumps(bad)])
    explainer = LLMExplainer(client=client, max_retries=0)
    with pytest.raises(ValueError):
        explainer.explain(_sample_payload())


def test_explainer_rejects_ml_jargon():
    bad = json.loads(_good_response())
    bad["confidence_statement"] = (
        "The posterior mean for SGLT-2 is 2.1 and Thompson sampling "
        "supports the choice."
    )
    client = StubClient(responses=[json.dumps(bad)])
    explainer = LLMExplainer(client=client, max_retries=0)
    with pytest.raises(ValueError):
        explainer.explain(_sample_payload())


def test_explainer_retries_on_schema_violation():
    bad = json.dumps({"recommendation_summary": "x"})
    good = _good_response()
    client = StubClient(responses=[bad, good])
    explainer = LLMExplainer(client=client, max_retries=2)
    out = explainer.explain(_sample_payload())
    assert client.calls == 2
    assert out["recommendation_summary"].startswith("For this")
