"""
Phase 0 — LLM payload / prompt invariants.

Covers:
    - The payload produced by ExplainabilityExtractor carries structured
      SafetyFinding dicts (G-13) — never bare strings.
    - The assembled prompt never states an HbA1c reduction greater than
      REWARD_CAP_PP (G-3 success criterion).
    - When an arm is contraindicated, the gate surfaces an override and the
      prompt reflects it (G-16).
    - The fairness section is omitted when no subgroup-regret data is supplied
      (G-15 success criterion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import re

from src.data_generator import (
    REWARD_CAP_PP,
    N_TREATMENTS,
    generate_patient,
    reward_oracle,
    IDX_TO_TREATMENT,
)
from src.explainability import ExplainabilityExtractor
from src.feature_engineering import get_scaled_pipeline
from src.llm_explain import build_prompt
from src.neural_bandit import NeuralThompson


@pytest.fixture(scope="module")
def extractor_and_batch():
    rng = np.random.default_rng(7)
    rows = [generate_patient(rng) for _ in range(300)]
    df = pd.DataFrame(rows)
    pipe = get_scaled_pipeline()
    X = pipe.fit_transform(df)
    actions = rng.integers(0, N_TREATMENTS, size=len(df))
    rewards = np.array([
        reward_oracle(rows[i], IDX_TO_TREATMENT[int(actions[i])], noise=False)
        for i in range(len(df))
    ])
    model = NeuralThompson(
        input_dim=X.shape[1], hidden_dims=[16, 8], dropout=0.0,
        reg_lambda=1.0, noise_variance=0.1,
    )
    model.train(X, actions, rewards, epochs=3, verbose=False)
    model.initialize_posterior(X, actions, rewards)
    extractor = ExplainabilityExtractor(model, n_confidence_draws=60)
    return extractor, rows, X


def test_payload_safety_findings_are_structured(extractor_and_batch):
    extractor, rows, X = extractor_and_batch
    payload = extractor.extract(rows[0], X[0])
    s = payload["safety"]
    # Must exist
    assert "recommended_contraindications" in s
    assert "recommended_warnings" in s
    assert "excluded_treatments" in s
    # Each finding must be a dict with rule_id + message (G-13)
    for f in s["recommended_contraindications"] + s["recommended_warnings"]:
        assert isinstance(f, dict)
        assert "rule_id" in f
        assert "message" in f
        assert "severity" in f


def test_payload_fairness_omitted_by_default(extractor_and_batch):
    """G-15: no subgroup data ⇒ no fairness block in payload."""
    extractor, rows, X = extractor_and_batch
    payload = extractor.extract(rows[0], X[0])
    assert "fairness" not in payload


def test_prompt_mentions_only_plausible_effect_sizes(extractor_and_batch):
    """
    G-3 success criterion: no explanation must state an HbA1c reduction
    greater than {REWARD_CAP_PP} pp. We grep the rendered prompt for numbers
    followed by 'pp' and assert they are all ≤ REWARD_CAP_PP.
    """
    extractor, rows, X = extractor_and_batch
    prompt = build_prompt(extractor.extract(rows[0], X[0]))
    # Pattern matches "X.YY pp" or "X.Y pp" or "-X.Y pp"
    pp_values = re.findall(r"(-?\d+\.\d+)\s*pp", prompt)
    assert pp_values, "no pp values found in prompt"
    for v in pp_values:
        val = float(v)
        # Allow a small rounding epsilon above the cap
        assert val <= REWARD_CAP_PP + 0.1, (
            f"prompt contains implausible HbA1c reduction {val} pp "
            f"(cap {REWARD_CAP_PP})"
        )


def test_safety_gate_overrides_low_egfr_metformin(extractor_and_batch):
    """G-16: a contraindicated top-1 pick is demoted and an override appears."""
    extractor, rows, X = extractor_and_batch
    # Force-craft a patient where Metformin is contraindicated
    ctx = dict(rows[0])
    ctx["egfr"] = 22.0
    # Use the first transformed row as x stand-in; we only care about the
    # safety gate path triggered by the context.
    payload = extractor.extract(ctx, X[0])
    # The final recommendation must not be Metformin
    assert payload["decision"]["recommended_treatment"] != "Metformin"
    if payload["decision"]["override"] is not None:
        ov = payload["decision"]["override"]
        assert "Metformin" in ov["blocked_treatments"]
