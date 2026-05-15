"""
Phase 2 — closed-form attribution / contrast / uncertainty invariants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    N_TREATMENTS, IDX_TO_TREATMENT,
    generate_patient, reward_oracle,
)
from src.feature_engineering import get_scaled_pipeline, ALL_FEATURES
from src.interpretability import (
    AttributionEngine,
    integrated_gradients,
    uncertainty_decomposition,
)
from src.neural_bandit import NeuralThompson


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(17)
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
    return model, X, ALL_FEATURES


def test_integrated_gradients_shape(trained):
    model, X, feat_names = trained
    attr = integrated_gradients(model, X[0], arm=0, n_steps=8)
    assert attr.shape == (X.shape[1],)
    assert np.isfinite(attr).all()


def test_attribution_completeness(trained):
    """
    IG completeness axiom: Σ_i IG_i ≈ φ(x)·μ − φ(baseline)·μ.
    Exact for linear networks; approximate for deep ReLU nets.
    """
    model, X, _ = trained
    x = X[0]
    baseline = np.zeros_like(x)
    mu = model.mu[0]

    import torch
    with torch.no_grad():
        phi_x = model.network.get_features(
            torch.FloatTensor(x.reshape(1, -1)).to(model.device)
        ).cpu().numpy().flatten()
        phi_b = model.network.get_features(
            torch.FloatTensor(baseline.reshape(1, -1)).to(model.device)
        ).cpu().numpy().flatten()
    target = float(phi_x @ mu - phi_b @ mu)

    attr = integrated_gradients(model, x, arm=0, n_steps=64)
    recovered = float(attr.sum())
    # Allow 15% slack for ReLU nonlinearity discretisation error
    assert abs(recovered - target) <= max(0.2, 0.15 * abs(target))


def test_attribution_engine_returns_expected_keys(trained):
    model, X, feat_names = trained
    engine = AttributionEngine(feature_names=feat_names, n_steps=8)
    out = engine.explain(
        model, X[0], top_treatment="Metformin", runner_up="GLP-1",
    )
    assert "attribution" in out
    assert "contrast" in out
    assert "uncertainty_drivers" in out
    assert isinstance(out["attribution"], dict)
    assert len(out["attribution"]) == len(feat_names)
    assert out["contrast"] is not None
    assert len(out["uncertainty_drivers"]) <= 5


def test_uncertainty_decomposition_shape(trained):
    model, X, feat_names = trained
    drivers = uncertainty_decomposition(
        model, X[0], arm=0, feature_names=feat_names, n_steps=4,
    )
    assert len(drivers) == len(feat_names)
    assert all("feature" in d and "contribution" in d for d in drivers)
    # Sorted by |contribution| descending
    contribs = [abs(d["contribution"]) for d in drivers]
    assert contribs == sorted(contribs, reverse=True)
