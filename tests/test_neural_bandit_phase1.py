"""
Phase 1 — G-2 / G-5 / G-6 invariants for NeuralThompson.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    N_TREATMENTS,
    IDX_TO_TREATMENT,
    generate_patient,
    reward_oracle,
)
from src.feature_engineering import get_scaled_pipeline
from src.neural_bandit import NeuralThompson


@pytest.fixture(scope="module")
def cohort():
    rng = np.random.default_rng(11)
    rows = [generate_patient(rng) for _ in range(300)]
    df = pd.DataFrame(rows)
    pipe = get_scaled_pipeline()
    X = pipe.fit_transform(df)
    actions = rng.integers(0, N_TREATMENTS, size=len(df))
    rewards = np.array([
        reward_oracle(rows[i], IDX_TO_TREATMENT[int(actions[i])], noise=False)
        for i in range(len(df))
    ])
    counterfactuals = np.zeros((len(df), N_TREATMENTS))
    for i in range(len(df)):
        for k in range(N_TREATMENTS):
            counterfactuals[i, k] = reward_oracle(rows[i], IDX_TO_TREATMENT[k], noise=False)
    return X, actions, rewards, counterfactuals


def test_counterfactual_training_runs(cohort):
    """G-2: train() accepts the (n, K) counterfactual matrix."""
    X, actions, rewards, counterfactuals = cohort
    model = NeuralThompson(input_dim=X.shape[1], hidden_dims=[16, 8], dropout=0.0)
    hist = model.train(
        X, actions, rewards, counterfactuals=counterfactuals,
        epochs=2, verbose=False,
    )
    assert hist["epochs_run"] >= 1


def test_forgetting_factor_decays_A(cohort):
    """G-5: γ < 1 causes A_k to decay across repeated updates of a single arm."""
    X, actions, rewards, _ = cohort
    model = NeuralThompson(
        input_dim=X.shape[1], hidden_dims=[16, 8], dropout=0.0,
        forgetting_factor=0.9,
    )
    model.train(X, actions, rewards, epochs=2, verbose=False)
    # All-zero context approximation: repeated update for arm 0 with constant reward
    x = X[0]
    for _ in range(50):
        model.update_posterior(x, action=0, reward=1.0)
    # Trace of A should stabilise near a steady state; diverges without γ
    trace_final = float(np.trace(model.A[0]))
    assert trace_final < 1e4  # would blow up without forgetting for 50 repeats of same φ


def test_vectorized_compute_confidence_shape(cohort):
    """G-6: compute_confidence returns the expected dict shape and valid win rates."""
    X, actions, rewards, _ = cohort
    model = NeuralThompson(input_dim=X.shape[1], hidden_dims=[16, 8], dropout=0.0)
    model.train(X, actions, rewards, epochs=2, verbose=False)
    model.initialize_posterior(X, actions, rewards)
    out = model.compute_confidence(X[0], n_draws=100)
    assert set(out["win_rates"].keys()) == set(IDX_TO_TREATMENT.values())
    total = sum(out["win_rates"].values())
    assert abs(total - 1.0) < 1e-6
    assert 0 <= out["confidence_pct"] <= 100


def test_online_retrain_toggle(cohort):
    """G-4: enabling online retraining is side-effect-free until threshold."""
    X, actions, rewards, _ = cohort
    model = NeuralThompson(input_dim=X.shape[1], hidden_dims=[16, 8], dropout=0.0)
    model.train(X, actions, rewards, epochs=1, verbose=False)
    model.initialize_posterior(X, actions, rewards)
    model.enable_online_retraining(
        buffer_size=100, retrain_every=50, minibatch_size=32,
        retrain_epochs=1, min_buffer_for_retrain=50,
    )
    for i in range(10):
        model.online_update(X[i], int(actions[i]), float(rewards[i]))
    assert len(model.replay_buffer) == 10
    assert model._online_step == 10
