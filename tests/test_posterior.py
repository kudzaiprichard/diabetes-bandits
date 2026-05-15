"""
Phase 0 — posterior invariants for NeuralThompson (G-1, G-6).

Covers:
    - Posterior precision matrices stay positive semi-definite after
      ``initialize_posterior`` (G-1) and subsequent ``update_posterior`` calls.
    - Thompson's ``select_action`` is deterministic under a fixed np.random seed
      (baseline for G-27's RNG plumbing).
    - ``noise_variance_from_residuals`` returns a positive float and mutates
      the attribute when requested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    N_TREATMENTS,
    generate_patient,
    reward_oracle,
)
from src.feature_engineering import get_scaled_pipeline, ALL_FEATURES
from src.neural_bandit import NeuralThompson


@pytest.fixture(scope="module")
def trained_thompson():
    """A small NeuralThompson trained on a synthetic cohort."""
    rng = np.random.default_rng(7)
    rows = [generate_patient(rng) for _ in range(400)]
    df = pd.DataFrame(rows)
    pipe = get_scaled_pipeline()
    X = pipe.fit_transform(df)

    # Random actions, oracle rewards (noiseless for test stability)
    actions = rng.integers(0, N_TREATMENTS, size=len(df))
    from src.data_generator import IDX_TO_TREATMENT
    rewards = np.array([
        reward_oracle(rows[i], IDX_TO_TREATMENT[int(actions[i])], noise=False)
        for i in range(len(df))
    ])

    model = NeuralThompson(
        input_dim=X.shape[1],
        hidden_dims=[16, 8],
        dropout=0.0,
        reg_lambda=1.0,
        noise_variance=0.1,
    )
    # Minimal training — 3 epochs is enough for this test
    model.train(X, actions, rewards, epochs=3, verbose=False)
    return model, X, actions, rewards


def test_initialize_posterior_is_psd(trained_thompson):
    """G-1: bootstrapping from data produces PSD precision matrices."""
    model, X, actions, rewards = trained_thompson
    model.initialize_posterior(X, actions, rewards)

    for k in range(N_TREATMENTS):
        A = model.A[k]
        # symmetry
        assert np.allclose(A, A.T, atol=1e-8)
        eigs = np.linalg.eigvalsh((A + A.T) / 2)
        assert eigs.min() >= 0.0 - 1e-8, (k, eigs.min())


def test_posterior_mean_becomes_nontrivial_after_bootstrap(trained_thompson):
    """G-1: after bootstrap, posterior means must differ from zero."""
    model, X, actions, rewards = trained_thompson
    model.reset_posterior()
    assert all(np.allclose(m, 0.0) for m in model.mu), \
        "precondition: posterior should start at zero"

    model.initialize_posterior(X, actions, rewards)
    assert any(np.linalg.norm(m) > 1e-3 for m in model.mu), \
        "post-bootstrap posterior must not stay at zero"


def test_thompson_select_action_deterministic_under_fixed_seed(trained_thompson):
    """
    Thompson is stochastic, but with a pre-set np.random seed two back-to-back
    ``select_action`` calls should return identical outputs — baseline for
    the RNG plumbing in G-27.
    """
    model, X, actions, rewards = trained_thompson
    model.initialize_posterior(X, actions, rewards)

    x = X[0]

    np.random.seed(1234)
    a1, r1 = model.select_action(x)
    np.random.seed(1234)
    a2, r2 = model.select_action(x)
    assert a1 == a2
    np.testing.assert_allclose(r1, r2)


def test_noise_variance_from_residuals(trained_thompson):
    """G-10: residual variance estimator is positive and sets attribute."""
    model, X, actions, rewards = trained_thompson
    before = model.noise_variance
    sigma2 = model.noise_variance_from_residuals(X, actions, rewards)
    assert sigma2 > 0.0
    assert model.noise_variance == sigma2
    # Restore for downstream tests in the session
    model.noise_variance = before


def test_posterior_update_preserves_psd(trained_thompson):
    """After a handful of online updates the precision must remain PSD."""
    model, X, actions, rewards = trained_thompson
    model.initialize_posterior(X, actions, rewards)

    rng = np.random.default_rng(0)
    for _ in range(25):
        i = int(rng.integers(0, X.shape[0]))
        model.update_posterior(X[i], int(actions[i]), float(rewards[i]))
    for k in range(N_TREATMENTS):
        eigs = np.linalg.eigvalsh((model.A[k] + model.A[k].T) / 2)
        assert eigs.min() >= 0.0 - 1e-8, (k, eigs.min())
