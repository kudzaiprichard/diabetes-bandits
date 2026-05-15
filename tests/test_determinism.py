"""
G-27: pin the deterministic-seed contract.

Given identical seeds and identical inputs, the same agent must produce
an identical action sequence. This protects against future refactors
that accidentally introduce non-reproducible randomness (e.g. reading
from an unseeded global RNG in a new code path).
"""
from __future__ import annotations

import numpy as np

from src.utils import seed_everything
from src.policies import LinUCBPolicy, EpsilonGreedyPolicy, BoltzmannPolicy


def _run_eps_greedy(seed: int, n: int = 200) -> list[int]:
    seed_everything(seed)
    policy = EpsilonGreedyPolicy(epsilon=0.3, decay=1.0)
    rewards = np.random.default_rng(0).normal(0.5, 0.2, size=(n, 5))
    return [policy.select_action(rewards[t]) for t in range(n)]


def _run_boltzmann(seed: int, n: int = 200) -> list[int]:
    seed_everything(seed)
    policy = BoltzmannPolicy(temperature=0.5)
    rewards = np.random.default_rng(0).normal(0.5, 0.2, size=(n, 5))
    return [policy.select_action(rewards[t]) for t in range(n)]


def _run_linucb(seed: int, n: int = 200, d: int = 8) -> list[int]:
    seed_everything(seed)
    policy = LinUCBPolicy(feature_dim=d, alpha=1.0)
    rng = np.random.default_rng(0)
    xs = rng.normal(0, 1, size=(n, d))
    actions = []
    for t in range(n):
        a = policy.select_action(np.zeros(5), x=xs[t])
        reward = float(rng.normal(0.5, 0.1))
        policy.update_model(xs[t], a, reward)
        actions.append(a)
    return actions


def test_epsilon_greedy_is_deterministic_under_seed():
    assert _run_eps_greedy(42) == _run_eps_greedy(42)


def test_boltzmann_is_deterministic_under_seed():
    assert _run_boltzmann(42) == _run_boltzmann(42)


def test_different_seeds_produce_different_sequences():
    a = _run_eps_greedy(1)
    b = _run_eps_greedy(2)
    assert a != b


def test_linucb_is_deterministic_under_seed():
    assert _run_linucb(42) == _run_linucb(42)
