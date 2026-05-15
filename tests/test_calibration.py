"""
Phase 1 — G-11 calibration utilities.
"""
from __future__ import annotations

import numpy as np

from src.evaluation import (
    confidence_label_accuracy,
    expected_calibration_error,
    reliability_diagram,
)


def test_ece_perfect_calibration_is_zero():
    """If predicted confidence exactly matches accuracy, ECE = 0."""
    rng = np.random.default_rng(0)
    conf = rng.integers(0, 101, size=5000)
    # Simulate correctness where P(correct|c) = c/100
    correctness = (rng.uniform(size=5000) < conf / 100.0).astype(int)
    ece = expected_calibration_error(conf, correctness, n_bins=10)
    assert ece < 0.05


def test_ece_uniformly_overconfident_is_large():
    """If the model is always 95% confident but only right 50% of the time, ECE ~ 0.45."""
    n = 5000
    conf = np.full(n, 95)
    rng = np.random.default_rng(1)
    correctness = (rng.uniform(size=n) < 0.5).astype(int)
    ece = expected_calibration_error(conf, correctness, n_bins=10)
    assert 0.4 <= ece <= 0.5


def test_reliability_diagram_shape():
    n = 200
    rng = np.random.default_rng(0)
    conf = rng.integers(0, 101, size=n)
    correct = rng.integers(0, 2, size=n)
    diag = reliability_diagram(conf, correct, n_bins=10)
    assert diag["bin_centers"].shape == (10,)
    assert diag["bin_counts"].sum() == n


def test_confidence_label_accuracy_buckets():
    labels = np.array(["HIGH", "HIGH", "MODERATE", "LOW", "LOW", "LOW"])
    correct = np.array([1, 1, 1, 0, 0, 1])
    out = confidence_label_accuracy(labels, correct)
    assert out["HIGH"]["n"] == 2 and out["HIGH"]["accuracy"] == 1.0
    assert out["MODERATE"]["n"] == 1 and out["MODERATE"]["accuracy"] == 1.0
    assert out["LOW"]["n"] == 3
    assert abs(out["LOW"]["accuracy"] - 1 / 3) < 1e-6
