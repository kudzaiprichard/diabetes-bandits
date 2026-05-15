"""
Phase 0 — feature pipeline round-trip test.

Asserts that the scaled pipeline produces consistent output shapes and that
the un-scaled pipeline returns the raw feature values as written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generator import generate_patient
from src.feature_engineering import (
    get_scaled_pipeline,
    get_unscaled_pipeline,
    ALL_FEATURES,
)


def _batch_df(n: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = [generate_patient(rng) for _ in range(n)]
    return pd.DataFrame(rows)


def test_scaled_pipeline_shape_and_finiteness():
    df = _batch_df()
    pipe = get_scaled_pipeline()
    X = pipe.fit_transform(df)
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(ALL_FEATURES)
    assert np.isfinite(X).all()


def test_unscaled_pipeline_preserves_values():
    df = _batch_df()
    pipe = get_unscaled_pipeline()
    X = pipe.fit_transform(df)
    assert X.shape == (len(df), len(ALL_FEATURES))
    # continuous columns should round-trip approximately
    idx_age = ALL_FEATURES.index("age")
    np.testing.assert_allclose(X[:, idx_age], df["age"].to_numpy(), rtol=1e-6)


def test_pipeline_handles_single_row():
    df = _batch_df(n=1)
    pipe = get_scaled_pipeline()
    X = pipe.fit_transform(df)
    assert X.shape == (1, len(ALL_FEATURES))
