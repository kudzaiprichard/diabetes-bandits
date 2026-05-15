"""
Shared test helpers: build tiny FeaturePipeline + NeuralThompson artefacts on
the fly so inference tests don't depend on ``models/`` existing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data_generator import generate_patient, reward_oracle, TREATMENTS
from src.feature_engineering import FeaturePipeline
from src.neural_bandit import NeuralThompson


def build_tiny_artefacts(
    tmp_dir: Path,
    n_patients: int = 400,
    seed: int = 7,
) -> Tuple[Path, Path, FeaturePipeline, NeuralThompson]:
    """
    Train a tiny NeuralThompson on a small synthetic cohort and save both
    the feature pipeline and the model checkpoint under ``tmp_dir``.

    Returns ``(model_path, pipeline_path, pipeline, model)``.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_patients):
        ctx = generate_patient(rng)
        a = int(rng.randint(0, len(TREATMENTS)))
        r = reward_oracle(ctx, TREATMENTS[a], noise=True)
        ctx.update({"action": a, "reward": r, "propensity": 0.2})
        rows.append(ctx)
    df = pd.DataFrame(rows)

    pipeline = FeaturePipeline(scale=True, add_interactions=True)
    X = pipeline.fit_transform(df)
    actions = df["action"].values
    rewards = df["reward"].values

    pipeline_path = tmp_dir / "feature_pipeline.joblib"
    pipeline.save(str(pipeline_path))

    model = NeuralThompson(input_dim=X.shape[1], hidden_dims=[16, 8], device="cpu")
    model.train(X, actions, rewards, epochs=3, verbose=False, val_fraction=0.1)
    model.initialize_posterior(X, actions, rewards)

    model_path = tmp_dir / "neural_thompson.pt"
    model.save(str(model_path))

    return model_path, pipeline_path, pipeline, model


def sample_patient(rng: np.random.RandomState) -> dict:
    """Generate a patient context dict suitable for PatientInput."""
    ctx = generate_patient(rng)
    # generate_patient already produces all 16 features + 4 optional safety flags
    # (safety flags default to 0 if absent — PatientInput handles it)
    return ctx
