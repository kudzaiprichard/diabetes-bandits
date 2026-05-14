"""
Feature Engineering for Diabetes Contextual Bandits

Handles:
- Feature scaling (StandardScaler for neural, passthrough for tree-based)
- Interaction features (clinically meaningful combinations)
- Feature selection and validation
- Train/test splitting with stratification
- Context vector construction for bandit consumption
- Pipeline persistence (save/load fitted pipeline)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional, List
from loguru import logger

from .constants import CONTEXT_FEATURES, TREATMENTS, N_TREATMENTS, TREATMENT_TO_IDX

# ─────────────────────────────────────────────────────────────────────────────
# RAW FEATURES (from data generator)
# ─────────────────────────────────────────────────────────────────────────────

CONTINUOUS_FEATURES = [
    "age", "bmi", "hba1c_baseline", "egfr", "diabetes_duration",
    "fasting_glucose", "c_peptide", "bp_systolic", "ldl", "hdl",
    "triglycerides", "alt",
]

BINARY_FEATURES = ["cvd", "ckd", "nafld", "hypertension"]

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTION FEATURES (clinically motivated)
# ─────────────────────────────────────────────────────────────────────────────

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically meaningful interaction features.

    These capture non-linear relationships that help the bandit
    distinguish which treatment is best for a given patient profile.
    """
    out = df.copy()

    # Kidney-adjusted age: elderly + CKD is a distinct clinical scenario
    out["age_x_ckd"] = out["age"] * out["ckd"]

    # Obesity severity with liver involvement
    out["bmi_x_nafld"] = out["bmi"] * out["nafld"]

    # Cardiovascular-renal overlap
    out["cvd_x_egfr"] = out["cvd"] * out["egfr"]

    # Beta-cell reserve relative to disease severity
    out["cpeptide_x_hba1c"] = out["c_peptide"] * out["hba1c_baseline"]

    # Disease progression indicator
    out["duration_x_hba1c"] = out["diabetes_duration"] * out["hba1c_baseline"]

    # Insulin resistance proxy (high fasting glucose + high BMI)
    out["fg_x_bmi"] = out["fasting_glucose"] * out["bmi"] / 100.0

    # Metabolic syndrome proxy
    out["tg_hdl_ratio"] = out["triglycerides"] / out["hdl"].clip(lower=1.0)

    # Renal risk score
    out["renal_risk"] = (
        (out["egfr"] < 60).astype(float)
        + (out["egfr"] < 30).astype(float)
        + out["ckd"]
        + out["hypertension"]
    )

    # Severity composite
    out["severity_score"] = (
        (out["hba1c_baseline"] - 7.0).clip(lower=0) / 3.0
        + (1.0 - out["c_peptide"].clip(upper=2.0) / 2.0)
        + out["diabetes_duration"] / 30.0
    )

    return out


INTERACTION_FEATURES = [
    "age_x_ckd", "bmi_x_nafld", "cvd_x_egfr", "cpeptide_x_hba1c",
    "duration_x_hba1c", "fg_x_bmi", "tg_hdl_ratio", "renal_risk",
    "severity_score",
]

ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES + INTERACTION_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class FeaturePipeline:
    """
    Unified feature preprocessing for all bandit models.

    Usage:
        # Training time — fit, transform, and save
        pipe = FeaturePipeline(scale=True)
        X_train, X_test, meta = pipe.fit_transform_split(df)
        pipe.save("models/feature_pipeline.joblib")

        # Inference time — load and transform
        pipe = FeaturePipeline.load("models/feature_pipeline.joblib")
        x = pipe.transform_single(patient_context_dict)
    """

    def __init__(
        self,
        scale: bool = True,
        add_interactions: bool = True,
        features: Optional[List[str]] = None,
    ):
        """
        Args:
            scale: StandardScale continuous features (True for neural, False for trees)
            add_interactions: compute interaction features
            features: override feature list (default: ALL_FEATURES)
        """
        self.scale = scale
        self.add_interactions = add_interactions
        self.features = features or ALL_FEATURES
        self.scaler: Optional[StandardScaler] = None
        self._continuous_idx: Optional[List[int]] = None
        self._fitted = False

    def _get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and optionally create interaction features."""
        if self.add_interactions:
            df = compute_interaction_features(df)

        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in dataframe: {missing}")

        return df[self.features].copy()

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit scaler on training data."""
        feat_df = self._get_feature_matrix(df)

        if self.scale:
            self._continuous_idx = [
                i for i, f in enumerate(self.features) if f in CONTINUOUS_FEATURES + INTERACTION_FEATURES
            ]
            self.scaler = StandardScaler()
            self.scaler.fit(feat_df.iloc[:, self._continuous_idx].to_numpy())

        self._fitted = True
        logger.info(f"FeaturePipeline fitted: {len(self.features)} features, scale={self.scale}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe to feature matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        feat_df = self._get_feature_matrix(df)
        X = feat_df.values.astype(np.float64)

        if self.scale and self.scaler is not None:
            X[:, self._continuous_idx] = self.scaler.transform(X[:, self._continuous_idx])

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def transform_single(self, context: Dict) -> np.ndarray:
        """
        Transform a single patient context dict into a feature vector.
        Used during online simulation / inference.
        """
        row_df = pd.DataFrame([context])
        return self.transform(row_df).flatten()

    def fit_transform_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify_col: str = "action",
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Full pipeline: interactions → split → fit on train → transform both.

        Returns:
            X_train: (n_train, d) scaled feature matrix
            X_test:  (n_test, d) scaled feature matrix
            meta: dict with {
                train_idx, test_idx,
                y_train, y_test (rewards),
                a_train, a_test (actions),
                p_train, p_test (propensities),
                feature_names
            }
        """
        if self.add_interactions:
            df = compute_interaction_features(df)

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed,
            stratify=df[stratify_col] if stratify_col in df.columns else None,
        )

        train_idx = train_df.index.values
        test_idx = test_df.index.values

        # Fit on train only
        self.fit(train_df)
        X_train = self.transform(train_df)
        X_test = self.transform(test_df)

        meta = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "y_train": train_df["reward"].values,
            "y_test": test_df["reward"].values,
            "a_train": train_df["action"].values,
            "a_test": test_df["action"].values,
            "p_train": train_df["propensity"].values,
            "p_test": test_df["propensity"].values,
            "feature_names": self.features,
        }

        # Include counterfactual rewards if available
        reward_cols = [f"reward_{i}" for i in range(N_TREATMENTS)]
        if all(c in train_df.columns for c in reward_cols):
            meta["cf_train"] = train_df[reward_cols].values
            meta["cf_test"] = test_df[reward_cols].values

        # Include optimal actions if available
        if "optimal_action" in train_df.columns:
            meta["opt_train"] = train_df["optimal_action"].values
            meta["opt_test"] = test_df["optimal_action"].values

        logger.info(
            f"Split: train={X_train.shape[0]}, test={X_test.shape[0]}, "
            f"features={X_train.shape[1]}"
        )
        return X_train, X_test, meta

    # ─────────────────────────────────────────────────────────────────────
    # PERSISTENCE — save / load fitted pipeline
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str = "models/feature_pipeline.joblib") -> None:
        """
        Save the fitted pipeline to disk.

        Persists all configuration and fitted state (scaler parameters,
        feature list, continuous indices) so the pipeline can be restored
        without access to the original training data.

        Args:
            path: output file path (.joblib recommended)
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted pipeline. Call fit() first.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        state = {
            "scale": self.scale,
            "add_interactions": self.add_interactions,
            "features": self.features,
            "scaler": self.scaler,
            "_continuous_idx": self._continuous_idx,
            "_fitted": self._fitted,
        }
        joblib.dump(state, path)
        logger.info(f"FeaturePipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """
        Load a previously saved fitted pipeline.

        Returns a fully reconstructed FeaturePipeline instance that is
        ready to call transform() or transform_single() immediately —
        no fitting required.

        Args:
            path: path to the saved .joblib file

        Returns:
            Fitted FeaturePipeline instance
        """
        state = joblib.load(path)

        pipe = cls(
            scale=state["scale"],
            add_interactions=state["add_interactions"],
            features=state["features"],
        )
        pipe.scaler = state["scaler"]
        pipe._continuous_idx = state["_continuous_idx"]
        pipe._fitted = state["_fitted"]

        logger.info(
            f"FeaturePipeline loaded from {path}: "
            f"{len(pipe.features)} features, scale={pipe.scale}"
        )
        return pipe


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(
    csv_path: str = "data/bandit_dataset.csv",
    scale: bool = True,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict, FeaturePipeline]:
    """
    One-liner to load data and get train/test splits.

    Returns:
        X_train, X_test, meta, pipeline
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    pipe = FeaturePipeline(scale=scale, add_interactions=True)
    X_train, X_test, meta = pipe.fit_transform_split(
        df, test_size=test_size, seed=seed
    )
    return X_train, X_test, meta, pipe


def get_unscaled_pipeline() -> FeaturePipeline:
    """Get a pipeline without scaling (for tree-based models like XGBoost)."""
    return FeaturePipeline(scale=False, add_interactions=True)


def get_scaled_pipeline() -> FeaturePipeline:
    """Get a pipeline with scaling (for neural bandits)."""
    return FeaturePipeline(scale=True, add_interactions=True)