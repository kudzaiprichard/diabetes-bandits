"""
XGBoost Reward Model for Diabetes Contextual Bandits

Approaches:
1. Separate model per treatment — trains K independent XGBoost regressors
2. Single model with treatment as feature — one model, action one-hot encoded

Both predict E[reward | context, action] which is the core of the
Direct Method (DM) for offline policy evaluation and can serve as
the reward estimator for bandit policies.

Usage:
    from src.reward_model import RewardModelEnsemble
    model = RewardModelEnsemble()
    model.fit(X_train, a_train, y_train)
    predicted_rewards = model.predict_all(X_test)  # (n, 5)
    best_actions = model.predict_best_action(X_test)  # (n,)
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from typing import Dict, Optional, Tuple, List
from loguru import logger
import os
import json

from src.data_generator import N_TREATMENTS, IDX_TO_TREATMENT, TREATMENTS


# ─────────────────────────────────────────────────────────────────────────────
# PER-TREATMENT ENSEMBLE (recommended approach)
# ─────────────────────────────────────────────────────────────────────────────

class RewardModelEnsemble:
    """
    K separate XGBoost regressors, one per treatment.

    For each treatment k, we train on the subset of data where action==k,
    predicting the observed reward. At inference, all K models score the
    same context to produce the full reward vector.
    """

    def __init__(self, xgb_params: Optional[Dict] = None):
        self.xgb_params = xgb_params or {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }
        self.models: Dict[int, xgb.XGBRegressor] = {}
        self._fitted = False
        self._feature_names: Optional[List[str]] = None

    def fit(
        self,
        X: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        feature_names: Optional[List[str]] = None,
        eval_X: Optional[np.ndarray] = None,
        eval_actions: Optional[np.ndarray] = None,
        eval_rewards: Optional[np.ndarray] = None,
    ) -> "RewardModelEnsemble":
        """
        Train one XGBoost model per treatment.

        Args:
            X: (n, d) feature matrix
            actions: (n,) action indices
            rewards: (n,) observed rewards
            feature_names: optional feature name list
            eval_*: optional validation set for early stopping
        """
        self._feature_names = feature_names

        for k in range(N_TREATMENTS):
            mask = actions == k
            X_k = X[mask]
            y_k = rewards[mask]

            if len(X_k) == 0:
                logger.warning(f"No samples for treatment {k} ({IDX_TO_TREATMENT[k]})")
                continue

            model = xgb.XGBRegressor(**self.xgb_params)

            fit_params = {}
            if eval_X is not None and eval_actions is not None and eval_rewards is not None:
                eval_mask = eval_actions == k
                if eval_mask.sum() > 0:
                    fit_params["eval_set"] = [(eval_X[eval_mask], eval_rewards[eval_mask])]
                    fit_params["verbose"] = False

            model.fit(X_k, y_k, **fit_params)
            self.models[k] = model

            logger.info(
                f"  {IDX_TO_TREATMENT[k]:<12} "
                f"n_train={len(X_k):>5}  "
                f"mean_reward={y_k.mean():.3f}"
            )

        self._fitted = True
        logger.info(f"RewardModelEnsemble fitted: {len(self.models)} models")
        return self

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """
        Predict expected reward for all treatments.

        Args:
            X: (n, d) feature matrix

        Returns:
            (n, K) matrix of predicted rewards
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        n = X.shape[0]
        predictions = np.zeros((n, N_TREATMENTS))

        for k, model in self.models.items():
            predictions[:, k] = model.predict(X)

        return predictions

    def predict_best_action(self, X: np.ndarray) -> np.ndarray:
        """Predict the best treatment for each patient."""
        rewards = self.predict_all(X)
        return np.argmax(rewards, axis=1)

    def predict_single(self, x: np.ndarray) -> np.ndarray:
        """Predict rewards for a single patient context vector."""
        return self.predict_all(x.reshape(1, -1)).flatten()

    def evaluate(
        self,
        X: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        counterfactuals: Optional[np.ndarray] = None,
        optimal_actions: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Evaluate reward model quality.

        Returns dict with:
            - per_treatment_rmse: RMSE for each treatment's observed data
            - overall_rmse: pooled RMSE on observed (action, reward) pairs
            - policy_accuracy: % of time greedy policy matches oracle optimal
            - policy_value: average counterfactual reward of greedy policy
            - oracle_value: average reward of oracle policy
        """
        pred_all = self.predict_all(X)
        n = len(X)

        # Per-treatment RMSE on observed data
        per_treatment_rmse = {}
        residuals_all = []
        for k in range(N_TREATMENTS):
            mask = actions == k
            if mask.sum() == 0:
                continue
            pred_k = pred_all[mask, k]
            true_k = rewards[mask]
            rmse = np.sqrt(np.mean((pred_k - true_k) ** 2))
            per_treatment_rmse[IDX_TO_TREATMENT[k]] = round(rmse, 4)
            residuals_all.extend((pred_k - true_k).tolist())

        overall_rmse = np.sqrt(np.mean(np.array(residuals_all) ** 2))

        result = {
            "per_treatment_rmse": per_treatment_rmse,
            "overall_rmse": round(overall_rmse, 4),
        }

        # Policy evaluation against oracle
        greedy_actions = np.argmax(pred_all, axis=1)

        if optimal_actions is not None:
            accuracy = (greedy_actions == optimal_actions).mean()
            result["policy_accuracy"] = round(accuracy, 4)

        if counterfactuals is not None:
            policy_value = counterfactuals[np.arange(n), greedy_actions].mean()
            oracle_value = counterfactuals.max(axis=1).mean()
            result["policy_value"] = round(policy_value, 4)
            result["oracle_value"] = round(oracle_value, 4)
            result["regret"] = round(oracle_value - policy_value, 4)

        return result

    def feature_importance(self, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Get top feature importances per treatment model."""
        importances = {}
        for k, model in self.models.items():
            scores = model.feature_importances_
            if self._feature_names:
                pairs = list(zip(self._feature_names, scores))
            else:
                pairs = [(f"f{i}", s) for i, s in enumerate(scores)]
            pairs.sort(key=lambda x: x[1], reverse=True)
            importances[IDX_TO_TREATMENT[k]] = pairs[:top_k]
        return importances

    def save(self, directory: str = "models/reward_model") -> None:
        """Save all models to directory."""
        os.makedirs(directory, exist_ok=True)
        for k, model in self.models.items():
            path = os.path.join(directory, f"xgb_treatment_{k}.json")
            model.save_model(path)

        meta = {
            "xgb_params": self.xgb_params,
            "feature_names": self._feature_names,
            "n_models": len(self.models),
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved {len(self.models)} models to {directory}/")

    def load(self, directory: str = "models/reward_model") -> "RewardModelEnsemble":
        """Load models from directory."""
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)

        self._feature_names = meta.get("feature_names")
        self.xgb_params = meta.get("xgb_params", self.xgb_params)

        for k in range(N_TREATMENTS):
            path = os.path.join(directory, f"xgb_treatment_{k}.json")
            if os.path.exists(path):
                model = xgb.XGBRegressor()
                model.load_model(path)
                self.models[k] = model

        self._fitted = True
        logger.info(f"Loaded {len(self.models)} models from {directory}/")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODEL APPROACH (alternative)
# ─────────────────────────────────────────────────────────────────────────────

class RewardModelSingle:
    """
    Single XGBoost model with treatment one-hot encoded as features.

    Simpler but may underperform the ensemble since treatment-specific
    feature interactions are harder to learn in one tree.
    """

    def __init__(self, xgb_params: Optional[Dict] = None):
        self.xgb_params = xgb_params or {
            "n_estimators": 800,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }
        self.model: Optional[xgb.XGBRegressor] = None
        self._fitted = False
        self._n_base_features: int = 0

    def _augment(self, X: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Append one-hot treatment encoding to feature matrix."""
        n = X.shape[0]
        ohe = np.zeros((n, N_TREATMENTS))
        ohe[np.arange(n), actions.astype(int)] = 1.0
        return np.hstack([X, ohe])

    def _augment_all(self, X: np.ndarray) -> np.ndarray:
        """Create copies of X for each treatment (for prediction)."""
        n = X.shape[0]
        X_all = np.tile(X, (N_TREATMENTS, 1))
        actions_all = np.repeat(np.arange(N_TREATMENTS), n)
        return self._augment(X_all, actions_all)

    def fit(self, X: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> "RewardModelSingle":
        self._n_base_features = X.shape[1]
        X_aug = self._augment(X, actions)
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_aug, rewards)
        self._fitted = True
        logger.info(f"RewardModelSingle fitted: {X_aug.shape[1]} features")
        return self

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Predict rewards for all treatments. Returns (n, K)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        n = X.shape[0]
        predictions = np.zeros((n, N_TREATMENTS))

        for k in range(N_TREATMENTS):
            ohe = np.zeros((n, N_TREATMENTS))
            ohe[:, k] = 1.0
            X_aug = np.hstack([X, ohe])
            predictions[:, k] = self.model.predict(X_aug)

        return predictions

    def predict_best_action(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_all(X), axis=1)