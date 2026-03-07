"""
Offline Policy Evaluation for Diabetes Contextual Bandits

Implements:
1. Custom counterfactual evaluation — using ground-truth potential outcomes
2. Policy comparison framework — statistical tests between policies
3. Per-subgroup analysis — evaluate policies on clinical subgroups

All evaluation works OFFLINE using logged data + counterfactuals.
No interaction with the reward oracle needed.

Usage:
    from src.evaluation import OfflinePolicyEvaluator
    evaluator = OfflinePolicyEvaluator(X, actions, rewards, propensities, counterfactuals)
    results = evaluator.evaluate_policy(policy_action_probs)
    comparison = evaluator.compare_policies({"xgb": probs_xgb, "neural": probs_neural})
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from loguru import logger
from scipy import stats
import warnings

from src.data_generator import N_TREATMENTS, IDX_TO_TREATMENT, TREATMENTS

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM OPE ESTIMATORS (always available, no OBP dependency)
# ─────────────────────────────────────────────────────────────────────────────

def inverse_propensity_scoring(
    actions: np.ndarray,
    rewards: np.ndarray,
    pscores: np.ndarray,
    policy_probs: np.ndarray,
) -> Dict[str, float]:
    """
    Inverse Propensity Scoring (IPS).

    V_IPS = (1/n) * sum( reward_t * pi(a_t|x_t) / mu(a_t|x_t) )

    Where pi is the evaluation policy, mu is the logging policy.
    """
    n = len(actions)
    pi_a = policy_probs[np.arange(n), actions]  # eval policy prob for logged action
    weights = pi_a / np.clip(pscores, 1e-6, None)

    value = (rewards * weights).mean()
    # Variance for confidence interval
    weighted_rewards = rewards * weights
    se = weighted_rewards.std() / np.sqrt(n)

    return {
        "value": round(value, 4),
        "std_error": round(se, 4),
        "ci_lower": round(value - 1.96 * se, 4),
        "ci_upper": round(value + 1.96 * se, 4),
        "effective_n": round(1.0 / (weights ** 2).mean(), 1),
    }


def self_normalized_ips(
    actions: np.ndarray,
    rewards: np.ndarray,
    pscores: np.ndarray,
    policy_probs: np.ndarray,
) -> Dict[str, float]:
    """
    Self-Normalized IPS (SNIPS).

    V_SNIPS = sum(w_t * r_t) / sum(w_t)

    Lower variance than IPS, slightly biased.
    """
    n = len(actions)
    pi_a = policy_probs[np.arange(n), actions]
    weights = pi_a / np.clip(pscores, 1e-6, None)

    value = (rewards * weights).sum() / weights.sum()

    # Bootstrap confidence interval
    bootstrap_values = []
    for _ in range(200):
        idx = np.random.choice(n, size=n, replace=True)
        w_b = weights[idx]
        r_b = rewards[idx]
        bootstrap_values.append((r_b * w_b).sum() / w_b.sum())
    se = np.std(bootstrap_values)

    return {
        "value": round(value, 4),
        "std_error": round(se, 4),
        "ci_lower": round(value - 1.96 * se, 4),
        "ci_upper": round(value + 1.96 * se, 4),
    }


def direct_method(
    reward_predictions: np.ndarray,
    policy_probs: np.ndarray,
) -> Dict[str, float]:
    """
    Direct Method (DM).

    V_DM = (1/n) * sum_t( sum_a( pi(a|x_t) * hat{r}(x_t, a) ) )

    Uses a reward model to estimate counterfactual outcomes.
    """
    # reward_predictions: (n, K), policy_probs: (n, K)
    per_patient_value = (policy_probs * reward_predictions).sum(axis=1)
    value = per_patient_value.mean()
    se = per_patient_value.std() / np.sqrt(len(per_patient_value))

    return {
        "value": round(value, 4),
        "std_error": round(se, 4),
        "ci_lower": round(value - 1.96 * se, 4),
        "ci_upper": round(value + 1.96 * se, 4),
    }


def doubly_robust(
    actions: np.ndarray,
    rewards: np.ndarray,
    pscores: np.ndarray,
    policy_probs: np.ndarray,
    reward_predictions: np.ndarray,
) -> Dict[str, float]:
    """
    Doubly Robust (DR) estimator.

    V_DR = V_DM + (1/n) * sum( w_t * (r_t - hat{r}(x_t, a_t)) )

    Combines DM and IPS. Consistent if either the reward model
    or the propensity model is correct.
    """
    n = len(actions)
    pi_a = policy_probs[np.arange(n), actions]
    weights = pi_a / np.clip(pscores, 1e-6, None)

    # DM component
    dm_values = (policy_probs * reward_predictions).sum(axis=1)

    # Correction term
    predicted_for_logged = reward_predictions[np.arange(n), actions]
    correction = weights * (rewards - predicted_for_logged)

    dr_values = dm_values + correction
    value = dr_values.mean()
    se = dr_values.std() / np.sqrt(n)

    return {
        "value": round(value, 4),
        "std_error": round(se, 4),
        "ci_lower": round(value - 1.96 * se, 4),
        "ci_upper": round(value + 1.96 * se, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COUNTERFACTUAL EVALUATION (ground-truth, uses potential outcomes)
# ─────────────────────────────────────────────────────────────────────────────

def counterfactual_policy_value(
    counterfactuals: np.ndarray,
    policy_actions: np.ndarray,
) -> Dict[str, float]:
    """
    Exact policy value using ground-truth counterfactual rewards.

    This is the gold standard — only possible with synthetic data
    where we know all potential outcomes.
    """
    n = len(policy_actions)
    policy_rewards = counterfactuals[np.arange(n), policy_actions]
    oracle_rewards = counterfactuals.max(axis=1)

    value = policy_rewards.mean()
    oracle_value = oracle_rewards.mean()
    regret = oracle_value - value
    accuracy = (policy_actions == counterfactuals.argmax(axis=1)).mean()

    return {
        "policy_value": round(value, 4),
        "oracle_value": round(oracle_value, 4),
        "regret": round(regret, 4),
        "accuracy": round(accuracy, 4),
        "relative_efficiency": round(value / oracle_value, 4) if oracle_value > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE POLICY EVALUATOR (main class)
# ─────────────────────────────────────────────────────────────────────────────

class OfflinePolicyEvaluator:
    """
    Comprehensive offline policy evaluation.

    Brings together custom estimators and counterfactual ground truth.
    """

    def __init__(
        self,
        X: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        pscores: np.ndarray,
        counterfactuals: Optional[np.ndarray] = None,
        reward_predictions: Optional[np.ndarray] = None,
    ):
        """
        Args:
            X: (n, d) context features
            actions: (n,) logged actions
            rewards: (n,) observed rewards
            pscores: (n,) propensity scores from logging policy
            counterfactuals: (n, K) ground-truth rewards for all treatments
            reward_predictions: (n, K) reward model predictions (for DM/DR)
        """
        self.X = X
        self.actions = actions.astype(int)
        self.rewards = rewards
        self.pscores = pscores
        self.counterfactuals = counterfactuals
        self.reward_predictions = reward_predictions
        self.n = len(actions)

        logger.info(
            f"OfflinePolicyEvaluator: n={self.n}, "
            f"counterfactuals={'yes' if counterfactuals is not None else 'no'}, "
            f"reward_model={'yes' if reward_predictions is not None else 'no'}"
        )

    def set_reward_predictions(self, predictions: np.ndarray) -> None:
        """Set or update reward model predictions."""
        self.reward_predictions = predictions

    def evaluate_policy(
        self,
        policy_probs: np.ndarray,
        policy_name: str = "policy",
    ) -> Dict[str, Dict]:
        """
        Full evaluation of a single policy.

        Args:
            policy_probs: (n, K) action probabilities under evaluation policy
            policy_name: name for logging

        Returns:
            Dict with results from each estimator
        """
        results = {}

        # ── Custom estimators ──
        results["ips"] = inverse_propensity_scoring(
            self.actions, self.rewards, self.pscores, policy_probs
        )
        results["snips"] = self_normalized_ips(
            self.actions, self.rewards, self.pscores, policy_probs
        )

        if self.reward_predictions is not None:
            results["dm"] = direct_method(self.reward_predictions, policy_probs)
            results["dr"] = doubly_robust(
                self.actions, self.rewards, self.pscores,
                policy_probs, self.reward_predictions,
            )

        # ── Counterfactual ground truth ──
        if self.counterfactuals is not None:
            greedy_actions = np.argmax(policy_probs, axis=1)
            results["counterfactual"] = counterfactual_policy_value(
                self.counterfactuals, greedy_actions
            )

        logger.info(f"Evaluated '{policy_name}': {len(results)} estimators")
        return results

    def compare_policies(
        self,
        policies: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Compare multiple policies side by side.

        Args:
            policies: dict mapping policy_name → (n, K) action probabilities

        Returns:
            DataFrame with one row per policy, columns for each estimator
        """
        rows = []
        for name, probs in policies.items():
            result = self.evaluate_policy(probs, policy_name=name)
            row = {"policy": name}

            # Flatten nested results
            for estimator, metrics in result.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        row[f"{estimator}_{metric}"] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by counterfactual value if available
        if "counterfactual_policy_value" in df.columns:
            df = df.sort_values("counterfactual_policy_value", ascending=False)

        return df

    def subgroup_analysis(
        self,
        policy_probs: np.ndarray,
        subgroup_labels: np.ndarray,
        subgroup_names: Optional[Dict[int, str]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate policy performance across patient subgroups.

        Args:
            policy_probs: (n, K) action probabilities
            subgroup_labels: (n,) integer subgroup IDs
            subgroup_names: optional mapping of ID → name

        Returns:
            DataFrame with per-subgroup metrics
        """
        if self.counterfactuals is None:
            raise ValueError("Subgroup analysis requires counterfactuals")

        greedy_actions = np.argmax(policy_probs, axis=1)
        unique_groups = np.unique(subgroup_labels)

        rows = []
        for g in unique_groups:
            mask = subgroup_labels == g
            n_g = mask.sum()
            if n_g == 0:
                continue

            cf_g = self.counterfactuals[mask]
            act_g = greedy_actions[mask]

            metrics = counterfactual_policy_value(cf_g, act_g)
            metrics["subgroup"] = subgroup_names.get(g, str(g)) if subgroup_names else str(g)
            metrics["n_patients"] = n_g

            # Action distribution within subgroup
            for k in range(N_TREATMENTS):
                metrics[f"pct_{IDX_TO_TREATMENT[k]}"] = round((act_g == k).mean(), 4)

            rows.append(metrics)

        return pd.DataFrame(rows).sort_values("regret")

    def statistical_test(
        self,
        policy_a_probs: np.ndarray,
        policy_b_probs: np.ndarray,
        n_bootstrap: int = 1000,
    ) -> Dict:
        """
        Statistical comparison between two policies.

        Uses bootstrap + paired t-test on counterfactual values.
        """
        if self.counterfactuals is None:
            raise ValueError("Statistical test requires counterfactuals")

        actions_a = np.argmax(policy_a_probs, axis=1)
        actions_b = np.argmax(policy_b_probs, axis=1)

        values_a = self.counterfactuals[np.arange(self.n), actions_a]
        values_b = self.counterfactuals[np.arange(self.n), actions_b]
        diff = values_a - values_b

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(values_a, values_b)

        # Bootstrap CI for difference
        boot_diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(self.n, size=self.n, replace=True)
            boot_diffs.append(diff[idx].mean())
        boot_diffs = np.array(boot_diffs)

        return {
            "mean_diff": round(diff.mean(), 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant_at_05": p_value < 0.05,
            "ci_lower": round(np.percentile(boot_diffs, 2.5), 4),
            "ci_upper": round(np.percentile(boot_diffs, 97.5), 4),
            "policy_a_value": round(values_a.mean(), 4),
            "policy_b_value": round(values_b.mean(), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: GREEDY PROBS FROM ACTION ARRAY
# ─────────────────────────────────────────────────────────────────────────────

def actions_to_probs(actions: np.ndarray, n_actions: int = N_TREATMENTS) -> np.ndarray:
    """
    Convert deterministic action array to probability matrix.
    Puts probability 1.0 on the chosen action, 0 elsewhere.
    """
    n = len(actions)
    probs = np.zeros((n, n_actions))
    probs[np.arange(n), actions.astype(int)] = 1.0
    return probs


def softmax_probs(
    reward_matrix: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    """
    Convert reward predictions to softmax probabilities.

    Args:
        reward_matrix: (n, K)
        temperature: softmax temperature

    Returns:
        (n, K) probability matrix
    """
    logits = reward_matrix / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)