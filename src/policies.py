"""
Bandit Policy Classes for Diabetes Treatment Selection

Standalone policy wrappers that can use ANY reward estimator
(XGBoost, Neural, VW) underneath. These are thin strategy layers
that take predicted rewards and add exploration.

Policies:
- GreedyPolicy:         pure exploitation
- EpsilonGreedyPolicy:  epsilon-greedy with optional decay
- BoltzmannPolicy:      softmax / temperature-based exploration
- UCBPolicy:            Upper Confidence Bound (requires uncertainty)
- ThompsonPolicy:       Thompson Sampling (requires posterior)
- LinUCBPolicy:         Linear UCB on raw features (no neural network)
- RandomPolicy:         uniform random baseline

All policies implement:
    select_action(reward_estimates, **kwargs) → action_index
    select_action_batch(reward_matrix, **kwargs) → action_indices

Usage:
    from src.policies import EpsilonGreedyPolicy
    policy = EpsilonGreedyPolicy(epsilon=0.1, decay=0.999)
    action = policy.select_action(predicted_rewards)
"""

import numpy as np
from typing import Optional, Dict, Tuple
from loguru import logger

from src.data_generator import N_TREATMENTS, IDX_TO_TREATMENT


# ─────────────────────────────────────────────────────────────────────────────
# BASE POLICY
# ─────────────────────────────────────────────────────────────────────────────

class BasePolicy:
    """Abstract base for all policies."""

    def __init__(self, name: str = "base"):
        self.name = name
        self._step = 0
        self._action_counts = np.zeros(N_TREATMENTS, dtype=int)

    def select_action(self, reward_estimates: np.ndarray, **kwargs) -> int:
        """
        Select a single action given reward estimates.

        Args:
            reward_estimates: (K,) estimated reward for each treatment

        Returns:
            action index
        """
        raise NotImplementedError

    def select_action_batch(self, reward_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        Select actions for a batch.

        Args:
            reward_matrix: (n, K) estimated rewards

        Returns:
            (n,) action indices
        """
        return np.array([
            self.select_action(reward_matrix[i], **kwargs)
            for i in range(reward_matrix.shape[0])
        ])

    def update(self, action: int, reward: float) -> None:
        """Update internal state after observing (action, reward)."""
        self._step += 1
        self._action_counts[action] += 1

    def reset(self) -> None:
        """Reset policy state."""
        self._step = 0
        self._action_counts = np.zeros(N_TREATMENTS, dtype=int)

    @property
    def action_distribution(self) -> Dict[str, float]:
        """Current empirical action distribution."""
        total = max(self._action_counts.sum(), 1)
        return {
            IDX_TO_TREATMENT[k]: round(self._action_counts[k] / total, 4)
            for k in range(N_TREATMENTS)
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM POLICY (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class RandomPolicy(BasePolicy):
    """Uniform random action selection. Baseline for comparison."""

    def __init__(self):
        super().__init__(name="random")

    def select_action(self, reward_estimates: np.ndarray, **kwargs) -> int:
        action = np.random.randint(N_TREATMENTS)
        self.update(action, 0)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY POLICY
# ─────────────────────────────────────────────────────────────────────────────

class GreedyPolicy(BasePolicy):
    """Always pick the action with highest estimated reward."""

    def __init__(self):
        super().__init__(name="greedy")

    def select_action(self, reward_estimates: np.ndarray, **kwargs) -> int:
        action = int(np.argmax(reward_estimates))
        self.update(action, 0)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# EPSILON-GREEDY POLICY
# ─────────────────────────────────────────────────────────────────────────────

class EpsilonGreedyPolicy(BasePolicy):
    """
    Epsilon-greedy with optional decay.

    epsilon_t = max(epsilon_min, epsilon * decay^t)
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        decay: float = 1.0,
        epsilon_min: float = 0.01,
    ):
        super().__init__(name=f"eps_greedy(e={epsilon})")
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min

    @property
    def current_epsilon(self) -> float:
        return max(self.epsilon_min, self.epsilon * (self.decay ** self._step))

    def select_action(self, reward_estimates: np.ndarray, **kwargs) -> int:
        eps = self.current_epsilon
        if np.random.random() < eps:
            action = np.random.randint(N_TREATMENTS)
        else:
            action = int(np.argmax(reward_estimates))
        self.update(action, 0)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# BOLTZMANN / SOFTMAX POLICY
# ─────────────────────────────────────────────────────────────────────────────

class BoltzmannPolicy(BasePolicy):
    """
    Softmax exploration with temperature.

    P(a) = exp(reward_a / tau) / sum(exp(reward_k / tau))

    High tau → more uniform (exploration)
    Low tau → more greedy (exploitation)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        temperature_decay: float = 1.0,
        temperature_min: float = 0.1,
    ):
        super().__init__(name=f"boltzmann(tau={temperature})")
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min

    @property
    def current_temperature(self) -> float:
        return max(
            self.temperature_min,
            self.temperature * (self.temperature_decay ** self._step),
        )

    def select_action(self, reward_estimates: np.ndarray, **kwargs) -> int:
        tau = self.current_temperature
        # Numerically stable softmax
        logits = reward_estimates / tau
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        action = np.random.choice(N_TREATMENTS, p=probs)
        self.update(action, 0)
        return int(action)

    def get_probs(self, reward_estimates: np.ndarray) -> np.ndarray:
        """Get action probabilities without sampling."""
        tau = self.current_temperature
        logits = reward_estimates / tau
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()


# ─────────────────────────────────────────────────────────────────────────────
# UCB POLICY
# ─────────────────────────────────────────────────────────────────────────────

class UCBPolicy(BasePolicy):
    """
    Upper Confidence Bound policy.

    Requires uncertainty estimates alongside reward predictions.
    UCB score = reward + alpha * uncertainty

    Can be used with any model that provides uncertainty
    (neural UCB, bootstrap, etc).
    """

    def __init__(self, alpha: float = 1.0, alpha_decay: float = 1.0):
        super().__init__(name=f"ucb(alpha={alpha})")
        self.alpha = alpha
        self.alpha_decay = alpha_decay

    @property
    def current_alpha(self) -> float:
        return self.alpha * (self.alpha_decay ** self._step)

    def select_action(
        self,
        reward_estimates: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        **kwargs,
    ) -> int:
        if uncertainties is None:
            # Fallback to count-based UCB
            total = max(self._step, 1)
            uncertainties = np.sqrt(2 * np.log(total) / (self._action_counts + 1))

        ucb_scores = reward_estimates + self.current_alpha * uncertainties
        action = int(np.argmax(ucb_scores))
        self.update(action, 0)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# THOMPSON SAMPLING POLICY
# ─────────────────────────────────────────────────────────────────────────────

class ThompsonPolicy(BasePolicy):
    """
    Thompson Sampling with Gaussian posterior.

    Maintains per-treatment running mean and variance estimates.
    Samples from posterior and picks the highest sample.

    For use when you don't have a neural network posterior —
    just tracks observed rewards per treatment.
    """

    def __init__(self, prior_mean: float = 3.0, prior_var: float = 2.0):
        super().__init__(name="thompson")
        self.prior_mean = prior_mean
        self.prior_var = prior_var

        # Per-treatment sufficient statistics
        self._sum_rewards = np.zeros(N_TREATMENTS)
        self._sum_sq_rewards = np.zeros(N_TREATMENTS)
        self._counts = np.zeros(N_TREATMENTS)

    def update(self, action: int, reward: float) -> None:
        super().update(action, reward)
        self._sum_rewards[action] += reward
        self._sum_sq_rewards[action] += reward ** 2
        self._counts[action] += 1

    def select_action(self, reward_estimates: np.ndarray, **kwargs) -> int:
        """
        Ignores reward_estimates, uses own posterior instead.
        Pass reward_estimates=np.zeros(K) if calling standalone.
        """
        samples = np.zeros(N_TREATMENTS)
        for k in range(N_TREATMENTS):
            if self._counts[k] < 2:
                # Prior
                samples[k] = np.random.normal(self.prior_mean, np.sqrt(self.prior_var))
            else:
                # Posterior
                n = self._counts[k]
                mean = self._sum_rewards[k] / n
                var = (self._sum_sq_rewards[k] / n - mean ** 2) / n
                var = max(var, 1e-4)
                samples[k] = np.random.normal(mean, np.sqrt(var))

        action = int(np.argmax(samples))
        # Note: don't call super().update here — caller should call update() with actual reward
        self._step += 1
        self._action_counts[action] += 1
        return action

    def reset(self) -> None:
        super().reset()
        self._sum_rewards = np.zeros(N_TREATMENTS)
        self._sum_sq_rewards = np.zeros(N_TREATMENTS)
        self._counts = np.zeros(N_TREATMENTS)


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR UCB (feature-based, no neural network)
# ─────────────────────────────────────────────────────────────────────────────

class LinUCBPolicy(BasePolicy):
    """
    Linear UCB (disjoint model).

    Per-treatment linear model with UCB exploration.
    Directly operates on feature vectors — no neural network needed.

    A_k = I + sum(x_t x_t^T)  for action k
    b_k = sum(r_t * x_t)      for action k
    theta_k = A_k^{-1} b_k
    UCB_k = theta_k^T x + alpha * sqrt(x^T A_k^{-1} x)
    """

    def __init__(self, feature_dim: int, alpha: float = 1.0):
        super().__init__(name=f"linucb(alpha={alpha})")
        self.feature_dim = feature_dim
        self.alpha = alpha

        self.A = [np.eye(feature_dim) for _ in range(N_TREATMENTS)]
        self.A_inv = [np.eye(feature_dim) for _ in range(N_TREATMENTS)]
        self.b = [np.zeros(feature_dim) for _ in range(N_TREATMENTS)]
        self.theta = [np.zeros(feature_dim) for _ in range(N_TREATMENTS)]

    def select_action(
        self,
        reward_estimates: np.ndarray,
        x: Optional[np.ndarray] = None,
        **kwargs,
    ) -> int:
        """
        Args:
            reward_estimates: ignored (LinUCB uses own estimates)
            x: (d,) feature vector (REQUIRED)
        """
        if x is None:
            raise ValueError("LinUCBPolicy requires feature vector x")

        ucb_scores = np.zeros(N_TREATMENTS)
        for k in range(N_TREATMENTS):
            pred = self.theta[k] @ x
            uncertainty = np.sqrt(x @ self.A_inv[k] @ x)
            ucb_scores[k] = pred + self.alpha * uncertainty

        action = int(np.argmax(ucb_scores))
        self._step += 1
        self._action_counts[action] += 1
        return action

    def update_model(self, x: np.ndarray, action: int, reward: float) -> None:
        """Update linear model for chosen action."""
        k = action
        self.A[k] += np.outer(x, x)

        # Sherman-Morrison for A_inv
        x_col = x.reshape(-1, 1)
        A_inv = self.A_inv[k]
        numerator = A_inv @ x_col @ x_col.T @ A_inv
        denominator = 1.0 + x_col.T @ A_inv @ x_col
        self.A_inv[k] = A_inv - numerator / denominator.item()

        self.b[k] += reward * x
        self.theta[k] = self.A_inv[k] @ self.b[k]

    def reset(self) -> None:
        super().reset()
        d = self.feature_dim
        self.A = [np.eye(d) for _ in range(N_TREATMENTS)]
        self.A_inv = [np.eye(d) for _ in range(N_TREATMENTS)]
        self.b = [np.zeros(d) for _ in range(N_TREATMENTS)]
        self.theta = [np.zeros(d) for _ in range(N_TREATMENTS)]

    def pretrain(
        self,
        X: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        """
        G-19: batched pre-training on logged data.

        Replays (x, a, r) rows through the Sherman-Morrison update so that
        the online comparison against NeuralThompson starts from the same
        information state. Without this, LinUCB begins every online
        experiment at a cold prior and is unfairly handicapped.
        """
        n = X.shape[0]
        for i in range(n):
            self.update_model(X[i], int(actions[i]), float(rewards[i]))
        logger.info(
            f"LinUCB pretrained on {n} rows "
            f"(posteriors for {N_TREATMENTS} arms bootstrapped)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# POLICY FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_policy(name: str, **kwargs) -> BasePolicy:
    """Factory function to create policies by name."""
    policies = {
        "random": RandomPolicy,
        "greedy": GreedyPolicy,
        "epsilon_greedy": EpsilonGreedyPolicy,
        "boltzmann": BoltzmannPolicy,
        "ucb": UCBPolicy,
        "thompson": ThompsonPolicy,
        "linucb": LinUCBPolicy,
    }
    if name not in policies:
        raise ValueError(f"Unknown policy: {name}. Available: {list(policies.keys())}")
    return policies[name](**kwargs)