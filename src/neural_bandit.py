"""
Neural Contextual Bandits for Diabetes Treatment Selection

Implements:
1. NeuralGreedy     — Deep reward network, greedy action selection
2. NeuralEpsilon    — Deep reward network + epsilon-greedy exploration
3. NeuralUCB        — Upper Confidence Bound via last-layer uncertainty
4. NeuralThompson   — Thompson Sampling via last-layer posterior

All share a common reward network architecture. The exploration
strategies differ in how they use the network's predictions and
uncertainty estimates to select actions.

The reward network predicts E[reward | context] for each treatment
via K output heads (one per treatment).

Usage:
    from src.neural_bandit import NeuralUCB
    bandit = NeuralUCB(input_dim=25, hidden_dims=[128, 64])
    bandit.train(X_train, a_train, r_train, epochs=50)
    actions = bandit.select_actions(X_test)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from loguru import logger
import os
import json
import copy

from src.data_generator import N_TREATMENTS, IDX_TO_TREATMENT


# ─────────────────────────────────────────────────────────────────────────────
# REWARD NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class RewardNetwork(nn.Module):
    """
    Multi-head reward prediction network.

    Architecture:
        Input → [shared hidden layers] → K separate output heads

    The shared layers learn a patient representation. Each head
    specializes in predicting the reward for one treatment.
    The last shared layer's activations are used for uncertainty
    estimation in NeuralUCB and NeuralThompson.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        n_treatments: int = N_TREATMENTS,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.n_treatments = n_treatments

        # Shared backbone
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1]  # last hidden layer dim

        # Per-treatment output heads
        self.heads = nn.ModuleList([
            nn.Linear(self.feature_dim, 1) for _ in range(n_treatments)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, input_dim)

        Returns:
            (batch, K) predicted rewards for each treatment
        """
        features = self.backbone(x)
        outputs = [head(features).squeeze(-1) for head in self.heads]
        return torch.stack(outputs, dim=1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract last hidden layer features (for uncertainty estimation).

        Args:
            x: (batch, input_dim)

        Returns:
            (batch, feature_dim) last-layer representations
        """
        with torch.no_grad():
            return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# BASE NEURAL BANDIT
# ─────────────────────────────────────────────────────────────────────────────

class NeuralBanditBase:
    """
    Base class for neural contextual bandits.

    Handles training, prediction, evaluation. Subclasses implement
    the action selection strategy.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        device: str = "auto",
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = RewardNetwork(
            input_dim, self.hidden_dims, dropout
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        self._train_losses: List[float] = []
        self._val_losses: List[float] = []

        logger.info(
            f"{self.__class__.__name__} initialized: "
            f"input={input_dim}, hidden={self.hidden_dims}, device={self.device}"
        )

    def train(
        self,
        X: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        epochs: int = 50,
        val_fraction: float = 0.1,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the reward network.

        Only updates the head corresponding to the observed action
        for each example (partial feedback).
        """
        n = X.shape[0]
        n_val = int(n * val_fraction)
        indices = np.random.permutation(n)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_t = torch.FloatTensor(X[train_idx]).to(self.device)
        a_t = torch.LongTensor(actions[train_idx]).to(self.device)
        r_t = torch.FloatTensor(rewards[train_idx]).to(self.device)

        X_v = torch.FloatTensor(X[val_idx]).to(self.device)
        a_v = torch.LongTensor(actions[val_idx]).to(self.device)
        r_v = torch.FloatTensor(rewards[val_idx]).to(self.device)

        train_ds = TensorDataset(X_t, a_t, r_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # ── Train ──
            self.network.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, a_batch, r_batch in train_loader:
                self.optimizer.zero_grad()
                pred_all = self.network(X_batch)  # (batch, K)

                # Select predictions for observed actions only
                pred = pred_all[torch.arange(len(a_batch)), a_batch]
                loss = nn.MSELoss()(pred, r_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            self._train_losses.append(train_loss)

            # ── Validate ──
            self.network.eval()
            with torch.no_grad():
                pred_val = self.network(X_v)
                pred_val_obs = pred_val[torch.arange(len(a_v)), a_v]
                val_loss = nn.MSELoss()(pred_val_obs, r_v).item()

            self._val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            # ── Early stopping ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.network.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"  Epoch {epoch + 1:>3}/{epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if patience_counter >= early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            self.network.load_state_dict(best_state)

        logger.info(
            f"Training complete: best_val_loss={best_val_loss:.4f}, "
            f"epochs_run={epoch + 1}"
        )

        return {
            "best_val_loss": best_val_loss,
            "epochs_run": epoch + 1,
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
        }

    def predict_rewards(self, X: np.ndarray) -> np.ndarray:
        """
        Predict expected rewards for all treatments.

        Returns:
            (n, K) reward predictions
        """
        self.network.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            preds = self.network(X_t).cpu().numpy()
        return preds

    def predict_rewards_single(self, x: np.ndarray) -> np.ndarray:
        """Predict rewards for a single context. Returns (K,)."""
        return self.predict_rewards(x.reshape(1, -1)).flatten()

    def select_action(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Select action for a single context.
        Subclasses override this for different exploration strategies.

        Returns:
            (action_index, info_dict_or_scores)
        """
        raise NotImplementedError

    def select_actions(self, X: np.ndarray) -> np.ndarray:
        """Select actions for a batch. Returns (n,) action indices."""
        return np.array([self.select_action(X[i])[0] for i in range(X.shape[0])])

    def evaluate(
        self,
        X: np.ndarray,
        counterfactuals: np.ndarray,
        optimal_actions: Optional[np.ndarray] = None,
    ) -> Dict:
        """Evaluate policy against oracle."""
        actions = self.select_actions(X)
        n = len(X)

        policy_value = counterfactuals[np.arange(n), actions].mean()
        oracle_value = counterfactuals.max(axis=1).mean()

        result = {
            "policy_value": round(policy_value, 4),
            "oracle_value": round(oracle_value, 4),
            "regret": round(oracle_value - policy_value, 4),
        }

        if optimal_actions is not None:
            result["accuracy"] = round((actions == optimal_actions).mean(), 4)

        for k in range(N_TREATMENTS):
            result[f"pct_{IDX_TO_TREATMENT[k]}"] = round((actions == k).mean(), 4)

        return result

    def save(self, path: str = "models/neural_bandit.pt") -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            },
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
        }, path)
        logger.info(f"Saved model to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._train_losses = checkpoint.get("train_losses", [])
        self._val_losses = checkpoint.get("val_losses", [])
        logger.info(f"Loaded model from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL GREEDY
# ─────────────────────────────────────────────────────────────────────────────

class NeuralGreedy(NeuralBanditBase):
    """Pure exploitation: always pick the treatment with highest predicted reward."""

    def select_action(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        rewards = self.predict_rewards_single(x)
        return int(np.argmax(rewards)), rewards


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL EPSILON-GREEDY
# ─────────────────────────────────────────────────────────────────────────────

class NeuralEpsilon(NeuralBanditBase):
    """
    Epsilon-greedy exploration with optional decay.

    With probability epsilon, choose a random action.
    With probability 1-epsilon, choose the greedy action.
    """

    def __init__(
        self,
        input_dim: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        **kwargs,
    ):
        super().__init__(input_dim, **kwargs)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self._step = 0

    def select_action(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        rewards = self.predict_rewards_single(x)
        current_eps = max(self.epsilon_min, self.epsilon * (self.epsilon_decay ** self._step))
        self._step += 1

        if np.random.random() < current_eps:
            action = np.random.randint(N_TREATMENTS)
        else:
            action = int(np.argmax(rewards))

        return action, rewards

    @property
    def current_epsilon(self) -> float:
        return max(self.epsilon_min, self.epsilon * (self.epsilon_decay ** self._step))


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL UCB
# ─────────────────────────────────────────────────────────────────────────────

class NeuralUCB(NeuralBanditBase):
    """
    Neural Upper Confidence Bound.

    Uses the last hidden layer features to estimate uncertainty.
    Maintains a running covariance matrix (like LinUCB) on top of
    the neural network's learned features.

    UCB score = predicted_reward + alpha * uncertainty

    Reference: Zhou et al., "Neural Contextual Bandits with UCB-based Exploration" (2020)
    """

    def __init__(
        self,
        input_dim: int,
        alpha: float = 1.0,
        reg_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(input_dim, **kwargs)
        self.alpha = alpha
        self.reg_lambda = reg_lambda

        # Per-treatment covariance matrices: A_k = lambda * I + sum(phi * phi^T)
        feat_dim = self.network.feature_dim
        self.A = [
            self.reg_lambda * np.eye(feat_dim) for _ in range(N_TREATMENTS)
        ]
        self.A_inv = [
            np.eye(feat_dim) / self.reg_lambda for _ in range(N_TREATMENTS)
        ]

    def update_covariance(self, x: np.ndarray, action: int) -> None:
        """Update covariance matrix for chosen action after observing reward."""
        self.network.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x.reshape(1, -1)).to(self.device)
            phi = self.network.get_features(x_t).cpu().numpy().flatten()

        self.A[action] += np.outer(phi, phi)
        # Sherman-Morrison update for inverse
        phi_col = phi.reshape(-1, 1)
        A_inv = self.A_inv[action]
        numerator = A_inv @ phi_col @ phi_col.T @ A_inv
        denominator = 1.0 + phi_col.T @ A_inv @ phi_col
        self.A_inv[action] = A_inv - numerator / denominator.item()

    def _compute_ucb(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute UCB scores for all treatments."""
        rewards = self.predict_rewards_single(x)

        self.network.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x.reshape(1, -1)).to(self.device)
            phi = self.network.get_features(x_t).cpu().numpy().flatten()

        uncertainties = np.zeros(N_TREATMENTS)
        for k in range(N_TREATMENTS):
            # Uncertainty = phi^T A_inv phi
            uncertainties[k] = np.sqrt(phi @ self.A_inv[k] @ phi)

        ucb_scores = rewards + self.alpha * uncertainties
        return ucb_scores, uncertainties

    def select_action(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        ucb_scores, _ = self._compute_ucb(x)
        return int(np.argmax(ucb_scores)), ucb_scores

    def online_update(self, x: np.ndarray, action: int, reward: float) -> None:
        """Update both network and covariance after observing (x, a, r)."""
        self.update_covariance(x, action)

    def reset_covariance(self) -> None:
        """Reset covariance matrices (e.g., after retraining network)."""
        feat_dim = self.network.feature_dim
        self.A = [self.reg_lambda * np.eye(feat_dim) for _ in range(N_TREATMENTS)]
        self.A_inv = [np.eye(feat_dim) / self.reg_lambda for _ in range(N_TREATMENTS)]

    def save(self, path: str = "models/neural_ucb.pt") -> None:
        """Save model checkpoint including covariance matrices."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "alpha": self.alpha,
                "reg_lambda": self.reg_lambda,
            },
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
            # Covariance state
            "A": [a.copy() for a in self.A],
            "A_inv": [a.copy() for a in self.A_inv],
        }, path)
        logger.info(f"Saved NeuralUCB (with covariance) to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint including covariance matrices."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._train_losses = checkpoint.get("train_losses", [])
        self._val_losses = checkpoint.get("val_losses", [])

        # Restore covariance state
        if "A" in checkpoint and "A_inv" in checkpoint:
            self.A = checkpoint["A"]
            self.A_inv = checkpoint["A_inv"]
            logger.info(f"Loaded NeuralUCB (with covariance) from {path}")
        else:
            logger.warning(
                f"Loaded NeuralUCB from {path} — no covariance found, "
                f"using fresh prior. Rebuild with update_covariance() if needed."
            )


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL THOMPSON SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

class NeuralThompson(NeuralBanditBase):
    """
    Neural Thompson Sampling.

    Maintains a Bayesian linear regression on top of the neural
    network's last-layer features. Samples from the posterior
    to select actions.

    For each treatment k:
        posterior ~ N(mu_k, sigma^2 * A_k^{-1})

    Includes compute_confidence() which draws multiple posterior
    samples to estimate how consistently the model favours each
    treatment — producing a direct percentage confidence score.

    Reference: Zhang et al., "Neural Thompson Sampling" (2021)
    """

    def __init__(
        self,
        input_dim: int,
        reg_lambda: float = 1.0,
        noise_variance: float = 0.25,
        **kwargs,
    ):
        super().__init__(input_dim, **kwargs)
        self.reg_lambda = reg_lambda
        self.noise_variance = noise_variance

        feat_dim = self.network.feature_dim

        # Per-treatment posterior parameters
        self.A = [self.reg_lambda * np.eye(feat_dim) for _ in range(N_TREATMENTS)]
        self.A_inv = [np.eye(feat_dim) / self.reg_lambda for _ in range(N_TREATMENTS)]
        self.b = [np.zeros(feat_dim) for _ in range(N_TREATMENTS)]
        self.mu = [np.zeros(feat_dim) for _ in range(N_TREATMENTS)]

    def update_posterior(self, x: np.ndarray, action: int, reward: float) -> None:
        """Update posterior for the chosen action after observing reward."""
        self.network.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x.reshape(1, -1)).to(self.device)
            phi = self.network.get_features(x_t).cpu().numpy().flatten()

        k = action
        self.A[k] += np.outer(phi, phi)
        self.b[k] += reward * phi

        # Update inverse via Sherman-Morrison
        phi_col = phi.reshape(-1, 1)
        A_inv = self.A_inv[k]
        numerator = A_inv @ phi_col @ phi_col.T @ A_inv
        denominator = 1.0 + phi_col.T @ A_inv @ phi_col
        self.A_inv[k] = A_inv - numerator / denominator.item()

        # Update mean
        self.mu[k] = self.A_inv[k] @ self.b[k]

    def select_action(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """Sample from posterior and pick best action."""
        self.network.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x.reshape(1, -1)).to(self.device)
            phi = self.network.get_features(x_t).cpu().numpy().flatten()

        sampled_rewards = np.zeros(N_TREATMENTS)
        for k in range(N_TREATMENTS):
            # Sample theta_k ~ N(mu_k, noise_var * A_k_inv)
            cov = self.noise_variance * self.A_inv[k]
            # Ensure positive semi-definite
            cov = (cov + cov.T) / 2
            cov += 1e-6 * np.eye(len(cov))
            try:
                theta_k = np.random.multivariate_normal(self.mu[k], cov)
            except np.linalg.LinAlgError:
                theta_k = self.mu[k]
            sampled_rewards[k] = phi @ theta_k

        return int(np.argmax(sampled_rewards)), sampled_rewards

    def compute_confidence(
        self,
        x: np.ndarray,
        n_draws: int = 200,
    ) -> Dict:
        """
        Estimate confidence by drawing from the posterior multiple times
        and counting how often each treatment wins.

        This is the most principled confidence measure — it directly
        reflects the model's own uncertainty about which treatment is best
        for this specific patient.

        Args:
            x: transformed feature vector for a single patient
            n_draws: number of posterior draws (higher = more stable)

        Returns:
            Dict with:
                win_rates: {treatment_name: fraction} — how often each wins
                recommended: treatment with highest win rate
                recommended_win_rate: float — the winning percentage (0-1)
                confidence_pct: int — win rate as a percentage (0-100)
                confidence_label: str — HIGH (85+), MODERATE (60-84), LOW (<60)
                posterior_means: {treatment_name: float} — deterministic estimates
                mean_gap: float — gap between 1st and 2nd posterior means
        """
        self.network.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x.reshape(1, -1)).to(self.device)
            phi = self.network.get_features(x_t).cpu().numpy().flatten()

        # ── Precompute covariance matrices (reused across draws) ──
        covs = []
        for k in range(N_TREATMENTS):
            cov = self.noise_variance * self.A_inv[k]
            cov = (cov + cov.T) / 2
            cov += 1e-6 * np.eye(len(cov))
            covs.append(cov)

        # ── Draw from posterior n_draws times ──
        win_counts = np.zeros(N_TREATMENTS)

        for _ in range(n_draws):
            sampled_rewards = np.zeros(N_TREATMENTS)
            for k in range(N_TREATMENTS):
                try:
                    theta_k = np.random.multivariate_normal(self.mu[k], covs[k])
                except np.linalg.LinAlgError:
                    theta_k = self.mu[k]
                sampled_rewards[k] = phi @ theta_k
            winner = int(np.argmax(sampled_rewards))
            win_counts[winner] += 1

        win_rates = win_counts / n_draws

        # ── Posterior means (deterministic) ──
        posterior_means = np.array([self.mu[k] @ phi for k in range(N_TREATMENTS)])
        sorted_means = np.sort(posterior_means)[::-1]
        mean_gap = sorted_means[0] - sorted_means[1]

        # ── Build result ──
        recommended_idx = int(np.argmax(win_rates))
        recommended = IDX_TO_TREATMENT[recommended_idx]
        recommended_win_rate = win_rates[recommended_idx]
        confidence_pct = int(round(recommended_win_rate * 100))

        if confidence_pct >= 85:
            confidence_label = "HIGH"
        elif confidence_pct >= 60:
            confidence_label = "MODERATE"
        else:
            confidence_label = "LOW"

        return {
            "win_rates": {
                IDX_TO_TREATMENT[k]: round(float(win_rates[k]), 3)
                for k in range(N_TREATMENTS)
            },
            "recommended": recommended,
            "recommended_idx": recommended_idx,
            "recommended_win_rate": round(float(recommended_win_rate), 3),
            "confidence_pct": confidence_pct,
            "confidence_label": confidence_label,
            "posterior_means": {
                IDX_TO_TREATMENT[k]: round(float(posterior_means[k]), 2)
                for k in range(N_TREATMENTS)
            },
            "mean_gap": round(float(mean_gap), 2),
            "n_draws": n_draws,
        }

    def online_update(self, x: np.ndarray, action: int, reward: float) -> None:
        """Update posterior after observing (x, a, r)."""
        self.update_posterior(x, action, reward)

    def reset_posterior(self) -> None:
        """Reset posterior parameters."""
        feat_dim = self.network.feature_dim
        self.A = [self.reg_lambda * np.eye(feat_dim) for _ in range(N_TREATMENTS)]
        self.A_inv = [np.eye(feat_dim) / self.reg_lambda for _ in range(N_TREATMENTS)]
        self.b = [np.zeros(feat_dim) for _ in range(N_TREATMENTS)]
        self.mu = [np.zeros(feat_dim) for _ in range(N_TREATMENTS)]

    def save(self, path: str = "models/neural_thompson.pt") -> None:
        """Save model checkpoint including posterior parameters."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "reg_lambda": self.reg_lambda,
                "noise_variance": self.noise_variance,
            },
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
            # Posterior state
            "A": [a.copy() for a in self.A],
            "A_inv": [a.copy() for a in self.A_inv],
            "b": [b.copy() for b in self.b],
            "mu": [m.copy() for m in self.mu],
        }, path)
        logger.info(f"Saved NeuralThompson (with posterior) to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint including posterior parameters."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._train_losses = checkpoint.get("train_losses", [])
        self._val_losses = checkpoint.get("val_losses", [])

        # Restore posterior state
        if "A" in checkpoint and "mu" in checkpoint:
            self.A = checkpoint["A"]
            self.A_inv = checkpoint["A_inv"]
            self.b = checkpoint["b"]
            self.mu = checkpoint["mu"]
            logger.info(f"Loaded NeuralThompson (with posterior) from {path}")
        else:
            logger.warning(
                f"Loaded NeuralThompson from {path} — no posterior found, "
                f"using fresh prior. Rebuild with update_posterior() if needed."
            )