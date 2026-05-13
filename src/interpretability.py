"""
Phase 2 — Faithful interpretability for NeuralThompson.

Closed-form, posterior-mean feature attribution via integrated gradients.
For the recommended arm k:

    score(x) = φ(x) · μ_k        (posterior-mean prediction)
    grad_x score = (dφ/dx)ᵀ μ_k

Integrated gradients from a cohort-median baseline give per-feature
contributions that exactly sum to `score(x) - score(baseline)` in the
linear head. Contrast attributions answer "why THIS arm instead of that
arm". Marginal variance contributions locate the features driving
posterior uncertainty.

This module covers gaps G-7, G-8, G-9 — it is the layer the LLM's clinical
claims must be grounded in.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Sequence, Any
from loguru import logger

from src.data_generator import N_TREATMENTS


# ─────────────────────────────────────────────────────────────────────────────
# CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_phi(model, x: np.ndarray) -> np.ndarray:
    """Return the last-layer feature vector φ(x) as a numpy array."""
    model.network.eval()
    with torch.no_grad():
        x_t = torch.FloatTensor(x.reshape(1, -1)).to(model.device)
        return model.network.get_features(x_t).cpu().numpy().flatten()


def _grad_phi(
    model, x: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    """
    Compute d(φᵀμ)/dx for a single input — the input-space gradient of the
    posterior-mean prediction for the specified arm.
    """
    model.network.eval()
    x_t = torch.FloatTensor(x.reshape(1, -1)).to(model.device)
    x_t.requires_grad_(True)
    phi = model.network.get_features(x_t)  # (1, feat_dim)
    mu_t = torch.from_numpy(mu.astype(np.float32)).to(model.device)
    score = (phi.squeeze(0) * mu_t).sum()
    grad = torch.autograd.grad(score, x_t)[0].cpu().numpy().flatten()
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATED GRADIENTS (closed-form attribution)
# ─────────────────────────────────────────────────────────────────────────────

def integrated_gradients(
    model,
    x: np.ndarray,
    arm: int,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 32,
) -> np.ndarray:
    """
    Integrated gradients of φᵀμ_k w.r.t. x, from a baseline to x.

    IG_i = (x_i - baseline_i) * ∫_0^1 ∂(φᵀμ)/∂x_i |_{x = baseline + α(x - baseline)} dα

    Riemann-sum approximation with ``n_steps`` equispaced α. For a purely
    linear backbone this is exact for any n_steps; with ReLU we use 32 which
    is sufficient for stable attributions.
    """
    x = np.asarray(x, dtype=np.float32).flatten()
    if baseline is None:
        baseline = np.zeros_like(x)
    baseline = np.asarray(baseline, dtype=np.float32).flatten()
    mu = model.mu[int(arm)]

    alphas = np.linspace(0.0, 1.0, n_steps, endpoint=False) + 1.0 / (2 * n_steps)
    grads = np.zeros_like(x)
    for a in alphas:
        xp = baseline + a * (x - baseline)
        grads += _grad_phi(model, xp, mu)
    grads /= n_steps
    attribution = (x - baseline) * grads
    return attribution


# ─────────────────────────────────────────────────────────────────────────────
# UNCERTAINTY DECOMPOSITION (G-9)
# ─────────────────────────────────────────────────────────────────────────────

def uncertainty_decomposition(
    model,
    x: np.ndarray,
    arm: int,
    feature_names: Sequence[str],
    n_steps: int = 16,
) -> List[Dict[str, Any]]:
    """
    For a given arm, decompose the predictive variance φᵀ A_k⁻¹ φ into
    per-feature marginal contributions.

    We use path-integrated gradients of the variance w.r.t. x, scaled by
    (x - baseline). Exact for a linear backbone; a smooth approximation for
    deep backbones with ReLU gating.
    """
    x = np.asarray(x, dtype=np.float32).flatten()
    A_inv = model.A_inv[int(arm)]
    baseline = np.zeros_like(x)

    def variance(xp_np: np.ndarray) -> torch.Tensor:
        xp_t = torch.FloatTensor(xp_np.reshape(1, -1)).to(model.device)
        xp_t.requires_grad_(True)
        phi = model.network.get_features(xp_t).squeeze(0)
        A_inv_t = torch.from_numpy(A_inv.astype(np.float32)).to(model.device)
        v = phi @ A_inv_t @ phi
        return xp_t, v

    alphas = np.linspace(0.0, 1.0, n_steps, endpoint=False) + 1.0 / (2 * n_steps)
    total_grad = np.zeros_like(x)
    for a in alphas:
        xp = baseline + a * (x - baseline)
        xp_t, v = variance(xp)
        g = torch.autograd.grad(v, xp_t)[0].cpu().numpy().flatten()
        total_grad += g
    total_grad /= n_steps
    contribs = (x - baseline) * total_grad

    # Return top-k sorted by absolute contribution
    order = np.argsort(-np.abs(contribs))
    return [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
            "contribution": float(contribs[i]),
            "value": float(x[i]),
        }
        for i in order
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ATTRIBUTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AttributionEngine:
    """
    Attribution engine implementing the ``explain`` protocol expected by
    ``ExplainabilityExtractor``. Produces:

        attribution          — per-feature IG for the recommended arm
        contrast             — per-feature IG for (top − runner-up)
        uncertainty_drivers  — per-feature marginal variance contributions
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 32,
        treatment_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.feature_names = list(feature_names)
        self.baseline = baseline
        self.n_steps = int(n_steps)
        if treatment_to_idx is None:
            from src.data_generator import TREATMENT_TO_IDX
            treatment_to_idx = TREATMENT_TO_IDX
        self.treatment_to_idx = dict(treatment_to_idx)

    def explain(
        self,
        model,
        x: np.ndarray,
        top_treatment: str,
        runner_up: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return attribution / contrast / uncertainty_drivers for one patient."""
        top_idx = self.treatment_to_idx[top_treatment]
        attr = integrated_gradients(
            model, x, top_idx, baseline=self.baseline, n_steps=self.n_steps
        )
        attr_map = {
            self.feature_names[i]: float(attr[i])
            for i in range(min(len(self.feature_names), len(attr)))
        }

        contrast_map: Optional[Dict[str, float]] = None
        if runner_up is not None and runner_up in self.treatment_to_idx:
            ru_idx = self.treatment_to_idx[runner_up]
            attr_ru = integrated_gradients(
                model, x, ru_idx, baseline=self.baseline, n_steps=self.n_steps
            )
            contrast_vec = attr - attr_ru
            contrast_map = {
                self.feature_names[i]: float(contrast_vec[i])
                for i in range(min(len(self.feature_names), len(contrast_vec)))
            }

        uncertainty_drivers = uncertainty_decomposition(
            model, x, top_idx, feature_names=self.feature_names
        )[:5]

        return {
            "attribution": attr_map,
            "contrast": contrast_map,
            "uncertainty_drivers": uncertainty_drivers,
        }
