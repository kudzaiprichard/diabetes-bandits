"""
Phase 4 — drift monitoring for online NeuralThompson deployments.

Running statistics over a rolling window on four streams:
    - context features (mean + std)
    - chosen action distribution
    - observed rewards
    - regret proxy (if oracle counterfactuals are available)

Raises a ``DriftAlert`` when any stream drifts by more than ``k`` std
devs over the rolling window. Alerts are structured so downstream
alerting (Slack, PagerDuty, ...) can consume them directly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
from loguru import logger


@dataclass
class DriftAlert:
    stream: str
    statistic: str
    baseline_mean: float
    baseline_std: float
    window_mean: float
    window_std: float
    z_score: float
    message: str

    def to_dict(self) -> Dict:
        return {
            "stream": self.stream,
            "statistic": self.statistic,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "window_mean": self.window_mean,
            "window_std": self.window_std,
            "z_score": self.z_score,
            "message": self.message,
        }


class DriftMonitor:
    """
    Rolling-window monitor that compares recent statistics to a fixed
    baseline captured from the first ``baseline_size`` observations.
    """

    def __init__(
        self,
        baseline_size: int = 2_000,
        window_size: int = 2_000,
        threshold_z: float = 3.0,
    ):
        self.baseline_size = int(baseline_size)
        self.window_size = int(window_size)
        self.threshold_z = float(threshold_z)

        self._buffers: Dict[str, Deque[float]] = {
            "context": deque(maxlen=window_size),
            "action": deque(maxlen=window_size),
            "reward": deque(maxlen=window_size),
            "regret": deque(maxlen=window_size),
        }
        self._baselines: Dict[str, Optional[Dict[str, float]]] = {
            k: None for k in self._buffers
        }
        self._counts: Dict[str, int] = {k: 0 for k in self._buffers}

    # ── observation ────────────────────────────────────────────────────────
    def observe(
        self,
        context_norm: Optional[float] = None,
        action: Optional[int] = None,
        reward: Optional[float] = None,
        regret: Optional[float] = None,
    ) -> List[DriftAlert]:
        alerts: List[DriftAlert] = []
        for stream, val in (
            ("context", context_norm),
            ("action", float(action) if action is not None else None),
            ("reward", reward),
            ("regret", regret),
        ):
            if val is None:
                continue
            self._buffers[stream].append(float(val))
            self._counts[stream] += 1
            self._maybe_capture_baseline(stream)
            alert = self._check(stream)
            if alert is not None:
                alerts.append(alert)
        return alerts

    # ── internals ──────────────────────────────────────────────────────────
    def _maybe_capture_baseline(self, stream: str) -> None:
        if self._baselines[stream] is None and len(self._buffers[stream]) >= self.baseline_size:
            vals = np.asarray(self._buffers[stream], dtype=float)
            self._baselines[stream] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1) + 1e-12),
            }
            logger.info(
                f"DriftMonitor baseline captured for {stream}: "
                f"mean={self._baselines[stream]['mean']:.3f}, "
                f"std={self._baselines[stream]['std']:.3f}"
            )

    # ── introspection ──────────────────────────────────────────────────────
    def current_z_scores(self) -> Dict[str, float]:
        """
        Current per-stream z-score of the rolling window mean against the
        captured baseline. Returns 0.0 for any stream whose baseline has not
        yet been captured (or whose buffer is empty). Never raises.

        Exposes the z-scores that ``_check`` uses internally, so callers can
        render the drift signal continuously — not only when the threshold is
        breached.
        """
        out: Dict[str, float] = {}
        for stream, buf in self._buffers.items():
            b = self._baselines.get(stream)
            if b is None or len(buf) == 0:
                out[stream] = 0.0
                continue
            try:
                vals = np.asarray(buf, dtype=float)
                m = float(vals.mean())
                denom = float(b["std"]) + 1e-12
                z = (m - float(b["mean"])) / denom
                out[stream] = float(z)
            except Exception:
                out[stream] = 0.0
        return out

    def _check(self, stream: str) -> Optional[DriftAlert]:
        b = self._baselines[stream]
        if b is None:
            return None
        buf = self._buffers[stream]
        if len(buf) < self.window_size:
            return None
        vals = np.asarray(buf, dtype=float)
        m, s = float(vals.mean()), float(vals.std(ddof=1) + 1e-12)
        z = (m - b["mean"]) / b["std"]
        if abs(z) < self.threshold_z:
            return None
        return DriftAlert(
            stream=stream, statistic="mean",
            baseline_mean=b["mean"], baseline_std=b["std"],
            window_mean=m, window_std=s, z_score=z,
            message=(
                f"Drift on {stream}: window mean {m:.3f} is {z:+.2f}σ "
                f"from baseline mean {b['mean']:.3f}"
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Champion / challenger harness
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChampionChallengerResult:
    n_rounds: int
    champion_regret: float
    challenger_regret: float
    improvement: float
    p_value: float
    promoted: bool
    reason: str


def run_champion_challenger(
    champion_agent,
    challenger_agent,
    contexts,                # iterable of feature vectors
    counterfactuals: np.ndarray,  # (n, K) oracle
    traffic_split: float = 0.5,
    significance: float = 0.05,
    seed: int = 0,
) -> ChampionChallengerResult:
    """
    Split-traffic champion/challenger evaluation. Each round routes to one
    agent based on ``traffic_split``; regret is collected against the
    counterfactual oracle. Promotion uses a one-sided Welch t-test on
    per-round regret.
    """
    from scipy import stats
    rng = np.random.default_rng(seed)

    champ_reg: List[float] = []
    chall_reg: List[float] = []

    X = np.asarray(contexts, dtype=np.float32)
    for t in range(X.shape[0]):
        x = X[t]
        optimal = float(counterfactuals[t].max())
        if rng.uniform() < traffic_split:
            a = int(challenger_agent.select_action(x)[0])
            r = float(counterfactuals[t, a])
            challenger_agent.online_update(x, a, r) if hasattr(
                challenger_agent, "online_update"
            ) else None
            chall_reg.append(optimal - r)
        else:
            a = int(champion_agent.select_action(x)[0])
            r = float(counterfactuals[t, a])
            champ_reg.append(optimal - r)

    champ_mean = float(np.mean(champ_reg)) if champ_reg else float("nan")
    chall_mean = float(np.mean(chall_reg)) if chall_reg else float("nan")
    improvement = champ_mean - chall_mean

    if len(champ_reg) > 1 and len(chall_reg) > 1:
        t_stat, p_two = stats.ttest_ind(
            champ_reg, chall_reg, equal_var=False, alternative="greater",
        )
        p_value = float(p_two)
    else:
        p_value = float("nan")

    promoted = bool(improvement > 0 and p_value < significance)
    reason = (
        f"challenger regret {chall_mean:.3f} vs champion {champ_mean:.3f} "
        f"(p={p_value:.3f}); {'PROMOTE' if promoted else 'HOLD'}"
    )
    return ChampionChallengerResult(
        n_rounds=X.shape[0],
        champion_regret=champ_mean,
        challenger_regret=chall_mean,
        improvement=improvement,
        p_value=p_value,
        promoted=promoted,
        reason=reason,
    )
