"""
Rich per-step event for continuous-learning streaming.

``LearningStepEvent`` exposes one event's worth of internal state during
continuous learning — the Thompson samples that caused the decision, the
post-update posterior, running aggregates, backbone health, drift
signals, and enough patient context to render each step in a UI.
Serialises cleanly to JSON, WebSocket frames, or Server-Sent-Events.

``LearningStream`` is the session-style helper that drives Thompson
sampling, observes reward from caller-supplied per-arm oracle rewards,
applies the online update, and emits one ``LearningStepEvent`` per
step. It keeps the per-stream running aggregates the event needs
(cumulative reward/regret, running accuracy, per-arm pull counts).

This module is additive: the existing ``update`` / ``LearningAck`` /
``LearningSession`` surface is untouched.
"""
from __future__ import annotations

import asyncio
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Union,
)

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from ._internal.constants import IDX_TO_TREATMENT, N_TREATMENTS
from ._internal.explainability import (
    SEVERITY_CONTRAINDICATION,
    SEVERITY_WARNING,
    collect_findings,
)
from .schemas import PatientInput

if TYPE_CHECKING:
    from .engine import InferenceEngine


Phase = Literal["Early", "Mid", "Late"]
SafetyStatus = Literal["CLEAR", "WARNING", "CONTRAINDICATION_FOUND"]
ConfidenceLabel = Literal["HIGH", "MODERATE", "LOW"]

_DRIFT_STREAMS: Tuple[str, ...] = ("context", "action", "reward")


# ─────────────────────────────────────────────────────────────────────────────
# EVENT
# ─────────────────────────────────────────────────────────────────────────────


class LearningStepEvent(BaseModel):
    """
    Rich per-step event emitted by :class:`LearningStream`.

    All per-arm dicts are keyed by treatment name (``Metformin``, ``GLP-1``,
    ``SGLT-2``, ``DPP-4``, ``Insulin``). All scalar fields are plain JSON
    primitives so the event serialises cleanly for SSE, WebSocket, console,
    or file sinks.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    # ── core decision ───────────────────────────────────────────────────
    step: int
    selectedIdx: int
    posteriorMeanArgmax: int
    explored: bool
    thompsonSamples: Dict[str, float]
    observedReward: float
    oracleOptimalIdx: int
    oracleOptimalReward: float
    regret: float

    # ── posterior state (post-update) ──────────────────────────────────
    posteriorMeans: Dict[str, float]
    posteriorUncertainty: Dict[str, float]
    winRates: Dict[str, float]
    confidencePct: int
    confidenceLabel: ConfidenceLabel
    meanGap: float
    nUpdatesPerArm: Dict[str, int]

    # ── running aggregates ─────────────────────────────────────────────
    cumulativeReward: float
    cumulativeRegret: float
    runningAccuracy: float
    runningMeanRewardPerArm: Dict[str, float]
    bestTreatmentIdx: int
    phase: Phase

    # ── backbone health ────────────────────────────────────────────────
    retrainFired: bool
    noiseVariance: float
    replayBufferSize: int

    # ── drift signals ──────────────────────────────────────────────────
    contextNorm: float
    driftAlerts: List[Dict[str, Any]] = Field(default_factory=list)
    driftStreams: Dict[str, float]

    # ── patient context ────────────────────────────────────────────────
    patientId: Optional[str] = None
    patientFeatures: Dict[str, Any]
    safetyStatus: SafetyStatus

    # ── serialisation helpers ──────────────────────────────────────────

    def to_sse(self) -> bytes:
        """One Server-Sent-Events ``data:`` frame (already terminated)."""
        return f"data: {self.model_dump_json()}\n\n".encode("utf-8")

    def to_ws(self) -> str:
        """One WebSocket text frame (plain JSON string)."""
        return self.model_dump_json()

    def to_console_line(self) -> str:
        """Compact one-line human-readable summary."""
        tag = "EXPLORE" if self.explored else "exploit"
        correct = "OK" if self.selectedIdx == self.oracleOptimalIdx else "--"
        treatment = IDX_TO_TREATMENT[self.selectedIdx]
        return (
            f"[{self.phase:<5s}] step {self.step:>4d} | "
            f"{tag:<7s} -> {treatment:<9s} {correct} | "
            f"r={self.observedReward:+6.3f} reg={self.regret:+6.3f} "
            f"cumReg={self.cumulativeRegret:+7.2f} acc={self.runningAccuracy:6.2%} "
            f"conf={self.confidencePct:>3d}% ({self.confidenceLabel:<8s}) "
            f"safety={self.safetyStatus}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STREAM
# ─────────────────────────────────────────────────────────────────────────────


OracleRewards = Union[np.ndarray, Iterable[float]]
_PatientArg = Union[Dict[str, Any], PatientInput]


class LearningStream:
    """
    Active Thompson-sampling learning stream.

    The stream Thompson-samples the current per-arm posterior, looks up the
    reward for the chosen arm from caller-supplied oracle rewards (an ndarray
    of shape ``(K,)`` or any iterable convertible to one), applies the
    online update on the engine's ``NeuralThompson``, and emits a
    :class:`LearningStepEvent` whose ``thompsonSamples`` are causally
    correct — drawn from the same ``phi``, ``mu``, ``A_inv`` and
    ``noise_variance`` that produced the real decision.

    Running aggregates (cumulative reward/regret, running accuracy, per-arm
    pull counts) are per-stream. The engine's global ``_n_updates`` still
    advances.

    Usage::

        with engine.learning_stream(total_steps=1000) as stream:
            for patient, oracle_rewards in source:
                event = stream.step(patient, oracle_rewards)
                yield event.to_sse()          # or .to_ws() or .to_console_line()

    Oracle rewards must be ordered by the same treatment index as
    ``inference.Treatment`` (Metformin=0, GLP-1=1, SGLT-2=2, DPP-4=3,
    Insulin=4).
    """

    def __init__(
        self,
        engine: "InferenceEngine",
        *,
        total_steps: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.engine = engine
        self.total_steps = total_steps
        self._rng = rng if rng is not None else np.random.default_rng()
        self._step = 0
        self._cum_reward = 0.0
        self._cum_regret = 0.0
        self._n_correct = 0
        self._sum_reward_per_arm = np.zeros(N_TREATMENTS, dtype=float)
        self._count_per_arm = np.zeros(N_TREATMENTS, dtype=int)
        self._n_updates_per_arm = np.zeros(N_TREATMENTS, dtype=int)
        self._closed = False

    # ── public API ─────────────────────────────────────────────────────

    def step(
        self,
        patient: _PatientArg,
        oracle_rewards: OracleRewards,
    ) -> LearningStepEvent:
        """Thompson-sample, observe, update, emit — blocking."""
        if self._closed:
            raise RuntimeError("LearningStream is closed")
        return self._run_step(patient, oracle_rewards)

    async def astep(
        self,
        patient: _PatientArg,
        oracle_rewards: OracleRewards,
    ) -> LearningStepEvent:
        """Async mirror of :meth:`step` — offloads to a worker thread."""
        return await asyncio.to_thread(self.step, patient, oracle_rewards)

    def stream_events(
        self,
        source: Iterable[Tuple[_PatientArg, OracleRewards]],
    ) -> Iterator[LearningStepEvent]:
        """Iterate ``(patient, oracle_rewards)`` tuples, yielding events."""
        for patient, oracle in source:
            yield self.step(patient, oracle)

    async def astream_events(
        self,
        source: Union[
            Iterable[Tuple[_PatientArg, OracleRewards]],
            AsyncIterable[Tuple[_PatientArg, OracleRewards]],
        ],
    ) -> AsyncIterator[LearningStepEvent]:
        """Async iterate ``(patient, oracle_rewards)`` tuples."""
        if hasattr(source, "__aiter__"):
            async for patient, oracle in source:  # type: ignore[union-attr]
                yield await self.astep(patient, oracle)
        else:
            for patient, oracle in source:  # type: ignore[union-attr]
                yield await self.astep(patient, oracle)

    def snapshot(self) -> Dict[str, Any]:
        """Running-aggregates snapshot without advancing the stream."""
        return {
            "steps": int(self._step),
            "cumulative_reward": float(self._cum_reward),
            "cumulative_regret": float(self._cum_regret),
            "running_accuracy": (
                float(self._n_correct / self._step) if self._step > 0 else 0.0
            ),
            "per_arm_pulls": {
                IDX_TO_TREATMENT[k]: int(self._count_per_arm[k])
                for k in range(N_TREATMENTS)
            },
        }

    def close(self) -> None:
        self._closed = True

    # ── dual-mode context management ───────────────────────────────────

    def __enter__(self) -> "LearningStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> "LearningStream":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── internals ──────────────────────────────────────────────────────

    def _run_step(
        self,
        patient: _PatientArg,
        oracle_rewards: OracleRewards,
    ) -> LearningStepEvent:
        engine = self.engine
        model = engine.model

        pi = engine._validate_patient(patient, context="patient")
        x = engine._transform(pi)

        if isinstance(oracle_rewards, np.ndarray):
            oracle_arr = np.asarray(oracle_rewards, dtype=float).flatten()
        else:
            oracle_arr = np.asarray(list(oracle_rewards), dtype=float).flatten()
        if oracle_arr.shape != (N_TREATMENTS,):
            raise ValueError(
                f"oracle_rewards must be shape ({N_TREATMENTS},), "
                f"got {oracle_arr.shape}"
            )

        with engine._lock:
            # ── pre-update feature extraction + Thompson draw ──────────
            model.network.eval()
            x_t = torch.from_numpy(x.astype(np.float32)).reshape(1, -1).to(
                model.device
            )
            with torch.no_grad():
                phi_pre = (
                    model.network.get_features(x_t).cpu().numpy().flatten()
                )

            means_pre = np.array(
                [model.mu[k] @ phi_pre for k in range(N_TREATMENTS)]
            )
            noise_var = float(model.noise_variance)
            thompson_var = np.array(
                [
                    max(noise_var * float(phi_pre @ model.A_inv[k] @ phi_pre),
                        1e-12)
                    for k in range(N_TREATMENTS)
                ]
            )
            thompson_std = np.sqrt(thompson_var)
            z = self._rng.standard_normal(N_TREATMENTS)
            thompson = means_pre + z * thompson_std
            selected_idx = int(np.argmax(thompson))
            posterior_mean_argmax = int(np.argmax(means_pre))
            explored = selected_idx != posterior_mean_argmax

            # ── observe & update ──────────────────────────────────────
            oracle_optimal_idx = int(np.argmax(oracle_arr))
            oracle_optimal_reward = float(oracle_arr[oracle_optimal_idx])
            observed_reward = float(oracle_arr[selected_idx])
            regret = float(oracle_optimal_reward - observed_reward)

            before_step = int(getattr(model, "_online_step", 0))
            model.online_update(x, selected_idx, observed_reward)
            after_step = int(getattr(model, "_online_step", 0))
            retrain_fired = bool(
                engine.config.online_retraining
                and engine.config.retrain_every > 0
                and after_step > 0
                and after_step != before_step
                and after_step % engine.config.retrain_every == 0
            )
            engine._n_updates += 1

            # ── post-update state ─────────────────────────────────────
            if retrain_fired:
                model.network.eval()
                with torch.no_grad():
                    phi_post = (
                        model.network.get_features(x_t).cpu().numpy().flatten()
                    )
            else:
                phi_post = phi_pre

            means_post = np.array(
                [model.mu[k] @ phi_post for k in range(N_TREATMENTS)]
            )
            uncertainty_post = np.array(
                [
                    max(float(phi_post @ model.A_inv[k] @ phi_post), 1e-12)
                    for k in range(N_TREATMENTS)
                ]
            )
            best_treatment_idx = int(np.argmax(means_post))

            confidence = model.compute_confidence(
                x, n_draws=engine.config.n_confidence_draws,
            )

            # ── drift ─────────────────────────────────────────────────
            context_norm = float(np.linalg.norm(x))
            drift_alerts: List[Dict[str, Any]] = []
            drift_streams: Dict[str, float] = {s: 0.0 for s in _DRIFT_STREAMS}
            if engine._drift is not None:
                alerts = engine._drift.observe(
                    context_norm=context_norm,
                    action=selected_idx,
                    reward=observed_reward,
                )
                drift_alerts = [a.to_dict() for a in alerts]
                try:
                    zs = engine._drift.current_z_scores()
                except Exception:
                    zs = {}
                drift_streams = {
                    s: float(zs.get(s, 0.0)) for s in _DRIFT_STREAMS
                }

            # ── safety status for the selected arm ────────────────────
            selected_treatment = IDX_TO_TREATMENT[selected_idx]
            findings = collect_findings(pi.context_dict()).get(
                selected_treatment, []
            )
            if any(f.severity == SEVERITY_CONTRAINDICATION for f in findings):
                safety_status: SafetyStatus = "CONTRAINDICATION_FOUND"
            elif any(f.severity == SEVERITY_WARNING for f in findings):
                safety_status = "WARNING"
            else:
                safety_status = "CLEAR"

            # ── running aggregates ────────────────────────────────────
            self._step += 1
            self._cum_reward += observed_reward
            self._cum_regret += regret
            self._sum_reward_per_arm[selected_idx] += observed_reward
            self._count_per_arm[selected_idx] += 1
            self._n_updates_per_arm[selected_idx] += 1
            if selected_idx == oracle_optimal_idx:
                self._n_correct += 1

            running_accuracy = (
                self._n_correct / self._step if self._step > 0 else 0.0
            )
            running_mean_per_arm = {
                IDX_TO_TREATMENT[k]: (
                    float(self._sum_reward_per_arm[k] / self._count_per_arm[k])
                    if self._count_per_arm[k] > 0 else 0.0
                )
                for k in range(N_TREATMENTS)
            }

            replay_size = (
                len(model.replay_buffer)
                if getattr(model, "replay_buffer", None) is not None
                else 0
            )

            names = [IDX_TO_TREATMENT[k] for k in range(N_TREATMENTS)]
            event = LearningStepEvent(
                step=int(self._step),
                selectedIdx=selected_idx,
                posteriorMeanArgmax=posterior_mean_argmax,
                explored=explored,
                thompsonSamples={
                    names[k]: float(thompson[k]) for k in range(N_TREATMENTS)
                },
                observedReward=observed_reward,
                oracleOptimalIdx=oracle_optimal_idx,
                oracleOptimalReward=oracle_optimal_reward,
                regret=regret,
                posteriorMeans={
                    names[k]: float(means_post[k]) for k in range(N_TREATMENTS)
                },
                posteriorUncertainty={
                    names[k]: float(uncertainty_post[k])
                    for k in range(N_TREATMENTS)
                },
                winRates=dict(confidence["win_rates"]),
                confidencePct=int(confidence["confidence_pct"]),
                confidenceLabel=confidence["confidence_label"],
                meanGap=float(confidence["mean_gap"]),
                nUpdatesPerArm={
                    names[k]: int(self._n_updates_per_arm[k])
                    for k in range(N_TREATMENTS)
                },
                cumulativeReward=float(self._cum_reward),
                cumulativeRegret=float(self._cum_regret),
                runningAccuracy=float(running_accuracy),
                runningMeanRewardPerArm=running_mean_per_arm,
                bestTreatmentIdx=best_treatment_idx,
                phase=self._compute_phase(),
                retrainFired=retrain_fired,
                noiseVariance=float(model.noise_variance),
                replayBufferSize=int(replay_size),
                contextNorm=context_norm,
                driftAlerts=drift_alerts,
                driftStreams=drift_streams,
                patientId=pi.patient_id,
                patientFeatures=pi.feature_dict(),
                safetyStatus=safety_status,
            )

        return event

    def _compute_phase(self) -> Phase:
        if self.total_steps is None or self.total_steps <= 0:
            return "Early"
        third = self.total_steps / 3.0
        if self._step <= third:
            return "Early"
        if self._step <= 2 * third:
            return "Mid"
        return "Late"


__all__ = ["LearningStepEvent", "LearningStream", "Phase"]
