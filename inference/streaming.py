"""
LearningSession — a bounded, context-managed window of continuous-learning
updates.

NeuralThompson updates are not rollback-able, so the session is a lifecycle
wrapper (flush, checkpoint, metrics) — not a transaction. See design §6.3.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterable, Dict, Iterable, List, Optional, Union

from loguru import logger

from .schemas import LearningAck, LearningRecord

if TYPE_CHECKING:
    from .engine import InferenceEngine


@dataclass
class _SessionMetrics:
    n_updates: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_retrains: int = 0
    n_drift_alerts: int = 0
    started_at: float = field(default_factory=time.monotonic)
    total_latency_s: float = 0.0

    def snapshot(self) -> Dict[str, Any]:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        return {
            "n_updates": self.n_updates,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_retrains": self.n_retrains,
            "n_drift_alerts": self.n_drift_alerts,
            "elapsed_s": elapsed,
            "avg_latency_ms": 1000.0 * self.total_latency_s
            / max(self.n_updates, 1),
            "throughput_per_s": self.n_updates / elapsed,
        }


class LearningSession:
    """
    Synchronous context-managed learning window.

    Use via ``with engine.learning_session() as session``; on exit flushes
    checkpoints (if configured) and emits a metrics summary.
    """

    def __init__(
        self,
        engine: "InferenceEngine",
        *,
        checkpoint_every: Optional[int] = None,
        emit_metrics: bool = True,
    ):
        self.engine = engine
        self.checkpoint_every = checkpoint_every
        self.emit_metrics = emit_metrics
        self.metrics = _SessionMetrics()
        self._closed = False
        self._last_checkpoint_count = 0

    def push(self, record: Union[Dict[str, Any], LearningRecord]) -> LearningAck:
        if self._closed:
            raise RuntimeError("LearningSession is closed")
        t0 = time.monotonic()
        ack = self.engine.update(record)
        self.metrics.total_latency_s += (time.monotonic() - t0)
        self._observe(ack)
        self._maybe_checkpoint()
        return ack

    def push_many(
        self,
        records: Iterable[Union[Dict[str, Any], LearningRecord]],
    ) -> List[LearningAck]:
        return [self.push(r) for r in records]

    def flush(self) -> Dict[str, Any]:
        """Persist a checkpoint (if enabled) and return current metrics."""
        self._maybe_checkpoint(force=True)
        return self.metrics.snapshot()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._maybe_checkpoint(force=True)
        if self.emit_metrics:
            snap = self.metrics.snapshot()
            logger.info(f"LearningSession closed: {snap}")

    def _observe(self, ack: LearningAck) -> None:
        self.metrics.n_updates += 1
        if ack.accepted:
            self.metrics.n_accepted += 1
        else:
            self.metrics.n_rejected += 1
        if ack.backbone_retrained:
            self.metrics.n_retrains += 1
        self.metrics.n_drift_alerts += len(ack.drift_alerts)

    def _maybe_checkpoint(self, force: bool = False) -> None:
        if self.checkpoint_every is None:
            return
        new_updates = self.metrics.n_accepted - self._last_checkpoint_count
        if force and new_updates == 0:
            return
        if not force and new_updates < self.checkpoint_every:
            return
        try:
            self.engine.checkpoint()
            self._last_checkpoint_count = self.metrics.n_accepted
        except Exception as e:
            logger.warning(f"Checkpoint failed: {e}")


class AsyncLearningSession:
    """Async mirror of ``LearningSession`` — offloads updates to a thread."""

    def __init__(
        self,
        engine: "InferenceEngine",
        *,
        checkpoint_every: Optional[int] = None,
        emit_metrics: bool = True,
    ):
        self.engine = engine
        self.checkpoint_every = checkpoint_every
        self.emit_metrics = emit_metrics
        self.metrics = _SessionMetrics()
        self._closed = False
        self._last_checkpoint_count = 0

    async def push(
        self, record: Union[Dict[str, Any], LearningRecord],
    ) -> LearningAck:
        if self._closed:
            raise RuntimeError("AsyncLearningSession is closed")
        t0 = time.monotonic()
        ack = await self.engine.aupdate(record)
        self.metrics.total_latency_s += (time.monotonic() - t0)
        self._observe(ack)
        await self._maybe_checkpoint()
        return ack

    async def push_many(
        self,
        records: Union[
            Iterable[Union[Dict[str, Any], LearningRecord]],
            AsyncIterable[Union[Dict[str, Any], LearningRecord]],
        ],
    ) -> List[LearningAck]:
        acks: List[LearningAck] = []
        if hasattr(records, "__aiter__"):
            async for r in records:  # type: ignore[union-attr]
                acks.append(await self.push(r))
        else:
            for r in records:  # type: ignore[union-attr]
                acks.append(await self.push(r))
        return acks

    async def flush(self) -> Dict[str, Any]:
        await self._maybe_checkpoint(force=True)
        return self.metrics.snapshot()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._maybe_checkpoint(force=True)
        if self.emit_metrics:
            snap = self.metrics.snapshot()
            logger.info(f"AsyncLearningSession closed: {snap}")

    def _observe(self, ack: LearningAck) -> None:
        self.metrics.n_updates += 1
        if ack.accepted:
            self.metrics.n_accepted += 1
        else:
            self.metrics.n_rejected += 1
        if ack.backbone_retrained:
            self.metrics.n_retrains += 1
        self.metrics.n_drift_alerts += len(ack.drift_alerts)

    async def _maybe_checkpoint(self, force: bool = False) -> None:
        if self.checkpoint_every is None:
            return
        new_updates = self.metrics.n_accepted - self._last_checkpoint_count
        if force and new_updates == 0:
            return
        if not force and new_updates < self.checkpoint_every:
            return
        import asyncio
        try:
            await asyncio.to_thread(self.engine.checkpoint)
            self._last_checkpoint_count = self.metrics.n_accepted
        except Exception as e:
            logger.warning(f"Async checkpoint failed: {e}")


__all__ = ["LearningSession", "AsyncLearningSession"]
