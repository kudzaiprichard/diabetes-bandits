"""LearningSession lifecycle: flush, metrics, checkpointing, async mirror."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from inference import (
    InferenceConfig,
    InferenceEngine,
    LearningSession,
)

from tests.inference_fixtures import build_tiny_artefacts, sample_patient


@pytest.fixture(scope="module")
def _artefacts(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("streaming_artefacts")
    model_path, pipeline_path, *_ = build_tiny_artefacts(tmp)
    return {"model_path": model_path, "pipeline_path": pipeline_path, "tmp": tmp}


@pytest.fixture
def engine(_artefacts):
    cfg = InferenceConfig(
        model_path=_artefacts["model_path"],
        pipeline_path=_artefacts["pipeline_path"],
        n_confidence_draws=16,
        online_retraining=True,
        replay_buffer_size=200,
        retrain_every=5,
        min_buffer_for_retrain=5,
        minibatch_size=8,
        drift_enabled=False,
        device="cpu",
        checkpoint_dir=_artefacts["tmp"],
    )
    return InferenceEngine.from_config(cfg)


def _make_records(n: int, rng: np.random.RandomState):
    return [
        {"patient": sample_patient(rng), "action": i % 5, "reward": 1.0}
        for i in range(n)
    ]


def test_learning_session_collects_metrics(engine):
    rng = np.random.RandomState(21)
    with engine.learning_session(emit_metrics=False) as session:
        for rec in _make_records(6, rng):
            session.push(rec)
        snap = session.flush()
    assert snap["n_updates"] == 6
    assert snap["n_accepted"] == 6
    assert snap["n_rejected"] == 0
    assert snap["avg_latency_ms"] >= 0


def test_learning_session_counts_rejections(engine):
    rng = np.random.RandomState(22)
    with engine.learning_session(emit_metrics=False) as session:
        session.push({"patient": sample_patient(rng), "action": 0, "reward": 1.0})
        session.push({"patient": sample_patient(rng), "action": 0, "reward": 99.0})
    assert session.metrics.n_accepted == 1
    assert session.metrics.n_rejected == 1


def test_learning_session_checkpoints_on_exit(engine, _artefacts):
    rng = np.random.RandomState(23)
    before = {p.name for p in Path(_artefacts["tmp"]).glob("*.pt")}
    with engine.learning_session(
        checkpoint_every=5, emit_metrics=False,
    ) as session:
        for rec in _make_records(5, rng):
            session.push(rec)
    after = {p.name for p in Path(_artefacts["tmp"]).glob("*.pt")}
    assert after - before, "expected at least one new checkpoint file"


def test_learning_session_raises_after_close(engine):
    rng = np.random.RandomState(24)
    sess = LearningSession(engine, emit_metrics=False)
    sess.push({"patient": sample_patient(rng), "action": 0, "reward": 1.0})
    sess.close()
    with pytest.raises(RuntimeError):
        sess.push({"patient": sample_patient(rng), "action": 0, "reward": 1.0})


# ── async ────────────────────────────────────────────────────────────────────

def test_async_learning_session():
    async def _go(engine):
        rng = np.random.RandomState(25)
        async with engine.alearning_session(emit_metrics=False) as session:
            for rec in _make_records(4, rng):
                await session.push(rec)
            snap = await session.flush()
        return snap

    # Build engine inside the test (async fixture overhead not needed)
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    mp, pp, *_ = build_tiny_artefacts(tmp)
    cfg = InferenceConfig(
        model_path=mp, pipeline_path=pp,
        n_confidence_draws=16, online_retraining=False,
        drift_enabled=False, device="cpu",
    )
    engine = InferenceEngine.from_config(cfg)
    snap = asyncio.run(_go(engine))
    assert snap["n_updates"] == 4
    assert snap["n_accepted"] == 4
