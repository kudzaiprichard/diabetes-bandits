"""
inference/examples/fastapi_app.py — a complete runnable FastAPI server
wrapping ``InferenceEngine``.

Endpoints:

    GET   /health                 engine readiness + snapshot
    POST  /predict                single prediction (optional ?explain=true)
    POST  /predict/batch          batch prediction
    POST  /learn                  single update
    POST  /learn/stream           SSE stream of LearningAck events
    POST  /learn/stream/rich      SSE stream of LearningStepEvent frames
                                  (body: NDJSON of {patient, oracle_rewards})
    POST  /learn/stream/simulate  Self-contained SSE stream — the endpoint
                                  generates its own patients; body carries
                                  only the simulation parameters.

Run (from the repo root):

    uvicorn inference.examples.fastapi_app:app --reload

Environment:
    BANDITS_MODEL_PATH, BANDITS_PIPELINE_PATH, BANDITS_LLM_*
    (see inference/USAGE.md §6)
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import numpy as np

from inference import (
    ConfigurationError,
    InferenceConfig,
    InferenceEngine,
    LearningRecord,
    ModelError,
    PatientInput,
    ValidationError,
)
from inference._internal.constants import N_TREATMENTS, REWARD_CAP_PP


# @asynccontextmanager
# async def _lifespan(app: FastAPI):
#     """Lazy engine construction so a misconfigured deployment fails at startup."""
#     try:
#         cfg = InferenceConfig.load()
#         app.state.engine = InferenceEngine.from_config(cfg)
#     except ConfigurationError as e:
#         raise RuntimeError(f"Engine startup failed: {e}") from e
#     yield
#     # No teardown needed — engine is stateless aside from the model.

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Lazy engine construction so a misconfigured deployment fails at startup."""
    try:
        cfg = InferenceConfig(
            model_path=r"C:\Users\Administrator\Documents\Projects\Metis\bandits\models\neural_thompson.pt",
            pipeline_path=r"C:\Users\Administrator\Documents\Projects\Metis\bandits\models\feature_pipeline.joblib",
        )
        app.state.engine = InferenceEngine.from_config(cfg)
    except ConfigurationError as e:
        raise RuntimeError(f"Engine startup failed: {e}") from e
    yield
    # No teardown needed — engine is stateless aside from the model.

app = FastAPI(title="Diabetes Bandits Inference", lifespan=_lifespan)


# ─── routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    engine: InferenceEngine = app.state.engine
    return {"ready": engine.ready, **engine.snapshot()}


@app.post("/predict")
async def predict(patient: PatientInput, explain: bool = False):
    engine: InferenceEngine = app.state.engine
    try:
        result = await engine.apredict(patient.model_dump(), explain=explain)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except ModelError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result.model_dump()


@app.post("/predict/batch")
async def predict_batch(patients: List[PatientInput], explain: bool = False):
    engine: InferenceEngine = app.state.engine
    results = await asyncio.to_thread(
        engine.predict_batch, [p.model_dump() for p in patients], explain,
    )
    return {"results": [r.model_dump() for r in results]}


@app.post("/learn")
async def learn(record: LearningRecord):
    engine: InferenceEngine = app.state.engine
    ack = await engine.aupdate(record.model_dump())
    return ack.model_dump()


@app.post("/learn/stream")
async def learn_stream(request: Request):
    """
    SSE stream of LearningAck events — consume a newline-delimited JSON body.
    Each line is one ``LearningRecord``; each event yields one ack.
    """
    engine: InferenceEngine = app.state.engine

    async def _events():
        async with engine.alearning_session(emit_metrics=True) as session:
            async for raw in _ndjson_stream(request):
                try:
                    ack = await session.push(raw)
                except Exception as e:  # final safety net — SSE must not 500
                    yield _sse({
                        "accepted": False,
                        "validation_errors": [{"msg": str(e)}],
                    })
                    continue
                yield _sse(ack.model_dump(mode="json"))
            snap = await session.flush()
            yield _sse({"event": "summary", **snap})

    return StreamingResponse(_events(), media_type="text/event-stream")


@app.post("/learn/stream/rich")
async def learn_stream_rich(request: Request, total_steps: int | None = None):
    """
    SSE stream of ``LearningStepEvent`` frames — richer than ``/learn/stream``.

    Request body: newline-delimited JSON lines, each one of shape::

        {"patient": {...PatientInput fields...},
         "oracle_rewards": [r0, r1, r2, r3, r4]}

    ``oracle_rewards`` is a length-5 vector in the canonical treatment order
    (Metformin, GLP-1, SGLT-2, DPP-4, Insulin). Each line drives exactly one
    Thompson-sampling step on the engine; the emitted event carries the full
    per-step state the UI needs to render learning in real time.

    A closing ``{"event":"summary", ...}`` frame is emitted on stream end.
    """
    engine: InferenceEngine = app.state.engine

    async def _events():
        async with engine.alearning_stream(total_steps=total_steps) as stream:
            async for raw in _ndjson_stream(request):
                try:
                    patient = raw["patient"]
                    oracle = np.asarray(raw["oracle_rewards"], dtype=float)
                    if oracle.shape != (N_TREATMENTS,):
                        raise ValueError(
                            f"oracle_rewards must have length {N_TREATMENTS}"
                        )
                    event = await stream.astep(patient, oracle)
                except Exception as e:
                    yield _sse({
                        "event": "error",
                        "message": str(e),
                    })
                    continue
                yield event.to_sse()
            yield _sse({"event": "summary", **stream.snapshot()})

    return StreamingResponse(_events(), media_type="text/event-stream")


class SimulateParams(BaseModel):
    """Parameters for ``POST /learn/stream/simulate``."""

    n_steps: int = Field(default=500, ge=1, le=100_000)
    shift_at: Optional[int] = Field(default=None, ge=0)
    bmi_shift: float = Field(default=6.0, ge=-30.0, le=30.0)
    seed: int = 0


@app.post("/learn/stream/simulate")
async def learn_stream_simulate(params: SimulateParams = SimulateParams()):
    """
    Self-contained SSE stream of ``LearningStepEvent`` frames.

    The endpoint synthesises its own patients (inline generator, zero src/
    dependency) and drives :meth:`InferenceEngine.alearning_stream`. No
    request body is required beyond the simulation parameters. Useful for
    demos, load tests, and end-to-end dashboards that want a live feed
    without having to produce patient data themselves.

    Closes with a ``{"event":"summary", ...}`` frame.
    """
    engine: InferenceEngine = app.state.engine

    async def _events():
        rng = np.random.default_rng(params.seed)
        async with engine.alearning_stream(total_steps=params.n_steps) as stream:
            for i in range(params.n_steps):
                ctx = _simulate_patient(rng, patient_id=f"SIM-{i:06d}")
                if params.shift_at is not None and i >= params.shift_at:
                    ctx["bmi"] = float(
                        min(ctx["bmi"] + params.bmi_shift, 78.0)
                    )
                oracle = _oracle_rewards(ctx, rng)
                try:
                    event = await stream.astep(ctx, oracle)
                except Exception as e:
                    yield _sse({"event": "error", "step": i + 1, "message": str(e)})
                    continue
                yield event.to_sse()
            yield _sse({"event": "summary", **stream.snapshot()})

    return StreamingResponse(_events(), media_type="text/event-stream")


# ─── helpers ─────────────────────────────────────────────────────────────────

async def _ndjson_stream(request: Request):
    """Yield one dict per newline-delimited JSON line in the request body."""
    buf = ""
    async for chunk in request.stream():
        buf += chunk.decode("utf-8")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    if buf.strip():
        try:
            yield json.loads(buf)
        except json.JSONDecodeError:
            pass


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, default=str)}\n\n".encode("utf-8")


# ─── inline patient generator + toy oracle ────────────────────────────────────
# Kept deliberately small — this file must have zero src/ dependency.

def _simulate_patient(
    rng: np.random.Generator, *, patient_id: Optional[str] = None,
) -> dict:
    """Sample one plausible patient that passes ``PatientInput`` validation."""
    ctx = {
        "age": int(np.clip(rng.normal(60.0, 11.0), 30, 85)),
        "bmi": float(np.clip(rng.normal(32.0, 5.0), 18.0, 72.0)),
        "hba1c_baseline": float(np.clip(rng.normal(8.2, 1.2), 5.0, 14.0)),
        "egfr": float(np.clip(rng.normal(78.0, 22.0), 15.0, 140.0)),
        "diabetes_duration": float(np.clip(rng.exponential(5.5), 0.0, 40.0)),
        "fasting_glucose": float(np.clip(rng.normal(155.0, 32.0), 70.0, 400.0)),
        "c_peptide": float(np.clip(rng.normal(2.1, 0.8), 0.2, 6.0)),
        "bp_systolic": float(np.clip(rng.normal(135.0, 15.0), 90.0, 200.0)),
        "ldl": float(np.clip(rng.normal(112.0, 26.0), 50.0, 250.0)),
        "hdl": float(np.clip(rng.normal(45.0, 10.0), 20.0, 100.0)),
        "triglycerides": float(np.clip(rng.normal(185.0, 60.0), 50.0, 800.0)),
        "alt": float(np.clip(rng.normal(32.0, 12.0), 5.0, 200.0)),
        "cvd": int(rng.random() < 0.25),
        "ckd": int(rng.random() < 0.15),
        "nafld": int(rng.random() < 0.35),
        "hypertension": int(rng.random() < 0.55),
    }
    if patient_id is not None:
        ctx["patient_id"] = patient_id
    return ctx


def _oracle_rewards(ctx: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Toy oracle producing a length-5 per-arm reward vector in canonical
    treatment order (Metformin, GLP-1, SGLT-2, DPP-4, Insulin). Encodes
    just enough signal for the model to learn something without recreating
    the real ``reward_oracle`` from src/.
    """
    bmi = float(ctx["bmi"])
    egfr = float(ctx["egfr"])
    hba1c = float(ctx["hba1c_baseline"])
    cvd = int(ctx["cvd"])
    ckd = int(ctx["ckd"])

    base = np.array([
        1.0 + (-1.2 if egfr < 30 else 0.0),                          # Metformin
        1.2 + (0.6 if bmi >= 35.0 else 0.0),                         # GLP-1
        1.1 + (0.4 if cvd else 0.0) + (-0.7 if egfr < 30 else 0.0),  # SGLT-2
        0.75,                                                        # DPP-4
        0.9 + (0.6 if hba1c >= 9.0 else 0.0) + (0.3 if ckd else 0.0),# Insulin
    ], dtype=float)
    noise = rng.normal(0.0, 0.15, size=N_TREATMENTS)
    return np.clip(base + noise, 0.0, float(REWARD_CAP_PP))
