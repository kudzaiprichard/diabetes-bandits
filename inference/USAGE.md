# Inference Module — Usage Guide

The `inference/` package is a single façade for prediction and continuous learning on the six-phase diabetes-bandits stack. Callers never touch `FeaturePipeline`, `NeuralThompson`, `ExplainabilityExtractor`, or `LLMExplainer` directly.

---

## 1. Install & setup

### 1.1 Environment

Assumes the Conda env from `environment.yml` is active:

```bash
conda env create -f environment.yml
conda activate diabetes-bandits
```

LLM explanations require `google-genai` (optional — the rest of the engine works without it):

```bash
pip install google-genai
```

### 1.2 Required artefacts

The engine loads a fitted `FeaturePipeline` and a trained `NeuralThompson`:

```
models/
  feature_pipeline.joblib
  neural_thompson.pt
```

To regenerate them from scratch:

```bash
python -m src.data_generator      # writes data/bandit_dataset.csv
python -m src.cli train            # trains + saves the neural bandit + pipeline
```

---

## 2. Single prediction

```python
from inference import InferenceEngine, InferenceConfig

engine = InferenceEngine.from_config(InferenceConfig.load())

patient = {
    "age": 62, "bmi": 34.2, "hba1c_baseline": 8.9, "egfr": 85.0,
    "diabetes_duration": 6.0, "fasting_glucose": 180.0, "c_peptide": 1.4,
    "bp_systolic": 140.0, "ldl": 120.0, "hdl": 45.0,
    "triglycerides": 200.0, "alt": 30.0,
    "cvd": 1, "ckd": 0, "nafld": 1, "hypertension": 1,
    "patient_id": "PID-0001",
}

result = engine.predict(patient)
print(result.recommended, result.confidence_pct, result.safety_status)
```

### 2.1 With explanation (stub — no API key required)

```python
cfg = InferenceConfig.load(llm_enabled=True, llm_provider="stub")
engine = InferenceEngine.from_config(cfg)
result = engine.predict(patient, explain=True)
print(result.explanation["recommendation_summary"])
```

### 2.2 With real Gemini explanation

Set `GEMINI_API_KEY` (or `BANDITS_LLM_API_KEY`) and choose the Gemini provider:

```python
cfg = InferenceConfig.load(llm_enabled=True, llm_provider="gemini")
engine = InferenceEngine.from_config(cfg)
result = engine.predict(patient, explain=True)
```

`explain=True` soft-fails (returns `explanation=None`) if the LLM errors out. Pass `explain="require"` to propagate the exception.

### 2.3 Batch prediction

```python
import pandas as pd
df = pd.DataFrame([patient, patient, patient])
results = engine.predict_batch(df)
for r in results:
    if r.accepted:
        print(r.recommended)
    else:
        print("rejected:", r.validation_errors)
```

Rows that fail schema validation do **not** stop the batch — they return sentinel `PredictionResult(accepted=False, validation_errors=...)`.

---

## 3. Continuous learning

### 3.1 One record at a time

```python
record = {
    "patient": patient,
    "action": 1,              # or: "treatment": "GLP-1"
    "reward": 1.2,
    "observed_at": "2026-04-15T10:00:00Z",
    "source": "ehr:site-3",
}
ack = engine.update(record)
assert ack.accepted
print(ack.n_updates_so_far, ack.backbone_retrained, ack.drift_alerts)
```

### 3.2 Batch or streaming

```python
acks = list(engine.update_many(records))
# or ingest from CSV
acks = list(engine.ingest_csv("data/new_outcomes.csv"))
```

### 3.3 Session-scoped window

```python
with engine.learning_session(checkpoint_every=1000) as session:
    for rec in live_feed():
        ack = session.push(rec)
        if ack.drift_alerts:
            notify_ops(ack.drift_alerts)
    print(session.flush())        # metrics snapshot
```

On exit: the session flushes any pending checkpoint and logs an aggregate metrics summary (`n_updates`, `n_retrains`, `n_drift_alerts`, `avg_latency_ms`, `throughput_per_s`).

### 3.4 Delayed-feedback pattern

Outcomes often arrive hours or days after the prediction. Join the raw outcome stream to your prior predictions by `patient_id`:

```python
pending = {}  # patient_id -> prediction context

def on_prediction(ctx):
    r = engine.predict(ctx)
    pending[ctx["patient_id"]] = (ctx, r.recommended_idx)
    return r

def on_outcome(patient_id, reward):
    ctx, action_idx = pending.pop(patient_id)
    engine.update({"patient": ctx, "action": action_idx, "reward": reward})
```

---

## 4. Async / FastAPI

```python
from fastapi import FastAPI, HTTPException
from inference import (
    InferenceEngine, InferenceConfig,
    PatientInput, LearningRecord, ValidationError,
)

app = FastAPI()
engine = InferenceEngine.from_config(InferenceConfig.load())


@app.post("/predict")
async def predict(patient: PatientInput, explain: bool = False):
    try:
        return await engine.apredict(patient.model_dump(), explain=explain)
    except ValidationError as e:
        raise HTTPException(422, detail=e.errors())


@app.post("/learn")
async def learn(record: LearningRecord):
    return await engine.aupdate(record.model_dump())


@app.get("/health")
def health():
    return engine.snapshot()
```

A complete runnable example with SSE streaming and lifespan-managed startup is in `inference/examples/fastapi_app.py`.

### 4.1 Why `asyncio.to_thread`?

Every async method offloads sync work to `asyncio.to_thread`. Sherman-Morrison updates are fast (microseconds), but the G-4 periodic backbone retrain can take 1–3 seconds — isolating it keeps the event loop free.

---

## 5. CLI / scripting

A shell demo is in `inference/examples/cli_example.sh`. The gist:

```bash
export BANDITS_MODEL_PATH=models/neural_thompson.pt
export BANDITS_PIPELINE_PATH=models/feature_pipeline.joblib
export BANDITS_LLM_ENABLED=true
export BANDITS_LLM_PROVIDER=stub

python - <<'PY'
from inference import InferenceEngine
engine = InferenceEngine.from_env()
print(engine.predict({...}).model_dump_json(indent=2))
PY
```

---

## 6. Configuration reference

All fields of `InferenceConfig`. Env vars use `BANDITS_` + upper-cased name (e.g. `BANDITS_LLM_ENABLED`).

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `model_path` | Path | `models/neural_thompson.pt` | NeuralThompson `.pt` checkpoint |
| `pipeline_path` | Path | `models/feature_pipeline.joblib` | Saved `FeaturePipeline` |
| `data_dir` | Path | `data/` | Location for CSVs, logs |
| `n_confidence_draws` | int | 200 | Posterior samples per prediction (G-6) |
| `attribution_enabled` | bool | True | Enable IG attribution (G-7..9) |
| `llm_enabled` | bool | False | Required for `explain=True` |
| `llm_provider` | `"gemini"\|"stub"\|"none"` | `"none"` | `stub` = deterministic test client |
| `llm_api_key` | SecretStr\|None | None | Falls back to `GEMINI_API_KEY` |
| `llm_model_name` | str | `"gemini-2.0-flash"` | |
| `llm_max_retries` | int | 2 | Repair-retry count |
| `llm_temperature` | float | 0.3 | |
| `online_retraining` | bool | True | G-4 replay buffer + periodic retrain |
| `replay_buffer_size` | int | 50_000 | |
| `retrain_every` | int | 2_000 | Updates between backbone retrains |
| `min_buffer_for_retrain` | int | 2_000 | |
| `minibatch_size` | int | 1_024 | |
| `retrain_epochs` | int | 1 | Epochs of backbone fine-tune per retrain |
| `drift_enabled` | bool | True | |
| `drift_baseline_size` | int | 2_000 | Baseline window size |
| `drift_window_size` | int | 2_000 | Rolling window size |
| `drift_threshold_z` | float | 3.0 | Alert if |z| above threshold |
| `device` | `"auto"\|"cpu"\|"cuda"` | `"auto"` | |
| `seed` | int | 42 | |
| `checkpoint_dir` | Path\|None | None | Default = parent of `model_path` |
| `checkpoint_every` | int\|None | None | For `learning_session(checkpoint_every=...)` |

### 6.1 YAML config file

Set `BANDITS_CONFIG_FILE=configs/inference.yaml` or pass `InferenceConfig.load(file=...)`:

```yaml
# configs/inference.yaml
llm_enabled: true
llm_provider: gemini
n_confidence_draws: 200
drift_threshold_z: 2.5
```

Priority: explicit kwargs to `.load()` > env vars > YAML file > defaults.

---

## 7. Error handling

Every engine error descends from `InferenceError`.

| Exception | Fires when | Caller action |
| --- | --- | --- |
| `ConfigurationError` | Missing artefact / bad config / pipeline schema mismatch | Fix config, restart |
| `ValidationError` | Input failed Pydantic validation | Return 422 upstream; do not retry |
| `ModelError` | Torch raised, NaN output, shape mismatch | Log, 500 upstream; retry once |
| `ExplanationError` | LLM retries exhausted, jargon guard tripped | Ignored unless `explain="require"` |

`ValidationError.errors()` returns a list of `{loc, msg, type}` dicts (framework-agnostic — FastAPI-compatible).

---

## 8. What this module will not do

Out of scope (see `docs/inference_module_design.md §9`):

- Training, model saving, model selection.
- Feature-engineering changes (retrain+save the pipeline separately).
- Persistent storage of predictions/updates (caller's responsibility).
- Counterfactual OPE on live data (no counterfactuals in production).
- Transactional update rollback (Sherman-Morrison isn't reversible).
- Multi-tenant routing (one engine instance = one tenant).
