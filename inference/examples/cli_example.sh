#!/usr/bin/env bash
# cli_example.sh — end-to-end shell demo of the inference engine.
#
# Shows:
#   1. Three-layer config (env vars)
#   2. Single-patient predict
#   3. Batch predict (newline-delimited JSON)
#   4. Continuous-learning ingest from CSV
#
# Run from the repo root:
#   bash inference/examples/cli_example.sh
set -euo pipefail

export BANDITS_MODEL_PATH="${BANDITS_MODEL_PATH:-models/neural_thompson.pt}"
export BANDITS_PIPELINE_PATH="${BANDITS_PIPELINE_PATH:-models/feature_pipeline.joblib}"
export BANDITS_N_CONFIDENCE_DRAWS="${BANDITS_N_CONFIDENCE_DRAWS:-100}"
export BANDITS_LLM_ENABLED="${BANDITS_LLM_ENABLED:-true}"
export BANDITS_LLM_PROVIDER="${BANDITS_LLM_PROVIDER:-stub}"
export BANDITS_DEVICE="${BANDITS_DEVICE:-cpu}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ─── 1. Single predict ─────────────────────────────────────────────────────
echo "== single predict =="
python - <<'PY'
from inference import InferenceEngine

engine = InferenceEngine.from_env()
patient = {
    "age": 62, "bmi": 34.2, "hba1c_baseline": 8.9, "egfr": 85.0,
    "diabetes_duration": 6.0, "fasting_glucose": 180.0, "c_peptide": 1.4,
    "bp_systolic": 140.0, "ldl": 120.0, "hdl": 45.0,
    "triglycerides": 200.0, "alt": 30.0,
    "cvd": 1, "ckd": 0, "nafld": 1, "hypertension": 1,
    "patient_id": "PID-0001",
}
result = engine.predict(patient, explain=True)
print(f"recommended={result.recommended} "
      f"confidence={result.confidence_pct}% "
      f"safety={result.safety_status}")
print("-- explanation --")
print(result.explanation["recommendation_summary"])
PY

# Expected:
#   recommended=<one of the five treatments>
#   confidence=<0–100>%
#   safety=<CLEAR|WARNING|CONTRAINDICATION_FOUND>
#   <stub explanation paragraph>

# ─── 2. Batch predict from NDJSON ─────────────────────────────────────────
echo
echo "== batch predict =="
NDJSON="$TMP/batch.ndjson"
python - > "$NDJSON" <<'PY'
import json, random
random.seed(0)
for i in range(3):
    print(json.dumps({
        "age": 55 + i, "bmi": 30 + i, "hba1c_baseline": 8.0 + 0.2*i,
        "egfr": 90 - 10*i, "diabetes_duration": 5.0,
        "fasting_glucose": 160.0, "c_peptide": 1.3, "bp_systolic": 130.0,
        "ldl": 110.0, "hdl": 50.0, "triglycerides": 180.0, "alt": 25.0,
        "cvd": i % 2, "ckd": 0, "nafld": 1, "hypertension": 1,
        "patient_id": f"PID-{i:04d}",
    }))
PY

python - <<PY
import json
from inference import InferenceEngine
engine = InferenceEngine.from_env()
rows = [json.loads(line) for line in open("$NDJSON")]
for r in engine.predict_batch(rows):
    print(f"{r.patient_id} -> {r.recommended} (conf={r.confidence_pct}%)")
PY

# Expected: three lines of "PID-000X -> <treatment> (conf=XX%)"

# ─── 3. Continuous learning via CSV ingest ────────────────────────────────
echo
echo "== ingest_csv =="
CSV="$TMP/updates.csv"
python - > "$CSV" <<'PY'
import csv, sys, random
random.seed(1)
fields = [
    "age","bmi","hba1c_baseline","egfr","diabetes_duration",
    "fasting_glucose","c_peptide","bp_systolic","ldl","hdl","triglycerides","alt",
    "cvd","ckd","nafld","hypertension","action","reward",
]
w = csv.DictWriter(sys.stdout, fieldnames=fields)
w.writeheader()
for i in range(5):
    w.writerow({
        "age": 55, "bmi": 32, "hba1c_baseline": 8.5, "egfr": 90,
        "diabetes_duration": 6, "fasting_glucose": 170, "c_peptide": 1.3,
        "bp_systolic": 130, "ldl": 110, "hdl": 50, "triglycerides": 180,
        "alt": 25, "cvd": 1, "ckd": 0, "nafld": 1, "hypertension": 1,
        "action": i % 5, "reward": round(1.0 + 0.1*i, 2),
    })
PY

python - <<PY
from inference import InferenceEngine
engine = InferenceEngine.from_env()
acks = list(engine.ingest_csv("$CSV"))
for ack in acks:
    print(f"accepted={ack.accepted} n={ack.n_updates_so_far} "
          f"retrain={ack.backbone_retrained} drift={len(ack.drift_alerts)}")
PY

# Expected: five lines of "accepted=True n=X retrain=False drift=0"
echo
echo "done."
