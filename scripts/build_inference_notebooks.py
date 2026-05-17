"""Generate notebooks/12 and notebooks/13 using nbformat.

Run once:
    python scripts/build_inference_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"


def _md(text: str) -> nbf.notebooknode.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def _code(src: str) -> nbf.notebooknode.NotebookNode:
    return nbf.v4.new_code_cell(src)


# ── Notebook 12: prediction demo ─────────────────────────────────────────────

def build_12() -> nbf.notebooknode.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md("# 12 — Inference: Prediction Demo\n\n"
            "End-to-end tour of the `inference` package for single and batch prediction. "
            "Uses `generate_patient` to synthesise cases on the fly — no CSV required.\n\n"
            "LLM explanations use the built-in `StubClient` (deterministic, offline). "
            "To run against Gemini, set `BANDITS_LLM_PROVIDER=gemini` and `GEMINI_API_KEY`."),
        _md("## 1. Setup"),
        _code(
            "from pathlib import Path\n"
            "import os, sys\n"
            "repo_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "os.chdir(repo_root)\n"
            "sys.path.insert(0, str(repo_root))\n"
            "\n"
            "import numpy as np\n"
            "from src.data_generator import generate_patient, TREATMENTS\n"
            "from inference import (\n"
            "    InferenceEngine, InferenceConfig,\n"
            "    PatientInput, ValidationError,\n"
            ")\n"
        ),
        _code(
            "cfg = InferenceConfig.load(\n"
            "    llm_enabled=True,\n"
            "    llm_provider='stub',   # change to 'gemini' when a key is set\n"
            "    n_confidence_draws=200,\n"
            "    device='cpu',\n"
            ")\n"
            "engine = InferenceEngine.from_config(cfg)\n"
            "engine.snapshot()"
        ),
        _md("## 2. Single prediction — raw output"),
        _code(
            "rng = np.random.RandomState(42)\n"
            "patient = generate_patient(rng)\n"
            "patient['patient_id'] = 'DEMO-0001'\n"
            "patient"
        ),
        _code(
            "result = engine.predict(patient, explain=False)\n"
            "print(f'Recommended: {result.recommended}')\n"
            "print(f'Confidence:  {result.confidence_pct}% ({result.confidence_label})')\n"
            "print(f'Safety:      {result.safety_status}')\n"
            "print(f'Runner-up:   {result.runner_up} '\n"
            "      f'(win rate {result.runner_up_win_rate:.2%})')\n"
            "print()\n"
            "print('Posterior means (pp HbA1c reduction):')\n"
            "for t, v in result.posterior_means.items():\n"
            "    print(f'  {t:<10s} {v:+.2f}')"
        ),
        _md("## 3. Prediction with LLM explanation (stub)"),
        _code(
            "result = engine.predict(patient, explain=True)\n"
            "print('RECOMMENDATION SUMMARY:')\n"
            "print(result.explanation['recommendation_summary'])\n"
            "print()\n"
            "print('SAFETY ASSESSMENT:')\n"
            "print(result.explanation['safety_assessment'])\n"
            "print()\n"
            "print('MONITORING NOTE:')\n"
            "print(result.explanation['monitoring_note'])"
        ),
        _md("## 4. Safety gate — eGFR < 30 forces an override\n\n"
            "Metformin is contraindicated at eGFR < 30 (lactic-acidosis risk). "
            "Force a low eGFR and confirm the gate excludes Metformin from the "
            "final recommendation."),
        _code(
            "patient_ckd = dict(patient)\n"
            "patient_ckd['egfr'] = 20.0     # severe CKD\n"
            "patient_ckd['ckd'] = 1\n"
            "result_ckd = engine.predict(patient_ckd)\n"
            "\n"
            "print(f'Final recommendation: {result_ckd.recommended}')\n"
            "print(f'Safety status:        {result_ckd.safety_status}')\n"
            "if result_ckd.override is not None:\n"
            "    print()\n"
            "    print('-- OVERRIDE FIRED --')\n"
            "    print(f\"Original:  {result_ckd.override['original_treatment']}\")\n"
            "    print(f\"Final:     {result_ckd.override['final_treatment']}\")\n"
            "    print(f\"Reason:    {result_ckd.override['reason']}\")\n"
            "    print(f\"Blocked:   {result_ckd.override['blocked_treatments']}\")\n"
            "\n"
            "assert result_ckd.recommended != 'Metformin', 'safety gate failed'\n"
            "print()\n"
            "print('assertion passed: Metformin is not the final recommendation')"
        ),
        _md("## 5. Batch prediction with an invalid row\n\n"
            "Invalid rows are returned as sentinel `PredictionResult(accepted=False, ...)` "
            "— the batch does not raise."),
        _code(
            "good_rows = [generate_patient(rng) for _ in range(3)]\n"
            "bad_row = {'age': 200, 'bmi': 34}     # malformed\n"
            "\n"
            "results = engine.predict_batch([*good_rows, bad_row])\n"
            "for i, r in enumerate(results):\n"
            "    if r.accepted:\n"
            "        print(f'{i}: ✓ {r.recommended} (conf={r.confidence_pct}%)')\n"
            "    else:\n"
            "        print(f'{i}: ✗ rejected -> {len(r.validation_errors)} field errors')"
        ),
        _md("## 6. Validation error surface\n\n"
            "`ValidationError.errors()` is structured like Pydantic's own — "
            "FastAPI-compatible for 422 responses."),
        _code(
            "try:\n"
            "    engine.predict({'age': 200, 'bmi': 34})\n"
            "except ValidationError as e:\n"
            "    for err in e.errors()[:3]:\n"
            "        print(err)"
        ),
        _md("---\n\nDone. Notebook 13 demonstrates the continuous-learning surface on a synthetic stream."),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


# ── Notebook 13: continuous-learning demo ────────────────────────────────────

def build_13() -> nbf.notebooknode.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md("# 13 — Inference: Continuous Learning Demo\n\n"
            "End-to-end tour of the `inference` continuous-learning surface on a "
            "synthetic stream generated via `generate_patient` + `reward_oracle`. "
            "The centrepiece is **`LearningStream`** — the rich per-step event "
            "channel used for real-time observability.\n\n"
            "### Sections\n"
            "1. Setup\n"
            "2. `update()` — the simple single-record surface\n"
            "3. `LearningStream` — the rich per-step event (every one of the 31 "
            "fields on `LearningStepEvent`)\n"
            "4. Real-time learning run with a BMI distribution shift:\n"
            "   - step-by-step printed output\n"
            "   - posterior means diverging over time\n"
            "   - posterior uncertainty shrinking over time\n"
            "   - rolling explore-vs-exploit window\n"
            "   - cumulative reward / regret / running accuracy\n"
            "   - drift z-scores + alerts firing after the shift\n"
            "   - backbone-health timeline (retrains, replay buffer, noise variance)\n"
            "5. SSE frame preview — what an HTTP streaming client would receive"),
        _md("## 1. Setup"),
        _code(
            "from pathlib import Path\n"
            "import os, sys\n"
            "repo_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "os.chdir(repo_root)\n"
            "sys.path.insert(0, str(repo_root))\n"
            "\n"
            "from collections import deque\n"
            "\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import clear_output, display\n"
            "\n"
            "from src.data_generator import generate_patient, reward_oracle, TREATMENTS\n"
            "from inference import (\n"
            "    InferenceEngine, InferenceConfig,\n"
            "    LearningStream, LearningStepEvent,\n"
            ")\n"
        ),
        _code(
            "cfg = InferenceConfig.load(\n"
            "    online_retraining=True,\n"
            "    replay_buffer_size=2000,\n"
            "    retrain_every=200,\n"
            "    min_buffer_for_retrain=200,\n"
            "    minibatch_size=64,\n"
            "    drift_enabled=True,\n"
            "    drift_baseline_size=200,\n"
            "    drift_window_size=200,\n"
            "    drift_threshold_z=2.5,\n"
            "    n_confidence_draws=64,\n"
            "    device='cpu',\n"
            ")\n"
            "engine = InferenceEngine.from_config(cfg)\n"
            "engine.snapshot()"
        ),
        _code(
            "rng = np.random.RandomState(0)\n"
            "\n"
            "def sample_patient(rng, patient_id=None):\n"
            "    ctx = generate_patient(rng)\n"
            "    if patient_id is not None:\n"
            "        ctx['patient_id'] = patient_id\n"
            "    return ctx\n"
            "\n"
            "def oracle_vector(ctx, noise=True):\n"
            "    \"\"\"Per-arm counterfactual rewards in canonical treatment order.\"\"\"\n"
            "    return np.array(\n"
            "        [reward_oracle(ctx, t, noise=noise) for t in TREATMENTS],\n"
            "        dtype=float,\n"
            "    )\n"
        ),
        _md("## 2. `update()` — the simple single-record surface\n\n"
            "The familiar ack-based API: submit `(patient, action, reward)`, get "
            "back a `LearningAck`. Use this when you already know the action and "
            "the observed reward."),
        _code(
            "ctx = sample_patient(rng, patient_id='SINGLE-001')\n"
            "action = int(rng.randint(0, len(TREATMENTS)))\n"
            "reward = float(reward_oracle(ctx, TREATMENTS[action], noise=True))\n"
            "ack = engine.update({'patient': ctx, 'action': action, 'reward': reward})\n"
            "print(f'accepted={ack.accepted} n={ack.n_updates_so_far} '\n"
            "      f'posterior_updated={ack.posterior_updated} '\n"
            "      f'retrain={ack.backbone_retrained}')"
        ),
        _md("## 3. `LearningStream` — the rich per-step event\n\n"
            "`LearningStream` drives Thompson sampling itself (you supply only the "
            "patient and a per-arm oracle reward vector), applies the online "
            "update, and emits a `LearningStepEvent` capturing the **full** "
            "internal state that drove the decision:\n\n"
            "| Group | Fields |\n"
            "|---|---|\n"
            "| core decision | `step`, `selectedIdx`, `posteriorMeanArgmax`, `explored`, `thompsonSamples`, `observedReward`, `oracleOptimalIdx`, `oracleOptimalReward`, `regret` |\n"
            "| posterior state | `posteriorMeans`, `posteriorUncertainty`, `winRates`, `confidencePct`, `confidenceLabel`, `meanGap`, `nUpdatesPerArm` |\n"
            "| running aggregates | `cumulativeReward`, `cumulativeRegret`, `runningAccuracy`, `runningMeanRewardPerArm`, `bestTreatmentIdx`, `phase` |\n"
            "| backbone health | `retrainFired`, `noiseVariance`, `replayBufferSize` |\n"
            "| drift signals | `contextNorm`, `driftAlerts`, `driftStreams` |\n"
            "| patient context | `patientId`, `patientFeatures`, `safetyStatus` |\n\n"
            "`thompsonSamples` is *causally correct* — it's the exact per-arm "
            "draw (from the same `phi`, `mu`, `A_inv`, `noise_variance`) that "
            "produced `selectedIdx`."),
        _code(
            "# One event — every field on LearningStepEvent\n"
            "preview_rng = np.random.default_rng(42)\n"
            "with engine.learning_stream(total_steps=5, rng=preview_rng) as stream:\n"
            "    ctx = sample_patient(rng, patient_id='DEMO-PREVIEW')\n"
            "    event = stream.step(ctx, oracle_vector(ctx))\n"
            "\n"
            "payload = event.model_dump()\n"
            "print(f'{len(payload)} fields on LearningStepEvent\\n')\n"
            "for k, v in payload.items():\n"
            "    if isinstance(v, dict):\n"
            "        inner = ', '.join(f\"{ik}={iv!r}\" for ik, iv in v.items())\n"
            "        print(f'  {k}: {{{inner}}}')\n"
            "    else:\n"
            "        print(f'  {k}: {v!r}')"
        ),
        _md("### 3.1 Compact console line\n\n"
            "`event.to_console_line()` is a one-line human-readable summary. "
            "Useful for dev loops, log sinks, or a CLI dashboard."),
        _code("print(event.to_console_line())"),
        _md("## 4. Real-time learning run with a BMI distribution shift\n\n"
            "Stream 600 synthetic patients with the display updating as each "
            "patient arrives — `clear_output(wait=True)` for a live tail of "
            "recent console lines, plus three charts (cumulative reward, "
            "cumulative regret, posterior means) that redraw every 10 steps. "
            "A BMI shift is injected at step 300 to trigger the drift monitor.\n\n"
            "The collected `events` list feeds the post-loop sections 4.1–4.7 "
            "that visualise each field group in full."),
        _code(
            "N = 600\n"
            "shift_start = 300\n"
            "redraw_every = 10\n"
            "tail_lines = 12          # recent console lines kept on screen\n"
            "\n"
            "events: list[LearningStepEvent] = []\n"
            "recent = deque(maxlen=tail_lines)\n"
            "stream_rng = np.random.default_rng(7)\n"
            "\n"
            "plt.ioff()\n"
            "fig_live, axes_live = plt.subplots(1, 3, figsize=(15, 4))\n"
            "\n"
            "def _redraw(axes, events, shift_start, N, step_now):\n"
            "    steps_so_far = np.array([e.step for e in events])\n"
            "    cum_r = np.array([e.cumulativeReward for e in events])\n"
            "    cum_reg = np.array([e.cumulativeRegret for e in events])\n"
            "    pmeans_live = {t: np.array([e.posteriorMeans[t] for e in events])\n"
            "                   for t in TREATMENTS}\n"
            "    for ax in axes: ax.clear()\n"
            "    axes[0].plot(steps_so_far, cum_r, color='#2ca02c', linewidth=1.3)\n"
            "    axes[0].set_title(f'cumulativeReward  (step {step_now}/{N})')\n"
            "    axes[0].set_xlabel('step'); axes[0].grid(alpha=0.3)\n"
            "    axes[1].plot(steps_so_far, cum_reg, color='#d62728', linewidth=1.3)\n"
            "    axes[1].set_title('cumulativeRegret')\n"
            "    axes[1].set_xlabel('step'); axes[1].grid(alpha=0.3)\n"
            "    for t in TREATMENTS:\n"
            "        axes[2].plot(steps_so_far, pmeans_live[t], label=t, linewidth=1.0)\n"
            "    axes[2].set_title('posteriorMeans')\n"
            "    axes[2].set_xlabel('step')\n"
            "    axes[2].legend(loc='best', fontsize=7); axes[2].grid(alpha=0.3)\n"
            "    for ax in axes:\n"
            "        ax.axvline(shift_start, color='grey', linestyle='--', alpha=0.5)\n"
            "    fig_live.tight_layout()\n"
            "\n"
            "try:\n"
            "    with engine.learning_stream(total_steps=N, rng=stream_rng) as stream:\n"
            "        for i in range(N):\n"
            "            ctx = sample_patient(rng)\n"
            "            if i >= shift_start:\n"
            "                ctx['bmi'] = min(float(ctx['bmi']) + 6.0, 79.0)\n"
            "            event = stream.step(ctx, oracle_vector(ctx))\n"
            "            events.append(event)\n"
            "            recent.append(event.to_console_line())\n"
            "\n"
            "            if (i + 1) % redraw_every == 0 or i == N - 1:\n"
            "                _redraw(axes_live, events, shift_start, N, i + 1)\n"
            "\n"
            "            clear_output(wait=True)\n"
            "            for line in recent:\n"
            "                print(line)\n"
            "            display(fig_live)\n"
            "finally:\n"
            "    plt.ion()\n"
            "    plt.close(fig_live)\n"
            "\n"
            "snap = stream.snapshot()\n"
            "print()\n"
            "print(f'Final snapshot: {snap}')"
        ),
        _md("### 4.1 Posterior means diverging over time\n\n"
            "`posteriorMeans` is the per-arm mean HbA1c-reduction estimate after "
            "each update. The five curves should fan apart as the model learns "
            "which treatments work."),
        _code(
            "steps = np.array([e.step for e in events])\n"
            "pmeans = {t: np.array([e.posteriorMeans[t] for e in events]) for t in TREATMENTS}\n"
            "\n"
            "fig, ax = plt.subplots(1, 1, figsize=(10, 4))\n"
            "for t in TREATMENTS:\n"
            "    ax.plot(steps, pmeans[t], label=t, linewidth=1.2)\n"
            "ax.axvline(shift_start, color='grey', linestyle='--', alpha=0.5, label='BMI shift')\n"
            "ax.set_xlabel('step')\n"
            "ax.set_ylabel('posterior mean (pp HbA1c reduction)')\n"
            "ax.set_title('posteriorMeans — arms diverge as the model learns')\n"
            "ax.legend(loc='best', fontsize=8)\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("### 4.2 Posterior uncertainty shrinking over time\n\n"
            "`posteriorUncertainty` is `φᵀ A⁻¹ φ` per arm — the last-layer "
            "Bayesian-regression variance proxy. For arms that get pulled, it "
            "shrinks; arms that are never chosen stay at the prior."),
        _code(
            "punc = {t: np.array([e.posteriorUncertainty[t] for e in events]) for t in TREATMENTS}\n"
            "\n"
            "fig, ax = plt.subplots(1, 1, figsize=(10, 4))\n"
            "for t in TREATMENTS:\n"
            "    ax.plot(steps, punc[t], label=t, linewidth=1.2)\n"
            "ax.axvline(shift_start, color='grey', linestyle='--', alpha=0.5)\n"
            "ax.set_xlabel('step')\n"
            "ax.set_ylabel('posteriorUncertainty (φᵀ A⁻¹ φ)')\n"
            "ax.set_title('posteriorUncertainty — shrinks on pulled arms')\n"
            "ax.set_yscale('log')\n"
            "ax.legend(loc='best', fontsize=8)\n"
            "ax.grid(alpha=0.3, which='both')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("### 4.3 Rolling explore-vs-exploit window\n\n"
            "`event.explored` is `True` when the Thompson draw picked a different "
            "arm than the posterior-mean argmax — i.e. real exploration, not "
            "exploit. A rolling mean over a 50-step window shows how exploration "
            "naturally tapers as the posterior sharpens."),
        _code(
            "window = 50\n"
            "explored = np.array([1.0 if e.explored else 0.0 for e in events])\n"
            "roll = np.convolve(explored, np.ones(window) / window, mode='valid')\n"
            "roll_steps = steps[window - 1:]\n"
            "\n"
            "fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))\n"
            "ax.plot(roll_steps, roll, color='#9467bd', linewidth=1.4)\n"
            "ax.axvline(shift_start, color='grey', linestyle='--', alpha=0.5)\n"
            "ax.axhline(0.5, color='red', linestyle=':', alpha=0.3, label='50% band')\n"
            "ax.fill_between(roll_steps, roll, alpha=0.2, color='#9467bd')\n"
            "ax.set_ylim(0, 1)\n"
            "ax.set_xlabel('step')\n"
            "ax.set_ylabel(f'rolling explore fraction (w={window})')\n"
            "ax.set_title('explored — rolling exploration vs exploitation')\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "print(f'total explored: {int(explored.sum())} / {len(explored)} '\n"
            "      f'({explored.mean():.1%})')"
        ),
        _md("### 4.4 Running aggregates — cumulative reward, regret, accuracy"),
        _code(
            "cum_r = np.array([e.cumulativeReward for e in events])\n"
            "cum_reg = np.array([e.cumulativeRegret for e in events])\n"
            "acc = np.array([e.runningAccuracy for e in events])\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))\n"
            "axes[0].plot(steps, cum_r, color='#2ca02c')\n"
            "axes[0].set_title('cumulativeReward')\n"
            "axes[0].set_xlabel('step'); axes[0].grid(alpha=0.3)\n"
            "axes[1].plot(steps, cum_reg, color='#d62728')\n"
            "axes[1].set_title('cumulativeRegret')\n"
            "axes[1].set_xlabel('step'); axes[1].grid(alpha=0.3)\n"
            "axes[2].plot(steps, acc, color='#1f77b4')\n"
            "axes[2].axhline(1 / len(TREATMENTS), color='grey', linestyle=':', alpha=0.6,\n"
            "                label=f'random = {1/len(TREATMENTS):.0%}')\n"
            "axes[2].set_ylim(0, 1)\n"
            "axes[2].set_title('runningAccuracy vs oracle')\n"
            "axes[2].set_xlabel('step'); axes[2].legend(); axes[2].grid(alpha=0.3)\n"
            "for ax in axes:\n"
            "    ax.axvline(shift_start, color='grey', linestyle='--', alpha=0.5)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("### 4.5 Drift signals — z-scores + alerts after the BMI shift\n\n"
            "`driftStreams` exposes the rolling-window z-score for each monitored "
            "signal (context-norm, action, reward) vs the baseline captured from "
            "the first `drift_baseline_size` observations. The shaded region "
            "marks the shifted BMI regime, and orange ticks show every step "
            "where a `DriftAlert` fired."),
        _code(
            "drift_ctx = np.array([e.driftStreams.get('context', 0.0) for e in events])\n"
            "drift_act = np.array([e.driftStreams.get('action', 0.0) for e in events])\n"
            "drift_rew = np.array([e.driftStreams.get('reward', 0.0) for e in events])\n"
            "alert_steps = [e.step for e in events if e.driftAlerts]\n"
            "\n"
            "fig, ax = plt.subplots(1, 1, figsize=(10, 3.8))\n"
            "ax.axvspan(shift_start, N, alpha=0.12, color='orange', label='BMI shifted')\n"
            "ax.plot(steps, drift_ctx, label='context', color='#1f77b4')\n"
            "ax.plot(steps, drift_act, label='action', color='#2ca02c')\n"
            "ax.plot(steps, drift_rew, label='reward', color='#d62728')\n"
            "for s in alert_steps:\n"
            "    ax.axvline(s, color='#ff7f0e', alpha=0.25, linestyle=':')\n"
            "ax.axhline(cfg.drift_threshold_z, color='black', linestyle='--', alpha=0.4,\n"
            "           label=f'±{cfg.drift_threshold_z}σ threshold')\n"
            "ax.axhline(-cfg.drift_threshold_z, color='black', linestyle='--', alpha=0.4)\n"
            "ax.set_xlabel('step')\n"
            "ax.set_ylabel('drift z-score')\n"
            "ax.set_title('driftStreams — rolling z-scores (ticks = driftAlerts fired)')\n"
            "ax.legend(loc='best', fontsize=8)\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "print(f'drift alerts fired on {len(alert_steps)} steps')"
        ),
        _md("### 4.6 Backbone health — retrains, replay buffer, noise variance\n\n"
            "`retrainFired` is `True` on steps where the shared backbone was "
            "retrained on the replay buffer (G-4). `replayBufferSize` grows up "
            "to the configured cap, and `noiseVariance` is the posterior noise "
            "parameter σ²."),
        _code(
            "replay = np.array([e.replayBufferSize for e in events])\n"
            "noise_var = np.array([e.noiseVariance for e in events])\n"
            "retrain_steps = [e.step for e in events if e.retrainFired]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(13, 3.2))\n"
            "axes[0].plot(steps, replay, color='#8c564b')\n"
            "for rs in retrain_steps:\n"
            "    axes[0].axvline(rs, color='#1f77b4', alpha=0.5)\n"
            "axes[0].axvline(shift_start, color='grey', linestyle='--', alpha=0.5)\n"
            "axes[0].set_title('replayBufferSize (blue = retrainFired)')\n"
            "axes[0].set_xlabel('step'); axes[0].grid(alpha=0.3)\n"
            "\n"
            "axes[1].plot(steps, noise_var, color='#17becf')\n"
            "axes[1].axvline(shift_start, color='grey', linestyle='--', alpha=0.5)\n"
            "axes[1].set_title('noiseVariance (σ²)')\n"
            "axes[1].set_xlabel('step'); axes[1].grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "print(f'retrains fired: {len(retrain_steps)}')"
        ),
        _md("### 4.7 Per-arm pulls + running reward — where did the traffic go?\n\n"
            "`nUpdatesPerArm` and `runningMeanRewardPerArm` track traffic "
            "allocation and realised performance per treatment."),
        _code(
            "last = events[-1]\n"
            "names = list(last.nUpdatesPerArm.keys())\n"
            "pulls = [last.nUpdatesPerArm[t] for t in names]\n"
            "mrew = [last.runningMeanRewardPerArm[t] for t in names]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))\n"
            "axes[0].bar(names, pulls, color='#4c72b0')\n"
            "axes[0].set_title('nUpdatesPerArm (final)')\n"
            "axes[0].set_ylabel('pulls')\n"
            "axes[1].bar(names, mrew, color='#55a868')\n"
            "axes[1].set_title('runningMeanRewardPerArm (final)')\n"
            "axes[1].set_ylabel('mean reward (pp HbA1c)')\n"
            "for ax in axes:\n"
            "    ax.grid(alpha=0.3, axis='y')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "print(f'bestTreatmentIdx (last): {last.bestTreatmentIdx} '\n"
            "      f'({TREATMENTS[last.bestTreatmentIdx]})')\n"
            "print(f'confidencePct (last): {last.confidencePct}% ({last.confidenceLabel})')\n"
            "print(f'meanGap (last): {last.meanGap:.4f}')\n"
            "print(f'phase (last): {last.phase}')\n"
            "print(f'safetyStatus (last): {last.safetyStatus}')"
        ),
        _md("## 5. SSE frame preview\n\n"
            "`event.to_sse()` returns the exact bytes a browser `EventSource` "
            "would read off the wire. See `inference/examples/fastapi_app.py` "
            "for a runnable `/learn/stream/rich` endpoint wired to "
            "`engine.alearning_stream(...)`."),
        _code(
            "frame = events[-1].to_sse()\n"
            "print(frame[:400].decode('utf-8'), '...' if len(frame) > 400 else '')"
        ),
        _md("## 6. Final engine snapshot"),
        _code("engine.snapshot()"),
        _md("---\n\nDone. `LearningStream` complements — does not replace — "
            "`update()` / `learning_session()`. Reach for it when you need "
            "real-time step-by-step visibility into the decision, posterior, "
            "aggregates, drift, and backbone health."),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    nbf.write(build_12(), NOTEBOOKS / "12_inference_prediction_demo.ipynb")
    nbf.write(build_13(), NOTEBOOKS / "13_inference_continuous_learning_demo.ipynb")
    print("wrote notebooks/12 and notebooks/13")


if __name__ == "__main__":
    main()
