"""
Contextual Bandit Dataset Generator for Type 2 Diabetes Treatment Selection

Purpose: Generate data specifically structured for contextual bandit training,
         offline evaluation, and online simulation.

Key design choices:
- NO normalization — natural treatment-context interactions preserved
- Full counterfactual rewards (all 5 potential outcomes per patient)
- Logging policy with recorded propensity scores
- Reward oracle function for online simulation
- Realistic patient distributions (HbA1c ~8.5, BMI balanced)
- Balanced optimal action distribution (~15-25% per treatment)
- Strong contextual signals with clear niches and anti-niches

Author: Refactored for bandit research
Date: 2025
"""
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TREATMENTS = ["Metformin", "GLP-1", "SGLT-2", "DPP-4", "Insulin"]
N_TREATMENTS = len(TREATMENTS)
TREATMENT_TO_IDX = {t: i for i, t in enumerate(TREATMENTS)}
IDX_TO_TREATMENT = {i: t for i, t in enumerate(TREATMENTS)}

CONTEXT_FEATURES = [
    "age", "bmi", "hba1c_baseline", "egfr", "diabetes_duration",
    "fasting_glucose", "c_peptide", "cvd", "ckd", "nafld",
    "hypertension", "bp_systolic", "ldl", "hdl", "triglycerides", "alt",
]

# Treatment-specific noise std devs (in abstract-score units; scaled to pp below)
TREATMENT_NOISE = {
    "Metformin": 0.3,
    "GLP-1": 0.4,
    "SGLT-2": 0.4,
    "DPP-4": 0.3,
    "Insulin": 0.5,
}

# G-3: Rescale the abstract "benefit score" produced by the hand-written oracle
# into clinically plausible HbA1c percentage-point reductions. With SCALE=0.25
# the ideal Metformin patient (score ~6) predicts ~1.5 pp, the ideal Insulin
# patient (score ~10) predicts ~2.5 pp, and the hard ceiling is 3.0 pp — all
# consistent with published effect sizes. Noise scales proportionally, keeping
# signal-to-noise unchanged.
REWARD_SCALE = 0.25
REWARD_CAP_PP = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# REWARD ORACLE
# ─────────────────────────────────────────────────────────────────────────────

def reward_oracle(context: Dict, treatment: str, noise: bool = True) -> float:
    """
    Compute expected HbA1c reduction for a patient-treatment pair.

    Design principles:
    - Each treatment has a STRONG niche where it dominates (reward ~1.5-2.5 pp)
    - Each treatment has clear ANTI-niches where it's poor (reward ~0-0.5 pp)
    - Niches are sized to produce ~15-25% optimal per treatment
    - Output is scaled to HbA1c percentage-point reductions, capped at 3.0 pp
    - Penalties are aggressive to create separation

    Treatment niches (designed for balanced patient distribution):
      Metformin: young (<60), lean (BMI<32), early (<7yr), good kidneys (eGFR>60)
      GLP-1:     obese (BMI>35), NAFLD, CVD co-benefit
      SGLT-2:    CVD present, moderate-good kidneys (eGFR>30), cardiorenal
      DPP-4:     elderly (>60), CKD or low eGFR, moderate disease
      Insulin:   severe HbA1c (>10), beta-cell failure (low C-peptide), long duration

    Args:
        context: dict of patient features
        treatment: treatment name string
        noise: whether to add stochastic noise

    Returns:
        HbA1c reduction in percentage points (higher = better, clipped to
        [0, REWARD_CAP_PP]).
    """
    age = context["age"]
    bmi = context["bmi"]
    hba1c = context["hba1c_baseline"]
    egfr = context["egfr"]
    duration = context["diabetes_duration"]
    fg = context["fasting_glucose"]
    cpep = context["c_peptide"]
    cvd = context["cvd"]
    ckd = context["ckd"]
    nafld = context["nafld"]

    # ── Shared baseline: small, reflects disease severity ──
    base = 0.15 * (hba1c - 7.0)
    base = np.clip(base, -0.2, 0.8)

    if treatment == "Metformin":
        # NICHE: young, lean, early-stage, good kidneys
        # ~20% of patients: age<60 AND BMI<32 AND duration<7 AND eGFR>60
        r = 1.5  # decent base — first-line drug

        # Core niche: need multiple factors together
        young = age < 60
        lean = bmi < 32
        early = duration < 7
        good_kidney = egfr > 60
        good_beta = cpep > 1.2

        niche_score = young + lean + early + good_kidney + good_beta
        if niche_score >= 4:
            r += 4.5  # strong niche match
        elif niche_score == 3:
            r += 2.5
        elif niche_score == 2:
            r += 1.0

        r += 0.5 * (hba1c < 9.0)

        # Penalties
        r -= 3.5 * (egfr < 30)
        r -= 2.0 * (egfr < 45) * (egfr >= 30)
        r -= 1.5 * (bmi > 37)
        r -= 1.5 * (duration > 12)
        r -= 1.0 * (age > 70)
        r -= 1.0 * (cpep < 0.7)
        r -= 0.8 * (hba1c > 10)

    elif treatment == "GLP-1":
        # NICHE: obese (BMI>35), especially with NAFLD or CVD
        # ~20% of patients: BMI>35 (about 55% have this, but need NAFLD/CVD combo)
        r = 0.5  # low base

        # Core niche: obesity + metabolic complications
        obese = bmi > 35
        very_obese = bmi > 39
        has_nafld = nafld == 1
        has_cvd = cvd == 1

        if very_obese and has_nafld:
            r += 5.5
        elif very_obese:
            r += 3.5
        elif obese and has_nafld and has_cvd:
            r += 4.5
        elif obese and has_nafld:
            r += 3.0
        elif obese and has_cvd:
            r += 2.5
        elif obese:
            r += 1.5  # obesity alone gives moderate benefit

        r += 0.5 * (cpep > 1.0)
        r += 0.5 * (hba1c > 9)

        # Penalties — lean patients should NOT get GLP-1
        r -= 5.0 * (bmi < 28)
        r -= 3.0 * (bmi < 32) * (bmi >= 28)
        r -= 1.0 * (age > 78)
        r -= 0.5 * (cpep < 0.5)

    elif treatment == "SGLT-2":
        # NICHE: CVD present with decent kidneys, cardiorenal protection
        # ~18% of patients: CVD=1 (~35%) AND eGFR>30 (~85%) → ~30%, but penalties thin it
        r = 0.5  # low base

        # Core niche: cardiovascular + renal
        has_cvd = cvd == 1
        has_ckd = ckd == 1
        decent_kidney = egfr >= 30
        good_kidney = egfr >= 45

        if has_cvd and good_kidney:
            r += 5.0
        elif has_cvd and decent_kidney:
            r += 3.5
        elif has_cvd:
            r += 1.5

        # Renoprotective benefit (independent of CVD)
        if has_ckd and decent_kidney and not has_cvd:
            r += 3.0
        elif has_ckd and decent_kidney and has_cvd:
            r += 1.5  # stacks with CVD bonus

        r += 0.5 * (bmi > 30) * has_cvd
        r += 0.3 * (cpep > 1.0)

        # Penalties — no CV/renal indication → poor choice
        r -= 4.0 * (1 - has_cvd) * (1 - has_ckd)
        r -= 2.5 * (egfr < 25)
        r -= 0.5 * (bmi < 27) * (1 - has_cvd)

    elif treatment == "DPP-4":
        # NICHE: elderly (>60), CKD or low eGFR, moderate disease
        # ~18% of patients: age>60 (~40%) AND (CKD=1 OR eGFR<60) (~30%) → ~12%,
        #                    plus age>60 alone gives moderate benefit
        r = 0.8  # slightly higher base — well tolerated

        # Core niche: age + kidney impairment
        elderly = age > 60
        very_elderly = age > 70
        has_ckd = ckd == 1
        low_egfr = egfr < 60
        mod_disease = 7.5 < hba1c < 10.0
        mod_cpep = 0.7 < cpep < 1.5

        if very_elderly and (has_ckd or low_egfr):
            r += 5.5
        elif very_elderly:
            r += 3.5
        elif elderly and (has_ckd or low_egfr):
            r += 4.0
        elif elderly:
            r += 2.0
        elif has_ckd or low_egfr:
            r += 2.0

        r += 1.0 * mod_disease
        r += 0.8 * mod_cpep
        r += 0.5 * (low_egfr and egfr >= 30)

        # Penalties — young / severe patients shouldn't get DPP-4
        r -= 4.0 * (age < 45)
        r -= 2.5 * (age < 55) * (age >= 45)
        r -= 1.5 * (hba1c > 10.5)
        r -= 1.0 * (1 - has_ckd) * (age < 60) * (egfr >= 60)
        r -= 0.5 * (cpep < 0.5)

    elif treatment == "Insulin":
        # NICHE: severe/advanced disease, beta-cell failure, long duration
        # ~20% of patients: hba1c>10 (~25%) OR cpep<0.8 (~20%), with overlap
        r = 0.0  # zero base — truly only for severe patients

        # Core niche: disease severity + beta-cell failure
        severe_hba1c = hba1c > 10
        very_severe_hba1c = hba1c > 11.5
        low_cpep = cpep < 0.8
        very_low_cpep = cpep < 0.5
        long_duration = duration > 15
        high_fg = fg > 220

        if very_low_cpep and very_severe_hba1c:
            r += 7.0
        elif very_low_cpep and severe_hba1c:
            r += 5.5
        elif very_low_cpep:
            r += 4.5
        elif low_cpep and very_severe_hba1c:
            r += 5.0
        elif low_cpep and severe_hba1c:
            r += 4.0
        elif severe_hba1c and long_duration:
            r += 3.5
        elif severe_hba1c:
            r += 2.0
        elif low_cpep:
            r += 2.5

        r += 1.5 * long_duration
        r += 1.0 * high_fg
        r += 0.5 * (duration > 10) * (duration <= 15)

        # Penalties — unnecessary insulin is harmful
        r -= 4.5 * (hba1c < 8.0) * (cpep > 1.5)
        r -= 3.0 * (hba1c < 9.0) * (cpep > 1.2)
        r -= 2.0 * (duration < 3) * (cpep > 1.0)
        r -= 1.5 * (age > 75)
        r -= 0.5 * (cpep > 2.0)

    else:
        raise ValueError(f"Unknown treatment: {treatment}")

    reward = base + r

    if noise:
        sigma = TREATMENT_NOISE[treatment]
        reward += np.random.normal(0, sigma)

    # G-3: rescale abstract benefit → HbA1c pp reduction, cap at clinical ceiling
    reward *= REWARD_SCALE
    return float(np.clip(reward, 0.0, REWARD_CAP_PP))


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_patient(rng: np.random.RandomState) -> Dict:
    """
    Generate a single realistic T2D patient context.

    Distribution targets (to support balanced treatment niches):
      Age:      mean ~58, range 25-85, ~40% over 60
      BMI:      mean ~32, range 22-45, ~40% over 35, ~25% under 30
      HbA1c:    mean ~8.5, range 6.5-14, ~25% over 10, ~35% under 8
      eGFR:     mean ~85, range 15-130, ~25% under 60
      Duration: mean ~8, range 0-30, ~30% under 3, ~15% over 15
      C-peptide: mean ~1.3, range 0.1-3, ~20% under 0.8
      CVD:      ~35%
      CKD:      ~25% (correlated with eGFR)
      NAFLD:    ~40% (correlated with BMI)
    """
    age = int(np.clip(rng.normal(58, 12), 25, 85))

    # BMI: more spread, ~25% lean/normal, ~35% overweight, ~40% obese
    bmi_cat = rng.choice(
        ["lean", "normal", "overweight", "obese", "severe"],
        p=[0.10, 0.15, 0.30, 0.28, 0.17],
    )
    bmi_params = {
        "lean": (25, 1.5, 22, 27),
        "normal": (28, 1.5, 26, 30),
        "overweight": (32, 1.5, 30, 35),
        "obese": (37, 1.5, 35, 40),
        "severe": (41, 1.5, 39, 45),
    }
    mu, sig, lo, hi = bmi_params[bmi_cat]
    bmi = float(np.clip(rng.normal(mu, sig), lo, hi))

    # HbA1c: realistic T2D distribution, mean ~8.5
    # Use shifted gamma to get right-skewed distribution centered around 8.5
    hba1c = float(np.clip(rng.gamma(3, 0.8) + 6.5, 6.5, 14.0))

    # eGFR: age-dependent decline
    egfr_mu = 100 - (age - 40) * 0.6
    egfr = float(np.clip(rng.normal(egfr_mu, 20), 15, 130))

    # Diabetes duration: mix of stages
    dur_cat = rng.choice(
        ["new", "recent", "established", "longterm"],
        p=[0.20, 0.30, 0.35, 0.15],
    )
    dur_ranges = {"new": (0, 3), "recent": (3, 7), "established": (7, 15), "longterm": (15, 30)}
    lo, hi = dur_ranges[dur_cat]
    duration = float(rng.uniform(lo, hi))

    # Fasting glucose: correlated with HbA1c
    fg_mu = 70 + (hba1c - 5) * 28
    fg = float(np.clip(rng.normal(fg_mu, 20), 80, 350))

    # C-peptide: declines with duration and severity
    cpep_mu = 1.8 - (duration / 30) * 1.0 - (hba1c - 7) * 0.1
    cpep = float(np.clip(rng.normal(cpep_mu, 0.35), 0.1, 3.0))

    # Comorbidities
    cvd = int(rng.random() < 0.35)
    ckd = int(egfr < 55 or rng.random() < 0.12)
    nafld = int((bmi > 33 and rng.random() < 0.7) or rng.random() < 0.15)
    hypertension = int(rng.random() < (0.3 + 0.3 * (bmi > 30) + 0.2 * (age > 60)))

    # Vitals and labs
    bp_sys = float(np.clip(120 + (bmi - 25) * 0.8 + (age - 40) * 0.3 + rng.normal(0, 10), 90, 200))
    ldl = float(np.clip(100 + (bmi - 25) * 2 + rng.normal(0, 30), 50, 250))
    hdl = float(np.clip(60 - (bmi - 25) * 0.5 + rng.normal(0, 10), 20, 100))
    tg = float(np.clip(120 + (bmi - 25) * 3 + rng.normal(0, 40), 50, 500))
    alt = float(np.clip(rng.exponential(25), 10, 100))

    return {
        "age": age,
        "bmi": round(bmi, 1),
        "hba1c_baseline": round(hba1c, 2),
        "egfr": round(egfr, 1),
        "diabetes_duration": round(duration, 1),
        "fasting_glucose": round(fg, 1),
        "c_peptide": round(cpep, 2),
        "cvd": cvd,
        "ckd": ckd,
        "nafld": nafld,
        "hypertension": hypertension,
        "bp_systolic": round(bp_sys, 1),
        "ldl": round(ldl, 1),
        "hdl": round(hdl, 1),
        "triglycerides": round(tg, 1),
        "alt": round(alt, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING POLICIES
# ─────────────────────────────────────────────────────────────────────────────

def uniform_logging_policy(context: Dict) -> np.ndarray:
    """Each treatment equally likely. Simplest for IPS/DR evaluation."""
    return np.ones(N_TREATMENTS) / N_TREATMENTS


def clinical_logging_policy(context: Dict) -> np.ndarray:
    """
    Realistic clinical policy: biased toward guidelines but imperfect.
    """
    scores = np.ones(N_TREATMENTS) * 1.0

    # Metformin bias for early/mild
    if context["hba1c_baseline"] < 9 and context["egfr"] > 60:
        scores[0] += 3.0
    # GLP-1 bias for obese
    if context["bmi"] > 35:
        scores[1] += 2.5
    # SGLT-2 bias for CVD
    if context["cvd"] == 1:
        scores[2] += 2.5
    # DPP-4 bias for elderly + CKD
    if context["age"] > 60 and (context["ckd"] == 1 or context["egfr"] < 60):
        scores[3] += 2.5
    # Insulin bias for severe
    if context["hba1c_baseline"] > 10 or context["c_peptide"] < 0.6:
        scores[4] += 3.0

    temperature = 1.5
    exp_scores = np.exp(scores / temperature)
    probs = exp_scores / exp_scores.sum()
    return probs


LOGGING_POLICIES = {
    "uniform": uniform_logging_policy,
    "clinical": clinical_logging_policy,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_bandit_dataset(
    n_patients: int = 20000,
    logging_policy: str = "clinical",
    seed: int = 42,
    include_counterfactuals: bool = True,
) -> pd.DataFrame:
    """
    Generate a contextual bandit dataset.

    Returns:
        DataFrame with context features, action, reward, propensity,
        counterfactual rewards, optimal action, and regret.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    policy_fn = LOGGING_POLICIES[logging_policy]
    records = []

    for i in range(n_patients):
        pid = f"P{str(i + 1).zfill(6)}"
        ctx = generate_patient(rng)

        probs = policy_fn(ctx)
        action = int(rng.choice(N_TREATMENTS, p=probs))
        action_name = IDX_TO_TREATMENT[action]
        propensity = float(probs[action])

        observed_reward = reward_oracle(ctx, action_name, noise=True)

        row = {"patient_id": pid, **ctx}
        row["action"] = action
        row["action_name"] = action_name
        row["reward"] = round(observed_reward, 3)
        row["propensity"] = round(propensity, 4)
        row["propensity_all"] = json.dumps([round(float(p), 4) for p in probs])

        if include_counterfactuals:
            potential_rewards = []
            for t_idx, t_name in IDX_TO_TREATMENT.items():
                r = reward_oracle(ctx, t_name, noise=False)
                row[f"reward_{t_idx}"] = round(r, 3)
                potential_rewards.append(r)

            optimal_action = int(np.argmax(potential_rewards))
            optimal_reward = potential_rewards[optimal_action]
            row["optimal_action"] = optimal_action
            row["optimal_action_name"] = IDX_TO_TREATMENT[optimal_action]
            row["optimal_reward"] = round(optimal_reward, 3)
            row["regret"] = round(optimal_reward - reward_oracle(ctx, action_name, noise=False), 3)

        records.append(row)

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print summary statistics to validate the dataset."""
    n = len(df)
    print(f"\n{'=' * 70}")
    print(f"BANDIT DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Patients: {n:,}")
    print(f"  Features: {len(CONTEXT_FEATURES)}")
    print(f"  Treatments: {N_TREATMENTS}")

    print(f"\n── Action distribution (logging policy) ──")
    for i, t in IDX_TO_TREATMENT.items():
        count = (df["action"] == i).sum()
        pct = count / n * 100
        avg_r = df[df["action"] == i]["reward"].mean()
        print(f"  {t:<12} n={count:>5} ({pct:>5.1f}%)  avg_reward={avg_r:.3f}")

    reward_cols = [f"reward_{i}" for i in range(N_TREATMENTS)]
    if all(c in df.columns for c in reward_cols):
        print(f"\n── Expected reward by treatment (counterfactual, no noise) ──")
        for i, t in IDX_TO_TREATMENT.items():
            col = f"reward_{i}"
            print(f"  {t:<12} mean={df[col].mean():.3f}  std={df[col].std():.3f}  "
                  f"min={df[col].min():.3f}  max={df[col].max():.3f}")

        print(f"\n── Optimal action distribution (ground truth) ──")
        for i, t in IDX_TO_TREATMENT.items():
            count = (df["optimal_action"] == i).sum()
            pct = count / n * 100
            print(f"  {t:<12} optimal for {count:>5} patients ({pct:.1f}%)")

        avg_regret = df["regret"].mean()
        print(f"\n  Logging policy avg regret: {avg_regret:.3f}")
        print(f"  Logging policy avg reward: {df['reward'].mean():.3f}")
        print(f"  Oracle avg reward:         {df['optimal_reward'].mean():.3f}")

        cf = df[reward_cols].values
        sorted_cf = np.sort(cf, axis=1)[:, ::-1]
        gap = sorted_cf[:, 0] - sorted_cf[:, 1]
        print(f"\n── Reward separation ──")
        print(f"  Mean gap (best vs 2nd): {gap.mean():.3f}")
        print(f"  Median gap:             {np.median(gap):.3f}")
        print(f"  Patients with gap > 0.5: {(gap > 0.5).mean()*100:.1f}%")
        print(f"  Patients with gap > 1.0: {(gap > 1.0).mean()*100:.1f}%")

    print(f"\n── Feature distributions ──")
    for feat in CONTEXT_FEATURES:
        vals = df[feat]
        print(f"  {feat:<20} mean={vals.mean():>8.2f}  std={vals.std():>7.2f}  "
              f"min={vals.min():>7.1f}  max={vals.max():>7.1f}")

    # Balance check
    print(f"\n── Balance check ──")
    if all(c in df.columns for c in reward_cols):
        opt_counts = df['optimal_action'].value_counts()
        min_pct = opt_counts.min() / n * 100
        max_pct = opt_counts.max() / n * 100
        if min_pct >= 10 and max_pct <= 30:
            print(f"  ✅ BALANCED: all treatments between {min_pct:.1f}% and {max_pct:.1f}%")
        elif min_pct >= 5:
            print(f"  ⚠️  ACCEPTABLE: range {min_pct:.1f}% to {max_pct:.1f}%")
        else:
            print(f"  ❌ IMBALANCED: range {min_pct:.1f}% to {max_pct:.1f}%")

        if gap.mean() > 1.5:
            print(f"  ✅ GOOD SEPARATION: mean gap = {gap.mean():.3f}")
        elif gap.mean() > 0.5:
            print(f"  ⚠️  MODERATE SEPARATION: mean gap = {gap.mean():.3f}")
        else:
            print(f"  ❌ POOR SEPARATION: mean gap = {gap.mean():.3f}")

    print(f"{'=' * 70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Anchor path to project root (two levels up from src/data_generator.py)
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)

    df = generate_bandit_dataset(
        n_patients=20000,
        logging_policy="clinical",
        seed=42,
        include_counterfactuals=True,
    )

    out_path = DATA_DIR / "bandit_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")

    print_dataset_summary(df)
    print("Done. Dataset ready for bandit training.")