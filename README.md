# Diabetes Contextual Bandits

**Personalized Type 2 Diabetes Treatment Selection using Contextual Armed Bandits**

> Given a patient's clinical context (age, BMI, HbA1c, eGFR, comorbidities, etc.), learn which of 5 treatments produces the best outcome — balancing exploration of uncertain options with exploitation of known-good ones.

**Author:** Kudzai P. Matizirofa (R191582L)  
**Date:** 2024–2025

---

## Problem

Selecting the right glucose-lowering medication for a Type 2 Diabetes patient depends on dozens of interacting clinical factors. Standard guidelines provide general rules, but the optimal choice for a specific patient profile is often unclear. This project frames treatment selection as a **contextual bandit** problem, where:

- **Context** = patient features (16 clinical variables + 9 engineered interactions)
- **Actions** = 5 treatments: Metformin, GLP-1 RA, SGLT-2i, DPP-4i, Insulin
- **Reward** = HbA1c reduction (higher is better)
- **Goal** = learn a policy that maximizes expected reward by choosing the best treatment per patient

---

## Approach

The project implements and compares **three families** of contextual bandit algorithms, evaluated both offline (using logged data) and online (via simulation).

### Models

| Family | Model | Key Idea |
|--------|-------|----------|
| **Tree-based** | XGBoost Ensemble | 5 separate reward regressors (one per treatment), greedy/softmax policy on top |
| **Linear online** | LinUCB | Linear reward model with UCB exploration, fully online, no pre-training |
| **VW** | Vowpal Wabbit `cb_explore_adf` | Online linear bandit with softmax, epsilon-greedy, bag, or cover exploration |
| **Neural** | NeuralGreedy | Multi-head deep reward network, pure exploitation |
| **Neural** | NeuralEpsilon | Same network + epsilon-greedy exploration with decay |
| **Neural** | NeuralUCB | Last-layer uncertainty via covariance matrix + UCB score |
| **Neural** | NeuralThompson | Bayesian linear posterior on learned features, Thompson sampling |

### Evaluation

| Method | Type | Description |
|--------|------|-------------|
| **Counterfactual** | Offline | Ground-truth policy value using all 5 potential outcomes per patient (only possible with synthetic data) |
| **IPS / SNIPS** | Offline (OPE) | Inverse propensity scoring — reweights logged data to estimate a new policy's value |
| **DM** | Offline (OPE) | Direct method — uses a reward model to predict counterfactual outcomes |
| **DR** | Offline (OPE) | Doubly robust — combines IPS and DM for lower variance |
| **Online Simulation** | Online | Bandit interacts with reward oracle in real time, tracks cumulative regret |

---

## Project Structure

```
diabetes-bandits/
├── README.md
├── environment.yml              # Conda environment specification
│
├── data/
│   ├── bandit_dataset.csv       # Full dataset (20K patients, counterfactuals, propensities)
│   ├── obp/                     # Numpy arrays for Open Bandit Pipeline
│   │   ├── context.npy
│   │   ├── action.npy
│   │   ├── reward.npy
│   │   ├── pscore.npy
│   │   └── expected_reward.npy
│   └── vw_train.txt             # Vowpal Wabbit format
│
├── src/                         # Python package — all reusable code
│   ├── __init__.py              # Re-exports everything for clean imports
│   ├── data_generator.py        # Synthetic data generator + reward oracle
│   ├── feature_engineering.py   # Preprocessing, scaling, interaction features
│   ├── reward_model.py          # XGBoost reward models (ensemble + single)
│   ├── vw_bandit.py             # Vowpal Wabbit contextual bandit wrapper
│   ├── neural_bandit.py         # PyTorch neural bandits (Greedy/Eps/UCB/Thompson)
│   ├── policies.py              # Policy classes (7 strategies + factory)
│   ├── online_simulator.py      # Online simulation engine
│   ├── evaluation.py            # Offline policy evaluation (OPE + custom)
│   └── utils.py                 # Plotting, logging, helpers
│
├── notebooks/                   # Jupyter experiments (run in order)
│   ├── 01_data_exploration.ipynb
│   ├── 02_reward_modeling.ipynb
│   ├── 03_vw_contextual_bandit.ipynb
│   ├── 04_neural_bandit.ipynb
│   ├── 05_online_simulation.ipynb
│   ├── 06_offline_evaluation.ipynb
│   └── 07_model_comparison.ipynb
│
├── models/                      # Saved model checkpoints
│   ├── reward_model/            # XGBoost ensemble (5 JSON files + meta)
│   ├── vw_bandit.model          # VW model binary
│   ├── neural_greedy.pt         # PyTorch checkpoints
│   ├── neural_epsilon.pt
│   ├── neural_ucb.pt
│   └── neural_thompson.pt
│
└── results/                     # Outputs
    ├── *.png                    # Plots (regret curves, heatmaps, comparisons)
    ├── *.csv                    # Per-round online simulation data
    └── *.json                   # Evaluation metrics and experiment logs
```

---

## Source Modules

### `data_generator.py`
Generates 20,000 synthetic T2D patients with clinically grounded reward functions. Key components:
- **`reward_oracle(context, treatment, noise)`** — the ground-truth reward function that bandits try to learn. Returns expected HbA1c reduction based on patient-treatment match. Callable with `noise=True` for stochastic simulation or `noise=False` for expected values.
- **`generate_patient(rng)`** — samples a realistic patient context with correlated features (e.g., eGFR declines with age, C-peptide drops with disease duration).
- **`generate_bandit_dataset()`** — produces the full dataset with a clinical logging policy, propensity scores, counterfactual rewards for all 5 treatments, and oracle-optimal actions.
- **Exports** to CSV, OBP numpy format, and VW text format.

### `feature_engineering.py`
`FeaturePipeline` class that handles the full preprocessing chain:
- 12 continuous features + 4 binary comorbidity flags
- 9 clinically motivated interaction features (e.g., `bmi_x_nafld`, `severity_score`, `tg_hdl_ratio`)
- Optional `StandardScaler` (enabled for neural models, disabled for trees)
- Train/test splitting with stratification, preserving counterfactuals and propensities in metadata

### `reward_model.py`
Two XGBoost approaches for estimating `E[reward | context, treatment]`:
- **`RewardModelEnsemble`** — trains 5 independent XGBoost regressors, one per treatment. Recommended approach: each model specializes in its treatment's feature interactions.
- **`RewardModelSingle`** — single XGBoost with one-hot treatment encoding. Simpler but typically underperforms.
- Both support: prediction for all treatments, greedy action selection, evaluation against oracle, feature importance extraction, save/load.

### `vw_bandit.py`
Wraps Vowpal Wabbit's `--cb_explore_adf` (action-dependent features):
- 5 exploration strategies: epsilon-greedy, softmax, bag, cover, squareCB
- Formats patient contexts and treatment actions into VW's multi-line example format
- Supports both online learning (`learn_one`) and batch training (`train_batch`)
- `vw_exploration_sweep()` — grid search across 11 strategy configurations

### `neural_bandit.py`
PyTorch neural contextual bandits sharing a common `RewardNetwork`:
- **Architecture:** shared backbone (configurable hidden layers + BatchNorm + Dropout) → 5 per-treatment output heads
- **NeuralGreedy** — pure exploitation baseline
- **NeuralEpsilon** — epsilon-greedy with configurable decay schedule
- **NeuralUCB** — maintains per-treatment covariance matrices on last-layer features, selects via UCB score. Uses Sherman-Morrison rank-1 updates for efficient online covariance tracking.
- **NeuralThompson** — maintains Bayesian linear posterior on last-layer features, samples from posterior to select actions. Based on the Neural Thompson Sampling paper (Zhang et al., 2021).

### `policies.py`
Standalone policy strategies that work with any reward estimator:
- **RandomPolicy** — uniform random baseline
- **GreedyPolicy** — always picks highest estimated reward
- **EpsilonGreedyPolicy** — epsilon-greedy with configurable decay
- **BoltzmannPolicy** — softmax with temperature annealing
- **UCBPolicy** — requires uncertainty estimates alongside reward predictions
- **ThompsonPolicy** — Gaussian posterior Thompson sampling (non-neural)
- **LinUCBPolicy** — disjoint linear UCB, fully online with Sherman-Morrison updates
- `create_policy(name)` — factory function

### `online_simulator.py`
Runs the full online contextual bandit loop:
1. Sample a patient from `generate_patient()`
2. Each registered agent selects a treatment
3. Reward observed from `reward_oracle()` (with noise)
4. Each agent updates its model
5. Track reward, regret, accuracy per round

Agent builders for every model type: `make_reward_model_agent`, `make_neural_bandit_agent`, `make_vw_agent`, `make_linucb_agent`, `make_random_agent`, `make_oracle_agent`. All agents run on the same patient stream for fair comparison. Provides windowed metrics for learning curve visualization.

### `evaluation.py`
Offline policy evaluation without interacting with the environment:
- **Custom estimators** (always available): IPS, SNIPS, DM, DR with confidence intervals
- **OBP integration** (when installed): wraps Open Bandit Pipeline's estimators
- **Counterfactual evaluation**: exact policy value using ground-truth potential outcomes
- **`OfflinePolicyEvaluator`** — main class that brings all estimators together
- **`compare_policies()`** — side-by-side table of multiple policies
- **`subgroup_analysis()`** — per-subgroup evaluation (e.g., by BMI group, age group)
- **`statistical_test()`** — bootstrap + paired t-test for pairwise policy comparison

### `utils.py`
Plotting and project helpers:
- 10+ plot functions: cumulative regret/reward curves, learning curves, policy comparison bars, action distribution stacked bars, feature importance, predicted vs actual scatter, training loss, OPE estimator comparison with CI, subgroup heatmaps
- `seed_everything(42)` — reproducibility across numpy, random, torch
- `timer("label")` — context manager for timing code blocks
- `save_results()` — JSON export with numpy type conversion
- Color schemes for treatments and agents

---

## Notebooks

Run notebooks **in order** from the `notebooks/` directory. Each notebook imports from `src/` and builds on previous results.

### `01_data_exploration.ipynb`
Verifies dataset quality before any modeling.
- Feature distributions, correlations, missing values
- Logging policy action distribution and propensity score validation
- Counterfactual reward analysis: which treatments work best for which patients?
- Oracle optimal action distribution: what the perfect policy would choose
- Reward gap analysis: how separable are the best treatments per patient?
- Dataset quality checklist (16 cells)

### `02_reward_modeling.ipynb`
Trains and evaluates XGBoost reward models.
- Ensemble (5 models) vs single model comparison
- Per-treatment RMSE, predicted vs actual plots
- Feature importance per treatment model
- Greedy policy evaluation against oracle (confusion matrix)
- Hyperparameter sensitivity: max_depth, n_estimators sweeps
- Subgroup accuracy analysis (BMI × HbA1c heatmap)
- Saves best model to `models/reward_model/` (17 cells)

### `03_vw_contextual_bandit.ipynb`
Trains Vowpal Wabbit contextual bandits with multiple exploration strategies.
- Softmax, epsilon-greedy, bag, cover trained and evaluated
- Full exploration sweep across 11 configurations
- Learning rate and number-of-passes sensitivity
- Action probability inspection (how VW distributes probability mass)
- Saves best VW model (16 cells)

### `04_neural_bandit.ipynb`
Trains all 4 PyTorch neural bandit variants.
- NeuralGreedy, NeuralEpsilon, NeuralUCB, NeuralThompson
- Training curves, action distributions, side-by-side comparison
- Architecture sensitivity sweep (7 layer configurations)
- NeuralUCB alpha sweep, NeuralThompson noise variance sweep
- Reward prediction quality vs oracle (scatter + correlation)
- Saves all 4 models (17 cells)

### `05_online_simulation.ipynb`
Simulates online learning with exploration vs exploitation.
- 9 agents compared on the same patient stream (10K and 50K rounds)
- Agents: Random, Oracle, XGB+Greedy, XGB+EpsGreedy, XGB+Boltzmann, LinUCB, VW, NeuralUCB, NeuralThompson
- Cumulative regret and reward curves
- Windowed learning curves (accuracy and regret over time)
- Action distribution shift visualization (exploration → exploitation)
- Converged performance analysis (last 2000 rounds)
- Extended 50K-round simulation for long-term behavior (19 cells)

### `06_offline_evaluation.ipynb`
Formal offline policy evaluation using OPE estimators.
- 8 policies × 5 estimators (IPS, SNIPS, DM, DR, Counterfactual)
- Estimator agreement heatmap and correlation analysis
- Pairwise statistical significance tests (bootstrap + paired t-test, p-value matrix)
- Subgroup analysis: BMI, HbA1c, Age groups with treatment selection heatmaps
- Estimator reliability: bias check against counterfactual ground truth
- Identifies most reliable OPE estimator (19 cells)

### `07_model_comparison.ipynb`
Final head-to-head comparison bringing everything together.
- All 7 models trained and evaluated offline on the same test set
- Master bar charts: policy value, regret, accuracy
- Per-treatment accuracy heatmap across all models
- Confusion matrices for top 3 models
- 20K-round online simulation with all agents
- Offline vs online ranking comparison (do they agree?)
- Subgroup accuracy heatmap: all models × 8 clinical subgroups
- Relative efficiency chart (% of oracle performance)
- Strengths/weaknesses table per model
- Final recommendation with winner (19 cells)

---

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate diabetes-bandits
```

If `vowpalwabbit` fails during env creation, install it separately:

```bash
pip install vowpalwabbit obp torch loguru
```

### 2. Generate the dataset

```bash
cd diabetes-bandits
python -m src.data_generator
```

This creates `data/bandit_dataset.csv`, `data/obp/`, and `data/vw_train.txt`.

### 3. Run the notebooks

```bash
jupyter lab
```

Open notebooks in order: `01` → `02` → ... → `07`. Each notebook is self-contained but builds on models/insights from previous ones.

### 4. Using the modules directly

```python
# Clean imports via __init__.py
from src import reward_oracle, FeaturePipeline, NeuralUCB, OnlineSimulator

# Train a neural bandit
pipe = FeaturePipeline(scale=True, add_interactions=True)
X_train, X_test, meta = pipe.fit_transform_split(df)

bandit = NeuralUCB(input_dim=X_train.shape[1], hidden_dims=[128, 64], alpha=1.0)
bandit.train(X_train, meta['a_train'], meta['y_train'], epochs=60)
```

---

## Synthetic Data

The dataset contains **20,000 patients** with:

| Category | Features |
|----------|----------|
| **Demographics** | age, gender, ethnicity |
| **Metabolic** | BMI, HbA1c baseline, fasting glucose, C-peptide |
| **Renal** | eGFR |
| **Cardiovascular** | BP systolic, LDL, HDL, triglycerides |
| **Comorbidities** | CVD, CKD, NAFLD, hypertension |
| **Engineered** | 9 interaction features (e.g., `bmi_x_nafld`, `severity_score`) |

Each patient has:
- An **observed treatment** selected by a clinical logging policy (biased toward guidelines)
- A **propensity score** for the observed treatment
- **Counterfactual rewards** for all 5 treatments (ground truth, no noise)
- The **oracle-optimal treatment** and associated **regret**

The reward function encodes realistic clinical relationships: Metformin works best for early-stage, lean patients with good kidney function; GLP-1 excels for obese patients with NAFLD; SGLT-2 is optimal for cardiovascular disease; DPP-4 suits elderly patients with CKD; Insulin is needed for severe, advanced disease with beta-cell failure.

---

## Key Results

Results are generated by running the notebooks. Expected findings:

- **XGBoost Ensemble** provides a strong offline baseline due to its accurate reward predictions
- **NeuralUCB and NeuralThompson** achieve the best online performance by balancing exploration and exploitation
- **LinUCB** learns effectively from scratch (no pre-training) but converges more slowly
- **VW** is fast to train and competitive, especially with softmax or bag exploration
- **Exploration matters**: greedy policies plateau while UCB/Thompson continue improving
- **OPE estimators**: DR has the lowest bias; SNIPS is more stable than raw IPS
- **Subgroup analysis** reveals that models struggle most with borderline patients where multiple treatments have similar expected rewards

---

## Tools and Packages

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Runtime |
| PyTorch | latest | Neural contextual bandits |
| XGBoost | latest | Reward modeling |
| Vowpal Wabbit | latest | Online contextual bandits |
| Open Bandit Pipeline | latest | Offline policy evaluation |
| scikit-learn | latest | Preprocessing, metrics |
| pandas / numpy | latest | Data manipulation |
| matplotlib / seaborn | latest | Visualization |
| loguru | latest | Logging |

---

## References

- Zhou et al., *"Neural Contextual Bandits with UCB-based Exploration"*, ICLR 2020
- Zhang et al., *"Neural Thompson Sampling"*, ICLR 2021
- Li et al., *"A Contextual-Bandit Approach to Personalized News Article Recommendation"*, WWW 2010 (LinUCB)
- Saito et al., *"Open Bandit Pipeline"*, 2021
- Agarwal et al., *"A Reductions Approach to Fair and Robust Contextual Bandits"*, ICML 2018
- ADA Standards of Medical Care in Diabetes, 2024
- Vowpal Wabbit documentation: https://vowpalwabbit.org/

---

## License

This project is for academic and research purposes.