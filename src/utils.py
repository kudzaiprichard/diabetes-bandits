"""
Utility Functions for Diabetes Contextual Bandits

Plotting:
- Learning curves (reward, regret over time)
- Policy comparison bar charts
- Action distribution plots
- Feature importance visualization
- Subgroup heatmaps
- Training loss curves

Helpers:
- Logging setup
- Reproducibility (seed everything)
- Timer context manager
- Results export
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from loguru import logger


def _savefig(fig: plt.Figure, save_path: str) -> None:
    """Save *fig* to *save_path*, creating parent directories if needed."""
    Path(save_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
from contextlib import contextmanager
import time
import os
import json
import sys

from src.data_generator import N_TREATMENTS, IDX_TO_TREATMENT, TREATMENTS

# ─────────────────────────────────────────────────────────────────────────────
# STYLE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TREATMENT_COLORS = {
    "Metformin": "#2196F3",
    "GLP-1": "#4CAF50",
    "SGLT-2": "#FF9800",
    "DPP-4": "#9C27B0",
    "Insulin": "#F44336",
}

AGENT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def setup_plotting(style: str = "seaborn-v0_8-whitegrid", figsize: Tuple = (10, 6)):
    """Configure matplotlib defaults."""
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8")
    plt.rcParams.update({
        "figure.figsize": figsize,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE SIMULATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_regret(
    results: Dict[str, pd.DataFrame],
    title: str = "Cumulative Regret",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cumulative regret curves for all agents."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, df) in enumerate(results.items()):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(df["round"], df["cumulative_regret"], label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    if save_path:
        _savefig(fig, save_path)
        logger.info(f"Saved plot: {save_path}")
    plt.tight_layout()
    return fig


def plot_cumulative_reward(
    results: Dict[str, pd.DataFrame],
    title: str = "Cumulative Reward",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cumulative reward curves for all agents."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, df) in enumerate(results.items()):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(df["round"], df["cumulative_reward"], label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


def plot_learning_curves(
    windowed: Dict[str, pd.DataFrame],
    metric: str = "avg_reward",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot windowed learning curves.

    Args:
        windowed: output from OnlineSimulator.get_windowed_metrics()
        metric: one of "avg_reward", "avg_regret", "accuracy"
    """
    title = title or f"Learning Curve — {metric}"
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, df) in enumerate(windowed.items()):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(df["round"], df[metric], label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Round")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


def plot_regret_and_accuracy(
    windowed: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side regret and accuracy learning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, (name, df) in enumerate(windowed.items()):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax1.plot(df["round"], df["avg_regret"], label=name, color=color, linewidth=1.5)
        ax2.plot(df["round"], df["accuracy"], label=name, color=color, linewidth=1.5)

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Average Regret (windowed)")
    ax1.set_title("Regret Over Time")
    ax1.legend()

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy (windowed)")
    ax2.set_title("Accuracy Over Time")
    ax2.legend()

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# POLICY COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_policy_comparison(
    summary: pd.DataFrame,
    metric: str = "avg_regret",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing policies on a single metric."""
    title = title or f"Policy Comparison — {metric}"
    fig, ax = plt.subplots(figsize=(10, 6))

    agents = summary["agent"].tolist()
    values = summary[metric].tolist()
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(len(agents))]

    bars = ax.barh(agents, values, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


def plot_action_distribution(
    summary: pd.DataFrame,
    title: str = "Action Distribution by Agent",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Stacked bar chart of action distributions per agent."""
    fig, ax = plt.subplots(figsize=(12, 6))

    agents = summary["agent"].tolist()
    treatment_cols = [t for t in TREATMENTS if t in summary.columns]

    if not treatment_cols:
        logger.warning("No treatment columns found in summary")
        return fig

    bottom = np.zeros(len(agents))
    for treatment in treatment_cols:
        values = summary[treatment].values
        color = TREATMENT_COLORS.get(treatment, "#888888")
        ax.barh(agents, values, left=bottom, label=treatment, color=color, edgecolor="white", height=0.6)
        bottom += values

    ax.set_xlabel("Proportion")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# REWARD MODEL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    importances: Dict[str, List[Tuple[str, float]]],
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot feature importances per treatment model."""
    n_treatments = len(importances)
    fig, axes = plt.subplots(1, n_treatments, figsize=(4 * n_treatments, 6), sharey=False)

    if n_treatments == 1:
        axes = [axes]

    for ax, (treatment, features) in zip(axes, importances.items()):
        names = [f[0] for f in features[:top_k]][::-1]
        scores = [f[1] for f in features[:top_k]][::-1]
        color = TREATMENT_COLORS.get(treatment, "#888888")
        ax.barh(names, scores, color=color, edgecolor="white")
        ax.set_title(treatment)
        ax.set_xlabel("Importance")

    fig.suptitle("Feature Importance by Treatment Model", fontsize=14, y=1.02)

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


def plot_predicted_vs_actual(
    predicted: np.ndarray,
    actual: np.ndarray,
    actions: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of predicted vs actual rewards per treatment."""
    fig, axes = plt.subplots(1, N_TREATMENTS, figsize=(4 * N_TREATMENTS, 4), sharey=True)

    for k in range(N_TREATMENTS):
        mask = actions == k
        ax = axes[k]
        color = TREATMENT_COLORS.get(IDX_TO_TREATMENT[k], "#888888")

        ax.scatter(actual[mask], predicted[mask, k], alpha=0.3, s=10, color=color)
        lims = [
            min(actual[mask].min(), predicted[mask, k].min()),
            max(actual[mask].max(), predicted[mask, k].max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
        ax.set_title(IDX_TO_TREATMENT[k])
        ax.set_xlabel("Actual")
        if k == 0:
            ax.set_ylabel("Predicted")

    fig.suptitle("Predicted vs Actual Rewards", fontsize=14, y=1.02)

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


def plot_training_loss(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Neural Bandit Training",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SUBGROUP HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_subgroup_heatmap(
    subgroup_df: pd.DataFrame,
    metric: str = "regret",
    title: str = "Policy Performance by Subgroup",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of policy metric across patient subgroups."""
    fig, ax = plt.subplots(figsize=(10, 6))

    treatment_cols = [c for c in subgroup_df.columns if c.startswith("pct_")]
    if treatment_cols:
        pivot = subgroup_df.set_index("subgroup")[treatment_cols]
        pivot.columns = [c.replace("pct_", "") for c in pivot.columns]
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
        ax.set_title(f"{title} — Treatment Selection by Subgroup")
    else:
        logger.warning("No treatment percentage columns for heatmap")

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# OPE ESTIMATOR COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_ope_comparison(
    eval_results: Dict[str, Dict],
    policy_name: str = "Policy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot OPE estimator values with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 5))

    estimators = []
    values = []
    ci_lows = []
    ci_highs = []

    for est_name, metrics in eval_results.items():
        if isinstance(metrics, dict) and "value" in metrics:
            estimators.append(est_name.upper())
            values.append(metrics["value"])
            ci_lows.append(metrics.get("ci_lower", metrics["value"]))
            ci_highs.append(metrics.get("ci_upper", metrics["value"]))
        elif isinstance(metrics, dict) and "policy_value" in metrics:
            estimators.append(est_name.upper())
            values.append(metrics["policy_value"])
            ci_lows.append(metrics["policy_value"])
            ci_highs.append(metrics["policy_value"])

    y_pos = np.arange(len(estimators))
    errors_low = [v - cl for v, cl in zip(values, ci_lows)]
    errors_high = [ch - v for v, ch in zip(values, ci_highs)]

    ax.barh(y_pos, values, xerr=[errors_low, errors_high],
            color=AGENT_COLORS[:len(estimators)], edgecolor="white",
            height=0.5, capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(estimators)
    ax.set_xlabel("Estimated Policy Value")
    ax.set_title(f"OPE Estimates — {policy_name}")

    if save_path:
        _savefig(fig, save_path)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info(f"Seeds set to {seed}")


@contextmanager
def timer(label: str = ""):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"[TIMER] {label}: {elapsed:.2f}s")


def save_results(
    results: Dict,
    path: str = "results/experiment.json",
) -> None:
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return obj

    cleaned = json.loads(json.dumps(results, default=convert))
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)
    logger.info(f"Saved results to {path}")


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru for the project."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        level=level,
    )
    logger.add(
        "results/experiment.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
        level="DEBUG",
        rotation="10 MB",
    )


def ensure_dirs() -> None:
    """Create project directories if they don't exist."""
    for d in ["data", "data/obp", "models", "results"]:
        os.makedirs(d, exist_ok=True)