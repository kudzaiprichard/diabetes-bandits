"""
Phase 4 / G-23 — Minimal Typer CLI wrapping the main training, simulation,
and explanation entry points. MLflow (G-24) is best-effort: if the module
is missing, runs are logged to a JSON file in ``runs/``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

try:
    import typer
    TYPER_AVAILABLE = True
except ImportError:  # pragma: no cover
    TYPER_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False


def _log_run(metrics: dict, params: dict, tag: str) -> None:
    """Log to MLflow if available; otherwise write a JSON run file."""
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=tag):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
        return
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    out = runs_dir / f"{tag}.json"
    out.write_text(json.dumps({"params": params, "metrics": metrics}, indent=2))
    print(f"Logged run to {out} (MLflow not installed)")


if TYPER_AVAILABLE:
    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command()
    def train(
        data_path: str = "data/bandit_dataset.csv",
        epochs: int = 30,
        hidden_dim: int = 128,
        out: str = "models/neural_thompson.pt",
        counterfactual: bool = True,
        tag: str = "train",
    ) -> None:
        """Train NeuralThompson on the stored bandit dataset."""
        import numpy as np
        import pandas as pd
        from src.data_generator import N_TREATMENTS
        from src.feature_engineering import get_scaled_pipeline
        from src.neural_bandit import NeuralThompson

        df = pd.read_csv(data_path)
        pipe = get_scaled_pipeline()
        X = pipe.fit_transform(df)
        actions = df["action"].astype(int).to_numpy()
        rewards = df["reward"].astype(float).to_numpy()
        cf = None
        if counterfactual:
            cols = [f"reward_{k}" for k in range(N_TREATMENTS)]
            if all(c in df.columns for c in cols):
                cf = df[cols].to_numpy(dtype=float)

        model = NeuralThompson(
            input_dim=X.shape[1], hidden_dims=[hidden_dim, hidden_dim // 2],
        )
        hist = model.train(
            X, actions, rewards, epochs=epochs, counterfactuals=cf, verbose=False
        )
        sigma2 = model.noise_variance_from_residuals(X, actions, rewards)
        model.initialize_posterior(X, actions, rewards)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        model.save(out)
        _log_run(
            metrics={
                "best_val_loss": hist["best_val_loss"],
                "noise_variance": sigma2,
                "epochs_run": hist["epochs_run"],
            },
            params={
                "epochs": epochs, "hidden_dim": hidden_dim,
                "counterfactual": counterfactual, "data_path": data_path,
            },
            tag=tag,
        )
        typer.echo(f"Saved model to {out} (val_loss={hist['best_val_loss']:.4f})")

    @app.command()
    def simulate(
        rounds: int = 5_000,
        data_path: str = "data/bandit_dataset.csv",
        model_path: Optional[str] = None,
        tag: str = "simulate",
    ) -> None:
        """Run an online simulation and log regret."""
        import numpy as np
        import pandas as pd
        from src.data_generator import (
            N_TREATMENTS, IDX_TO_TREATMENT, generate_patient, reward_oracle,
        )
        from src.feature_engineering import get_scaled_pipeline
        from src.neural_bandit import NeuralThompson

        df = pd.read_csv(data_path)
        pipe = get_scaled_pipeline()
        pipe.fit(df)

        model = NeuralThompson(input_dim=pipe.transform(df.head(1)).shape[1])
        if model_path and os.path.exists(model_path):
            model.load(model_path)

        rng = np.random.default_rng(0)
        regrets = []
        for _ in range(rounds):
            ctx = generate_patient(rng)
            x = pipe.transform(pd.DataFrame([ctx]))[0]
            rewards_all = np.array([
                reward_oracle(ctx, IDX_TO_TREATMENT[k], noise=False)
                for k in range(N_TREATMENTS)
            ])
            a = int(model.select_action(x)[0])
            r = float(reward_oracle(ctx, IDX_TO_TREATMENT[a], noise=True))
            regrets.append(rewards_all.max() - rewards_all[a])
            model.online_update(x, a, r)
        _log_run(
            metrics={"mean_regret": float(np.mean(regrets)),
                     "cum_regret": float(np.sum(regrets))},
            params={"rounds": rounds, "model_path": model_path or "fresh"},
            tag=tag,
        )
        typer.echo(f"Simulated {rounds} rounds, mean regret="
                   f"{np.mean(regrets):.3f}")

    @app.command()
    def explain(
        model_path: str,
        patient_json: str,
        api_key: Optional[str] = None,
    ) -> None:
        """Generate a clinical explanation for a single patient."""
        import pandas as pd
        from src.feature_engineering import get_scaled_pipeline
        from src.neural_bandit import NeuralThompson
        from src.explainability import ExplainabilityExtractor

        with open(patient_json) as fh:
            ctx = json.load(fh)

        df = pd.DataFrame([ctx])
        pipe = get_scaled_pipeline()
        pipe.fit(df)
        x = pipe.transform(df)[0]

        model = NeuralThompson(input_dim=x.shape[0])
        model.load(model_path)

        extractor = ExplainabilityExtractor(model)
        payload = extractor.extract(ctx, x)
        typer.echo(json.dumps(payload, indent=2, default=str))

    def main() -> None:
        app()

else:  # pragma: no cover
    def main() -> None:
        raise SystemExit("typer is required for the CLI: pip install typer")


if __name__ == "__main__":
    main()
