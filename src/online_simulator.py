"""
Online Simulation Engine for Diabetes Contextual Bandits

Simulates the full online learning loop:
    1. Patient arrives (context sampled)
    2. Bandit selects treatment (exploration vs exploitation)
    3. Reward observed from oracle
    4. Model updated
    5. Track cumulative regret, reward, accuracy over time

Supports:
- Any combination of reward model + policy
- Neural bandits with online updates
- LinUCB purely online
- Side-by-side comparison of multiple agents
- Windowed metrics for learning curve analysis

Usage:
    from src.online_simulator import OnlineSimulator, SimulationAgent
    sim = OnlineSimulator(n_rounds=10000, pipeline=scaled_pipe, unscaled_pipeline=unscaled_pipe)
    sim.add_agent("neural_ucb", model=neural_ucb_model, pipeline=pipe)
    sim.add_agent("linucb", policy=linucb_policy, pipeline=pipe)
    results = sim.run()
    sim.plot_results()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from loguru import logger
import time

from src.data_generator import (
    reward_oracle, generate_patient, TREATMENTS,
    N_TREATMENTS, IDX_TO_TREATMENT, TREATMENT_TO_IDX,
)
from src.feature_engineering import FeaturePipeline


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION AGENT (wrapper for any bandit approach)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationAgent:
    """
    Wraps any bandit model/policy for use in the simulator.

    Must provide either:
        - select_fn: callable(x_features) → action_index
        - update_fn: callable(x_features, action, reward) → None

    Or provide a model object with select_action() and online_update() methods.

    Args:
        use_scaled: if True, agent receives scaled features (neural models).
                    if False, agent receives unscaled features (tree-based models).
    """
    name: str
    select_fn: Optional[Callable] = None
    update_fn: Optional[Callable] = None
    use_scaled: bool = True  # False for tree-based (XGBoost) agents

    # Tracked metrics
    rewards: List[float] = field(default_factory=list)
    regrets: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    optimal_actions: List[int] = field(default_factory=list)
    cumulative_reward: float = 0.0
    cumulative_regret: float = 0.0

    def select_action(self, x: np.ndarray) -> int:
        if self.select_fn is None:
            raise ValueError(f"Agent '{self.name}' has no select_fn")
        return self.select_fn(x)

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        if self.update_fn is not None:
            self.update_fn(x, action, reward)

    def record(self, action: int, reward: float, optimal_reward: float, optimal_action: int) -> None:
        self.actions.append(action)
        self.rewards.append(reward)
        self.optimal_actions.append(optimal_action)

        regret = optimal_reward - reward
        self.regrets.append(regret)
        self.cumulative_reward += reward
        self.cumulative_regret += regret

    def reset(self) -> None:
        self.rewards = []
        self.regrets = []
        self.actions = []
        self.optimal_actions = []
        self.cumulative_reward = 0.0
        self.cumulative_regret = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# AGENT BUILDERS (convenience functions)
# ─────────────────────────────────────────────────────────────────────────────

def make_reward_model_agent(
    name: str,
    reward_model,
    policy,
    pipeline: FeaturePipeline,
) -> SimulationAgent:
    """
    Create agent from a reward model (XGBoost) + policy.

    Sets use_scaled=False so the simulator passes unscaled features,
    matching the features the XGBoost model was trained on.
    """
    def select_fn(x: np.ndarray) -> int:
        rewards = reward_model.predict_all(x.reshape(1, -1)).flatten()
        return policy.select_action(rewards)

    def update_fn(x: np.ndarray, action: int, reward: float) -> None:
        policy.update(action, reward)

    return SimulationAgent(name=name, select_fn=select_fn, update_fn=update_fn, use_scaled=False)


def make_neural_bandit_agent(
    name: str,
    neural_bandit,
    pipeline: FeaturePipeline,
) -> SimulationAgent:
    """
    Create agent from a neural bandit (NeuralUCB, NeuralThompson, etc).

    Sets use_scaled=True so the simulator passes scaled features,
    matching the features the neural model was trained on.
    """
    def select_fn(x: np.ndarray) -> int:
        action, _ = neural_bandit.select_action(x)
        return action

    def update_fn(x: np.ndarray, action: int, reward: float) -> None:
        if hasattr(neural_bandit, "online_update"):
            neural_bandit.online_update(x, action, reward)

    return SimulationAgent(name=name, select_fn=select_fn, update_fn=update_fn, use_scaled=True)


def make_linucb_agent(
    name: str,
    linucb_policy,
) -> SimulationAgent:
    """
    Create agent from LinUCB policy (fully online, no pretrained model).

    Uses scaled features for better linear model performance.
    """
    def select_fn(x: np.ndarray) -> int:
        return linucb_policy.select_action(np.zeros(N_TREATMENTS), x=x)

    def update_fn(x: np.ndarray, action: int, reward: float) -> None:
        linucb_policy.update_model(x, action, reward)

    return SimulationAgent(name=name, select_fn=select_fn, update_fn=update_fn, use_scaled=True)


def make_random_agent() -> SimulationAgent:
    """Uniform random baseline agent."""
    def select_fn(x: np.ndarray) -> int:
        return np.random.randint(N_TREATMENTS)

    return SimulationAgent(name="random", select_fn=select_fn, update_fn=None, use_scaled=True)


def make_oracle_agent() -> SimulationAgent:
    """
    Oracle agent that always picks the best treatment.
    Used as upper bound reference — requires context dict (set during sim).
    """
    _current_context = {}

    def set_context(ctx: Dict):
        _current_context.update(ctx)

    def select_fn(x: np.ndarray) -> int:
        rewards = [reward_oracle(_current_context, t, noise=False) for t in TREATMENTS]
        return int(np.argmax(rewards))

    agent = SimulationAgent(name="oracle", select_fn=select_fn, update_fn=None, use_scaled=True)
    agent._set_context = set_context
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class OnlineSimulator:
    """
    Runs the online contextual bandit simulation.

    Each round:
        1. Sample a patient context
        2. Transform context to both scaled and unscaled features
        3. Each agent selects an action using the correct feature version
        4. Reward is observed from the oracle
        5. Each agent updates its model
        6. Metrics are recorded

    Args:
        pipeline: scaled FeaturePipeline (scale=True) for neural/LinUCB agents
        unscaled_pipeline: unscaled FeaturePipeline (scale=False) for XGBoost agents.
                           If None, falls back to pipeline for all agents.
    """

    def __init__(
        self,
        n_rounds: int = 10000,
        pipeline: Optional[FeaturePipeline] = None,
        unscaled_pipeline: Optional[FeaturePipeline] = None,
        seed: int = 42,
        log_interval: int = 1000,
    ):
        self.n_rounds = n_rounds
        self.pipeline = pipeline
        self.unscaled_pipeline = unscaled_pipeline
        self.seed = seed
        self.log_interval = log_interval

        self.agents: Dict[str, SimulationAgent] = {}
        self._contexts: List[Dict] = []
        self._rng = np.random.RandomState(seed)

    def add_agent(self, agent: SimulationAgent) -> None:
        """Register an agent for simulation."""
        self.agents[agent.name] = agent
        logger.info(f"Added agent: {agent.name}")

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run the full simulation.

        Returns:
            Dict mapping agent_name → DataFrame of per-round metrics
        """
        if not self.agents:
            raise ValueError("No agents added. Call add_agent() first.")
        if self.pipeline is None:
            raise ValueError("Pipeline required. Pass pipeline to constructor.")

        np.random.seed(self.seed)
        rng = self._rng

        # Reset all agents
        for agent in self.agents.values():
            agent.reset()

        logger.info(f"Starting simulation: {self.n_rounds} rounds, {len(self.agents)} agents")
        start_time = time.time()

        for t in range(self.n_rounds):
            # ── 1. Sample patient ──
            context = generate_patient(rng)
            self._contexts.append(context)

            # ── 2. Compute oracle optimal ──
            oracle_rewards = [reward_oracle(context, tr, noise=False) for tr in TREATMENTS]
            optimal_action = int(np.argmax(oracle_rewards))
            optimal_reward = oracle_rewards[optimal_action]

            # ── 3. Transform context to scaled and unscaled features ──
            x_scaled = self.pipeline.transform_single(context)
            x_unscaled = (
                self.unscaled_pipeline.transform_single(context)
                if self.unscaled_pipeline is not None
                else x_scaled
            )

            # ── 4. Each agent acts ──
            for name, agent in self.agents.items():
                # Set context for oracle agent if applicable
                if hasattr(agent, "_set_context"):
                    agent._set_context(context)

                # Route correct feature version based on agent type
                x = x_scaled if agent.use_scaled else x_unscaled

                action = agent.select_action(x)

                # ── 5. Observe reward ──
                reward = reward_oracle(context, IDX_TO_TREATMENT[action], noise=True)

                # ── 6. Update agent ──
                agent.update(x, action, reward)

                # ── 7. Record metrics ──
                agent.record(action, reward, optimal_reward, optimal_action)

            # ── Logging ──
            if (t + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                lines = [f"\n  Round {t + 1}/{self.n_rounds} ({elapsed:.1f}s)"]
                for name, agent in self.agents.items():
                    avg_r = np.mean(agent.rewards[-self.log_interval:])
                    avg_reg = np.mean(agent.regrets[-self.log_interval:])
                    acc = np.mean(
                        np.array(agent.actions[-self.log_interval:])
                        == np.array(agent.optimal_actions[-self.log_interval:])
                    )
                    lines.append(
                        f"    {name:<25} "
                        f"avg_reward={avg_r:.3f}  "
                        f"avg_regret={avg_reg:.3f}  "
                        f"accuracy={acc:.3f}  "
                        f"cum_regret={agent.cumulative_regret:.1f}"
                    )
                logger.info("\n".join(lines))

        elapsed = time.time() - start_time
        logger.info(f"Simulation complete in {elapsed:.1f}s")

        # Build result DataFrames
        results = {}
        for name, agent in self.agents.items():
            results[name] = pd.DataFrame({
                "round": np.arange(1, self.n_rounds + 1),
                "action": agent.actions,
                "reward": agent.rewards,
                "regret": agent.regrets,
                "optimal_action": agent.optimal_actions,
                "correct": np.array(agent.actions) == np.array(agent.optimal_actions),
                "cumulative_reward": np.cumsum(agent.rewards),
                "cumulative_regret": np.cumsum(agent.regrets),
            })

        return results

    def get_summary(self) -> pd.DataFrame:
        """Summary statistics across all agents."""
        rows = []
        for name, agent in self.agents.items():
            n = len(agent.rewards)
            if n == 0:
                continue

            actions = np.array(agent.actions)
            action_dist = {
                IDX_TO_TREATMENT[k]: round((actions == k).mean(), 4)
                for k in range(N_TREATMENTS)
            }

            rows.append({
                "agent": name,
                "total_reward": round(agent.cumulative_reward, 2),
                "total_regret": round(agent.cumulative_regret, 2),
                "avg_reward": round(np.mean(agent.rewards), 4),
                "avg_regret": round(np.mean(agent.regrets), 4),
                "accuracy": round(
                    (np.array(agent.actions) == np.array(agent.optimal_actions)).mean(), 4
                ),
                **action_dist,
            })

        return pd.DataFrame(rows).sort_values("total_regret")

    def get_windowed_metrics(
        self, window: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling-window metrics for learning curve plots.

        Returns dict mapping agent_name → DataFrame with columns:
            round, window_avg_reward, window_avg_regret, window_accuracy
        """
        windowed = {}
        for name, agent in self.agents.items():
            rewards = np.array(agent.rewards)
            regrets = np.array(agent.regrets)
            correct = np.array(agent.actions) == np.array(agent.optimal_actions)

            n = len(rewards)
            if n < window:
                continue

            rounds = []
            avg_rewards = []
            avg_regrets = []
            accuracies = []

            for i in range(window, n + 1, window // 5):
                start = max(0, i - window)
                rounds.append(i)
                avg_rewards.append(rewards[start:i].mean())
                avg_regrets.append(regrets[start:i].mean())
                accuracies.append(correct[start:i].mean())

            windowed[name] = pd.DataFrame({
                "round": rounds,
                "avg_reward": avg_rewards,
                "avg_regret": avg_regrets,
                "accuracy": accuracies,
            })

        return windowed


# ─────────────────────────────────────────────────────────────────────────────
# QUICK COMPARISON (convenience function)
# ─────────────────────────────────────────────────────────────────────────────

def quick_compare(
    agents: List[SimulationAgent],
    pipeline: FeaturePipeline,
    unscaled_pipeline: Optional[FeaturePipeline] = None,
    n_rounds: int = 10000,
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Quick side-by-side comparison of multiple agents.

    Returns:
        (per_round_results, summary_df)
    """
    sim = OnlineSimulator(
        n_rounds=n_rounds,
        pipeline=pipeline,
        unscaled_pipeline=unscaled_pipeline,
        seed=seed,
    )
    for agent in agents:
        sim.add_agent(agent)

    results = sim.run()
    summary = sim.get_summary()
    return results, summary