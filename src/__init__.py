"""
Diabetes Contextual Bandits — src package

Modules:
    data_generator      - Synthetic data + reward oracle
    feature_engineering  - Preprocessing, scaling, interaction features
    reward_model         - XGBoost reward estimators
    neural_bandit        - PyTorch neural bandits (UCB, Thompson, etc.)
    policies             - Exploration strategies (epsilon-greedy, UCB, etc.)
    online_simulator     - Online learning simulation engine
    evaluation           - Offline policy evaluation
    explainability       - Model decision extraction for LLM explanation
    llm_explain          - Gemini-powered clinical explanation generator
    utils                - Plotting, logging, helpers
"""

# ── Data & Oracle ──
from src.data_generator import (
    reward_oracle,
    generate_patient,
    generate_bandit_dataset,
    TREATMENTS,
    N_TREATMENTS,
    TREATMENT_TO_IDX,
    IDX_TO_TREATMENT,
    CONTEXT_FEATURES,
    REWARD_SCALE,
    REWARD_CAP_PP,
)

# ── Feature Engineering ──
from src.feature_engineering import (
    FeaturePipeline,
    load_and_prepare,
    get_scaled_pipeline,
    get_unscaled_pipeline,
    ALL_FEATURES,
    CONTINUOUS_FEATURES,
    BINARY_FEATURES,
    INTERACTION_FEATURES,
)

# ── Reward Models ──
from src.reward_model import RewardModelEnsemble, RewardModelSingle

# ── Neural Bandits ──
from src.neural_bandit import (
    NeuralGreedy,
    NeuralEpsilon,
    NeuralUCB,
    NeuralThompson,
)

# ── Policies ──
from src.policies import (
    RandomPolicy,
    GreedyPolicy,
    EpsilonGreedyPolicy,
    BoltzmannPolicy,
    UCBPolicy,
    ThompsonPolicy,
    LinUCBPolicy,
    create_policy,
)

# ── Online Simulation ──
from src.online_simulator import (
    OnlineSimulator,
    SimulationAgent,
    make_reward_model_agent,
    make_neural_bandit_agent,
    make_linucb_agent,
    make_random_agent,
    make_oracle_agent,
    quick_compare,
)

# ── Evaluation ──
from src.evaluation import (
    OfflinePolicyEvaluator,
    actions_to_probs,
    softmax_probs,
    counterfactual_policy_value,
    expected_calibration_error,
    reliability_diagram,
    confidence_label_accuracy,
    truncated_ips,
    switch_estimator,
)

# ── Interpretability (Phase 2 — G-7/8/9) ──
from src.interpretability import (
    AttributionEngine,
    integrated_gradients,
    uncertainty_decomposition,
)

# ── Explainability ──
from src.explainability import (
    ExplainabilityExtractor,
    SafetyFinding,
    SEVERITY_CONTRAINDICATION,
    SEVERITY_WARNING,
    RecommendationOverride,
    SubgroupRegret,
    apply_safety_gate,
    get_findings,
    collect_findings,
    build_fairness_report,
    # back-compat wrappers
    check_contraindications,
    check_warnings,
    run_safety_checks,
    check_fairness,
)

# ── LLM Explanation (lazy: don't fail if google-genai is missing) ──
try:
    from src.llm_explain import (
        LLMExplainer,
        LLMClient,
        GeminiClient,
        build_prompt,
    )
except ImportError:  # pragma: no cover — genai may not be installed
    LLMExplainer = None       # type: ignore
    LLMClient = None          # type: ignore
    GeminiClient = None       # type: ignore
    build_prompt = None       # type: ignore

# ── Monitoring (Phase 4) ──
from src.monitoring import (
    DriftMonitor,
    DriftAlert,
    ChampionChallengerResult,
    run_champion_challenger,
)

# ── Utils ──
from src.utils import (
    setup_plotting,
    seed_everything,
    timer,
    save_results,
    ensure_dirs,
    TREATMENT_COLORS,
)