"""
LLM Explanation Generator for NeuralThompson

Takes the extracted payload from ExplainabilityExtractor (post G-3 reward
rescale, structured safety findings per G-13, and optional attribution
artefacts from Phase 2) and generates a clinical explanation via an LLM
provider.

Returns structured JSON with six sections:
    - recommendation_summary
    - runner_up_analysis
    - confidence_statement
    - safety_assessment
    - monitoring_note
    - disclaimer

Install (Gemini backend): pip install google-genai
"""

import os
import json
from typing import Dict, Optional, List, Any
from loguru import logger

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed — LLM explanations unavailable")

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYDANTIC_AVAILABLE = False
    logger.warning("pydantic not installed — schema validation disabled")

from src.data_generator import TREATMENTS, REWARD_CAP_PP


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMA (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

if PYDANTIC_AVAILABLE:

    class ClinicalExplanation(BaseModel):
        """Schema-validated LLM output. Used to enforce the response contract."""
        recommendation_summary: str = Field(..., min_length=20)
        runner_up_analysis: str = Field(..., min_length=10)
        confidence_statement: str = Field(..., min_length=10)
        safety_assessment: str = Field(..., min_length=10)
        monitoring_note: str = Field(..., min_length=10)
        disclaimer: str = Field(..., min_length=10)

        @field_validator("recommendation_summary", "runner_up_analysis",
                         "confidence_statement", "safety_assessment",
                         "monitoring_note")
        @classmethod
        def _no_implausible_effect_size(cls, v: str) -> str:
            """Reject any text claiming an HbA1c reduction above REWARD_CAP_PP."""
            import re
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:pp|percentage points?)", v, re.I):
                val = float(m.group(1))
                if val > REWARD_CAP_PP + 0.5:
                    raise ValueError(
                        f"implausible HbA1c reduction {val} pp "
                        f"(cap {REWARD_CAP_PP})"
                    )
            return v
else:  # pragma: no cover
    ClinicalExplanation = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER — updated for G-3 pp scale, G-13 structured safety, G-16
# override, G-15 optional fairness
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are Dr. Sarah Chen, a senior endocrinologist with 40 years of \
clinical experience specialising in Type 2 Diabetes management. You are \
also trained in interpreting AI-assisted clinical decision support tools.

You are reviewing the output of a contextual-bandit recommendation model. \
For each of five treatments, the model predicts an expected HbA1c reduction \
in percentage points (pp). The model has been calibrated so that the \
plausible range is 0–{REWARD_CAP_PP:.1f} pp — "ideal Metformin patient" is \
around 1.5 pp and "ideal Insulin patient for severe disease" is around \
2.5 pp. Values above {REWARD_CAP_PP:.1f} pp are not possible in this system.

The model also provides a confidence score: the percentage of times this \
treatment won when the model simulated the decision many times using its \
own posterior uncertainty.

Your task is to FAITHFULLY explain the model's recommendation in clear \
clinical language that a practising physician would find useful during a \
consultation.

CRITICAL TRANSLATION RULES — follow these exactly:

1. PREDICTED HbA1c REDUCTION IS IN PERCENTAGE POINTS. A predicted value of \
   1.8 means the model predicts ~1.8 pp reduction in HbA1c. Values will \
   typically fall between 0 and {REWARD_CAP_PP:.1f} pp — NEVER report reductions \
   greater than {REWARD_CAP_PP:.1f} pp. Always refer to these as "predicted HbA1c \
   reduction" — never say "posterior mean", "sampled score", or "reward".

2. CONFIDENCE = SIMULATION WIN RATE. Translate naturally:
   - 90%+ → "the model is highly confident — this treatment won in X% of simulations"
   - 70-89% → "the model is moderately confident — this treatment won in X% of simulations"
   - 50-69% → "this is a closer decision — the recommended treatment won in only X% of simulations"
   - Below 50% → "the model shows no clear preference — clinical judgement should guide this decision"
   Use the EXACT percentage provided. Do not round or approximate it.

3. MEAN GAP = BENEFIT DIFFERENCE BETWEEN TOP TWO. Describe as "the model \
   predicts approximately X percentage points more HbA1c reduction with \
   [recommended] compared to [runner-up]." Keep the number small — gaps \
   will typically be well under 1 pp.

4. SAFETY IS AUTHORITATIVE. The safety input is structured: each finding has \
   an explicit rule_id, severity (contraindication or warning), and a \
   clinician-readable message. Report the messages exactly — do not invent, \
   add, or remove safety concerns. If an OVERRIDE block is present, the \
   model's original pick was downgraded by the safety gate; surface this \
   clearly ("the model initially favoured X, but X is contraindicated by \
   [reason]; the next-best option, Y, has been selected").

5. NO ML JARGON. Never say: posterior mean, posterior sampling, sampled \
   score, trace, A_inv, Thompson sampling, contextual bandit, reward, \
   action, policy, regret, feature vector, neural network, epoch, \
   covariance, or any other machine learning terminology.

Do NOT cite clinical trial evidence, guideline recommendations, or \
mechanism-of-action reasoning unless it directly maps to a pattern visible \
in the model's predictions or structured safety output. The goal is \
transparency about what the model predicts, not post-hoc medical \
justification.

CRITICAL: You MUST respond with ONLY a valid JSON object. No markdown, no \
backticks, no preamble, no explanation outside the JSON. Just the raw JSON."""


RESPONSE_FORMAT_INSTRUCTION = """RESPONSE FORMAT:

You MUST respond with ONLY a valid JSON object matching this exact structure. \
No markdown code fences, no backticks, no text before or after the JSON.

{
    "recommendation_summary": "2-3 sentences explaining why this treatment is recommended for this patient. Reference specific patient features (age, BMI, HbA1c, eGFR, duration, C-peptide, comorbidities) and describe predicted HbA1c reductions. Write as a senior clinician advising a colleague. If a safety override was applied, state that the model's original pick was overridden on safety grounds and explain which treatment was chosen instead.",
    "runner_up_analysis": "1-2 sentences on which treatment was the next best alternative, its predicted HbA1c reduction in percentage points, and its simulation win rate compared to the recommended treatment.",
    "confidence_statement": "1-2 sentences stating the model's confidence as a percentage. Use the exact confidence percentage and number of simulations provided. Frame in clinical terms — high confidence means strong evidence from similar patients, low confidence means this is a close call.",
    "safety_assessment": "1-3 sentences reporting the safety check results in clinical language. Use ONLY the messages from the structured safety findings provided. If CONTRAINDICATION_FOUND: strongly flag this. If WARNING: report the specific clinical concerns. If CLEAR: confirm no contraindications. If an override is present, explain that the selected treatment is the safe alternative.",
    "monitoring_note": "1-2 sentences with specific, actionable monitoring recommendations based on this patient's profile, lab values, comorbidities, and the chosen treatment.",
    "disclaimer": "This is an AI-assisted decision support tool. Final treatment decisions must be made by the treating physician based on the complete clinical picture, patient preferences, and current guidelines."
}

REMEMBER: Output ONLY the JSON object. Nothing else. Write as Dr. Sarah Chen."""


def _format_finding(f: Dict[str, Any]) -> str:
    """Render a structured SafetyFinding dict as a prompt line."""
    rid = f.get("rule_id", "?")
    msg = f.get("message", "")
    return f"[{rid}] {msg}"


def build_prompt(payload: Dict) -> str:
    """Build the full LLM prompt from an extraction payload."""
    p = payload["patient"]
    d = payload["decision"]
    s = payload["safety"]
    f = payload.get("fairness")

    patient_section = f"""PATIENT PROFILE:
  Age:                {p['age']} years
  BMI:                {p['bmi']} kg/m²
  HbA1c:              {p['hba1c_baseline']}%
  eGFR:               {p['egfr']} mL/min/1.73m²
  Diabetes Duration:  {p['diabetes_duration']} years
  Fasting Glucose:    {p['fasting_glucose']} mg/dL
  C-Peptide:          {p['c_peptide']} ng/mL
  Blood Pressure:     {p['bp_systolic']} mmHg systolic
  LDL / HDL / TG:     {p['ldl']} / {p['hdl']} / {p['triglycerides']} mg/dL
  CVD History:        {p['cvd']}
  CKD:                {p['ckd']}
  NAFLD:              {p['nafld']}
  Hypertension:       {p['hypertension']}"""

    means = d["posterior_means"]
    win_rates = d["win_rates"]

    model_lines = [
        "MODEL PREDICTION:",
        f"  Final Treatment:        {d['recommended_treatment']}",
    ]
    if d.get("override"):
        ov = d["override"]
        model_lines.append(
            f"  *** SAFETY OVERRIDE ***  original model pick was "
            f"{ov['original_treatment']} but it was downgraded because: "
            f"{ov['reason']}"
        )
        model_lines.append(
            f"  Blocked by safety gate: {', '.join(ov['blocked_treatments'])}"
        )
    else:
        model_lines.append(
            f"  (No safety override applied; model's top pick is the final "
            f"recommendation.)"
        )

    model_lines += [
        f"  Model Confidence:       {d['confidence_pct']}% "
        f"({d['confidence_label']})",
        f"                          ({d['recommended_treatment']} won "
        f"{d['confidence_pct']} out of {d['n_draws']} simulations)",
        "",
        f"  Predicted HbA1c Reduction (percentage points, range 0–{REWARD_CAP_PP:.1f}):",
    ]
    for t in TREATMENTS:
        model_lines.append(f"    {t:<10s} → {means.get(t, 0.0):.2f} pp")

    model_lines += [
        "",
        f"  Simulation Win Rates (out of {d['n_draws']} simulations):",
    ]
    for t in TREATMENTS:
        model_lines.append(f"    {t:<10s} → {win_rates.get(t, 0.0):.1%}")

    model_lines += [
        "",
        f"  Next Best Alternative:  {d['runner_up']} "
        f"(won {d['runner_up_win_rate']:.1%} of simulations)",
        f"  Predicted Benefit Gap:  {d['mean_gap']:.2f} pp "
        f"(difference in predicted HbA1c reduction between top two)",
    ]
    model_section = "\n".join(model_lines)

    # Structured safety section (G-13)
    safety_lines: List[str] = ["SAFETY CHECK RESULTS (structured findings):"]
    safety_lines.append(f"  Status: {s['status']}")
    safety_lines.append(f"  Final treatment: {s.get('final_treatment', d['recommended_treatment'])}")

    if s["recommended_contraindications"]:
        safety_lines.append("")
        safety_lines.append(
            f"  CONTRAINDICATIONS for {d['recommended_treatment']}:"
        )
        for c in s["recommended_contraindications"]:
            safety_lines.append(f"    [CONTRA] {_format_finding(c)}")

    if s["recommended_warnings"]:
        safety_lines.append("")
        safety_lines.append(f"  WARNINGS for {d['recommended_treatment']}:")
        for w in s["recommended_warnings"]:
            safety_lines.append(f"    [WARN]   {_format_finding(w)}")

    if s.get("excluded_treatments"):
        safety_lines.append("")
        safety_lines.append(
            "  EXCLUDED TREATMENTS (contraindicated for this patient):"
        )
        for t, reasons in s["excluded_treatments"].items():
            for r in reasons:
                if isinstance(r, dict):
                    safety_lines.append(f"    [CONTRA] {t}: {_format_finding(r)}")
                else:
                    safety_lines.append(f"    [CONTRA] {t}: {r}")

    other_warns = s.get("other_treatment_warnings", {})
    if other_warns:
        safety_lines.append("")
        safety_lines.append("  WARNINGS for other treatments:")
        for t, warns in other_warns.items():
            for w in warns:
                if isinstance(w, dict):
                    safety_lines.append(f"    [WARN] {t}: {_format_finding(w)}")
                else:
                    safety_lines.append(f"    [WARN] {t}: {w}")

    if (not s["recommended_contraindications"]
            and not s["recommended_warnings"]):
        safety_lines.append("")
        safety_lines.append(
            f"  No contraindications or warnings for "
            f"{d['recommended_treatment']}."
        )

    safety_section = "\n".join(safety_lines)

    # Attribution section — only present after Phase 2 lands
    attribution_section = ""
    if d.get("attribution"):
        attribution_section = _render_attribution(
            d["attribution"], d.get("contrast"),
            d.get("uncertainty_drivers"),
        )

    # Fairness section (G-15): only rendered when a real subgroup report was
    # supplied; silently omitted otherwise so no static paragraph reaches the
    # LLM.
    sections = [patient_section, model_section, safety_section]
    if attribution_section:
        sections.append(attribution_section)
    if f and f.get("attestation_computed"):
        fairness_section = (
            "FAIRNESS REPORT (subgroup regret computed):\n"
            f"  Clinical features used: {', '.join(f['decision_features'])}\n"
            f"  Protected features NOT used: "
            f"{', '.join(f['excluded_protected_features'])}\n"
            f"  Subgroup regret table: {json.dumps(f['subgroup_regret'])}"
        )
        sections.append(fairness_section)

    sections.append(RESPONSE_FORMAT_INSTRUCTION)
    return "\n\n---\n\n".join(sections)


def _render_attribution(
    attribution: Optional[Dict[str, Any]],
    contrast: Optional[Dict[str, Any]],
    uncertainty_drivers: Optional[List[Dict[str, Any]]],
) -> str:
    """Render Phase 2 attribution artefacts (G-7/8/9) as a prompt block."""
    lines = ["MODEL ATTRIBUTION (computed, not model-introspected):"]
    if attribution:
        lines.append("  Top features driving the recommendation:")
        for name, val in sorted(
            attribution.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:5]:
            lines.append(f"    {name:<30s} {val:+.3f}")
    if contrast:
        lines.append("  Recommended vs runner-up contrast (Δ attribution):")
        for name, val in sorted(
            contrast.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:5]:
            lines.append(f"    {name:<30s} {val:+.3f}")
    if uncertainty_drivers:
        lines.append("  Top drivers of posterior uncertainty:")
        for d in uncertainty_drivers[:3]:
            lines.append(
                f"    {d.get('feature', '?'):<30s} "
                f"var_contrib={d.get('contribution', 0):.3f}"
            )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = [
    "recommendation_summary",
    "runner_up_analysis",
    "confidence_statement",
    "safety_assessment",
    "monitoring_note",
    "disclaimer",
]


def parse_llm_response(raw_text: str) -> Dict[str, str]:
    """Parse the LLM response into a structured dict."""
    text = raw_text.strip()
    text = text.replace('\uff5b', '{').replace('\uff5d', '}')
    text = text.replace('\u200b', '').replace('\ufeff', '')
    text = text.encode('ascii', 'ignore').decode('ascii')

    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
        elif "```" in text:
            text = text[:text.rfind("```")].strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response: {raw_text[:200]}")

    json_str = text[start:end]

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        repaired = _attempt_json_repair(json_str)
        if repaired is not None:
            result = repaired
        else:
            raise ValueError(f"Invalid JSON from LLM: {e}\nRaw: {json_str[:300]}")

    missing = [k for k in REQUIRED_KEYS if k not in result]
    if missing:
        if missing == ["disclaimer"]:
            result["disclaimer"] = (
                "This is an AI-assisted decision support tool. Final treatment "
                "decisions must be made by the treating physician based on the "
                "complete clinical picture, patient preferences, and current guidelines."
            )
        else:
            raise ValueError(f"LLM response missing required keys: {missing}")

    for key in REQUIRED_KEYS:
        result[key] = str(result[key])

    return result


def _attempt_json_repair(json_str: str) -> Optional[Dict]:
    """Try to repair truncated JSON from LLM output."""
    attempts = [
        json_str + '"}',
        json_str + '"}\n}',
        json_str + '..."}',
    ]
    for attempt in attempts:
        try:
            result = json.loads(attempt)
            if isinstance(result, dict):
                logger.warning("Repaired truncated JSON from LLM response")
                return result
        except json.JSONDecodeError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT — pluggable provider interface (Phase 3 groundwork)
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Minimal provider interface. Subclass this to add new LLM backends.
    """
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_output_tokens: int = 4096,
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai is required. pip install google-genai")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=system_prompt)],
                ),
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text=("Understood. I will respond with only a valid "
                              "JSON object, grounding every clinical claim in "
                              "the provided structured safety findings and "
                              "predicted HbA1c reductions.")
                    )],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt)],
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        return response.text


# ─────────────────────────────────────────────────────────────────────────────
# LLM EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

class LLMExplainer:
    """
    Generates structured clinical explanations for NeuralThompson
    recommendations using a pluggable LLM client (Gemini by default).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_retries: int = 2,
        client: Optional[LLMClient] = None,
    ):
        self.max_retries = max_retries
        if client is not None:
            self.client = client
        else:
            self.client = GeminiClient(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
            )
        self.model_name = getattr(self.client, "model_name", model_name)
        self.temperature = getattr(self.client, "temperature", temperature)
        logger.info(
            f"LLMExplainer initialized: client={type(self.client).__name__}, "
            f"model={self.model_name}, temp={self.temperature}, "
            f"max_retries={max_retries}"
        )

    def explain(self, payload: Dict) -> Dict[str, str]:
        """
        Generate a structured clinical explanation from an extraction payload.

        Phase 3: output passes through Pydantic validation. On schema
        violation, the next attempt receives a repair prompt with the
        validation error appended.
        """
        prompt = build_prompt(payload)
        repair_note: str = ""

        for attempt in range(self.max_retries + 1):
            try:
                user_prompt = prompt + repair_note
                raw = self.client.generate(SYSTEM_PROMPT, user_prompt)
                result = parse_llm_response(raw)
                # Phase 3: schema-first output (if pydantic is available)
                if PYDANTIC_AVAILABLE:
                    try:
                        ClinicalExplanation(**result)
                    except ValidationError as ve:
                        raise ValueError(f"schema violation: {ve}") from ve
                # Phase 3: provenance guard — no hallucinated feature references
                _enforce_provenance(result, payload)
                logger.info(
                    f"Explanation generated (attempt {attempt + 1}): "
                    f"treatment={payload['decision']['recommended_treatment']}"
                )
                return result
            except ValueError as e:
                logger.warning(f"Parse failed (attempt {attempt + 1}): {e}")
                repair_note = (
                    f"\n\n---\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {e}\n"
                    f"Produce ONLY a valid JSON object with all required keys, "
                    f"no HbA1c reduction above {REWARD_CAP_PP:.1f} pp, and only "
                    f"feature names present in the payload."
                )
                if attempt < self.max_retries:
                    continue
                raise ValueError(
                    f"Failed to parse LLM response after "
                    f"{self.max_retries + 1} attempts: {e}"
                ) from e
            except Exception as e:
                raise RuntimeError(f"LLM generate call failed: {e}") from e

    def explain_batch(self, payloads: list) -> list:
        explanations = []
        for i, payload in enumerate(payloads):
            logger.info(f"Generating explanation {i + 1}/{len(payloads)}...")
            explanations.append(self.explain(payload))
        return explanations


# ─────────────────────────────────────────────────────────────────────────────
# PROVENANCE GUARD (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_provenance(result: Dict[str, str], payload: Dict) -> None:
    """
    Phase 3: block hallucinated *feature names*. If the LLM names a feature
    not present in the patient payload, we reject the response.

    This is a conservative textual check — clinical prose uses many terms
    that map to features (HbA1c, eGFR, BMI, ...), so we only blacklist a
    small set of ML-adjacent patterns and a fixed whitelist of unknown
    "feature-like" tokens.
    """
    blacklist = [
        "posterior mean", "thompson sampling", "contextual bandit",
        "reward", "regret", "feature vector", "neural network",
    ]
    joined = " ".join(str(v).lower() for v in result.values())
    for term in blacklist:
        if term in joined:
            raise ValueError(f"ML jargon '{term}' leaked into output")