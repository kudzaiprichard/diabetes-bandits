"""
LLM Explanation Generator for NeuralThompson

Takes the extracted payload from ExplainabilityExtractor and
generates a clinical explanation via Google Gemini.

Uses the google-genai SDK (the current official Google Gen AI SDK).
Install: pip install google-genai

Returns structured JSON with six sections:
    - recommendation_summary
    - runner_up_analysis
    - confidence_statement
    - safety_assessment
    - monitoring_note
    - disclaimer

Usage:
    from src.llm_explain import LLMExplainer
    explainer = LLMExplainer(api_key="your-key")
    result = explainer.explain(payload)
    print(result["recommendation_summary"])
    print(result["safety_assessment"])
"""

import os
import json
from typing import Dict, Optional
from loguru import logger

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed — LLM explanations unavailable")

from src.data_generator import TREATMENTS


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Dr. Sarah Chen, a senior endocrinologist with 40 years of \
clinical experience specialising in Type 2 Diabetes management. You have treated \
over 30,000 patients across diverse populations, from newly diagnosed cases to \
complex multi-comorbidity presentations. You are also trained in interpreting \
AI-assisted clinical decision support tools and translating their outputs into \
language that fellow clinicians can immediately understand and act upon.

You are reviewing the output of an AI treatment recommendation model that \
predicts expected HbA1c reduction (in percentage points) for each of five \
treatment options given a patient's clinical profile. The model also provides \
a confidence score — the percentage of times this treatment won when the model \
simulated the decision 200 times using its own uncertainty estimates.

Your task is to FAITHFULLY explain the model's recommendation in clear clinical \
language that a practising physician would find useful during a consultation.

CRITICAL TRANSLATION RULES — follow these exactly:

1. POSTERIOR MEANS = PREDICTED HbA1c REDUCTION. The model's posterior mean \
   scores represent predicted HbA1c reduction in percentage points. A score \
   of 6.5 means the model predicts a 6.5 percentage point reduction in HbA1c. \
   A negative or near-zero score means the model predicts minimal or no benefit. \
   Always refer to these as "predicted HbA1c reduction" — never say "posterior \
   mean", "sampled score", or "reward".

2. CONFIDENCE = WIN RATE. The model provides a confidence percentage based on \
   how often the recommended treatment won across 200 internal simulations. \
   Translate this naturally: \
   - 90%+ → "the model is highly confident — this treatment won in X% of simulations" \
   - 70-89% → "the model is moderately confident — this treatment won in X% of simulations" \
   - 50-69% → "this is a closer decision — the recommended treatment won in only X% of simulations" \
   - Below 50% → "the model shows no clear preference — clinical judgement should guide this decision" \
   Use the EXACT percentage provided. Do not round or approximate it.

3. WIN RATES = TREATMENT COMPARISON. The model provides win rates for all \
   treatments (how often each won across simulations). Use these to explain \
   how close the alternatives are. For example: "SGLT-2 won 73% of simulations \
   while DPP-4 won 22%, indicating a clear but not overwhelming preference."

4. MEAN GAP = EXPECTED BENEFIT DIFFERENCE. The gap between the predicted \
   HbA1c reductions of the top two treatments. Describe this as "the model \
   predicts approximately X percentage points more HbA1c reduction with \
   [recommended] compared to [runner-up]."

5. SAFETY = REPORT EXACTLY. Report safety check results (contraindications \
   and warnings) EXACTLY as provided — do not add or remove safety concerns. \
   The safety checks are deterministic and authoritative. Use clinical language \
   but preserve the specific lab values and thresholds mentioned.

6. NO ML JARGON. Never use the following terms in your explanation: \
   posterior mean, posterior sampling, sampled score, trace, A_inv, \
   exploration flag, Thompson sampling, contextual bandit, reward, action, \
   policy, regret, feature vector, neural network, epoch, covariance, \
   win rate (use "simulations" instead), or any other machine learning \
   terminology. You are writing for clinicians, not data scientists.

Do NOT cite clinical trial evidence, guideline recommendations, or \
mechanism-of-action reasoning unless it directly maps to a pattern visible \
in the model's predictions. The goal is transparency about what the model \
predicts, not post-hoc medical justification.

CRITICAL: You MUST respond with ONLY a valid JSON object. No markdown, no \
backticks, no preamble, no explanation outside the JSON. Just the raw JSON."""


RESPONSE_FORMAT_INSTRUCTION = """RESPONSE FORMAT:

You MUST respond with ONLY a valid JSON object matching this exact structure. \
No markdown code fences, no backticks, no text before or after the JSON.

{
    "recommendation_summary": "2-3 sentences explaining why this treatment is recommended for this patient. Reference specific patient features (age, BMI, HbA1c, eGFR, duration, C-peptide, comorbidities) and describe predicted HbA1c reductions. Write as a senior clinician advising a colleague.",
    "runner_up_analysis": "1-2 sentences on which treatment was the next best alternative, its predicted HbA1c reduction, and how often it won in the model's simulations compared to the recommended treatment.",
    "confidence_statement": "1-2 sentences stating the model's confidence as a percentage. Use the exact confidence percentage and number of simulations provided. Frame in clinical terms — high confidence means strong evidence from similar patients, low confidence means this is a close call.",
    "safety_assessment": "1-3 sentences reporting the safety check results in clinical language. If CONTRAINDICATION_FOUND: strongly flag this. If WARNING: report the specific clinical concerns. If CLEAR: confirm no contraindications. Always include relevant warnings for other treatments if present.",
    "monitoring_note": "1-2 sentences with specific, actionable monitoring recommendations based on this patient's profile, lab values, comorbidities, and the chosen treatment.",
    "disclaimer": "This is an AI-assisted decision support tool. Final treatment decisions must be made by the treating physician based on the complete clinical picture, patient preferences, and current guidelines."
}

Example response for a CVD patient recommended SGLT-2 with 73% confidence:

{
    "recommendation_summary": "For this 63-year-old patient with established cardiovascular disease, moderate BMI of 29.0, HbA1c of 8.8%, and preserved renal function (eGFR 72), the model predicts SGLT-2 inhibitor therapy would achieve the greatest HbA1c reduction of approximately 5.9 percentage points. This is consistent with the cardiorenal benefit profile expected in patients with established CVD and adequate renal function.",
    "runner_up_analysis": "DPP-4 inhibitors were the closest alternative, with a predicted HbA1c reduction of approximately 3.2 percentage points. In the model's 200 internal simulations, DPP-4 won only 22% of the time compared to SGLT-2's 73%, indicating a clear preference for SGLT-2 in this patient profile.",
    "confidence_statement": "The model is moderately confident in this recommendation — SGLT-2 was the preferred treatment in 73% of 200 simulations. This indicates a solid preference, though clinical factors not captured by the model may also be relevant to the final decision.",
    "safety_assessment": "No contraindications were identified for SGLT-2 inhibitor therapy in this patient. No treatments are excluded due to contraindications for this patient profile. Standard precautions for SGLT-2 use apply.",
    "monitoring_note": "With eGFR of 72, renal function should be monitored every 3-6 months. Monitor for signs of volume depletion and urinary tract infections, particularly in the first few months of treatment. Recheck HbA1c at 3 months to assess treatment response.",
    "disclaimer": "This is an AI-assisted decision support tool. Final treatment decisions must be made by the treating physician based on the complete clinical picture, patient preferences, and current guidelines."
}

Example response for a severe patient recommended Insulin with 88% confidence:

{
    "recommendation_summary": "For this 58-year-old patient with severe, long-standing diabetes (HbA1c 13.0%, duration 19 years) and significant beta-cell depletion (C-peptide 0.15 ng/mL), the model predicts insulin therapy would achieve the greatest HbA1c reduction of approximately 7.8 percentage points. The severity of hyperglycaemia combined with near-absent endogenous insulin production makes this the only treatment predicted to achieve meaningful glycaemic control.",
    "runner_up_analysis": "SGLT-2 was the next alternative with a predicted HbA1c reduction of approximately 3.1 percentage points. In the model's simulations, SGLT-2 won only 8% of the time compared to insulin's 88%, confirming a strong preference for insulin given this disease severity.",
    "confidence_statement": "The model is highly confident in this recommendation — insulin was the preferred treatment in 88% of 200 simulations. The model has strong evidence from similar patient profiles that insulin achieves substantially better outcomes at this level of disease severity.",
    "safety_assessment": "No hard contraindications exist for insulin in this patient. However, the patient's BMI of 29.0 raises a clinical caution regarding insulin-associated weight gain. Given the severity of the disease, the clinical need for glycaemic control likely outweighs this concern. Metformin is contraindicated for patients with eGFR below 30.",
    "monitoring_note": "With HbA1c at 13.0%, frequent capillary glucose monitoring is critical during insulin initiation — at minimum 4 times daily for the first 2-4 weeks. Watch closely for hypoglycaemia. Recheck HbA1c at 3 months. Monitor weight at each visit.",
    "disclaimer": "This is an AI-assisted decision support tool. Final treatment decisions must be made by the treating physician based on the complete clinical picture, patient preferences, and current guidelines."
}

REMEMBER: Output ONLY the JSON object. Nothing else. Write as Dr. Sarah Chen — \
a senior endocrinologist explaining her assessment to a colleague."""


def build_prompt(payload: Dict) -> str:
    """
    Build the full LLM prompt from an extraction payload.

    Args:
        payload: output from ExplainabilityExtractor.extract()

    Returns:
        Formatted prompt string
    """
    p = payload["patient"]
    d = payload["decision"]
    s = payload["safety"]
    f = payload["fairness"]

    # Patient profile section
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

    # Model recommendation section
    means = d["posterior_means"]
    win_rates = d["win_rates"]

    model_section = f"""MODEL PREDICTION:
  Recommended Treatment:  {d['recommended_treatment']}
  Model Confidence:       {d['confidence_pct']}% ({d['confidence_label']})
                          ({d['recommended_treatment']} won {d['confidence_pct']} out of {d['n_draws']} simulations)

  Predicted HbA1c Reduction (percentage points):
    Metformin  → {means['Metformin']:.1f} pp
    GLP-1      → {means['GLP-1']:.1f} pp
    SGLT-2     → {means['SGLT-2']:.1f} pp
    DPP-4      → {means['DPP-4']:.1f} pp
    Insulin    → {means['Insulin']:.1f} pp

  Simulation Win Rates (how often each treatment won across {d['n_draws']} simulations):
    Metformin  → {win_rates['Metformin']:.1%}
    GLP-1      → {win_rates['GLP-1']:.1%}
    SGLT-2     → {win_rates['SGLT-2']:.1%}
    DPP-4      → {win_rates['DPP-4']:.1%}
    Insulin    → {win_rates['Insulin']:.1%}

  Next Best Alternative:    {d['runner_up']} (won {d['runner_up_win_rate']:.1%} of simulations)
  Predicted Benefit Gap:    {d['mean_gap']:.1f} percentage points (difference in predicted HbA1c reduction between top two)"""

    # Safety section
    safety_lines = [f"SAFETY CHECK RESULTS:"]
    safety_lines.append(f"  Status: {s['status']}")

    if s["recommended_contraindications"]:
        safety_lines.append(f"\n  CONTRAINDICATIONS for {d['recommended_treatment']}:")
        for c in s["recommended_contraindications"]:
            safety_lines.append(f"    ⛔ {c}")

    if s["recommended_warnings"]:
        safety_lines.append(f"\n  WARNINGS for {d['recommended_treatment']}:")
        for w in s["recommended_warnings"]:
            safety_lines.append(f"    ⚠️ {w}")

    if s["excluded_treatments"]:
        safety_lines.append(f"\n  EXCLUDED TREATMENTS (contraindicated for this patient):")
        for t, reasons in s["excluded_treatments"].items():
            for r in reasons:
                safety_lines.append(f"    ⛔ {t}: {r}")

    if s["all_warnings"]:
        other_warnings = {
            t: w for t, w in s["all_warnings"].items()
            if t != d["recommended_treatment"]
        }
        if other_warnings:
            safety_lines.append(f"\n  WARNINGS for other treatments:")
            for t, warns in other_warnings.items():
                for w in warns:
                    safety_lines.append(f"    ⚠️ {t}: {w}")

    if not s["recommended_contraindications"] and not s["recommended_warnings"]:
        safety_lines.append(
            f"\n  No contraindications or warnings for {d['recommended_treatment']}."
        )

    safety_section = "\n".join(safety_lines)

    # Fairness section
    fairness_section = f"""FAIRNESS ATTESTATION:
  {f['statement']}
  Clinical features used: {', '.join(f['decision_features'])}
  Protected features NOT used: {', '.join(f['excluded_protected_features'])}"""

    return (
        f"{patient_section}\n\n---\n\n"
        f"{model_section}\n\n---\n\n"
        f"{safety_section}\n\n---\n\n"
        f"{fairness_section}\n\n---\n\n"
        f"{RESPONSE_FORMAT_INSTRUCTION}"
    )


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
    """
    Parse the LLM response into a structured dict.

    Handles common LLM quirks: markdown fences, preamble text,
    trailing content after the JSON, unicode issues, and
    truncated responses.
    """
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
    """
    Try to repair truncated JSON from LLM output.
    """
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
# LLM EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

class LLMExplainer:
    """
    Generates structured clinical explanations for NeuralThompson
    recommendations using Google Gemini via the google-genai SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_retries: int = 2,
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai is required. Install with: "
                "pip install google-genai"
            )

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

        logger.info(f"LLMExplainer initialized: model={model_name}, temp={temperature}")

    def explain(self, payload: Dict) -> Dict[str, str]:
        """
        Generate a structured clinical explanation from an extraction payload.
        """
        prompt = build_prompt(payload)
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=SYSTEM_PROMPT)],
                        ),
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(
                                text="Understood. I will respond as Dr. Sarah Chen with only a valid JSON object, using clinical language throughout and no ML terminology."
                            )],
                        ),
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=prompt)],
                        ),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=4096,
                    ),
                )

                result = parse_llm_response(response.text)

                logger.info(
                    f"Explanation generated (attempt {attempt + 1}): "
                    f"treatment={payload['decision']['recommended_treatment']}"
                )
                return result

            except ValueError as e:
                last_error = e
                logger.warning(f"Parse failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    continue
                raise ValueError(
                    f"Failed to parse LLM response after {self.max_retries + 1} attempts: {e}"
                ) from e

            except Exception as e:
                raise RuntimeError(f"Gemini API call failed: {e}") from e

    def explain_batch(self, payloads: list) -> list:
        """
        Generate explanations for multiple patients.
        """
        explanations = []
        for i, payload in enumerate(payloads):
            logger.info(f"Generating explanation {i + 1}/{len(payloads)}...")
            explanation = self.explain(payload)
            explanations.append(explanation)
        return explanations