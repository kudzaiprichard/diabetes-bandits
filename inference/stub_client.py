"""
StubClient — a deterministic ``LLMClient`` implementation for tests, demos,
and offline notebooks.

Shipped in the package (not ``tests/``) so integration tests and downstream
users can opt into the pipeline without ``google-genai`` or a network.

The stub builds a valid ``ClinicalExplanation`` JSON by reading the payload's
recommended treatment and confidence, so every field passes the Pydantic
validator in ``src/llm_explain.py``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


class StubClient:
    """
    Returns a deterministic JSON explanation that satisfies
    ``ClinicalExplanation`` and the provenance guard.

    Designed to be passed as ``client=StubClient()`` into ``LLMExplainer``.
    """

    def __init__(self, extra_fields: Optional[Dict[str, str]] = None):
        self.extra_fields = dict(extra_fields or {})
        self.calls: list[Dict[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        recommended, confidence_pct, runner_up = _extract_from_prompt(user_prompt)
        result = {
            "recommendation_summary": (
                f"For this patient, the selected glucose-lowering therapy is "
                f"{recommended}. The predicted HbA1c reduction is modest and "
                f"appropriate for the clinical profile described."
            ),
            "runner_up_analysis": (
                f"The next best alternative would be {runner_up}, with a "
                f"slightly lower predicted HbA1c reduction."
            ),
            "confidence_statement": (
                f"The model's confidence is {confidence_pct}% based on repeated "
                f"simulations; interpret accordingly."
            ),
            "safety_assessment": (
                "Safety checks were applied per the structured findings provided; "
                "no additional concerns are introduced by this explanation."
            ),
            "monitoring_note": (
                "Monitor HbA1c at 3 months, renal function at 6 months, and review "
                "tolerability at the next visit."
            ),
            "disclaimer": (
                "This is an AI-assisted decision support tool. Final treatment "
                "decisions must be made by the treating physician based on the "
                "complete clinical picture, patient preferences, and current guidelines."
            ),
        }
        result.update(self.extra_fields)
        return json.dumps(result)


def _extract_from_prompt(prompt: str) -> tuple[str, int, str]:
    """Pull recommended treatment, confidence %, and runner-up from the prompt."""
    rec_match = re.search(r"Final Treatment:\s+(\S+)", prompt)
    recommended = rec_match.group(1) if rec_match else "Metformin"
    conf_match = re.search(r"Model Confidence:\s+(\d+)%", prompt)
    confidence = int(conf_match.group(1)) if conf_match else 75
    ru_match = re.search(r"Next Best Alternative:\s+(\S+)", prompt)
    runner_up = ru_match.group(1) if ru_match else "GLP-1"
    return recommended, confidence, runner_up


__all__ = ["StubClient"]
