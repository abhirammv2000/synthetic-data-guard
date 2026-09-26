"""
Meta-Validation Layer

Checks whether the LLM semantic validator's own reasoning is internally
consistent and grounded in real data.  Catches two classes of error:

  Level 2a, Contradictions: score vs anomaly-count mismatch,
                            overconfident hedging language.
  Level 2b, Hallucinations: references to columns that do not exist
                            in the dataset schema.
"""

import re
import time
from typing import Any, Dict, List

_VALID_COLUMNS = {
    "Time", "Amount", "Class",
    *(f"V{i}" for i in range(1, 29)),
}

_HEDGE_WORDS = re.compile(
    r"\b(might|possibly|unclear|perhaps|uncertain|not sure|could be)\b",
    re.IGNORECASE,
)


class MetaValidator:
    """Validates the LLM validator's output for self-consistency."""

    def validate(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        llm_result : dict
            The return value of ``LLMSemanticValidator.validate()``.

        Returns
        -------
        dict with keys: meta_validation_passed, contradictions_found,
             hallucinations_detected, confidence_adjustment,
             adjusted_score, execution_time.
        """
        t0 = time.perf_counter()

        contradictions: List[str] = []
        hallucinations: List[str] = []

        score = llm_result.get("score", 0.0)
        anomalies = llm_result.get("anomalies", [])
        raw = llm_result.get("raw_response", "")
        n_anomalies = len(anomalies)

        # ------------------------------------------------------------------
        # Contradiction checks
        # ------------------------------------------------------------------

        # High score but many anomalies
        if score > 0.85 and n_anomalies > 5:
            contradictions.append(
                f"Score is {score:.2f} (high) but {n_anomalies} anomalies "
                f"were reported: these are inconsistent."
            )

        # Low score but no anomalies
        if score < 0.4 and n_anomalies == 0:
            contradictions.append(
                f"Score is {score:.2f} (low) but zero anomalies were "
                f"listed: the LLM failed to justify its rating."
            )

        # Hedging language with overconfident score
        if raw and _HEDGE_WORDS.search(raw) and score > 0.9:
            contradictions.append(
                "Raw response contains hedging language ('might', 'possibly', "
                "etc.) yet the score exceeds 0.9: overconfidence detected."
            )

        # ------------------------------------------------------------------
        # Hallucination checks (fabricated column names)
        # ------------------------------------------------------------------
        for anomaly in anomalies:
            feat = anomaly.get("feature", "")
            if feat and feat not in _VALID_COLUMNS and feat != "parse_error":
                hallucinations.append(
                    f"Anomaly at row {anomaly.get('row_index', '?')} references "
                    f"column '{feat}' which does not exist in the schema."
                )

            # Check observation text for made-up column names
            obs = anomaly.get("observation", "")
            for token in re.findall(r"\b(V\d+|Amount|Time|Class)\b", obs):
                if token not in _VALID_COLUMNS:
                    hallucinations.append(
                        f"Observation mentions non-existent column '{token}'."
                    )

        # ------------------------------------------------------------------
        # Confidence adjustment
        # ------------------------------------------------------------------
        n_issues = len(contradictions) + len(hallucinations)

        if n_issues == 0:
            adjustment = 0.0
        elif n_issues <= 2:
            adjustment = -0.10
        else:
            adjustment = -0.25

        adjusted_score = max(0.0, min(1.0, score + adjustment))
        passed = (len(contradictions) == 0) and (len(hallucinations) == 0)

        elapsed = time.perf_counter() - t0

        return {
            "meta_validation_passed": passed,
            "contradictions_found": contradictions,
            "hallucinations_detected": hallucinations,
            "confidence_adjustment": round(adjustment, 4),
            "adjusted_score": round(adjusted_score, 4),
            "execution_time": round(elapsed, 6),
        }
