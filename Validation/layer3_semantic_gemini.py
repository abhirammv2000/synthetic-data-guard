"""
Layer 3: LLM Semantic Validator

Uses Gemini 2.0 Flash to reason about flagged synthetic records.
Falls back to a deterministic mock response when no API key is set.
"""

import time
import json
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from Validation.config import config


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a fraud-detection data quality auditor.
You will receive synthetic credit-card transaction records that have been
flagged by a statistical validator.  For each record decide whether the
feature values are plausible given these known patterns:

FRAUD patterns (Class=1):
  - V14 typically < -5, V3 < -5, V17 < -5, V12 < -5, V10 < -4
  - Amount usually < $50 (median ~$9.25)
  - High-value fraud (> $1000) is extremely rare

LEGITIMATE patterns (Class=0):
  - V-features centred near 0 with std ~1
  - Amount right-skewed, median ~$22, max ~$25,691
  - Time shows day/night cyclicality

For each record respond with a JSON object:
{
  "row_index": <int>,
  "is_anomalous": <bool>,
  "feature": "<most suspicious column>",
  "observation": "<one sentence explaining why>"
}

Return a JSON array of these objects, one per record reviewed.
Only flag records that are genuinely implausible, not merely unusual."""


def _build_prompt(rows: pd.DataFrame) -> str:
    """Format flagged rows into a prompt string."""
    lines = [_SYSTEM_PROMPT, "", "RECORDS TO REVIEW:", ""]
    for idx, row in rows.iterrows():
        summary = (
            f"Row {idx}: Class={int(row['Class'])}, "
            f"Amount={row['Amount']:.2f}, "
            f"V3={row.get('V3', 0):.3f}, V10={row.get('V10', 0):.3f}, "
            f"V12={row.get('V12', 0):.3f}, V14={row.get('V14', 0):.3f}, "
            f"V17={row.get('V17', 0):.3f}"
        )
        lines.append(summary)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock response generator (deterministic, no API call)
# ---------------------------------------------------------------------------

def _mock_review(rows: pd.DataFrame) -> tuple:
    """Return a realistic-looking mock result without calling any API."""
    anomalies: List[Dict[str, Any]] = []

    for idx, row in rows.iterrows():
        is_fraud = int(row["Class"]) == 1

        # Check for suspicious patterns
        if is_fraud and row["Amount"] > 500:
            anomalies.append({
                "row_index": int(idx),
                "feature": "Amount",
                "observation": (
                    f"Fraud transaction with Amount=${row['Amount']:.2f} is unusually "
                    f"high; real fraud median is $9.25. Possible CTGAN artefact."
                ),
            })
        elif is_fraud and row.get("V14", 0) > 0:
            anomalies.append({
                "row_index": int(idx),
                "feature": "V14",
                "observation": (
                    f"Fraud row has V14={row['V14']:.3f} (positive), but real fraud "
                    f"V14 is strongly negative (mean -6.97). Distribution mismatch."
                ),
            })
        elif not is_fraud and row.get("V14", 0) < -8:
            anomalies.append({
                "row_index": int(idx),
                "feature": "V14",
                "observation": (
                    f"Legitimate row has V14={row['V14']:.3f} which falls in the "
                    f"extreme fraud range. Likely mis-labelled synthetic record."
                ),
            })

    n_reviewed = len(rows)
    n_anomalies = len(anomalies)
    score = max(0.0, 1.0 - (n_anomalies / max(n_reviewed, 1)))

    raw = json.dumps(anomalies, indent=2)
    return anomalies, score, raw, "mock (no API key)"


# ---------------------------------------------------------------------------
# Live Gemini call
# ---------------------------------------------------------------------------

def _gemini_review(rows: pd.DataFrame) -> tuple:
    """Call Gemini 2.0 Flash and parse the response."""
    import google.genai as genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    prompt = _build_prompt(rows)

    response = client.models.generate_content(
        model=config.LAYER3_GEMINI_MODEL,
        contents=prompt,
    )
    raw_text = response.text

    # Parse JSON from response
    anomalies = []
    try:
        text = raw_text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            anomalies = [
                {
                    "row_index": int(item.get("row_index", -1)),
                    "feature": str(item.get("feature", "unknown")),
                    "observation": str(item.get("observation", "")),
                }
                for item in parsed
                if item.get("is_anomalous", True)
            ]
    except (json.JSONDecodeError, IndexError):
        anomalies = [{
            "row_index": -1,
            "feature": "parse_error",
            "observation": f"Could not parse LLM response: {raw_text[:200]}",
        }]

    n_reviewed = len(rows)
    n_anomalies = len(anomalies)
    score = max(0.0, 1.0 - (n_anomalies / max(n_reviewed, 1)))

    return anomalies, score, raw_text, config.LAYER3_GEMINI_MODEL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LLMSemanticValidator:
    """Layer 3: LLM-based semantic validation of synthetic records."""

    def validate(
        self,
        df: pd.DataFrame,
        flagged_indices: Optional[List[int]] = None,
        layer2_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate synthetic data using LLM semantic reasoning.

        Parameters
        ----------
        df : pd.DataFrame
            Synthetic dataset.
        flagged_indices : list[int], optional
            Row indices flagged by earlier layers. If None, a random
            sample of ``config.LAYER3_SAMPLE_SIZE`` rows is used.
        layer2_context : str, optional
            Extra context string (e.g. escalation info from RAG layer).

        Returns
        -------
        dict with keys: agent, passed, model_used, rows_reviewed,
             anomalies, score, raw_response, execution_time.
        """
        t0 = time.perf_counter()

        # Select rows to review
        sample_size = config.LAYER3_SAMPLE_SIZE
        if flagged_indices:
            indices = flagged_indices[:sample_size]
        else:
            indices = df.sample(n=min(sample_size, len(df)), random_state=7).index.tolist()

        rows = df.loc[indices].copy()

        # Append escalation context if provided
        if layer2_context:
            rows.attrs["escalation_context"] = layer2_context

        # Call LLM or mock
        if config.has_gemini():
            try:
                anomalies, score, raw, model_used = _gemini_review(rows)
            except Exception as e:
                print(f"  Gemini call failed ({e}), falling back to mock")
                anomalies, score, raw, model_used = _mock_review(rows)
        else:
            anomalies, score, raw, model_used = _mock_review(rows)

        elapsed = time.perf_counter() - t0
        passed = score >= 0.7

        return {
            "agent": "LLMSemanticValidator",
            "passed": passed,
            "model_used": model_used,
            "rows_reviewed": len(rows),
            "anomalies": anomalies,
            "score": round(score, 4),
            "raw_response": raw,
            "execution_time": round(elapsed, 4),
        }
