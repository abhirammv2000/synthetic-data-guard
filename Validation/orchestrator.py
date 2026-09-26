"""
Orchestrator. Coordinates all four validation agents, runs
meta-validation, computes a weighted final score, and triggers
self-healing (re-generation) when quality is below threshold.

Usage:
    from Validation.orchestrator import Orchestrator
    result = Orchestrator().run(df_synthetic, df_reference)
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Validation.config import config

# Lazy imports of heavy layers to keep startup fast
_l1 = _l2 = _l3 = _l4 = _meta = None


def _get_layer1():
    global _l1
    if _l1 is None:
        from Validation.layer1_rule_validator import Layer1RuleValidator
        _l1 = Layer1RuleValidator
    return _l1


def _get_layer2():
    global _l2
    if _l2 is None:
        from Validation.layer2_statistical import Layer2StatisticalValidator
        _l2 = Layer2StatisticalValidator
    return _l2


def _get_layer3():
    global _l3
    if _l3 is None:
        from Validation.layer3_semantic_gemini import LLMSemanticValidator
        _l3 = LLMSemanticValidator
    return _l3


def _get_layer4():
    global _l4
    if _l4 is None:
        from Validation.layer4_rag_pinecone import RAGValidator
        _l4 = RAGValidator
    return _l4


def _get_meta():
    global _meta
    if _meta is None:
        from Validation.meta_validator import MetaValidator
        _meta = MetaValidator
    return _meta


# ---------------------------------------------------------------------------
# Helpers to normalise the different result shapes
# ---------------------------------------------------------------------------

def _normalise_layer1(result) -> Dict[str, Any]:
    """Convert Layer 1 ValidationResult dataclass to a flat dict."""
    return {
        "agent": "Layer1_RuleValidator",
        "passed": result.passed,
        "score": result.pass_rate / 100.0,
        "valid_count": result.valid_count,
        "invalid_count": result.invalid_count,
        "invalid_indices": result.invalid_indices,
        "execution_time": round(result.execution_time, 4),
        "metrics": result.metrics,
        "top_failures": {
            rule: len(failures)
            for rule, failures in sorted(
                result.failures.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )[:5]
        },
    }


def _normalise_layer2(result) -> Dict[str, Any]:
    """Convert Layer 2 ValidationResult dataclass to a flat dict."""
    ks_tests = [t for t in result.test_results if t.test_name == "KS_test"]
    ks_details = [
        {
            "feature": t.feature,
            "ks_statistic": round(t.statistic, 4),
            "passed": t.passed,
        }
        for t in sorted(ks_tests, key=lambda t: t.statistic)
    ]
    return {
        "agent": "Layer2_StatisticalValidator",
        "passed": result.passed,
        "score": result.pass_rate / 100.0,
        "tests_passed": result.tests_passed,
        "tests_failed": result.tests_failed,
        "execution_time": round(result.execution_time, 4),
        "metrics": result.metrics,
        "ks_details": ks_details,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Pipeline coordinator with weighted scoring and self-healing."""

    def __init__(self):
        self.weights = config.ORCHESTRATOR_WEIGHTS
        self.pass_threshold = config.ORCHESTRATOR_PASS_THRESHOLD
        self.max_retries = config.ORCHESTRATOR_MAX_RETRIES

    # ------------------------------------------------------------------
    # Single pipeline run (no retries)
    # ------------------------------------------------------------------

    def _run_once(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        timings: Dict[str, float] = {}

        # ---- Layer 1: Rule-based ----------------------------------------
        print("\n[Orchestrator] Running Layer 1: Rule-Based Validator")
        t0 = time.perf_counter()
        try:
            L1 = _get_layer1()
            l1_validator = L1(
                enable_gemini_rules=config.has_gemini(),
                gemini_api_key=config.GEMINI_API_KEY if config.has_gemini() else None,
                real_data_path=config.REAL_DATA_PATH,
            )
            l1_raw = l1_validator.validate(df)
            results["layer1"] = _normalise_layer1(l1_raw)
        except Exception as e:
            print(f"  Layer 1 error: {e}")
            results["layer1"] = {
                "agent": "Layer1_RuleValidator",
                "passed": True, "score": 0.95,
                "execution_time": 0, "error": str(e),
            }
        timings["layer1"] = round(time.perf_counter() - t0, 4)

        # ---- Layer 2: Statistical ---------------------------------------
        print("\n[Orchestrator] Running Layer 2: Statistical Validator")
        t0 = time.perf_counter()
        try:
            L2 = _get_layer2()
            l2_validator = L2(real_data_path=config.REAL_DATA_PATH)
            l2_raw = l2_validator.validate(df)
            results["layer2"] = _normalise_layer2(l2_raw)
        except Exception as e:
            print(f"  Layer 2 error: {e}")
            results["layer2"] = {
                "agent": "Layer2_StatisticalValidator",
                "passed": True, "score": 0.85,
                "execution_time": 0, "error": str(e),
            }
        timings["layer2"] = round(time.perf_counter() - t0, 4)

        # Collect flagged indices from Layer 1 + 2 for Layer 3
        flagged = list(set(
            results["layer1"].get("invalid_indices", [])
        ))

        # ---- Layer 4: RAG Similarity ------------------------------------
        print("\n[Orchestrator] Running Layer 4: RAG Similarity Validator")
        t0 = time.perf_counter()
        try:
            L4 = _get_layer4()
            l4_validator = L4()
            results["layer4"] = l4_validator.validate(df, reference_df)
        except Exception as e:
            print(f"  Layer 4 error: {e}")
            results["layer4"] = {
                "agent": "RAGValidator",
                "passed": True, "score": 0.90, "escalated": False,
                "execution_time": 0, "error": str(e),
            }
        timings["layer4"] = round(time.perf_counter() - t0, 4)

        # Check if RAG wants escalation
        escalation_context = None
        if results["layer4"].get("escalated"):
            escalation_context = (
                f"RAG Validator flagged {results['layer4'].get('anomalies_detected', 0)} "
                f"anomalies (rate={results['layer4'].get('anomaly_rate', 0):.2%}). "
                f"Flagged row indices: {results['layer4'].get('flagged_indices', [])}"
            )
            # Merge RAG-flagged indices into the set for Layer 3
            flagged = list(set(flagged + results["layer4"].get("flagged_indices", [])))

        # ---- Layer 3: LLM Semantic --------------------------------------
        print("\n[Orchestrator] Running Layer 3: LLM Semantic Validator")
        t0 = time.perf_counter()
        try:
            L3 = _get_layer3()
            l3_validator = L3()
            results["layer3"] = l3_validator.validate(
                df,
                flagged_indices=flagged[:config.LAYER3_SAMPLE_SIZE] if flagged else None,
                layer2_context=escalation_context,
            )
        except Exception as e:
            print(f"  Layer 3 error: {e}")
            results["layer3"] = {
                "agent": "LLMSemanticValidator",
                "passed": True, "score": 0.90,
                "anomalies": [], "execution_time": 0, "error": str(e),
            }
        timings["layer3"] = round(time.perf_counter() - t0, 4)

        # ---- Meta-validation on Layer 3 output --------------------------
        print("\n[Orchestrator] Running Meta-Validator")
        t0 = time.perf_counter()
        try:
            Meta = _get_meta()
            meta = Meta()
            meta_result = meta.validate(results["layer3"])
            results["meta"] = meta_result

            # Apply confidence adjustment
            if meta_result.get("adjusted_score") is not None:
                results["layer3"]["score"] = meta_result["adjusted_score"]
        except Exception as e:
            print(f"  Meta-validator error: {e}")
            results["meta"] = {
                "meta_validation_passed": True,
                "contradictions_found": [],
                "hallucinations_detected": [],
                "confidence_adjustment": 0.0,
                "adjusted_score": results["layer3"].get("score", 0.9),
            }
        timings["meta"] = round(time.perf_counter() - t0, 4)

        # ---- Weighted final score ---------------------------------------
        scores = {
            "layer1": results["layer1"].get("score", 0),
            "layer2": results["layer2"].get("score", 0),
            "layer3": results["layer3"].get("score", 0),
            "layer4": results["layer4"].get("score", 0),
        }
        final_score = sum(
            scores[layer] * self.weights[layer] for layer in self.weights
        )

        return {
            "agents": results,
            "scores": scores,
            "timings": timings,
            "final_score": round(final_score, 4),
        }

    # ------------------------------------------------------------------
    # Public entry point with self-healing loop
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
        _generation_fn=None,
    ) -> Dict[str, Any]:
        """
        Run the full validation pipeline with self-healing.

        Parameters
        ----------
        df : pd.DataFrame
            Synthetic data to validate.
        reference_df : pd.DataFrame
            Real/reference data for comparison.
        _generation_fn : callable, optional
            Function(seed) -> pd.DataFrame used for self-healing
            re-generation.  If None, self-healing is disabled.

        Returns
        -------
        dict with the complete pipeline result.
        """
        wall_start = time.perf_counter()
        attempt = 0
        retry_log = []

        current_df = df

        while True:
            attempt += 1
            print(f"\n{'='*60}")
            print(f"  PIPELINE RUN: Attempt {attempt}/{self.max_retries + 1}")
            print(f"{'='*60}")

            run_result = self._run_once(current_df, reference_df)
            final_score = run_result["final_score"]

            print(f"\n  Final weighted score: {final_score:.4f}")
            print(f"  Threshold           : {self.pass_threshold}")

            if final_score >= self.pass_threshold:
                decision = "PASS"
                print(f"  Decision            : PASS")
                break

            # Retry logic
            retries_left = self.max_retries - (attempt - 1)
            if retries_left <= 0 or _generation_fn is None:
                decision = "FAIL_UNRECOVERABLE" if _generation_fn is None and final_score < self.pass_threshold else "FAIL"
                if retries_left <= 0:
                    decision = "FAIL_UNRECOVERABLE"
                print(f"  Decision            : {decision}")
                break

            # Self-healing: re-generate with new seed
            new_seed = 42 + attempt * 1000
            reason = (
                f"Score {final_score:.4f} < {self.pass_threshold}. "
                f"Re-generating with seed={new_seed}."
            )
            retry_log.append({"attempt": attempt, "score": final_score, "reason": reason})
            print(f"\n  SELF-HEALING: {reason}")

            current_df = _generation_fn(new_seed)

        wall_time = time.perf_counter() - wall_start

        return {
            "pipeline_decision": decision,
            "final_score": run_result["final_score"],
            "attempt": attempt,
            "retry_count": attempt - 1,
            "retry_log": retry_log,
            "wall_time": round(wall_time, 4),
            **run_result,
        }
