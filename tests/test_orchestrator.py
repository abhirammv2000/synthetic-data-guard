"""Validation/orchestrator.py's self-healing retry loop and pass/fail decision
are pure control flow around whatever _run_once() returns, so they're tested
here by monkeypatching _run_once to hand back a controlled sequence of scores,
without needing real data, API keys, or the four validation layers.
"""
import pandas as pd

from Validation.orchestrator import Orchestrator


def make_orchestrator(scores):
    """An Orchestrator whose _run_once() returns each score in `scores` in turn."""
    orch = Orchestrator()
    it = iter(scores)

    def fake_run_once(df, reference_df):
        return {"agents": {}, "scores": {}, "timings": {}, "final_score": next(it)}

    orch._run_once = fake_run_once
    return orch


EMPTY = pd.DataFrame()


def test_a_first_attempt_above_threshold_passes_without_retrying():
    orch = make_orchestrator([0.9])

    result = orch.run(EMPTY, EMPTY)

    assert result["pipeline_decision"] == "PASS"
    assert result["attempt"] == 1
    assert result["retry_count"] == 0


def test_a_low_score_triggers_self_healing_and_can_recover():
    orch = make_orchestrator([0.5, 0.9])

    result = orch.run(EMPTY, EMPTY, _generation_fn=lambda seed: EMPTY)

    assert result["pipeline_decision"] == "PASS"
    assert result["attempt"] == 2
    assert result["retry_count"] == 1
    assert len(result["retry_log"]) == 1


def test_retries_exhausted_without_recovering_fails_unrecoverable():
    # ORCHESTRATOR_MAX_RETRIES is 2, so 3 low scores in a row exhausts every attempt.
    orch = make_orchestrator([0.3, 0.3, 0.3])

    result = orch.run(EMPTY, EMPTY, _generation_fn=lambda seed: EMPTY)

    assert result["pipeline_decision"] == "FAIL_UNRECOVERABLE"
    assert result["attempt"] == 3
    assert result["retry_count"] == 2


def test_without_a_generation_fn_a_low_score_fails_on_the_first_attempt():
    # self-healing is opt-in: passing no _generation_fn means no retry, even though
    # ORCHESTRATOR_MAX_RETRIES is nonzero.
    orch = make_orchestrator([0.3])

    result = orch.run(EMPTY, EMPTY)

    assert result["pipeline_decision"] == "FAIL_UNRECOVERABLE"
    assert result["attempt"] == 1
    assert result["retry_count"] == 0


def test_each_regeneration_uses_a_new_seed_derived_from_the_attempt_number():
    seeds_used = []
    orch = make_orchestrator([0.3, 0.3, 0.9])

    orch.run(EMPTY, EMPTY, _generation_fn=lambda seed: seeds_used.append(seed) or EMPTY)

    assert seeds_used == [42 + 1 * 1000, 42 + 2 * 1000]
