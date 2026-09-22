"""Validation/meta_validator.py checks the LLM semantic validator's output for
internal consistency. It has no API or data dependency, so it's fully testable
in isolation, and it's the layer meant to catch the validator itself
hallucinating, so it's worth pinning down exactly what it catches.
"""
from Validation.meta_validator import MetaValidator


def make_llm_result(score, anomalies=None, raw_response=""):
    return {"score": score, "anomalies": anomalies or [], "raw_response": raw_response}


def test_clean_result_passes_with_no_adjustment():
    result = MetaValidator().validate(make_llm_result(score=0.7, anomalies=[{"feature": "V14", "row_index": 3, "observation": "V14 is unusually low"}]))

    assert result["meta_validation_passed"] is True
    assert result["contradictions_found"] == []
    assert result["hallucinations_detected"] == []
    assert result["confidence_adjustment"] == 0.0
    assert result["adjusted_score"] == 0.7


def test_high_score_with_many_anomalies_is_a_contradiction():
    anomalies = [{"feature": "Amount", "row_index": i, "observation": ""} for i in range(6)]

    result = MetaValidator().validate(make_llm_result(score=0.9, anomalies=anomalies))

    assert result["meta_validation_passed"] is False
    assert len(result["contradictions_found"]) == 1
    assert result["confidence_adjustment"] == -0.10
    assert result["adjusted_score"] == 0.80


def test_low_score_with_zero_anomalies_is_a_contradiction():
    result = MetaValidator().validate(make_llm_result(score=0.2, anomalies=[]))

    assert result["meta_validation_passed"] is False
    assert "zero anomalies" in result["contradictions_found"][0]


def test_hedging_language_with_an_overconfident_score_is_flagged():
    result = MetaValidator().validate(make_llm_result(score=0.95, raw_response="This might be fraud, but I am not sure."))

    assert result["meta_validation_passed"] is False
    assert any("hedging" in c for c in result["contradictions_found"])


def test_a_reference_to_a_nonexistent_column_is_a_hallucination():
    anomalies = [{"feature": "TransactionRisk", "row_index": 5, "observation": "TransactionRisk looks off"}]

    result = MetaValidator().validate(make_llm_result(score=0.6, anomalies=anomalies))

    assert result["meta_validation_passed"] is False
    assert len(result["hallucinations_detected"]) >= 1


def test_real_schema_columns_are_never_flagged_as_hallucinations():
    anomalies = [{"feature": "V14", "row_index": 1, "observation": "V14 and Amount both look unusual"}]

    result = MetaValidator().validate(make_llm_result(score=0.6, anomalies=anomalies))

    assert result["hallucinations_detected"] == []


def test_parse_error_feature_is_not_treated_as_a_hallucination():
    # layer3 uses "parse_error" as a sentinel feature name when it can't parse
    # the LLM's own output, so meta_validator must not report that as a
    # hallucinated column, or every parse failure would double-count as one.
    anomalies = [{"feature": "parse_error", "row_index": 0, "observation": ""}]

    result = MetaValidator().validate(make_llm_result(score=0.5, anomalies=anomalies))

    assert result["hallucinations_detected"] == []


def test_three_or_more_issues_gets_the_larger_penalty():
    anomalies = [{"feature": "Bogus1", "row_index": 0, "observation": ""}, {"feature": "Bogus2", "row_index": 1, "observation": ""}]

    result = MetaValidator().validate(make_llm_result(score=0.92, anomalies=anomalies, raw_response="might be fraud"))

    assert len(result["hallucinations_detected"]) + len(result["contradictions_found"]) >= 3
    assert result["confidence_adjustment"] == -0.25


def test_adjusted_score_is_clamped_to_zero_not_negative():
    anomalies = [{"feature": "Ghost", "row_index": 0, "observation": ""}, {"feature": "Ghost2", "row_index": 1, "observation": ""}]

    result = MetaValidator().validate(make_llm_result(score=0.05, anomalies=anomalies))

    assert result["adjusted_score"] >= 0.0
