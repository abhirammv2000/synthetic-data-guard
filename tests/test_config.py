"""Validation/config.py's has_gemini/has_pinecone/has_openai gate every layer's
choice between a real API call and mock mode, so a wrong result here means a
layer silently runs (or silently skips) against a real key.
"""
from Validation.config import Config


def test_no_keys_means_every_layer_reports_mock_mode():
    cfg = Config(GEMINI_API_KEY="", PINECONE_API_KEY="", OPENAI_API_KEY="")

    assert cfg.has_gemini() is False
    assert cfg.has_pinecone() is False
    assert cfg.has_openai() is False


def test_a_real_looking_key_is_detected():
    cfg = Config(GEMINI_API_KEY="AIzaSyExampleKeyValue123", PINECONE_API_KEY="", OPENAI_API_KEY="")

    assert cfg.has_gemini() is True


def test_a_too_short_value_does_not_count_as_a_real_key():
    # config.py's own rule is "len > 5", so a placeholder like "TODO" or "xxx"
    # left in .env should still be treated as not-configured.
    cfg = Config(GEMINI_API_KEY="xxxxx", PINECONE_API_KEY="", OPENAI_API_KEY="")

    assert cfg.has_gemini() is False


def test_orchestrator_weights_sum_to_one():
    # _run_once() computes final_score as a weighted sum of the four layer
    # scores. If the weights don't sum to 1, "final_score >= pass_threshold"
    # stops meaning what the threshold's docstring says it means.
    cfg = Config()

    assert abs(sum(cfg.ORCHESTRATOR_WEIGHTS.values()) - 1.0) < 1e-9
