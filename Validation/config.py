"""
Configuration for the validation system.

Loads API keys from .env (project root) via python-dotenv.
Falls back to empty strings when keys are absent; every layer
that needs an API key checks for this and switches to mock mode.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Locate project root (parent of Validation/) and load .env from there
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on real env vars

# ---------------------------------------------------------------------------
# Helper: resolve a path that may be relative to the project root
# ---------------------------------------------------------------------------

def _resolve(rel: str) -> str:
    """Return absolute path from a project-root-relative string."""
    p = _PROJECT_ROOT / rel
    return str(p)


@dataclass
class Config:
    """Central configuration. All paths are absolute."""

    # API Keys (empty string = mock mode)
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    PINECONE_API_KEY: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Data paths, resolved to absolute at construction time
    REAL_DATA_PATH: str = field(default_factory=lambda: _resolve("data/raw/creditcard.csv"))
    SYNTHETIC_DATA_PATH: str = field(default_factory=lambda: _resolve("data/synthetic/demo_synthetic.csv"))
    REFERENCE_DATA_PATH: str = field(default_factory=lambda: _resolve("data/synthetic/reference_500.csv"))
    OUTPUT_DIR: str = field(default_factory=lambda: _resolve("reports/validation"))

    # Layer 1: Rule Validator
    LAYER1_ENABLE_GEMINI_RULES: bool = True
    LAYER1_MAX_GEMINI_RULES: int = 10
    LAYER1_GEMINI_MODEL: str = "gemini-2.0-flash-exp"

    # Layer 2: Statistical Validator
    LAYER2_KS_THRESHOLD: float = 0.05
    LAYER2_CORRELATION_THRESHOLD: float = 0.1
    LAYER2_OUTLIER_ZSCORE_THRESHOLD: float = 3.0

    # Layer 3: Semantic Validator
    LAYER3_SAMPLE_SIZE: int = 10
    LAYER3_GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    LAYER3_TEMPERATURE: float = 0.0
    LAYER3_BATCH_SIZE: int = 10

    # Layer 4: RAG Validator
    LAYER4_EMBEDDING_MODEL: str = "models/text-embedding-004"
    LAYER4_PINECONE_INDEX: str = "fraud-reference"
    LAYER4_TOP_K_NEIGHBORS: int = 10
    LAYER4_SIMILARITY_THRESHOLD: float = 0.4
    LAYER4_SAMPLE_SIZE: int = 20

    # Meta Validator
    META_CONSISTENCY_RUNS: int = 3
    META_AGREEMENT_THRESHOLD: float = 0.85

    # Performance
    BATCH_SIZE: int = 10000
    PARALLEL_WORKERS: int = 2

    # Success criteria
    TARGET_PASS_RATE: float = 0.90
    TARGET_DETECTION_RATE: float = 0.98
    TARGET_FALSE_POSITIVE_RATE: float = 0.05

    # Orchestrator
    ORCHESTRATOR_WEIGHTS: dict = field(default_factory=lambda: {
        "layer1": 0.20,
        "layer2": 0.40,
        "layer3": 0.25,
        "layer4": 0.15,
    })
    ORCHESTRATOR_PASS_THRESHOLD: float = 0.75
    ORCHESTRATOR_MAX_RETRIES: int = 2

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def has_gemini(self) -> bool:
        return bool(self.GEMINI_API_KEY and len(self.GEMINI_API_KEY) > 5)

    def has_pinecone(self) -> bool:
        return bool(self.PINECONE_API_KEY and len(self.PINECONE_API_KEY) > 5)

    def has_openai(self) -> bool:
        return bool(self.OPENAI_API_KEY and len(self.OPENAI_API_KEY) > 5)


# Global singleton
config = Config()

# Ensure output directory exists
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def validate_config() -> bool:
    """Print diagnostic summary. Returns True if minimum viable."""
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"  Gemini API key : {'SET' if config.has_gemini() else 'NOT SET (mock mode)'}")
    print(f"  Pinecone API key: {'SET' if config.has_pinecone() else 'NOT SET (sklearn fallback)'}")
    print(f"  OpenAI API key : {'SET' if config.has_openai() else 'NOT SET (not used)'}")
    print(f"  Real data      : {config.REAL_DATA_PATH}")
    print(f"  Synthetic data : {config.SYNTHETIC_DATA_PATH}")
    print(f"  Output dir     : {config.OUTPUT_DIR}")

    real_exists = os.path.exists(config.REAL_DATA_PATH)
    print(f"  Real data found: {real_exists}")
    if not real_exists:
        print("  WARNING: creditcard.csv not found, demo_data.py will generate from distributions")
    print("=" * 60)
    return True
