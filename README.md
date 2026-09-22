# Synthetic Data Generation Platform with AI Hallucination Detection

A system that generates realistic synthetic banking transactions using CTGAN and validates them through a multi-agent AI pipeline with hierarchical hallucination detection, catching bad outputs from both the generative model and the LLM validator itself. Privacy protection (differential privacy, k-anonymity) is designed but not yet built; see [Architecture](#architecture).

Built on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, 0.17% fraud rate), whose only features are already PCA-anonymized (`V1` through `V28`) plus `Time`, `Amount` and `Class`, so there is no raw PII in the source data to begin with. Generates 100K synthetic transactions intended to be statistically identical to real data.

---

## Why This Exists

Real transaction data is sensitive, regulated, and hard to share. Synthetic data solves this, but only if it's actually good. Bad synthetic data (hallucinations) can look plausible while being statistically impossible.

This project tackles both problems: generate high-fidelity fake data, then deploy an autonomous AI system to catch anything the generator got wrong.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR AGENT                     │
│         Pipeline manager · Pass/fail decisions           │
│         Auto-triggers CTGAN re-generation if needed      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  CTGAN        │  │  Privacy     │  │  Quality     │  │
│  │  Generator    │──│  Layer       │──│  Report      │  │
│  │  (100K rows)  │  │  (PLANNED,  │  │  (JSON)      │  │
│  │              │  │  not built)  │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌────────────────── VALIDATION AGENTS ────────────────┐ │
│  │                                                      │ │
│  │  Agent 1: Rule Validator                             │ │
│  │  Business logic + LLM-generated rule discovery       │ │
│  │                                                      │ │
│  │  Agent 2: Statistical Validator                      │ │
│  │  KS tests, correlations, autonomous failure drill-   │ │
│  │  down and diagnosis                                  │ │
│  │                                                      │ │
│  │  Agent 3: LLM Semantic Validator (Gemini)            │ │
│  │  Natural language reasoning on flagged records        │ │
│  │  Can request context from Agent 2                    │ │
│  │                                                      │ │
│  │  Agent 4: RAG Similarity Validator (Pinecone)        │ │
│  │  Embedding search for nearest real neighbors         │ │
│  │  Escalates to Agent 3 when no close match found      │ │
│  │                                                      │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  META-VALIDATION: LLM Hallucination Check            │ │
│  │  Verifies the validator's own reasoning is grounded   │ │
│  │  in actual data (catches validator hallucinations)    │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Two-level hallucination detection:**
- **Level 1:** CTGAN output: catches impossible synthetic transactions
- **Level 2:** LLM validator: catches when the LLM validator hallucinates during its own validation reasoning (e.g., claims "Amount is negative" when it's actually 50.00)

---

## Tech Stack

| Category | Tools |
|---|---|
| **Generation** | Python, SDV/CTGAN, PyTorch |
| **GenAI/LLM** | Gemini (`gemini-2.0-flash-exp`) for rule discovery and semantic validation |
| **RAG** | Pinecone vector database, `text-embedding-004` |
| **Validation** | SciPy (KS tests), Scikit-learn (outlier detection) |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Deployment** | Docker, Docker Compose |

---

## Project Structure

This is the actual layout, not an aspirational one. `notebooks/` holds CTGAN training (there is no standalone `src/generation/` script yet), and there is no `src/privacy/`; k-anonymity and differential privacy are not implemented (see the architecture diagram above).

```
synthetic-data-guard/
├── data/
│   ├── raw/                       # Kaggle dataset (not committed)
│   └── synthetic/                 # Generated synthetic data
├── notebooks/
│   ├── 01_ctgan_first_pass.ipynb
│   └── 02_ctgan_production_500epochs.ipynb
├── scripts/
│   ├── verify_setup.py            # Environment verification
│   └── eda_baseline.py            # EDA + baseline statistics
├── Validation/                    # Agentic validation system
│   ├── orchestrator.py
│   ├── layer1_rule_validator.py
│   ├── layer2_statistical.py
│   ├── layer3_semantic_gemini.py
│   ├── layer4_rag_pinecone.py
│   ├── meta_validator.py
│   └── config.py
├── tests/                         # pytest, the layers above that need no API key or dataset
├── reports/
│   ├── eda/                       # EDA visualizations + baseline JSON
│   └── validation/                # Validation reports (JSON)
├── app.py                         # Streamlit dashboard
├── api.py                         # FastAPI endpoints
├── demo_data.py
├── requirements.txt
├── requirements-dev.txt
├── .env.template
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- ~2 GB disk space (PyTorch + SDV are large)

### Setup

```bash
# Clone the repo
git clone https://github.com/abhirammv2000/synthetic-data-guard.git
cd synthetic-data-guard

# Create virtual environment
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

1. Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in `data/raw/creditcard.csv`

### Verify Installation

```bash
python scripts/verify_setup.py
```

All checks should pass, including a CTGAN smoke test that trains on 500 rows and generates 5 synthetic records.

### API Keys (needed for validation agents)

Copy `.env.template` to `.env` and add your keys:

```bash
cp .env.template .env
```

```
GEMINI_API_KEY=your-key-here
PINECONE_API_KEY=your-key-here
PINECONE_ENVIRONMENT=your-environment
```

Without a `GEMINI_API_KEY`, Layer 1's rule discovery and Layer 3 fall back to mock mode (`Config.has_gemini()` returns `False`); without a `PINECONE_API_KEY`, Layer 4 falls back to scikit-learn nearest-neighbor search.

---

## Dataset

**Kaggle Credit Card Fraud Detection**: 284,807 European credit card transactions from September 2013.

| Column | Description |
|---|---|
| `Time` | Seconds elapsed from first transaction |
| `V1` to `V28` | PCA-transformed features (anonymized) |
| `Amount` | Transaction amount |
| `Class` | 0 = legitimate, 1 = fraud (0.17% fraud rate) |

The extreme class imbalance (492 fraud out of 284,807) is a key challenge, so synthetic generation must preserve this ratio, and the validation system must catch drift.

---

## How It Works

### 1. Generate
CTGAN learns the joint distribution of all 31 features from real data and generates 100K synthetic transactions. **A differential-privacy and k-anonymity layer is planned but not yet built** (see the architecture diagram above).

### 2. Validate
Four autonomous agents, coordinated by an orchestrator:

- **Rule Agent:** enforces business constraints (amount ranges, feature bounds) and discovers new rules by prompting Gemini to analyze the real dataset
- **Statistical Agent:** runs Kolmogorov-Smirnov tests, correlation comparisons, and distribution matching. Autonomously drills into failures to diagnose root causes
- **LLM Agent:** Gemini reads flagged transactions and reasons about whether they make business sense. Can pull additional context from the Statistical Agent
- **RAG Agent:** embeds synthetic records using `text-embedding-004`, searches Pinecone for nearest real neighbors. No close match → escalates to LLM Agent

Agents communicate with each other. The orchestrator can loop, retry, and trigger re-generation without human intervention.

### 3. Meta-Validate
A separate check verifies that the semantic validator's own reasoning is grounded in actual data, catching cases where the LLM hallucinates facts about the records it's reviewing.

---

## API (`api.py`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/generate` | Generate synthetic records |
| `GET` | `/validate/{batch_id}` | Run validation on a batch |
| `GET` | `/report/{batch_id}` | Get quality report |
| `GET` | `/health` | Health check |

---

## Testing

Automated coverage is currently limited to the parts of the pipeline that need no API key, no Pinecone index and no dataset: `Config`'s mock-mode gating, `MetaValidator`'s contradiction/hallucination checks, and `Orchestrator`'s self-healing retry and pass/fail decision logic (tested by substituting a controlled sequence of scores for `_run_once`, so the four heavy validation layers are never invoked).

```bash
pip install -r requirements-dev.txt
pytest
```

The four validation layers themselves (`layer1_rule_validator.py` through `layer4_rag_pinecone.py`) are exercised manually via `Validation/test_validation.py`, which needs the real dataset and, for Layer 1's Gemini rule discovery, a live API key. It is a CLI script with a cost-confirmation prompt, not an automated test suite.
