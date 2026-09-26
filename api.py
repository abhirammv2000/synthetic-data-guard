"""
FastAPI REST interface for the Synthetic Data Validation Platform.

Endpoints
---------
GET  /health    : service status
POST /generate  : create synthetic data
POST /validate  : generate + validate through the full pipeline
GET  /report    : last validation run summary

Run:
    uvicorn api:app --reload --port 8000
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo_data import generate_synthetic, generate_reference
from Validation.config import config
from Validation.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Synthetic Data Validation Platform",
    version="1.0.0",
    description="Multi-agent validation with hierarchical hallucination detection",
)

# In-memory store for last run
_last_report: dict = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    n_samples: int = 500
    fraud_rate: float = 0.00173
    seed: int = 42


class ValidateRequest(BaseModel):
    n_samples: int = 100
    fraud_rate: float = 0.00173
    seed: int = 42


# ---------------------------------------------------------------------------
# Logging middleware (simple stdout)
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {request.method} {request.url.path} -> {response.status_code} ({elapsed:.3f}s)")
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gemini_available": config.has_gemini(),
        "pinecone_available": config.has_pinecone(),
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        t0 = time.perf_counter()
        df = generate_synthetic(n=req.n_samples, fraud_rate=req.fraud_rate, seed=req.seed)
        elapsed = time.perf_counter() - t0

        preview = df.head(5).to_dict(orient="records")
        # Convert numpy types in preview
        for row in preview:
            for k, v in row.items():
                if hasattr(v, "item"):
                    row[k] = v.item()

        return {
            "status": "success",
            "n_samples": len(df),
            "fraud_count": int(df["Class"].sum()),
            "fraud_rate": round(float(df["Class"].mean()), 5),
            "generation_time": round(elapsed, 4),
            "preview": preview,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
def validate(req: ValidateRequest):
    global _last_report
    try:
        # Generate fresh data
        df = generate_synthetic(n=req.n_samples, fraud_rate=req.fraud_rate, seed=req.seed)
        ref = generate_reference(n=500, seed=0)

        # Self-healing generation function
        def regen(seed):
            return generate_synthetic(n=req.n_samples, fraud_rate=req.fraud_rate, seed=seed)

        # Run orchestrator
        orch = Orchestrator()
        result = orch.run(df, ref, _generation_fn=regen)

        # Strip raw_response (can be large) for the API response
        if "agents" in result and "layer3" in result["agents"]:
            result["agents"]["layer3"].pop("raw_response", None)

        _last_report = result
        return {"status": "success", **result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report")
def report():
    if not _last_report:
        raise HTTPException(status_code=404, detail="No validation has been run yet. POST /validate first.")

    # Return a clean summary
    agents = _last_report.get("agents", {})
    summary = {
        "pipeline_decision": _last_report.get("pipeline_decision"),
        "final_score": _last_report.get("final_score"),
        "attempt": _last_report.get("attempt"),
        "wall_time": _last_report.get("wall_time"),
        "agent_scores": _last_report.get("scores"),
        "agent_passed": {
            name: info.get("passed") for name, info in agents.items() if name != "meta"
        },
        "meta_validation": agents.get("meta", {}),
        "timings": _last_report.get("timings"),
    }
    return summary
