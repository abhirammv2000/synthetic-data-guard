"""
Layer 4: RAG Similarity Validator

Compares synthetic records against real reference data using cosine
similarity on normalised feature vectors.

Default path: sklearn (no external dependency).
Optional path: Pinecone vector DB when PINECONE_API_KEY is set.

Escalation: if the anomaly rate exceeds 15 % the orchestrator should
re-run Layer 3 with additional context.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from Validation.config import config

_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ---------------------------------------------------------------------------
# sklearn-based local similarity search
# ---------------------------------------------------------------------------

def _sklearn_similarity(
    synthetic: pd.DataFrame,
    reference: pd.DataFrame,
    sample_size: int,
    threshold: float,
    seed: int = 42,
) -> tuple:
    """Compute cosine similarity between sampled synthetic rows and reference."""
    # Sample synthetic rows
    n = min(sample_size, len(synthetic))
    sampled = synthetic.sample(n=n, random_state=seed)
    sampled_indices = sampled.index.tolist()

    # Fit scaler on reference, transform both
    scaler = StandardScaler()
    ref_features = reference[_FEATURE_COLS].values
    scaler.fit(ref_features)

    ref_scaled = scaler.transform(ref_features)
    syn_scaled = scaler.transform(sampled[_FEATURE_COLS].values)

    # Cosine similarity: each synthetic row vs all reference rows
    sim_matrix = cosine_similarity(syn_scaled, ref_scaled)  # (n, len(ref))
    max_similarities = sim_matrix.max(axis=1)  # best match per row

    flagged = []
    for i, (idx, max_sim) in enumerate(zip(sampled_indices, max_similarities)):
        if max_sim < threshold:
            flagged.append({
                "row_index": int(idx),
                "max_similarity": round(float(max_sim), 4),
            })

    return sampled_indices, max_similarities.tolist(), flagged


# ---------------------------------------------------------------------------
# Pinecone-based vector search (optional)
# ---------------------------------------------------------------------------

def _pinecone_similarity(
    synthetic: pd.DataFrame,
    reference: pd.DataFrame,
    sample_size: int,
    threshold: float,
    seed: int = 42,
) -> tuple:
    """Use Pinecone for nearest-neighbour search."""
    from pinecone import Pinecone

    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    index_name = config.LAYER4_PINECONE_INDEX
    dimension = len(_FEATURE_COLS)  # 29

    # Create index if it does not exist
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )

    index = pc.Index(index_name)

    # Normalise features
    scaler = StandardScaler()
    ref_features = reference[_FEATURE_COLS].values
    scaler.fit(ref_features)

    # Upsert reference vectors
    ref_scaled = scaler.transform(ref_features)
    vectors = [
        {"id": f"ref_{i}", "values": ref_scaled[i].tolist()}
        for i in range(len(ref_scaled))
    ]
    # Batch upsert
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[start : start + batch_size])

    # Sample and query
    n = min(sample_size, len(synthetic))
    sampled = synthetic.sample(n=n, random_state=seed)
    sampled_indices = sampled.index.tolist()
    syn_scaled = scaler.transform(sampled[_FEATURE_COLS].values)

    max_similarities = []
    flagged = []
    for i, (idx, vec) in enumerate(zip(sampled_indices, syn_scaled)):
        result = index.query(vector=vec.tolist(), top_k=1)
        top_score = result.matches[0].score if result.matches else 0.0
        max_similarities.append(top_score)
        if top_score < threshold:
            flagged.append({
                "row_index": int(idx),
                "max_similarity": round(float(top_score), 4),
            })

    return sampled_indices, max_similarities, flagged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RAGValidator:
    """Layer 4: RAG-based similarity validation."""

    def validate(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
        sample_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Synthetic dataset to validate.
        reference_df : pd.DataFrame
            Real (or high-quality reference) data for comparison.
        sample_size : int, optional
            Number of synthetic rows to sample (default from config).
        threshold : float, optional
            Minimum cosine similarity to consider a row non-anomalous.

        Returns
        -------
        dict with keys: agent, passed, rows_sampled, anomalies_detected,
             anomaly_rate, escalated, score, flagged_indices,
             execution_time, backend.
        """
        t0 = time.perf_counter()
        sample_size = sample_size or config.LAYER4_SAMPLE_SIZE
        threshold = threshold or config.LAYER4_SIMILARITY_THRESHOLD

        if config.has_pinecone():
            try:
                sampled_idx, sims, flagged = _pinecone_similarity(
                    df, reference_df, sample_size, threshold
                )
                backend = "pinecone"
            except Exception as e:
                print(f"  Pinecone failed ({e}), falling back to sklearn")
                sampled_idx, sims, flagged = _sklearn_similarity(
                    df, reference_df, sample_size, threshold
                )
                backend = "sklearn (pinecone fallback)"
        else:
            sampled_idx, sims, flagged = _sklearn_similarity(
                df, reference_df, sample_size, threshold
            )
            backend = "sklearn"

        n_sampled = len(sampled_idx)
        n_anomalies = len(flagged)
        anomaly_rate = n_anomalies / max(n_sampled, 1)
        escalated = anomaly_rate > 0.15
        score = max(0.0, 1.0 - anomaly_rate)
        passed = anomaly_rate <= 0.15

        elapsed = time.perf_counter() - t0

        return {
            "agent": "RAGValidator",
            "passed": passed,
            "rows_sampled": n_sampled,
            "anomalies_detected": n_anomalies,
            "anomaly_rate": round(anomaly_rate, 4),
            "escalated": escalated,
            "score": round(score, 4),
            "flagged_indices": [f["row_index"] for f in flagged],
            "execution_time": round(elapsed, 4),
            "backend": backend,
        }
