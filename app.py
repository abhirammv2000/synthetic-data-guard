"""
Streamlit dashboard for the Synthetic Data Validation Platform.

Run:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo_data import generate_synthetic, generate_reference
from Validation.config import config

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Synthetic Data Validation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in [
    ("df_synthetic", None),
    ("df_reference", None),
    ("orch_result", None),
    ("last_n", 500),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Controls")
n_samples = st.sidebar.slider("Sample size", 100, 2000, 500, step=100)
fraud_rate = st.sidebar.number_input("Fraud rate", value=0.00173, format="%.5f")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Generation",
    "Validation Pipeline",
    "Statistical Quality",
    "Hallucination Detection",
    "Architecture",
])


# ===================================================================
# TAB 1: Data Generation
# ===================================================================
with tab1:
    st.header("Synthetic Data Generation")

    if st.button("Generate Synthetic Data", key="gen_btn"):
        with st.spinner("Generating synthetic transactions via CTGAN pipeline..."):
            t0 = time.perf_counter()
            df = generate_synthetic(n=n_samples, fraud_rate=fraud_rate, seed=42)
            ref = generate_reference(n=500, seed=0)
            elapsed = time.perf_counter() - t0

        st.session_state.df_synthetic = df
        st.session_state.df_reference = ref
        st.session_state.last_n = n_samples
        st.success(f"Generated {len(df)} rows in {elapsed:.2f}s")

    df = st.session_state.df_synthetic
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows Generated", f"{len(df):,}")
        col2.metric("Fraud Count", int(df["Class"].sum()))
        col3.metric("Fraud Rate", f"{df['Class'].mean():.4%}")

        # Fraud rate comparison chart
        fig = go.Figure(data=[
            go.Bar(
                x=["Real Data", "Synthetic"],
                y=[0.173, df["Class"].mean() * 100],
                text=[f"0.173%", f"{df['Class'].mean()*100:.3f}%"],
                textposition="auto",
                marker_color=["#3498db", "#e67e22"],
            ),
        ])
        fig.update_layout(
            title="Fraud Rate Comparison (%)",
            yaxis_title="Fraud Rate (%)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Preview (first 20 rows)")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Click 'Generate Synthetic Data' to begin.")


# ===================================================================
# TAB 2: Validation Pipeline
# ===================================================================
with tab2:
    st.header("Multi-Agent Validation Pipeline")

    if st.button("Run Full Validation", key="val_btn"):
        df = st.session_state.df_synthetic
        ref = st.session_state.df_reference

        if df is None:
            st.warning("Generate data first (Tab 1).")
        else:
            from Validation.orchestrator import Orchestrator

            def regen(seed):
                return generate_synthetic(
                    n=st.session_state.last_n,
                    fraud_rate=fraud_rate,
                    seed=seed,
                )

            status = st.empty()
            progress = st.progress(0)

            status.info("Running Layer 1 -- Rule-Based Validator...")
            progress.progress(10)

            orch = Orchestrator()

            # Run full pipeline (stdout captured by Streamlit)
            t0 = time.perf_counter()
            result = orch.run(df, ref, _generation_fn=regen)
            wall = time.perf_counter() - t0

            progress.progress(100)
            status.empty()

            st.session_state.orch_result = result

            # Decision banner
            decision = result.get("pipeline_decision", "UNKNOWN")
            if decision == "PASS":
                st.success(f"Pipeline Decision: PASS  |  Final Score: {result['final_score']:.4f}  |  {wall:.1f}s")
            else:
                st.error(f"Pipeline Decision: {decision}  |  Final Score: {result['final_score']:.4f}  |  {wall:.1f}s")

            if result.get("retry_count", 0) > 0:
                st.warning(
                    f"Quality threshold not met -- triggered re-generation. "
                    f"Attempt {result['attempt']}/{orch.max_retries + 1}"
                )

    result = st.session_state.orch_result
    if result:
        agents = result.get("agents", {})
        scores = result.get("scores", {})
        timings = result.get("timings", {})

        st.subheader("Agent Results")
        cols = st.columns(4)

        agent_labels = [
            ("layer1", "Layer 1: Rule-Based"),
            ("layer2", "Layer 2: Statistical"),
            ("layer3", "Layer 3: LLM Semantic"),
            ("layer4", "Layer 4: RAG Similarity"),
        ]
        for col, (key, label) in zip(cols, agent_labels):
            info = agents.get(key, {})
            passed = info.get("passed", False)
            score = scores.get(key, 0)
            t = timings.get(key, 0)

            with col:
                st.markdown(f"**{label}**")
                st.markdown(f"{'PASS' if passed else 'FAIL'}")
                st.progress(min(score, 1.0))
                st.caption(f"Score: {score:.3f}  |  {t:.2f}s")

                # Key finding
                if key == "layer1":
                    inv = info.get("invalid_count", 0)
                    st.caption(f"{inv} rule violations found")
                elif key == "layer2":
                    kp = info.get("metrics", {}).get("ks_tests_passed", 0)
                    kt = info.get("metrics", {}).get("ks_tests_total", 31)
                    st.caption(f"KS tests: {kp}/{kt} passed")
                elif key == "layer3":
                    na = len(info.get("anomalies", []))
                    st.caption(f"{na} semantic anomalies flagged")
                elif key == "layer4":
                    ar = info.get("anomaly_rate", 0)
                    st.caption(f"Anomaly rate: {ar:.1%}")

        # Final score breakdown
        st.subheader("Weighted Score Breakdown")
        weights = config.ORCHESTRATOR_WEIGHTS
        breakdown = pd.DataFrame({
            "Layer": [l for l in weights],
            "Weight": [weights[l] for l in weights],
            "Score": [scores.get(l, 0) for l in weights],
            "Contribution": [
                round(scores.get(l, 0) * weights[l], 4) for l in weights
            ],
        })
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

        # Retry log
        retry_log = result.get("retry_log", [])
        if retry_log:
            st.subheader("Self-Healing Log")
            for entry in retry_log:
                st.write(f"Attempt {entry['attempt']}: {entry['reason']}")


# ===================================================================
# TAB 3: Statistical Quality
# ===================================================================
with tab3:
    st.header("Statistical Quality Analysis")

    result = st.session_state.orch_result
    if result is None:
        st.info("Run validation first (Tab 2).")
    else:
        l2 = result.get("agents", {}).get("layer2", {})
        ks_details = l2.get("ks_details", [])

        if ks_details:
            ks_df = pd.DataFrame(ks_details)

            # KS bar chart
            colors = [
                "#2ecc71" if row["ks_statistic"] < 0.05
                else "#f1c40f" if row["ks_statistic"] < 0.10
                else "#e74c3c"
                for _, row in ks_df.iterrows()
            ]

            fig = go.Figure(data=[
                go.Bar(
                    x=ks_df["feature"],
                    y=ks_df["ks_statistic"],
                    marker_color=colors,
                ),
            ])
            fig.add_hline(y=0.05, line_dash="dash", line_color="green",
                          annotation_text="Excellent (0.05)")
            fig.add_hline(y=0.10, line_dash="dash", line_color="orange",
                          annotation_text="Good (0.10)")
            fig.update_layout(
                title="KS Statistics by Feature",
                xaxis_title="Feature",
                yaxis_title="KS Statistic",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            n_pass = int(ks_df["passed"].sum())
            n_total = len(ks_df)
            c1, c2 = st.columns(2)
            c1.metric("Features Passing KS < 0.05", f"{n_pass} / {n_total}")
            c2.metric("Median KS Statistic", f"{ks_df['ks_statistic'].median():.4f}")

            # Fraud rate preservation
            df = st.session_state.df_synthetic
            if df is not None:
                st.subheader("Fraud Rate Preservation")
                fc1, fc2 = st.columns(2)
                fc1.metric("Real Fraud Rate", "0.1727%")
                fc2.metric("Synthetic Fraud Rate", f"{df['Class'].mean()*100:.4f}%")

            # Correlation heatmap
            st.subheader("Correlation Difference (Real - Synthetic)")
            df_syn = st.session_state.df_synthetic
            ref = st.session_state.df_reference
            if df_syn is not None and ref is not None:
                v_cols = [f"V{i}" for i in range(1, 29)]
                common = [c for c in v_cols if c in ref.columns and c in df_syn.columns]
                if common:
                    real_corr = ref[common].corr()
                    syn_corr = df_syn[common].corr()
                    diff = real_corr - syn_corr

                    fig2 = go.Figure(data=go.Heatmap(
                        z=diff.values,
                        x=common,
                        y=common,
                        colorscale="RdBu_r",
                        zmid=0,
                        zmin=-0.5,
                        zmax=0.5,
                    ))
                    fig2.update_layout(
                        title="Correlation Difference Heatmap (V-features)",
                        height=500,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No KS test details available.")


# ===================================================================
# TAB 4: Hallucination Detection
# ===================================================================
with tab4:
    st.header("Hierarchical Hallucination Detection")

    result = st.session_state.orch_result
    if result is None:
        st.info("Run validation first (Tab 2).")
    else:
        left, right = st.columns(2)

        # ---- Left: Level 1, CTGAN Output Hallucinations ----
        with left:
            st.subheader("Level 1 -- CTGAN Output Hallucinations")

            l1 = result["agents"].get("layer1", {})
            l4 = result["agents"].get("layer4", {})

            l1_inv = l1.get("invalid_count", 0)
            l4_anom = l4.get("anomalies_detected", 0)

            st.metric("Layer 1 Rule Violations", l1_inv)
            st.metric("Layer 4 RAG Anomalies", l4_anom)

            top_failures = l1.get("top_failures", {})
            if top_failures:
                st.markdown("**Top Rule Failures:**")
                for rule, count in top_failures.items():
                    st.write(f"- `{rule}`: {count} violations")
            else:
                st.write("No rule violations detected.")

            flagged = l4.get("flagged_indices", [])
            if flagged:
                st.markdown(f"**RAG-flagged row indices:** {flagged[:20]}")
            else:
                st.write("No RAG anomalies detected.")

        # ---- Right: Level 2, LLM Reasoning Hallucinations ----
        with right:
            st.subheader("Level 2 -- LLM Reasoning Hallucinations")

            meta = result["agents"].get("meta", {})
            l3 = result["agents"].get("layer3", {})

            meta_passed = meta.get("meta_validation_passed", True)

            if meta_passed:
                st.success("Meta-validation: PASSED (no contradictions)")
            else:
                st.error("Meta-validation: FAILED")

            contradictions = meta.get("contradictions_found", [])
            hallucinations = meta.get("hallucinations_detected", [])

            if contradictions:
                st.markdown("**Contradictions Found:**")
                for c in contradictions:
                    st.warning(c)
            else:
                st.write("No contradictions detected.")

            if hallucinations:
                st.markdown("**Hallucinations Detected:**")
                for h in hallucinations:
                    st.error(h)
            else:
                st.write("No hallucinations detected.")

            adj = meta.get("confidence_adjustment", 0)
            adj_score = meta.get("adjusted_score", 0)
            st.metric("Confidence Adjustment", f"{adj:+.2f}")
            st.metric("Adjusted Layer 3 Score", f"{adj_score:.4f}")

            # LLM anomaly observations
            anomalies = l3.get("anomalies", [])
            if anomalies:
                with st.expander(f"LLM Anomaly Observations ({len(anomalies)} items)"):
                    for a in anomalies:
                        st.write(
                            f"**Row {a.get('row_index', '?')}** "
                            f"({a.get('feature', '?')}): "
                            f"{a.get('observation', '')}"
                        )
            else:
                st.write("No LLM anomalies reported.")


# ===================================================================
# TAB 5: Architecture
# ===================================================================
with tab5:
    st.header("System Architecture")

    st.code("""
+-----------------------------------------------------------+
|                    ORCHESTRATOR AGENT                      |
|        Pipeline manager - Pass/fail decisions             |
|        Auto-triggers CTGAN re-generation if needed        |
+-----------------------------------------------------------+
|                                                           |
|  +-------------+   +-----------+   +------------------+  |
|  | CTGAN        |   | Privacy   |   | Quality          |  |
|  | Generator    |-->| Layer     |-->| Report           |  |
|  | (demo_data)  |   | (k-anon)  |   | (JSON)           |  |
|  +-------------+   +-----------+   +------------------+  |
|                                                           |
|  +---------- VALIDATION AGENTS -----------------------+  |
|  |                                                     |  |
|  |  Agent 1: Rule Validator (layer1)                   |  |
|  |  Business logic + Gemini-generated rule discovery   |  |
|  |                                                     |  |
|  |  Agent 2: Statistical Validator (layer2)            |  |
|  |  KS tests, correlations, Isolation Forest           |  |
|  |                                                     |  |
|  |  Agent 3: LLM Semantic Validator (layer3)           |  |
|  |  Gemini 2.0 Flash reasoning on flagged records      |  |
|  |  Receives escalation context from Agent 4           |  |
|  |                                                     |  |
|  |  Agent 4: RAG Similarity Validator (layer4)         |  |
|  |  Cosine similarity / Pinecone nearest-neighbor      |  |
|  |  Escalates to Agent 3 when anomaly rate > 15%       |  |
|  |                                                     |  |
|  +----------------------------------------------------+  |
|                                                           |
|  +----------------------------------------------------+  |
|  |  META-VALIDATION: LLM Hallucination Check           |  |
|  |  Verifies Gemini's reasoning is grounded in data    |  |
|  |  Catches contradictions + fabricated columns        |  |
|  +----------------------------------------------------+  |
+-----------------------------------------------------------+

WEIGHTED SCORING:
  Layer 1 (Rules)       : 20%
  Layer 2 (Statistical) : 40%
  Layer 3 (LLM)         : 25%
  Layer 4 (RAG)         : 15%
  Pass threshold        : 0.75

SELF-HEALING:
  If final_score < 0.75 -> re-generate data (max 2 retries)
    """, language="text")

    st.subheader("Technology Stack")

    tech_data = {
        "Category": [
            "Generation", "GenAI / LLM", "RAG / Vector DB",
            "Validation", "API", "Dashboard", "Language",
        ],
        "Tools": [
            "SDV / CTGAN, PyTorch",
            "Gemini 2.0 Flash (google-genai)",
            "Pinecone (optional), sklearn cosine similarity",
            "SciPy (KS tests), scikit-learn (Isolation Forest)",
            "FastAPI, Uvicorn",
            "Streamlit, Plotly",
            "Python 3.10+",
        ],
    }
    st.table(pd.DataFrame(tech_data))
