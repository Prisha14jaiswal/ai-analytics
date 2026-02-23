"""
Universal AI Analytics Assistant — Refactored Streamlit Application.

Structure:
1. Always-visible Upload Section
2. Tabbed Interface (Dataset Overview, Ask Questions, Train Model)
3. RAG and Agent logic consolidated under 'Ask Questions'
"""

import os
import sys
import re
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ── Env setup ────────────────────────────────────────────────────────────────
for env_path in [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path)
        break

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Analytics Assistant",
    page_icon="🧠",
    layout="wide",
)

# ── Styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 8px; padding: 16px;
        border-left: 4px solid #4f46e5; margin-bottom: 8px;
    }
    .metric-label { font-size: 0.85em; color: #6b7280; }
    .metric-value { font-size: 1.4em; font-weight: 700; color: #1f2937; }
    .stTabs [data-baseweb="tab-list"] { gap: 16px; margin-bottom: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        font-weight: 600; 
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Visualization Handler ──────────────────────────────────────────────

def render_visualization(df: pd.DataFrame, spec: dict):
    """Hardened Matplotlib renderer with validation and error handling."""
    try:
        chart_type = spec.get("chart_type", "bar")
        x_col = spec.get("x_column")
        y_col = spec.get("y_column")
        agg = spec.get("aggregation")
        title = spec.get("title", "Data Insight")

        if not x_col or x_col not in df.columns:
            st.error(f"❌ X-axis column '{x_col}' is invalid or missing.")
            return
        if y_col and y_col not in df.columns:
            st.error(f"❌ Y-axis column '{y_col}' is invalid.")
            return

        plot_df = df.copy()
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            if agg and y_col:
                data = plot_df.groupby(x_col)[y_col].agg(agg).sort_values(ascending=False).head(15)
                data.plot(kind="bar", ax=ax, color="#4f46e5")
                ax.set_ylabel(f"{agg.capitalize()} of {y_col}")
            else:
                data = plot_df[x_col].value_counts().head(15)
                data.plot(kind="bar", ax=ax, color="#4f46e5")
                ax.set_ylabel("Frequency")

        elif chart_type == "line":
            if agg and y_col:
                data = plot_df.groupby(x_col)[y_col].agg(agg)
                data.plot(kind="line", ax=ax, marker='o', color="#4f46e5")
                ax.set_ylabel(f"{agg.capitalize()} of {y_col}")
            else:
                data = plot_df[x_col].value_counts().sort_index()
                data.plot(kind="line", ax=ax, marker='o', color="#4f46e5")

        elif chart_type == "histogram":
            plot_df[x_col].plot(kind="hist", bins=20, ax=ax, color="#4f46e5", edgecolor="white")
            ax.set_xlabel(x_col)

        elif chart_type == "scatter" and y_col:
            plot_df.plot(x=x_col, y=y_col, kind="scatter", ax=ax, alpha=0.5, color="#4f46e5")

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Renderer Error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🧠 AI Analytics Assistant")
st.sidebar.caption("Universal Data Intelligence")

if "df" in st.session_state:
    df = st.session_state.df
    st.sidebar.success(f"✅ Dataset Active\n\n**{df.shape[0]:,}** rows\n**{df.shape[1]}** columns")
    if st.sidebar.button("🗑️ Clear Dataset"):
        for key in ["df", "summary", "summary_text", "rag_chain", "data_agent"]:
            st.session_state.pop(key, None)
        st.rerun()
else:
    st.sidebar.info("Upload a CSV to begin analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8em;'>"
    "Powered by LangChain · FAISS · Groq · scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Step 1: Always-visible Upload
# ══════════════════════════════════════════════════════════════════════════════

st.title("📁 Upload & Analyze")
uploaded = st.file_uploader("Drop your CSV file here", type=["csv"], label_visibility="collapsed")

if uploaded is not None:
    if "df" not in st.session_state or st.session_state.get("_last_uploaded") != uploaded.name:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.session_state._last_uploaded = uploaded.name
            # Reset dependent states
            for key in ["summary", "summary_text", "rag_chain", "data_agent"]:
                st.session_state.pop(key, None)
        except Exception as e:
            st.error(f"Critical upload error: {e}")
            st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Step 2: Tabs (Conditional)
# ══════════════════════════════════════════════════════════════════════════════

if "df" in st.session_state:
    df = st.session_state.df
    
    t1, t2, t3 = st.tabs(["📊 Dataset Overview", "💬 Ask Questions", "🤖 Train Model"])

    # ── TAB 1: Dataset Overview ──────────────────────────────────────────────
    with t1:
        if "summary" not in st.session_state:
            from schema_analyzer import analyze_schema, summary_to_text
            with st.spinner("Profiling dataset..."):
                summary = analyze_schema(df)
                st.session_state.summary = summary
                st.session_state.summary_text = summary_to_text(summary)
        
        summary = st.session_state.summary
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{summary['n_rows']:,}")
        c2.metric("Columns", summary["n_cols"])
        c3.metric("Memory", f"{summary['memory_usage_mb']} MB")
        c4.metric("Missing Values", f"{sum(summary['missing_values'].values()):,}")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Column Types")
            dtype_df = pd.DataFrame([(c, t) for c, t in summary["dtypes"].items()], columns=["Column", "Type"])
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)
            
        with col_b:
            cols_with_missing = [c for c, v in summary["missing_values"].items() if v > 0]
            if cols_with_missing:
                st.subheader("Missing Data")
                missing_counts = [summary["missing_values"][c] for c in cols_with_missing]
                missing_pcts = [f"{summary['missing_pct'][c]}%" for c in cols_with_missing]
                missing_df = pd.DataFrame({
                    "Column": cols_with_missing,
                    "Count": missing_counts,
                    "Percentage": missing_pcts
                })
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No missing values detected.")

        if summary["numeric_summary"]:
            st.subheader("Numeric Statistics")
            st.dataframe(pd.DataFrame(summary["numeric_summary"]).round(2), use_container_width=True)

        if summary["categorical_summary"]:
            st.subheader("Categorical Details")
            for col, info in summary["categorical_summary"].items():
                with st.expander(f"**{col}** — {info['unique_count']} unique values"):
                    st.dataframe(pd.DataFrame(list(info["top_values"].items()), columns=["Value", "Count"]), hide_index=True)

    # ── TAB 2: Ask Questions (RAG + Agent) ───────────────────────────────────
    with t2:
        mode = st.radio("Intelligence Mode", ["🐼 Data Agent (Math/Charts)", "📚 RAG (Schema/General)"], horizontal=True)
        
        if mode == "📚 RAG (Schema/General)":
            st.info("Best for questions about the dataset structure and broad summaries.")
            if "rag_chain" not in st.session_state:
                from rag.build_vector_store import build_vector_store
                from rag.rag_pipeline import build_rag_chain
                with st.spinner("Indexing report..."):
                    vs = build_vector_store(st.session_state.summary_text)
                    st.session_state.rag_chain = build_rag_chain(vs)
            
            q = st.text_input("Ask about the dataset metadata:", placeholder="What are the missing values?", key="rag_input")
            if st.button("Query RAG", type="primary"):
                with st.spinner("Retrieving..."):
                    ans = st.session_state.rag_chain.invoke(q)
                    st.markdown("### Answer")
                    st.markdown(ans)

        else:
            st.info("Best for complex calculations, aggregations, and creating visualizations.")
            if "data_agent" not in st.session_state:
                from agent.dataframe_agent import build_dataframe_agent
                with st.spinner("Activating agent..."):
                    st.session_state.data_agent = build_dataframe_agent(df)
            
            q = st.text_input("Ask for analysis or a chart:", placeholder="Show price distribution histogram", key="agent_input")
            if st.button("Run Analytics", type="primary"):
                with st.spinner("Computing..."):
                    from agent.dataframe_agent import ask_agent
                    answer = ask_agent(st.session_state.data_agent, q).strip()
                    def extract_spec(text):
                        text = re.sub(r'```(?:json)?', '', text).strip()
                        match = re.search(r'\{.*"visualization":\s*true.*\}', text, re.DOTALL)
                        if match:
                            try: return json.loads(match.group(0))
                            except: return None
                        return None
                    spec = extract_spec(answer)
                    if spec:
                        st.subheader("Visualization")
                        render_visualization(df, spec)
                    else:
                        st.subheader("Analysis")
                        st.markdown(answer)

    # ── TAB 3: Train Model ───────────────────────────────────────────────────
    with t3:
        st.subheader("🤖 Quick AutoML Classifier")
        from ml.auto_trainer import AVAILABLE_MODELS, prepare_data, train_model, results_to_text
        
        c1, c2, c3 = st.columns([2, 2, 1])
        target = c1.selectbox("Target Column", df.columns.tolist())
        model_name = c2.selectbox("Model Type", list(AVAILABLE_MODELS.keys()))
        test_size = c3.slider("Test Size", 0.1, 0.4, 0.2)
        
        if st.button("🚀 Train Now", type="primary"):
            try:
                X_tr, X_te, y_tr, y_te, info = prepare_data(df, target, test_size)
                res = train_model(model_name, X_tr, X_te, y_tr, y_te, info)
                
                st.success(f"✅ Training Complete — Accuracy: {res['accuracy']*100:.1f}%")
                
                m1, m2 = st.tabs(["Performance", "Feature Importance"])
                with m1:
                    st.code(results_to_text(res))
                    st.subheader("Classification Report")
                    st.dataframe(pd.DataFrame(res["classification_report"]).T.round(3), use_container_width=True)
                with m2:
                    if res.get("feature_importance"):
                        fi_df = pd.DataFrame(list(res["feature_importance"].items()), columns=["Feature", "Score"])
                        st.bar_chart(fi_df.set_index("Feature"), horizontal=True)
                    else:
                        st.info("Feature importance not supported for this model type.")
            except Exception as e:
                st.error(f"Training failed: {e}")

else:
    st.warning("Please upload a CSV file above to unlock the analytics dashboard.")
