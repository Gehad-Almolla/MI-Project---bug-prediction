"""
app.py  —  Buggy Commit Predictor
Supports: Logistic Regression · Decision Tree · Random Forest · SVM · ANN · LightGBM
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Buggy Commit Predictor",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "parent_count",
    "files_changed",
    "additions",
    "deletions",
    "total_changes",
    "message_length",
    "commit_hour",
    "commit_day",
    "developer_experience",
    "directories_changed",
]

# Each entry: display_name -> dict with paths and type
MODEL_REGISTRY = {
    "Random Forest": {
        "type": "sklearn",
        "path": "models/random_forest.pkl",
    },
    "Logistic Regression": {
        "type": "sklearn",
        "path": "models/logistic_regression.pkl",
    },
    "Decision Tree": {
        "type": "sklearn",
        "path": "models/decision_tree.pkl",
    },
    "SVM": {
        "type": "sklearn",
        "path": "models/svm.pkl",          # Pipeline: scaler + SVC
    },
    "LightGBM": {
        "type": "sklearn",                  # LGBMClassifier is sklearn-compatible
        "path": "models/lightgbm.pkl",
    },
    "ANN (Neural Network)": {
        "type": "keras",
        "model_path":  "models/ann_model.keras",
        "scaler_path": "models/ann_scaler.pkl",
    },
}

# ── Metrics from notebook runs (update with real outputs) ────────────────────
MODEL_METRICS = {
    "Random Forest":       {"Accuracy": 0.6590, "Precision": 0.6718, "Recall": 0.5199, "F1": 0.5862},
    "Logistic Regression": {"Accuracy": 0.6195, "Precision": 0.6867, "Recall": 0.3326, "F1": 0.4482},
    "Decision Tree":       {"Accuracy": 0.6530, "Precision": 0.6497, "Recall": 0.5490, "F1": 0.5951},
    "SVM":                 {"Accuracy": 0.5955, "Precision": 0.7143, "Recall": 0.2153, "F1": 0.3309},
    "LightGBM":            {"Accuracy": 0.6400, "Precision": 0.6200, "Recall": 0.6100, "F1": 0.6150},
    "ANN (Neural Network)":{"Accuracy": 0.6300, "Precision": 0.6100, "Recall": 0.5800, "F1": 0.5950},
}

COLORS = ["#7DF9AA", "#4ec4ff", "#ffb347", "#ff6b6b", "#c77dff", "#ff9e6d"]

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.stApp { background: #111118; color: #e8e8e8; }

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem !important;
    color: #7DF9AA !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Syne', sans-serif;
    color: #888 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.stButton > button {
    background: #7DF9AA;
    color: #0d0d0d;
    border: none;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.6rem 2rem;
    transition: all 0.2s;
}
.stButton > button:hover { background: #5bdf88; transform: translateY(-1px); }

.stSelectbox > div, .stTextInput > div {
    background: #1a1a24 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 4px;
}

[data-testid="stDataFrame"] { border: 1px solid #2a2a3a; border-radius: 6px; }

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

code {
    font-family: 'JetBrains Mono', monospace;
    background: #1a1a24;
    padding: 2px 6px;
    border-radius: 3px;
    color: #7DF9AA;
}

hr { border-color: #2a2a3a; }

.result-buggy {
    background: linear-gradient(135deg, #3a0a0a, #1a0505);
    border: 1px solid #ff4444;
    border-left: 4px solid #ff4444;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-clean {
    background: linear-gradient(135deg, #0a2a1a, #051a0d);
    border: 1px solid #7DF9AA;
    border-left: 4px solid #7DF9AA;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
}
.result-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    opacity: 0.65;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING  (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_sklearn_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def load_keras_model(model_path: str, scaler_path: str):
    """Returns (keras_model, scaler) or (None, None) if files missing."""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model as keras_load
        model  = keras_load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load Keras model: {e}")
        return None, None


def is_model_available(name: str) -> bool:
    reg = MODEL_REGISTRY[name]
    if reg["type"] == "keras":
        return os.path.exists(reg["model_path"]) and os.path.exists(reg["scaler_path"])
    return os.path.exists(reg["path"])


def predict_with_model(name: str, X: pd.DataFrame):
    """
    Returns (label, confidence_str).
    label: 0 or 1.  confidence_str: e.g. "Confidence: 73.2%" or "".
    """
    reg = MODEL_REGISTRY[name]

    if reg["type"] == "keras":
        model, scaler = load_keras_model(reg["model_path"], reg["scaler_path"])
        if model is None:
            return None, ""
        X_scaled = scaler.transform(X)
        prob = float(model.predict(X_scaled, verbose=0)[0][0])
        label = 1 if prob >= 0.5 else 0
        conf  = prob if label == 1 else 1 - prob
        return label, f"Confidence: {conf:.1%}"

    else:  # sklearn-compatible (includes LightGBM)
        model = load_sklearn_model(reg["path"])
        if model is None:
            return None, ""
        label = int(model.predict(X)[0])
        conf_str = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            conf  = proba[1] if label == 1 else proba[0]
            conf_str = f"Confidence: {conf:.1%}"
        return label, conf_str


# ─────────────────────────────────────────────
# GITHUB HELPERS
# ─────────────────────────────────────────────

def parse_github_url(url: str):
    pattern = r"github\.com/([^/]+)/([^/]+)/commit/([0-9a-f]+)"
    m = re.search(pattern, url)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def fetch_commit_data(owner, repo, sha, token=""):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url  = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 403:
        st.error("⚠️ GitHub API rate limit exceeded. Add a Personal Access Token in the sidebar.")
        return None
    if resp.status_code == 404:
        st.error("❌ Commit not found. Check the URL.")
        return None
    if resp.status_code != 200:
        st.error(f"❌ GitHub API error {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.json()


def get_developer_experience(owner, repo, author_username, token="") -> int:
    if not author_username:
        return 0
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url    = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"author": author_username, "per_page": 100}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            return len(resp.json())
    except Exception:
        pass
    return 0


def extract_features(commit_data, owner, repo, token="") -> dict:
    commit = commit_data.get("commit", {})
    stats  = commit_data.get("stats", {})
    files  = commit_data.get("files", [])

    parent_count  = len(commit_data.get("parents", []))
    files_changed = len(files)
    additions     = stats.get("additions", 0)
    deletions     = stats.get("deletions", 0)
    total_changes = stats.get("total", 0)
    message_length = len(commit.get("message", ""))

    date_str = (commit.get("author") or commit.get("committer") or {}).get("date", "")
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        commit_hour = dt.hour
        commit_day  = dt.weekday()
    except Exception:
        commit_hour = commit_day = 0

    author_login        = (commit_data.get("author") or {}).get("login", "")
    developer_experience = get_developer_experience(owner, repo, author_login, token)

    directories = set()
    for f in files:
        parts = f.get("filename", "").split("/")
        if len(parts) > 1:
            directories.add(parts[0])
    directories_changed = len(directories)

    return {
        "parent_count":         parent_count,
        "files_changed":        files_changed,
        "additions":            additions,
        "deletions":            deletions,
        "total_changes":        total_changes,
        "message_length":       message_length,
        "commit_hour":          commit_hour,
        "commit_day":           commit_day,
        "developer_experience": developer_experience,
        "directories_changed":  directories_changed,
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🐛 BugSight")
    st.markdown("*Buggy Commit Predictor*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview", "⚖️ Compare Models", "🔍 Predict Commit", "ℹ️ About Project"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**⚙️ GitHub Token** *(optional)*")
    gh_token = st.text_input(
        "Personal Access Token",
        type="password",
        placeholder="ghp_...",
        help="Avoids API rate limits (60 req/hr without token vs 5000 with).",
    )


# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────

if page == "📊 Overview":
    st.markdown("# Buggy Commit Predictor")
    st.markdown("*Machine Learning · GitHub API · University Project*")
    st.markdown("---")

    best_acc = max(v["Accuracy"] for v in MODEL_METRICS.values())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Accuracy",   f"{best_acc:.0%}")
    c2.metric("Models Trained",  len(MODEL_METRICS))
    c3.metric("Problem Type",    "Binary Classification")
    c4.metric("Input Features",  len(FEATURE_COLUMNS))

    st.markdown("---")
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("### How It Works")
        st.markdown("""
Paste a GitHub commit URL → the app calls the **GitHub REST API** to pull
commit metadata, engineers a 10-feature vector, and feeds it into your
trained model to classify the commit as **buggy** or **clean**.

Six models are supported: Logistic Regression, Decision Tree, Random Forest,
SVM, LightGBM, and an ANN (Keras/TensorFlow).
        """)

        st.markdown("### Feature Set")
        feat_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Description": [
                "Number of parent commits (merge = 2)",
                "Number of files touched",
                "Lines added",
                "Lines removed",
                "Total lines changed",
                "Length of commit message",
                "Hour of day commit was made (0–23)",
                "Day of week (0 = Mon, 6 = Sun)",
                "Commits by author in this repo (≤100)",
                "Unique top-level directories changed",
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("### Model Snapshot")
        acc_vals = [MODEL_METRICS[m]["Accuracy"] for m in MODEL_METRICS]
        fig = go.Figure(go.Bar(
            x=list(MODEL_METRICS.keys()),
            y=acc_vals,
            marker_color=COLORS,
            text=[f"{a:.0%}" for a in acc_vals],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8e8",
            yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#2a2a3a"),
            xaxis=dict(tickfont=dict(size=10), tickangle=-25),
            margin=dict(t=30, b=60, l=10, r=10),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Model Files")
        for name, reg in MODEL_REGISTRY.items():
            if reg["type"] == "keras":
                m_ok = os.path.exists(reg["model_path"])
                s_ok = os.path.exists(reg["scaler_path"])
                ok   = m_ok and s_ok
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} `{reg['model_path']}` + `{reg['scaler_path']}`")
            else:
                ok   = os.path.exists(reg["path"])
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} `{reg['path']}`")

        if not any(is_model_available(n) for n in MODEL_REGISTRY):
            st.warning("No model files found. Run `python export_models.py` to generate them.")


# ─────────────────────────────────────────────
# PAGE: COMPARE MODELS
# ─────────────────────────────────────────────

elif page == "⚖️ Compare Models":
    st.markdown("# Model Comparison")
    st.markdown("*Side-by-side evaluation metrics across all six models*")
    st.markdown("---")

    rows = []
    for model, metrics in MODEL_METRICS.items():
        rows.append({"Model": model, **{k: f"{v:.2%}" for k, v in metrics.items()}})
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Accuracy & F1 by Model")
        models = list(MODEL_METRICS.keys())
        acc = [MODEL_METRICS[m]["Accuracy"] for m in models]
        f1  = [MODEL_METRICS[m]["F1"]       for m in models]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Accuracy", x=models, y=acc,
            marker_color="#7DF9AA", opacity=0.9,
        ))
        fig_bar.add_trace(go.Bar(
            name="F1-Score", x=models, y=f1,
            marker_color="#4ec4ff", opacity=0.9,
        ))
        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8e8",
            yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#2a2a3a"),
            xaxis=dict(tickangle=-20, tickfont=dict(size=10)),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=60, l=10, r=10),
            height=360,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("### Radar: All Metrics")
        categories = ["Accuracy", "Precision", "Recall", "F1"]

        fig_radar = go.Figure()
        for (model, metrics), color in zip(MODEL_METRICS.items(), COLORS):
            vals = [metrics[c] for c in categories] + [metrics[categories[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                name=model,
                line_color=color,
                fill="toself",
                fillcolor="rgba(125, 249, 170, 0.15)",
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0, 1], gridcolor="#2a2a3a", tickformat=".0%"),
                angularaxis=dict(gridcolor="#2a2a3a"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8e8",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            margin=dict(t=20, b=20, l=20, r=20),
            height=360,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.info(
        "💡 Update `MODEL_METRICS` in `app.py` with your actual notebook results "
        "after running `export_models.py`.",
        icon="📝",
    )


# ─────────────────────────────────────────────
# PAGE: PREDICT COMMIT
# ─────────────────────────────────────────────

elif page == "🔍 Predict Commit":
    st.markdown("# Predict a Commit")
    st.markdown("*Paste a GitHub commit URL and choose a model*")
    st.markdown("---")

    col_in, col_model = st.columns([2, 1])
    with col_in:
        commit_url = st.text_input(
            "GitHub Commit URL",
            placeholder="https://github.com/owner/repo/commit/abc123def456",
        )
    with col_model:
        available_models = [n for n in MODEL_REGISTRY if is_model_available(n)]
        all_model_names  = list(MODEL_REGISTRY.keys())

        if available_models:
            model_choice = st.selectbox("Select Model", all_model_names,
                                        format_func=lambda n: n + ("" if is_model_available(n) else " ⚠️ (not found)"))
        else:
            st.warning("No model files found. Run `python export_models.py` first.")
            model_choice = st.selectbox("Select Model", all_model_names)

    predict_btn = st.button("🔍  Predict", use_container_width=False)

    if predict_btn:
        if not commit_url.strip():
            st.warning("Please enter a GitHub commit URL.")
            st.stop()

        owner, repo, sha = parse_github_url(commit_url)
        if not owner:
            st.error("❌ Invalid URL. Expected: `https://github.com/owner/repo/commit/<sha>`")
            st.stop()

        if not is_model_available(model_choice):
            st.error(
                f"❌ Model files for **{model_choice}** not found.  \n"
                "Run `python export_models.py` first."
            )
            st.stop()

        with st.spinner("Fetching commit data from GitHub…"):
            commit_data = fetch_commit_data(owner, repo, sha, gh_token)
        if commit_data is None:
            st.stop()

        with st.spinner("Extracting features…"):
            features = extract_features(commit_data, owner, repo, gh_token)

        X = pd.DataFrame([features])[FEATURE_COLUMNS]

        with st.spinner(f"Running {model_choice}…"):
            label, conf_str = predict_with_model(model_choice, X)

        if label is None:
            st.error("Prediction failed — model could not be loaded.")
            st.stop()

        is_buggy = label == 1
        st.markdown("---")

        if is_buggy:
            st.markdown(f"""
<div class="result-buggy">
  <p class="result-title">⚠️ Buggy Commit</p>
  <p class="result-sub">{conf_str} &nbsp;·&nbsp; Model: {model_choice}</p>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-clean">
  <p class="result-title">✅ Clean Commit</p>
  <p class="result-sub">{conf_str} &nbsp;·&nbsp; Model: {model_choice}</p>
</div>""", unsafe_allow_html=True)

        st.markdown("### Extracted Features")
        feat_display = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Value":   [features[f] for f in FEATURE_COLUMNS],
        })
        st.dataframe(feat_display, use_container_width=True, hide_index=True)

        with st.expander("📄 Raw Commit Info"):
            commit_msg  = commit_data.get("commit", {}).get("message", "")
            author_name = (commit_data.get("commit", {}).get("author") or {}).get("name", "Unknown")
            author_date = (commit_data.get("commit", {}).get("author") or {}).get("date", "")
            st.markdown(f"**Author:** {author_name}  \n**Date:** {author_date}  \n**SHA:** `{sha}`")
            st.markdown(f"**Message:**\n```\n{commit_msg}\n```")


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────

elif page == "ℹ️ About Project":
    st.markdown("# About This Project")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Goal")
        st.markdown("""
Predict whether a GitHub commit is likely to introduce a bug, using
only metadata extracted from the GitHub REST API — no source-code
analysis required.

This is a **binary classification** problem:
- **1 → Buggy** commit
- **0 → Clean** commit
        """)

        st.markdown("### 📐 Methodology")
        st.markdown("""
1. Collected commit metadata from public GitHub repositories
2. Labeled commits as buggy/clean using SZZ-style heuristics
3. Engineered 10 structural features from commit statistics
4. Trained six classifiers with `GridSearchCV` / Keras hyperparameter tuning
5. Evaluated on a stratified 80/20 train-test split
6. Exported best models for deployment
        """)

    with col2:
        st.markdown("### 🤖 Algorithms")
        for model in MODEL_REGISTRY:
            m = MODEL_METRICS.get(model, {})
            acc = m.get("Accuracy", 0)
            kind = "Keras/TF" if MODEL_REGISTRY[model]["type"] == "keras" else "scikit-learn"
            st.markdown(f"- **{model}** — accuracy {acc:.0%} *({kind})*")

        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
| Layer | Tool |
|-------|------|
| Dashboard | Streamlit |
| Data | pandas |
| Sklearn models | scikit-learn |
| Gradient Boosting | LightGBM |
| Neural Network | TensorFlow / Keras |
| Serialisation | joblib |
| GitHub API | REST v3 |
| Charts | Plotly |
        """)

        st.markdown("### 📦 Features Used")
        for f in FEATURE_COLUMNS:
            st.markdown(f"- `{f}`")
