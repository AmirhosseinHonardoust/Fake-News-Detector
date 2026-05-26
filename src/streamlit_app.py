#!/usr/bin/env python3
"""Streamlit app for interactive fake-news style-risk classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import streamlit as st

from detect_fake_news import classify_probability
from model_compat import load_pipeline as load_model_pipeline


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pipeline_path() -> Path:
    return project_root() / "outputs" / "pipeline.joblib"


def default_metrics_path() -> Path:
    return project_root() / "outputs" / "metrics.json"


@st.cache_resource
def load_pipeline(path: str):
    return load_model_pipeline(path)


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_short_input(text: str) -> bool:
    tokens = [token for token in text.strip().split() if token]
    return len(tokens) < 8 or len(text.strip()) < 50


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pipeline", default=str(default_pipeline_path()))
    args, _ = parser.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()
    metrics_path = default_metrics_path()

    st.set_page_config(page_title="Fake News Style-Risk Detector", layout="centered")
    st.title("Fake News Style-Risk Detector")
    st.caption("TF-IDF + Logistic Regression demo with honest validation reporting")

    with st.sidebar:
        st.subheader("Model")
        st.code(str(pipeline_path))
        st.write("Loaded:" if pipeline_path.exists() else "Missing:", pipeline_path.name)
        metrics = load_metrics(metrics_path)
        if metrics:
            test = metrics.get("holdout_test", {})
            st.subheader("Holdout metrics")
            st.write(f"Accuracy: **{test.get('accuracy', 0):.3f}**")
            st.write(f"Macro F1: **{test.get('macro_f1', 0):.3f}**")
            st.write(f"ROC-AUC: **{test.get('roc_auc', 0):.3f}**")
        st.warning(
            "This is an educational style-risk classifier. It can learn dataset/source artifacts "
            "and should not be used as a truth oracle."
        )

    if not pipeline_path.exists():
        st.error(
            "Model artifact not found. Run `python src/train_model.py` from the project root, "
            "then restart Streamlit."
        )
        st.stop()

    pipeline = load_pipeline(str(pipeline_path))
    text = st.text_area("Paste a headline or article excerpt:", height=220)
    threshold = st.slider("FAKE decision threshold", 0.05, 0.95, 0.50, 0.01)
    uncertainty_margin = st.slider(
        "UNCERTAIN band width",
        0.00,
        0.30,
        0.10,
        0.01,
        help="A band around the threshold where the app refuses to force a REAL/FAKE label.",
    )

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Paste some text first.")
            st.stop()

        prob_fake = float(pipeline.predict_proba([text])[0, 1])
        label = classify_probability(prob_fake, threshold, uncertainty_margin)
        half_margin = uncertainty_margin / 2
        lower = max(0.0, threshold - half_margin)
        upper = min(1.0, threshold + half_margin)

        st.metric("Prediction", label, help="This is a statistical style-risk prediction, not a fact-check.")
        st.write(f"Fake probability: **{prob_fake:.1%}**")
        st.caption(f"Decision rule: REAL < {lower:.0%}, UNCERTAIN = {lower:.0%}–{upper:.0%}, FAKE > {upper:.0%}.")

        if label == "UNCERTAIN":
            st.warning(
                "The model is close to the decision boundary. Treat this as low confidence and provide a longer excerpt if possible."
            )
        elif label == "FAKE":
            st.progress(prob_fake, text=f"Displayed-label confidence proxy: {prob_fake:.1%}")
        else:
            st.progress(1 - prob_fake, text=f"Displayed-label confidence proxy: {1 - prob_fake:.1%}")

        if is_short_input(text):
            st.warning(
                "The input is very short. The model works better with full headlines or article excerpts."
            )

        st.info(
            "For real-world use, verify claims against primary sources. This model was trained on a limited educational dataset."
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import joblib
import streamlit as st

# ---------- text cleaning ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- path helpers ----------
def project_root() -> Path:
    # this file is in src/, project root is parent directory
    return Path(__file__).resolve().parents[1]

def default_paths():
    root = project_root()
    out = root / "outputs"
    return {
        "pipeline": out / "pipeline.joblib",
        "model": out / "model.joblib",
        "vectorizer": out / "vectorizer.joblib",
    }

def load_pipeline_or_parts(pipeline_path: Path, model_path: Path, vectorizer_path: Path):
    if pipeline_path and pipeline_path.exists():
        return joblib.load(pipeline_path), None, None
    if model_path.exists() and vectorizer_path.exists():
        clf = joblib.load(model_path)
        vec = joblib.load(vectorizer_path)
        return None, clf, vec
    return None, None, None

# ---------- streamlit app ----------
def main():
    # parse CLI overrides but give safe defaults relative to repo root
    dp = default_paths()

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pipeline", default=str(dp["pipeline"]))
    ap.add_argument("--model", default=str(dp["model"]))
    ap.add_argument("--vectorizer", default=str(dp["vectorizer"]))
    args, _ = ap.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()
    model_path = Path(args.model).resolve()
    vectorizer_path = Path(args.vectorizer).resolve()

    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
    st.title("Fake News & Misinformation Detector")
    st.caption("TF-IDF + Logistic Regression (interpretable)")

    # sidebar: show where we look for files
    with st.sidebar:
        st.subheader("Model Artifacts")
        st.code(f"pipeline:  {pipeline_path}\nmodel:     {model_path}\nvectorizer:{vectorizer_path}")
        st.write(f"Exists → pipeline: **{pipeline_path.exists()}**, "
                 f"model: **{model_path.exists()}**, vectorizer: **{vectorizer_path.exists()}**")

    pipe, clf, vec = load_pipeline_or_parts(pipeline_path, model_path, vectorizer_path)
    if pipe is None and (clf is None or vec is None):
        st.error(
            "Model artifacts not found.\n\n"
            "• Ensure you trained and saved files to `outputs/`\n"
            "• Or run Streamlit with explicit paths, e.g.:\n"
            "  `streamlit run src/app.py -- --model C:/path/outputs/model.joblib --vectorizer C:/path/outputs/vectorizer.joblib`\n"
            "• From CLI, also try: `python -c \"import os; print(os.getcwd())\"` to see your working directory."
        )
        st.stop()

    txt = st.text_area("Paste headline or article text:", height=200)
    threshold = st.slider("FAKE decision threshold", 0.05, 0.95, 0.50, 0.01)

    if st.button("Analyze") and txt.strip():
        s = clean_text(txt)
        if pipe is not None:
            prob_fake = float(pipe.predict_proba([s])[0, 1])
        else:
            X = vec.transform([s])
            prob_fake = float(clf.predict_proba(X)[0, 1])

        label = "FAKE" if prob_fake >= threshold else "REAL"
        st.metric("Prediction", label)
        st.progress(prob_fake if label == "FAKE" else 1 - prob_fake,
                    text=f"Fake probability: {prob_fake:.1%} (threshold {threshold:.2f})")

if __name__ == "__main__":
    main()
