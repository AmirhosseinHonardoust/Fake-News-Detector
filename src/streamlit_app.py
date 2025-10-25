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
    st.title("📰 Fake News & Misinformation Detector")
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
