#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import joblib

from text_clean import clean_text

def load_pipeline_or_parts(pipeline_path: str | None,
                           model_path: str | None,
                           vectorizer_path: str | None):
    if pipeline_path:
        pipe = joblib.load(pipeline_path)
        return pipe, None, None
    # fallback to separate artifacts
    if not (model_path and vectorizer_path):
        raise ValueError(
            "Provide --pipeline OR both --model and --vectorizer."
        )
    clf = joblib.load(model_path)
    vec = joblib.load(vectorizer_path)
    return None, clf, vec

def main() -> None:
    ap = argparse.ArgumentParser(description="Detect fake news for a single text.")
    ap.add_argument("--pipeline", help="Path to pipeline.joblib (preferred).")
    ap.add_argument("--model", help="Path to model.joblib (fallback).")
    ap.add_argument("--vectorizer", help="Path to vectorizer.joblib (fallback).")
    ap.add_argument("--text", required=True, help="Headline or article text.")
    ap.add_argument("--threshold", type=float, default=0.40,
                    help="Decision threshold for FAKE (default: 0.40).")
    args = ap.parse_args()

    pipe, clf, vec = load_pipeline_or_parts(args.pipeline, args.model, args.vectorizer)

    s = clean_text(args.text)

    if pipe is not None:
        prob = float(pipe.predict_proba([s])[0, 1])
    else:
        X = vec.transform([s])
        prob = float(clf.predict_proba(X)[0, 1])

    label = "FAKE" if prob >= args.threshold else "REAL"
    print(f"Label: {label} | Fake probability: {prob:.3f} | Threshold: {args.threshold:.2f}")

if __name__ == "__main__":
    main()
