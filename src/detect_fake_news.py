#!/usr/bin/env python3
"""Command-line inference for the Fake News Style-Risk Detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_compat import load_pipeline


def default_pipeline_path() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "pipeline.joblib"


def classify_probability(
    prob_fake: float,
    threshold: float = 0.5,
    uncertainty_margin: float = 0.10,
) -> str:
    """Convert a fake probability into REAL, FAKE, or UNCERTAIN.

    The uncertainty margin is centered around the decision threshold. With the
    default threshold=0.50 and uncertainty_margin=0.10, probabilities from
    0.45 to 0.55 are labeled UNCERTAIN instead of forcing a brittle decision.
    """
    if not 0.0 <= prob_fake <= 1.0:
        raise ValueError("prob_fake must be between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if uncertainty_margin < 0.0:
        raise ValueError("uncertainty_margin must be non-negative")

    half_margin = uncertainty_margin / 2
    lower = max(0.0, threshold - half_margin)
    upper = min(1.0, threshold + half_margin)

    if lower <= prob_fake <= upper:
        return "UNCERTAIN"
    return "FAKE" if prob_fake > upper else "REAL"


def predict_one(
    pipeline_path: Path,
    text: str,
    threshold: float,
    uncertainty_margin: float = 0.10,
) -> dict[str, object]:
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline not found at {pipeline_path}. Train first with: python src/train_model.py"
        )
    pipeline = load_pipeline(pipeline_path)
    prob_fake = float(pipeline.predict_proba([text])[0, 1])
    label = classify_probability(prob_fake, threshold, uncertainty_margin)
    return {
        "label": label,
        "prob_fake": prob_fake,
        "threshold": threshold,
        "uncertainty_margin": uncertainty_margin,
        "model_path": str(pipeline_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify one headline/article as REAL, FAKE, or UNCERTAIN."
    )
    parser.add_argument("--pipeline", default=str(default_pipeline_path()), help="Path to outputs/pipeline.joblib.")
    parser.add_argument("--text", required=True, help="Headline or article text to classify.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for FAKE.")
    parser.add_argument(
        "--uncertainty-margin",
        type=float,
        default=0.10,
        help="Probability band around the threshold labeled UNCERTAIN. Default: 0.10.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    result = predict_one(
        Path(args.pipeline),
        args.text,
        args.threshold,
        args.uncertainty_margin,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Label: {result['label']} | Fake probability: {result['prob_fake']:.3f} | "
            f"Threshold: {result['threshold']:.2f} | "
            f"Uncertainty margin: +/-{result['uncertainty_margin'] / 2:.2f}"
        )


if __name__ == "__main__":
    main()
