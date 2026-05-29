#!/usr/bin/env python3
"""Train and evaluate the Fake News Detector with an honest validation protocol."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from text_clean import TextCleaner, clean_text

LABEL_NAMES: Final[list[str]] = ["REAL", "FAKE"]
LABEL_TO_ID: Final[dict[str, int]] = {"REAL": 0, "FAKE": 1}
ID_TO_LABEL: Final[dict[int, str]] = {0: "REAL", 1: "FAKE"}


@dataclass(frozen=True)
class DatasetProfile:
    total_rows: int
    rows_after_deduplication: int
    duplicate_rows_removed: int
    class_balance: dict[str, int]
    text_column_used: str
    title_column_used: bool


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def pick_text_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for candidate in ("combined_text", "text", "content", "article", "body"):
        if candidate in df.columns:
            return candidate
    if "title" in df.columns:
        return "title"
    raise ValueError(f"No suitable text column found. Available columns: {list(df.columns)}")


def build_labeled_frame(real_path: Path, fake_path: Path, text_col: str, include_title: bool) -> tuple[pd.DataFrame, DatasetProfile]:
    real = read_csv_any(real_path).copy()
    fake = read_csv_any(fake_path).copy()

    real_col = pick_text_column(real, text_col)
    fake_col = pick_text_column(fake, text_col)
    if real_col != fake_col:
        raise ValueError(f"Real and fake files resolved to different text columns: {real_col!r} vs {fake_col!r}")

    real["label"] = "REAL"
    fake["label"] = "FAKE"
    data = pd.concat([real, fake], ignore_index=True)

    base_text = data[real_col].fillna("").astype(str)
    if include_title and "title" in data.columns:
        title = data["title"].fillna("").astype(str)
        data["text_for_model"] = (title + " " + base_text).str.strip()
    else:
        data["text_for_model"] = base_text

    before = len(data)
    data["clean_text_for_dedupe"] = data["text_for_model"].map(clean_text)
    data = data[data["clean_text_for_dedupe"].str.len() > 0].copy()
    data = data.drop_duplicates(subset=["clean_text_for_dedupe", "label"]).reset_index(drop=True)
    after = len(data)

    profile = DatasetProfile(
        total_rows=before,
        rows_after_deduplication=after,
        duplicate_rows_removed=before - after,
        class_balance=data["label"].value_counts().reindex(LABEL_NAMES, fill_value=0).astype(int).to_dict(),
        text_column_used=real_col,
        title_column_used=bool(include_title and "title" in data.columns),
    )
    return data, profile


def build_pipeline(max_features: int, min_df: int, max_df: float, ngram_max: int, C: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("cleaner", TextCleaner()),
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    max_df=max_df,
                    max_features=max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=None,
                    random_state=42,
                    solver="saga",
                ),
            ),
        ]
    )


def metrics_at_threshold(y_true: np.ndarray, y_prob_fake: np.ndarray, threshold: float) -> dict[str, object]:
    y_pred = (y_prob_fake >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0),
    }


def evaluate_split(name: str, y_true: np.ndarray, y_prob_fake: np.ndarray, threshold: float) -> dict[str, object]:
    result = metrics_at_threshold(y_true, y_prob_fake, threshold)
    result["split"] = name
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob_fake))
        result["average_precision"] = float(average_precision_score(y_true, y_prob_fake))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def plot_confusion_matrix(cm: list[list[int]], out: Path, title: str) -> None:
    labels = LABEL_NAMES
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(arr, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            ax.text(col, row, str(arr[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob_fake: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob_fake)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob_fake):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve - Holdout Test")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_pr(y_true: np.ndarray, y_prob_fake: np.ndarray, out: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob_fake)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"AP = {average_precision_score(y_true, y_prob_fake):.3f}")
    ax.set_title("Precision-Recall Curve - Holdout Test")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_class_distribution(class_balance: dict[str, int], out: Path) -> None:
    labels = LABEL_NAMES
    counts = [class_balance.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts)
    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of samples")
    for bar, count in zip(bars, counts, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_confidence_distribution(y_prob_fake: np.ndarray, threshold: float, out: Path) -> None:
    confidence = np.maximum(y_prob_fake, 1 - y_prob_fake)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(confidence, bins=20, edgecolor="black")
    ax.axvline(max(threshold, 1 - threshold), linestyle="--", label="Decision threshold proxy")
    ax.set_title("Prediction Confidence Distribution - Holdout Test")
    ax.set_xlabel("Prediction confidence proxy")
    ax.set_ylabel("Number of samples")
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def artifact_environment() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
        "note": (
            "For best compatibility, load joblib artifacts with the same major/minor "
            "scikit-learn version used to train them, or retrain locally."
        ),
    }

def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def leakage_report(data: pd.DataFrame) -> dict[str, object]:
    report: dict[str, object] = {}
    by_label = data.groupby("label")

    report["subject_distribution"] = (
        data.groupby(["label", "subject"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index")
        if "subject" in data.columns
        else "subject column not available"
    )

    contains_reuters = data["text_for_model"].str.contains(r"\breuters\b", case=False, regex=True, na=False)
    data_with_rule = data.assign(contains_reuters=contains_reuters)
    report["contains_reuters_by_label"] = {
        label: {
            "count": int(group["contains_reuters"].sum()),
            "rate": float(group["contains_reuters"].mean()),
        }
        for label, group in data_with_rule.groupby("label")
    }
    reuters_rule_pred = np.where(contains_reuters.to_numpy(), LABEL_TO_ID["REAL"], LABEL_TO_ID["FAKE"])
    y_true = data["label"].map(LABEL_TO_ID).to_numpy()
    report["heuristic_contains_reuters_accuracy"] = float(accuracy_score(y_true, reuters_rule_pred))

    if "date" in data.columns:
        report["date_examples_by_label"] = {
            label: group["date"].dropna().astype(str).head(5).tolist()
            for label, group in by_label
        }

    report["interpretation"] = (
        "High performance on this dataset can be inflated by source/style artifacts. "
        "Use external validation before making real-world claims."
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a professional TF-IDF + Logistic Regression fake-news classifier.")
    parser.add_argument("--real", default="data/True.csv", help="Path to real-news CSV.")
    parser.add_argument("--fake", default="data/Fake.csv", help="Path to fake-news CSV.")
    parser.add_argument("--text-col", default="text", help="Preferred text column name.")
    parser.add_argument("--outdir", default="outputs", help="Directory for model artifacts and metrics.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="FAKE decision threshold.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of stratified CV folds on the training split.")
    parser.add_argument("--max-features", type=int, default=10000, help="Maximum TF-IDF features.")
    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2, 3], help="Maximum n-gram size.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency.")
    parser.add_argument("--max-df", type=float, default=0.9, help="Maximum document frequency.")
    parser.add_argument("--C", type=float, default=2.0, help="Logistic Regression inverse regularization strength.")
    parser.add_argument("--no-title", action="store_true", help="Do not prepend title when a title column exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir))
    charts_dir = ensure_dir(outdir / "charts")
    write_json(artifact_environment(), outdir / "artifact_environment.json")

    data, profile = build_labeled_frame(
        real_path=Path(args.real),
        fake_path=Path(args.fake),
        text_col=args.text_col,
        include_title=not args.no_title,
    )
    write_json(asdict(profile), outdir / "data_profile.json")
    write_json(leakage_report(data), outdir / "leakage_report.json")

    X = data["text_for_model"]
    y = data["label"].map(LABEL_TO_ID).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    pipeline = build_pipeline(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        C=args.C,
    )

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro", "roc_auc": "roc_auc"},
        n_jobs=1,
    )

    pipeline.fit(X_train, y_train)
    train_prob = pipeline.predict_proba(X_train)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_split("train", y_train, train_prob, args.threshold)
    test_metrics = evaluate_split("holdout_test", y_test, test_prob, args.threshold)

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "label_mapping": ID_TO_LABEL,
        "random_state": args.random_state,
        "threshold": args.threshold,
        "dataset_profile": asdict(profile),
        "validation_protocol": {
            "holdout_test_size": args.test_size,
            "split_strategy": "stratified train/test split",
            "cross_validation": f"{args.cv_folds}-fold StratifiedKFold on training split only",
        },
        "cross_validation_train_only": {
            metric.replace("test_", ""): {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for metric, values in cv_scores.items()
            if metric.startswith("test_")
        },
        "train": train_metrics,
        "holdout_test": test_metrics,
        "notes": [
            "Holdout metrics are more meaningful than training metrics, but the included dataset has known source/style leakage.",
            "See outputs/leakage_report.json before making real-world claims.",
        ],
    }
    write_json(metrics, outdir / "metrics.json")

    plot_confusion_matrix(test_metrics["confusion_matrix"], charts_dir / "confusion_matrix.png", "Confusion Matrix - Holdout Test")
    plot_roc(y_test, test_prob, charts_dir / "roc_curve.png")
    plot_pr(y_test, test_prob, charts_dir / "pr_curve.png")
    plot_class_distribution(profile.class_balance, charts_dir / "class_distribution.png")
    plot_confidence_distribution(test_prob, args.threshold, charts_dir / "confidence_distribution.png")

    predictions = pd.DataFrame(
        {
            "text": X_test.reset_index(drop=True),
            "true_label": [ID_TO_LABEL[int(v)] for v in y_test],
            "prob_fake": test_prob,
            "predicted_label": [ID_TO_LABEL[int(v)] for v in (test_prob >= args.threshold).astype(int)],
        }
    )
    predictions.to_csv(outdir / "holdout_predictions.csv", index=False)

    joblib.dump(pipeline, outdir / "pipeline.joblib")
    # Legacy artifacts are saved for compatibility, but the full pipeline is preferred.
    joblib.dump(pipeline.named_steps["tfidf"], outdir / "vectorizer.joblib")
    joblib.dump(pipeline.named_steps["classifier"], outdir / "model.joblib")

    print("Training complete")
    print(f"Rows after deduplication: {profile.rows_after_deduplication}")
    print(f"Holdout accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Holdout macro F1: {test_metrics['macro_f1']:.3f}")
    print(f"Holdout ROC-AUC: {test_metrics['roc_auc']:.3f}")
    print(f"Artifacts saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
