"""Leakage-aware baseline utilities for the fake-news style-risk demo."""

from __future__ import annotations

import re
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LABEL_TO_ID: Final[dict[str, int]] = {"REAL": 0, "FAKE": 1}
ID_TO_LABEL: Final[dict[int, str]] = {0: "REAL", 1: "FAKE"}
REUTERS_PATTERN: Final[re.Pattern[str]] = re.compile(r"\breuters\b", flags=re.IGNORECASE)


def _as_int_array(values: np.ndarray | pd.Series | list[int]) -> np.ndarray:
    """Return a one-dimensional integer numpy array."""
    arr = np.asarray(values, dtype=int)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional label array")
    return arr


def evaluate_baseline_scores(
    name: str,
    y_true: np.ndarray | pd.Series | list[int],
    y_prob_fake: np.ndarray | pd.Series | list[float],
    threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate a baseline using the same threshold metrics as the model.

    Baselines intentionally include simple and leakage-sensitive rules. They are
    not meant to replace the model; they provide context for interpreting high
    scores on a dataset with known source/style artifacts.
    """
    y = _as_int_array(y_true)
    prob = np.asarray(y_prob_fake, dtype=float)
    if prob.ndim != 1:
        raise ValueError("Expected a one-dimensional probability array")
    if len(y) != len(prob):
        raise ValueError("y_true and y_prob_fake must have the same length")
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty baseline")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if np.any((prob < 0.0) | (prob > 1.0)):
        raise ValueError("Baseline probabilities must be between 0 and 1")

    y_pred = (prob >= threshold).astype(int)
    metrics: dict[str, object] = {
        "name": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_fake": float(precision_score(y, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "predicted_fake_rate": float(y_pred.mean()),
    }

    if len(np.unique(y)) == 2 and len(np.unique(prob)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, prob))
        metrics["average_precision"] = float(average_precision_score(y, prob))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def majority_class_probabilities(y_train: np.ndarray | pd.Series | list[int], n_rows: int) -> np.ndarray:
    """Predict the majority training class for every row."""
    y = _as_int_array(y_train)
    if len(y) == 0:
        raise ValueError("y_train cannot be empty")
    if n_rows < 0:
        raise ValueError("n_rows cannot be negative")
    majority_label = int(pd.Series(y).mode().iloc[0])
    return np.full(n_rows, float(majority_label))


def empirical_prior_probabilities(y_train: np.ndarray | pd.Series | list[int], n_rows: int) -> np.ndarray:
    """Use the training fake-label rate as a constant probability."""
    y = _as_int_array(y_train)
    if len(y) == 0:
        raise ValueError("y_train cannot be empty")
    if n_rows < 0:
        raise ValueError("n_rows cannot be negative")
    return np.full(n_rows, float(y.mean()))


def reuters_heuristic_probabilities(texts: pd.Series | list[str]) -> np.ndarray:
    """Predict REAL when text contains 'Reuters', otherwise FAKE.

    This is a leakage-sensitive source/style baseline. Strong performance from
    this rule is evidence that the dataset contains source artifacts.
    """
    text_series = pd.Series(texts).fillna("").astype(str)
    contains_reuters = text_series.str.contains(REUTERS_PATTERN, regex=True, na=False)
    return np.where(contains_reuters.to_numpy(), 0.0, 1.0)


def subject_heuristic_probabilities(
    train_subjects: pd.Series | list[str],
    y_train: np.ndarray | pd.Series | list[int],
    test_subjects: pd.Series | list[str],
) -> np.ndarray:
    """Use per-subject fake rates learned from the training split."""
    y = _as_int_array(y_train)
    train = pd.Series(train_subjects).fillna("UNKNOWN").astype(str)
    test = pd.Series(test_subjects).fillna("UNKNOWN").astype(str)
    if len(train) != len(y):
        raise ValueError("train_subjects and y_train must have the same length")
    if len(y) == 0:
        raise ValueError("y_train cannot be empty")

    default_rate = float(y.mean())
    subject_rates = pd.DataFrame({"subject": train, "label": y}).groupby("subject")["label"].mean()
    return test.map(subject_rates).fillna(default_rate).to_numpy(dtype=float)


def build_baseline_report(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: np.ndarray | pd.Series | list[int],
    y_test: np.ndarray | pd.Series | list[int],
    threshold: float = 0.5,
    text_column: str = "text_for_model",
    subject_column: str = "subject",
) -> dict[str, object]:
    """Build baseline comparisons for the holdout split."""
    if text_column not in test_frame.columns:
        raise ValueError(f"Missing text column: {text_column}")

    baselines: dict[str, dict[str, object]] = {}
    n_test = len(test_frame)

    baseline_probs = {
        "majority_class": majority_class_probabilities(y_train, n_test),
        "empirical_prior": empirical_prior_probabilities(y_train, n_test),
        "reuters_heuristic": reuters_heuristic_probabilities(test_frame[text_column]),
    }

    if subject_column in train_frame.columns and subject_column in test_frame.columns:
        baseline_probs["subject_heuristic"] = subject_heuristic_probabilities(
            train_frame[subject_column],
            y_train,
            test_frame[subject_column],
        )

    for name, probabilities in baseline_probs.items():
        baselines[name] = evaluate_baseline_scores(name, y_test, probabilities, threshold=threshold)

    ranking_by_macro_f1 = sorted(
        baselines,
        key=lambda baseline_name: float(baselines[baseline_name]["macro_f1"]),
        reverse=True,
    )
    ranking_by_accuracy = sorted(
        baselines,
        key=lambda baseline_name: float(baselines[baseline_name]["accuracy"]),
        reverse=True,
    )

    return {
        "baselines": baselines,
        "ranking_by_macro_f1": ranking_by_macro_f1,
        "ranking_by_accuracy": ranking_by_accuracy,
        "interpretation": (
            "Leakage-aware baselines provide context for high model scores. "
            "If a simple source/style rule performs very well, the dataset is likely "
            "easy because of artifacts rather than true misinformation understanding."
        ),
    }


def baseline_report_to_frame(report: dict[str, object]) -> pd.DataFrame:
    """Convert a baseline report into a compact comparison table."""
    rows = []
    baselines = report.get("baselines", {})
    if not isinstance(baselines, dict):
        raise ValueError("report['baselines'] must be a dictionary")
    for name, metrics in baselines.items():
        if not isinstance(metrics, dict):
            raise ValueError("Each baseline entry must be a dictionary")
        rows.append(
            {
                "baseline": name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "precision_fake": metrics.get("precision_fake"),
                "recall_fake": metrics.get("recall_fake"),
                "f1_fake": metrics.get("f1_fake"),
                "roc_auc": metrics.get("roc_auc"),
                "average_precision": metrics.get("average_precision"),
                "predicted_fake_rate": metrics.get("predicted_fake_rate"),
            }
        )
    return pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
