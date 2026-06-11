from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

EXPECTED_OUTPUT_FILES = [
    "artifact_environment.json",
    "data_profile.json",
    "leakage_report.json",
    "metrics.json",
    "holdout_predictions.csv",
    "pipeline.joblib",
    "vectorizer.joblib",
    "model.joblib",
    "charts/class_distribution.png",
    "charts/confidence_distribution.png",
    "charts/confusion_matrix.png",
    "charts/pr_curve.png",
    "charts/roc_curve.png",
]


@pytest.fixture()
def smoke_outdir() -> Path:
    raw = os.environ.get("FAKE_NEWS_TRAINING_OUTDIR")
    if not raw:
        pytest.skip("FAKE_NEWS_TRAINING_OUTDIR is not set; CI validates training artifacts.")
    path = Path(raw)
    assert path.exists(), f"Training output directory does not exist: {path}"
    return path


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_training_smoke_outputs_exist(smoke_outdir: Path) -> None:
    missing = [name for name in EXPECTED_OUTPUT_FILES if not (smoke_outdir / name).exists()]
    assert missing == []


def test_metrics_json_has_expected_schema(smoke_outdir: Path) -> None:
    metrics = read_json(smoke_outdir / "metrics.json")

    assert metrics["model"] == "TF-IDF + Logistic Regression"
    assert metrics["label_mapping"] == {"0": "REAL", "1": "FAKE"}
    assert "validation_protocol" in metrics
    assert "cross_validation_train_only" in metrics
    assert "dataset_profile" in metrics
    assert "holdout_test" in metrics

    holdout = metrics["holdout_test"]
    assert isinstance(holdout, dict)
    for key in ["accuracy", "macro_f1", "precision_fake", "recall_fake", "roc_auc", "average_precision"]:
        assert key in holdout
        if holdout[key] is not None:
            assert 0.0 <= float(holdout[key]) <= 1.0

    confusion_matrix = holdout["confusion_matrix"]
    assert isinstance(confusion_matrix, list)
    assert len(confusion_matrix) == 2
    assert all(len(row) == 2 for row in confusion_matrix)


def test_leakage_report_has_expected_schema(smoke_outdir: Path) -> None:
    report = read_json(smoke_outdir / "leakage_report.json")

    assert "contains_reuters_by_label" in report
    assert "heuristic_contains_reuters_accuracy" in report
    assert "interpretation" in report
    assert 0.0 <= float(report["heuristic_contains_reuters_accuracy"]) <= 1.0

    by_label = report["contains_reuters_by_label"]
    assert isinstance(by_label, dict)
    assert {"REAL", "FAKE"}.issubset(by_label.keys())
    for label_stats in by_label.values():
        assert "count" in label_stats
        assert "rate" in label_stats
        assert 0.0 <= float(label_stats["rate"]) <= 1.0


def test_data_profile_has_expected_schema(smoke_outdir: Path) -> None:
    profile = read_json(smoke_outdir / "data_profile.json")

    assert profile["total_rows"] >= profile["rows_after_deduplication"] > 0
    assert profile["duplicate_rows_removed"] >= 0
    assert profile["text_column_used"]

    class_balance = profile["class_balance"]
    assert isinstance(class_balance, dict)
    assert class_balance["REAL"] > 0
    assert class_balance["FAKE"] > 0


def test_holdout_predictions_have_expected_schema(smoke_outdir: Path) -> None:
    with (smoke_outdir / "holdout_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) > 0
    required_columns = {"text", "true_label", "prob_fake", "predicted_label"}
    assert required_columns.issubset(rows[0].keys())

    for row in rows[:25]:
        assert row["true_label"] in {"REAL", "FAKE"}
        assert row["predicted_label"] in {"REAL", "FAKE"}
        assert 0.0 <= float(row["prob_fake"]) <= 1.0


def test_cli_prediction_json_has_expected_schema() -> None:
    raw = os.environ.get("FAKE_NEWS_PREDICTION_JSON")
    if not raw:
        pytest.skip("FAKE_NEWS_PREDICTION_JSON is not set; CI validates CLI prediction output.")

    prediction = read_json(Path(raw))
    assert prediction["label"] in {"REAL", "FAKE", "UNCERTAIN"}
    assert 0.0 <= float(prediction["prob_fake"]) <= 1.0
    assert 0.0 <= float(prediction["threshold"]) <= 1.0
    assert float(prediction["uncertainty_margin"]) >= 0.0
    assert prediction["model_path"].endswith("pipeline.joblib")
