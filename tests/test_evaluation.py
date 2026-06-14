import pandas as pd

from evaluation import out_of_source_evaluation, source_confounding_report
from train_model import build_pipeline


def _confounded_frame() -> pd.DataFrame:
    # Each source maps to exactly one label (like the bundled dataset).
    rows = []
    for i in range(20):
        rows.append({"text_for_model": f"reuters official statement {i}", "label": "REAL", "subject": "politicsNews"})
        rows.append({"text_for_model": f"shocking viral hoax claim {i}", "label": "FAKE", "subject": "News"})
    return pd.DataFrame(rows)


def _multi_source_frame() -> pd.DataFrame:
    # Both labels appear across several sources -> out-of-source split feasible.
    rows = []
    sources = ["wire_a", "wire_b", "blog_x", "blog_y"]
    for i in range(15):
        for src in sources:
            rows.append({"text_for_model": f"government policy economy report {src} {i}", "label": "REAL", "subject": src})
            rows.append({"text_for_model": f"miracle cure secret exposed {src} {i}", "label": "FAKE", "subject": src})
    return pd.DataFrame(rows)


def test_confounding_report_flags_perfect_confound():
    report = source_confounding_report(_confounded_frame(), group_col="subject")
    assert report["available"] is True
    assert report["confounding_score"] == 1.0
    assert report["out_of_source_split_feasible"] is False
    assert report["groups_with_real"] == 1
    assert report["groups_with_fake"] == 1


def test_confounding_report_marks_balanced_sources_feasible():
    report = source_confounding_report(_multi_source_frame(), group_col="subject")
    assert report["out_of_source_split_feasible"] is True
    assert report["groups_with_real"] >= 2
    assert report["groups_with_fake"] >= 2
    assert report["confounding_score"] < 1.0


def test_confounding_report_handles_missing_group_column():
    report = source_confounding_report(pd.DataFrame({"label": ["REAL"]}), group_col="subject")
    assert report["available"] is False


def test_out_of_source_evaluation_returns_none_when_confounded():
    result = out_of_source_evaluation(
        _confounded_frame(),
        make_pipeline=lambda: build_pipeline(100, 1, 1.0, 1, 1.0),
        text_col="text_for_model",
    )
    assert result is None


def test_out_of_source_evaluation_runs_on_multi_source_data():
    result = out_of_source_evaluation(
        _multi_source_frame(),
        make_pipeline=lambda: build_pipeline(200, 1, 1.0, 1, 1.0),
        text_col="text_for_model",
    )
    assert result is not None
    assert result["evaluation"] == "out_of_source_holdout"
    # Held-out groups must not appear in training.
    assert set(result["test_groups"]).isdisjoint(result["train_groups"])
    assert 0.0 <= result["accuracy"] <= 1.0
