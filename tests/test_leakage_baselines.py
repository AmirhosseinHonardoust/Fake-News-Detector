from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baselines import (
    baseline_report_to_frame,
    build_baseline_report,
    empirical_prior_probabilities,
    evaluate_baseline_scores,
    majority_class_probabilities,
    reuters_heuristic_probabilities,
    subject_heuristic_probabilities,
)


def test_reuters_heuristic_predicts_real_for_reuters_and_fake_otherwise():
    probs = reuters_heuristic_probabilities(
        [
            "Reuters reported that officials released a statement.",
            "Viral rumor claims a shocking secret was exposed.",
            None,
        ]
    )
    assert probs.tolist() == [0.0, 1.0, 1.0]


def test_majority_and_prior_baselines_are_deterministic():
    y_train = np.array([0, 0, 0, 1])
    assert majority_class_probabilities(y_train, 3).tolist() == [0.0, 0.0, 0.0]
    assert empirical_prior_probabilities(y_train, 2).tolist() == [0.25, 0.25]


def test_subject_heuristic_uses_training_subject_fake_rates_with_default():
    y_train = np.array([0, 0, 1, 1])
    probs = subject_heuristic_probabilities(
        train_subjects=["politics", "politics", "rumor", "rumor"],
        y_train=y_train,
        test_subjects=["politics", "rumor", "unknown"],
    )
    assert probs.tolist() == [0.0, 1.0, 0.5]


def test_evaluate_baseline_scores_validates_probabilities():
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_baseline_scores("bad", [0, 1], [0.2, 1.2])


def test_build_baseline_report_contains_leakage_sensitive_baselines():
    train_frame = pd.DataFrame(
        {
            "text_for_model": [
                "Reuters official statement",
                "Reuters government update",
                "viral fake rumor",
                "shocking fake claim",
            ],
            "subject": ["news", "news", "rumor", "rumor"],
        }
    )
    test_frame = pd.DataFrame(
        {
            "text_for_model": [
                "Reuters policy report",
                "celebrity hoax rumor",
            ],
            "subject": ["news", "rumor"],
        }
    )
    report = build_baseline_report(
        train_frame=train_frame,
        test_frame=test_frame,
        y_train=[0, 0, 1, 1],
        y_test=[0, 1],
    )

    assert set(report["baselines"]) == {
        "majority_class",
        "empirical_prior",
        "reuters_heuristic",
        "subject_heuristic",
    }
    assert report["baselines"]["reuters_heuristic"]["accuracy"] == 1.0
    assert report["ranking_by_macro_f1"][0] in {"reuters_heuristic", "subject_heuristic"}


def test_baseline_report_to_frame_sorts_by_macro_f1():
    report = {
        "baselines": {
            "weak": {"accuracy": 0.5, "macro_f1": 0.4},
            "strong": {"accuracy": 1.0, "macro_f1": 1.0},
        }
    }
    frame = baseline_report_to_frame(report)
    assert frame["baseline"].tolist() == ["strong", "weak"]
