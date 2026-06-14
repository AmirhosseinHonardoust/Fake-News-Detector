#!/usr/bin/env python3
"""Source/group leakage diagnostics and out-of-source evaluation.

The bundled dataset separates the classes almost entirely by wire source. The
helpers here make that explicit: they quantify how strongly a grouping column
(e.g. ``subject``) is confounded with the label, decide whether an honest
out-of-source split is even possible, and run that split when it is.

An out-of-source holdout puts *entire groups* in the test set that never appear
in training, which measures whether the model generalizes across sources rather
than memorizing source-specific style. It is only meaningful when both classes
appear in at least two groups; otherwise holding out a group yields a
single-class test set and any score is meaningless.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

LABEL_NAMES = ("REAL", "FAKE")


def source_confounding_report(
    data: pd.DataFrame,
    group_col: str = "subject",
    label_col: str = "label",
) -> dict[str, object]:
    """Quantify how strongly ``group_col`` is confounded with ``label_col``.

    ``confounding_score`` is the support-weighted mean, over groups, of the
    dominant class fraction within each group. For a binary label it ranges from
    0.5 (every group perfectly class-balanced; no confound) to 1.0 (every group
    is a single class; fully confounded, source alone determines the label).
    """
    if group_col not in data.columns:
        return {
            "group_col": group_col,
            "available": False,
            "interpretation": f"Group column {group_col!r} is not available in the data.",
        }

    crosstab = (
        pd.crosstab(data[group_col], data[label_col])
        .reindex(columns=list(LABEL_NAMES), fill_value=0)
        .astype(int)
    )
    group_totals = crosstab.sum(axis=1)
    dominant_fraction = crosstab.max(axis=1) / group_totals.replace(0, np.nan)
    confounding_score = float(
        np.average(dominant_fraction.fillna(0.0), weights=group_totals)
    )

    groups_with_real = int((crosstab["REAL"] > 0).sum())
    groups_with_fake = int((crosstab["FAKE"] > 0).sum())
    feasible = groups_with_real >= 2 and groups_with_fake >= 2

    if feasible:
        interpretation = (
            "Both classes appear in multiple groups, so an out-of-source holdout "
            "(groups held out of training) is feasible and was attempted."
        )
    else:
        interpretation = (
            "Source is confounded with the label: at least one class is confined to a "
            "single group, so any out-of-source holdout would produce a single-class "
            "test set. An honest out-of-source evaluation is impossible on this dataset; "
            "this is direct evidence that high accuracy reflects source recognition."
        )

    return {
        "group_col": group_col,
        "available": True,
        "n_groups": int(crosstab.shape[0]),
        "crosstab": {str(g): row.to_dict() for g, row in crosstab.iterrows()},
        "groups_with_real": groups_with_real,
        "groups_with_fake": groups_with_fake,
        "confounding_score": confounding_score,
        "out_of_source_split_feasible": feasible,
        "interpretation": interpretation,
    }


def _binary_metrics(
    y_true: np.ndarray, y_prob_fake: np.ndarray, threshold: float
) -> dict[str, object]:
    y_pred = (y_prob_fake >= threshold).astype(int)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_prob_fake)) if len(np.unique(y_true)) == 2 else None
    )
    return metrics


def out_of_source_evaluation(
    data: pd.DataFrame,
    make_pipeline: Callable[[], Pipeline],
    text_col: str,
    group_col: str = "subject",
    label_col: str = "label",
    label_to_id: dict[str, int] | None = None,
    threshold: float = 0.5,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, object] | None:
    """Train on a subset of groups and test on entirely held-out groups.

    Returns ``None`` when no split keeps both classes present in both train and
    test (i.e. when the source is confounded with the label).
    """
    if label_to_id is None:
        label_to_id = {"REAL": 0, "FAKE": 1}
    if group_col not in data.columns:
        return None

    X = data[text_col].to_numpy()
    y = data[label_col].map(label_to_id).to_numpy()
    groups = data[group_col].astype(str).to_numpy()

    splitter = GroupShuffleSplit(n_splits=8, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(X, y, groups):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        pipeline = make_pipeline()
        pipeline.fit(X[train_idx], y_train)
        prob_fake = pipeline.predict_proba(X[test_idx])[:, 1]
        result = _binary_metrics(y_test, prob_fake, threshold)
        result.update(
            {
                "evaluation": "out_of_source_holdout",
                "group_col": group_col,
                "train_groups": sorted(set(groups[train_idx])),
                "test_groups": sorted(set(groups[test_idx])),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
        return result
    return None
