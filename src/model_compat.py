#!/usr/bin/env python3
"""Model loading helpers with small scikit-learn compatibility repairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def ensure_sklearn_compatibility(pipeline: Any) -> Any:
    """Patch known cross-version sklearn pickle differences in-place.

    scikit-learn joblib/pickle artifacts are not guaranteed to be portable across
    versions. Some LogisticRegression artifacts saved by newer sklearn versions no
    longer contain the deprecated ``multi_class`` attribute, while older sklearn
    versions still read that attribute inside ``predict_proba``. Adding the binary
    default here makes the bundled artifact usable across more local environments.
    """
    classifier = None
    if hasattr(pipeline, "named_steps"):
        classifier = pipeline.named_steps.get("classifier")
    elif pipeline.__class__.__name__ == "LogisticRegression":
        classifier = pipeline

    if classifier is not None and classifier.__class__.__name__ == "LogisticRegression":
        if not hasattr(classifier, "multi_class"):
            classifier.multi_class = "auto"

    return pipeline


def load_pipeline(path: str | Path) -> Any:
    """Load a saved pipeline and apply compatibility repairs."""
    pipeline = joblib.load(path)
    return ensure_sklearn_compatibility(pipeline)
