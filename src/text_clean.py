#!/usr/bin/env python3
"""Shared text normalization utilities for training, CLI inference, and the Streamlit app."""

from __future__ import annotations

import re
from collections.abc import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\S+@\S+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
WHITESPACE_RE = re.compile(r"\s+")

# Source/style artifacts specific to the bundled dataset. Nearly every REAL
# article opens with a Reuters dateline such as "WASHINGTON (Reuters) - ..."
# and mentions "Reuters" in the body, while almost no FAKE article does. A
# classifier can therefore separate the classes by recognizing the wire source
# instead of any misinformation signal (see outputs/leakage_report.json).
# DATELINE_RE removes a leading "CITY (Reuters) -" dateline; REUTERS_RE removes
# any remaining standalone "Reuters" mentions.
DATELINE_RE = re.compile(r"^.{0,80}?\(reuters\)\s*-\s*", flags=re.IGNORECASE)
REUTERS_RE = re.compile(r"\breuters\b", flags=re.IGNORECASE)


def strip_source_artifacts(text: str | None) -> str:
    """Remove dataset-specific source markers (Reuters datelines and mentions).

    This is a *leakage control*. Removing the wire-source signal forces the
    model to rely on writing style rather than on recognizing the publisher,
    producing a more honest (and typically lower) accuracy estimate.
    """
    if not isinstance(text, str):
        return ""
    value = DATELINE_RE.sub("", text)
    value = REUTERS_RE.sub(" ", value)
    return WHITESPACE_RE.sub(" ", value).strip()


def clean_text(text: str | None) -> str:
    """Normalize one text value while preserving words useful for TF-IDF.

    The cleaner is intentionally conservative. It removes obvious noise such as URLs,
    email addresses, non-ASCII artifacts, and repeated whitespace, but it does not strip
    every punctuation mark before vectorization because punctuation boundaries can help
    tokenization.
    """
    if not isinstance(text, str):
        return ""

    value = text.lower()
    value = URL_RE.sub(" ", value)
    value = EMAIL_RE.sub(" ", value)
    value = NON_ASCII_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    return value


class TextCleaner(BaseEstimator, TransformerMixin):
    """scikit-learn compatible transformer for consistent preprocessing.

    When ``strip_source=True`` the cleaner also removes dataset-specific source
    artifacts (Reuters datelines/mentions). Because the flag lives on the
    transformer, the choice is baked into the saved pipeline and applied
    identically at training and inference time (no train/serve skew).
    """

    def __init__(self, strip_source: bool = False) -> None:
        self.strip_source = strip_source

    def fit(self, X: Iterable[object], y: object | None = None) -> TextCleaner:
        return self

    def transform(self, X: Iterable[object]) -> list[str]:
        # ``getattr`` (rather than ``self.strip_source``) keeps pipelines that were
        # pickled before this flag existed loadable; they default to no stripping.
        strip_source = getattr(self, "strip_source", False)
        cleaned = []
        for x in X:
            value = x if isinstance(x, str) else ""
            if strip_source:
                value = strip_source_artifacts(value)
            cleaned.append(clean_text(value))
        return cleaned
