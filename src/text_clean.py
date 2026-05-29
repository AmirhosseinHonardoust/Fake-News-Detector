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
    """scikit-learn compatible transformer for consistent preprocessing."""

    def fit(self, X: Iterable[object], y: object | None = None) -> TextCleaner:
        return self

    def transform(self, X: Iterable[object]) -> list[str]:
        return [clean_text(x if isinstance(x, str) else "") for x in X]
