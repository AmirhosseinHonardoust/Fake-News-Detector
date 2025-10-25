#!/usr/bin/env python3
"""
Utility functions for lightweight text normalization used by the fake-news detector.

Default behavior:
- Lowercase
- Remove URLs and emails
- Strip non-ASCII characters
- Collapse multiple whitespace into a single space
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Union, Optional

# Precompiled patterns (module-level so they compile once)
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
EXTRA_SPACE_RE = re.compile(r"\s+")

__all__ = [
    "clean_text",
    "clean_many",
]

def clean_text(
    text: Optional[str],
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_non_ascii: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """
    Clean a single text string with sensible defaults.

    Parameters
    ----------
    text : str | None
        Input text to normalize. Non-string values are treated as empty.
    lowercase : bool
        Convert to lowercase.
    remove_urls : bool
        Remove URL-like substrings.
    remove_emails : bool
        Remove email-like substrings.
    remove_non_ascii : bool
        Strip non-ASCII characters.
    collapse_whitespace : bool
        Replace runs of whitespace with a single space and strip ends.

    Returns
    -------
    str
        The normalized text (possibly empty string).
    """
    if not isinstance(text, str):
        return ""

    s = text

    if lowercase:
        s = s.lower()
    if remove_urls:
        s = URL_RE.sub(" ", s)
    if remove_emails:
        s = EMAIL_RE.sub(" ", s)
    if remove_non_ascii:
        s = NON_ASCII_RE.sub(" ", s)
    if collapse_whitespace:
        s = EXTRA_SPACE_RE.sub(" ", s).strip()

    return s


def clean_many(
    texts: Sequence[Optional[str]],
    **kwargs,
) -> List[str]:
    """
    Vectorized convenience to clean a list/sequence of texts.

    Parameters
    ----------
    texts : sequence of str | None
        Iterable of raw texts.
    **kwargs
        Passed through to `clean_text`.

    Returns
    -------
    list[str]
        Cleaned strings, same order as input.
    """
    return [clean_text(t, **kwargs) for t in texts]
