#!/usr/bin/env python3
"""
Small I/O utilities shared across training and app code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union, Mapping

PathLike = Union[str, Path]

__all__ = [
    "ensure_outdir",
    "save_json",
    "load_json",
]

def ensure_outdir(path: PathLike) -> Path:
    """
    Ensure a directory exists; create parents if needed.

    Parameters
    ----------
    path : str | Path
        Directory path to create/ensure.

    Returns
    -------
    Path
        Resolved directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def save_json(obj: Mapping[str, Any] | Any, path: PathLike, *, indent: int = 2) -> Path:
    """
    Save a Python object as pretty-printed JSON.

    Parameters
    ----------
    obj : Any
        JSON-serializable object.
    path : str | Path
        Output file path.
    indent : int
        JSON indent spacing.

    Returns
    -------
    Path
        The written file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    return p.resolve()


def load_json(path: PathLike) -> Any:
    """
    Load JSON from disk.

    Parameters
    ----------
    path : str | Path
        JSON file path.

    Returns
    -------
    Any
        Parsed JSON.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
