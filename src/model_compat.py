#!/usr/bin/env python3
"""Model loading helpers with integrity checks and sklearn compatibility repairs.

Security note
-------------
joblib/pickle artifacts execute arbitrary Python code when they are deserialized.
Loading a ``.joblib`` file is therefore equivalent to running whatever code it
contains. Only load artifacts that you produced yourself or obtained from a fully
trusted source; never load a pipeline downloaded from an untrusted location.

As a defense against accidental corruption or casual substitution, training writes
a SHA-256 sidecar (``<artifact>.sha256``) next to the saved pipeline, and
``load_pipeline`` verifies that sidecar when it is present. This detects a changed
or truncated file. It is NOT a defense against a determined attacker who can
replace both the artifact and its sidecar, and it is not a substitute for the
trust rule above. See ``docs/security.md`` for details.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib

_CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def write_checksum(path: str | Path) -> Path:
    """Write ``<path>.sha256`` containing the file's digest; return the sidecar path."""
    path = Path(path)
    sidecar = _sidecar_path(path)
    sidecar.write_text(sha256_file(path) + "\n", encoding="utf-8")
    return sidecar


def verify_checksum(path: str | Path) -> bool | None:
    """Verify a file against its ``.sha256`` sidecar.

    Returns ``True``/``False`` when a sidecar exists, or ``None`` when none is found.
    """
    path = Path(path)
    sidecar = _sidecar_path(path)
    if not sidecar.exists():
        return None
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip()
    return sha256_file(path) == expected


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


def load_pipeline(path: str | Path, *, verify: bool = True) -> Any:
    """Load a saved pipeline, verify its checksum sidecar, and apply compat repairs.

    When ``verify`` is True and a ``<path>.sha256`` sidecar exists, the file's digest
    must match or a ``ValueError`` is raised. When no sidecar is present, or when
    ``verify`` is False, the file is loaded without an integrity check.

    Reminder: loading a joblib artifact runs arbitrary code. Only load files you
    trust (see the module-level security note and ``docs/security.md``).
    """
    path = Path(path)
    if verify and verify_checksum(path) is False:
        raise ValueError(
            f"Checksum mismatch for {path}. The artifact does not match its "
            f"{path.name}.sha256 sidecar; it may be corrupted or modified. "
            "Retrain to regenerate it, or pass verify=False to bypass the check."
        )
    pipeline = joblib.load(path)
    return ensure_sklearn_compatibility(pipeline)
