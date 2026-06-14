from types import SimpleNamespace

import joblib

from model_compat import (
    ensure_sklearn_compatibility,
    load_pipeline,
    sha256_file,
    verify_checksum,
    write_checksum,
)
from train_model import build_pipeline


class FakeLogisticRegression:
    pass


FakeLogisticRegression.__name__ = "LogisticRegression"


def test_adds_missing_multi_class_to_logistic_classifier():
    classifier = FakeLogisticRegression()
    pipeline = SimpleNamespace(named_steps={"classifier": classifier})
    ensure_sklearn_compatibility(pipeline)
    assert classifier.multi_class == "auto"


def test_sha256_file_matches_and_changes_with_content(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"hello world")
    digest = sha256_file(a)
    assert len(digest) == 64
    assert sha256_file(a) == digest  # deterministic
    a.write_bytes(b"hello world!")
    assert sha256_file(a) != digest


def test_write_and_verify_checksum_roundtrip(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"some bytes")
    sidecar = write_checksum(target)
    assert sidecar.exists()
    assert verify_checksum(target) is True


def test_verify_checksum_none_when_missing(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"x")
    assert verify_checksum(target) is None


def test_verify_checksum_false_on_mismatch(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"x")
    write_checksum(target)
    target.write_bytes(b"tampered")  # change file, keep old sidecar
    assert verify_checksum(target) is False


def _dump_small_pipeline(path):
    pipeline = build_pipeline(max_features=50, min_df=1, max_df=1.0, ngram_max=1, C=1.0)
    pipeline.fit(["reuters policy", "viral hoax claim"], [0, 1])
    joblib.dump(pipeline, path)


def test_load_pipeline_ok_with_matching_sidecar(tmp_path):
    path = tmp_path / "pipeline.joblib"
    _dump_small_pipeline(path)
    write_checksum(path)
    loaded = load_pipeline(path)
    assert loaded.predict_proba(["reuters policy"]).shape == (1, 2)


def test_load_pipeline_raises_on_bad_sidecar(tmp_path):
    path = tmp_path / "pipeline.joblib"
    _dump_small_pipeline(path)
    (tmp_path / "pipeline.joblib.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    try:
        load_pipeline(path)
        raise AssertionError("expected ValueError on checksum mismatch")
    except ValueError:
        pass
    # verify=False bypasses the integrity check
    assert load_pipeline(path, verify=False) is not None


def test_load_pipeline_ok_without_sidecar(tmp_path):
    path = tmp_path / "pipeline.joblib"
    _dump_small_pipeline(path)
    assert load_pipeline(path) is not None
