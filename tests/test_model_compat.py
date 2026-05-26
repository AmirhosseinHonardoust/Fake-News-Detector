from types import SimpleNamespace

from model_compat import ensure_sklearn_compatibility


class FakeLogisticRegression:
    pass


FakeLogisticRegression.__name__ = "LogisticRegression"


def test_adds_missing_multi_class_to_logistic_classifier():
    classifier = FakeLogisticRegression()
    pipeline = SimpleNamespace(named_steps={"classifier": classifier})
    ensure_sklearn_compatibility(pipeline)
    assert classifier.multi_class == "auto"
