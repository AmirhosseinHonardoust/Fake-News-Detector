from sklearn.pipeline import Pipeline

from train_model import build_pipeline


def test_pipeline_can_fit_and_predict_small_sample():
    pipeline = build_pipeline(max_features=100, min_df=1, max_df=1.0, ngram_max=1, C=1.0)
    X = [
        "reuters official policy statement government economy",
        "reuters senate committee announces new bill",
        "shocking celebrity secret exposed fake claim",
        "viral hoax says impossible miracle happened",
    ]
    y = [0, 0, 1, 1]
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(["reuters government statement"])[0]
    assert isinstance(pipeline, Pipeline)
    assert probs.shape == (2,)
    assert abs(float(probs.sum()) - 1.0) < 1e-9
