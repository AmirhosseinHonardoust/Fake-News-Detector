# Model Card: Fake News Style-Risk Detector

## Model overview

This project trains a binary text classifier for educational fake-news style-risk demos.

- **Task:** Binary text classification
- **Labels:** `REAL`, `FAKE`
- **Model:** TF-IDF vectorizer + Logistic Regression
- **Preferred artifact:** `outputs/pipeline.joblib`
- **Framework:** scikit-learn

## Intended use

Good uses:

- Portfolio demonstration
- NLP/classification education
- Streamlit demo
- Experimenting with text preprocessing and evaluation workflows

Out-of-scope uses:

- Content moderation decisions
- Political/news credibility scoring without human review
- Any high-stakes decision

## Training data

The included dataset is a downsampled educational snapshot with real and fake news examples. It is not source-balanced and contains strong artifacts.

Current profile from `outputs/data_profile.json`:

- Rows before deduplication: 1,998
- Rows after deduplication: 1,991
- Duplicate rows removed: 7
- REAL rows after deduplication: 992
- FAKE rows after deduplication: 999

## Validation protocol

- Stratified train/test split
- 20% holdout test set
- 3-fold StratifiedKFold cross-validation on the training split only
- Threshold-based classification with default `p(fake) >= 0.5`
- Source-confounding diagnostic (`outputs/source_confounding_report.json`) and,
  when feasible, an out-of-source holdout where whole groups are held out of
  training

## Current metrics

From `outputs/metrics.json`:

| Metric | Leaked (default) | De-leaked (`--strip-source-artifacts`) |
|---|---:|---:|
| Holdout accuracy | 1.000 | 0.995 |
| Holdout macro F1 | 1.000 | 0.995 |
| Holdout ROC-AUC | 1.000 | 1.000 |

The de-leaked column removes the Reuters dateline and any "Reuters" mentions
before training. Reproduce with:

```bash
python src/train_model.py --outdir outputs                          # leaked
python src/train_model.py --outdir outputs_deleaked --strip-source-artifacts
```

## Critical limitation: source/style leakage

`outputs/leakage_report.json` shows that source artifacts alone are almost enough to separate the classes:

- 99.8% of REAL rows contain “Reuters”
- 0.8% of FAKE rows contain “Reuters”
- A simple heuristic `Reuters => REAL, otherwise FAKE` reaches about 99.5% accuracy

This strongly suggests that the model learns dataset/source artifacts in addition to, or instead of, misinformation patterns.

### The leakage is deeper than one token

Removing "Reuters" barely changes accuracy (1.000 → 0.995). Inspecting the
de-leaked model's top features shows why: the FAKE class is still separable
from boilerplate such as "Getty images" and "featured image" image-credit
footers, while the REAL class keys on wire-service house style ("said",
weekday datelines, "president donald"). In other words, the corpus is two
entirely different sources with two different writing styles, and stripping a
single marker does not remove that separation. A realistic evaluation requires
source-balanced data or an out-of-source split, not just token removal.

### Out-of-source evaluation is impossible on this dataset

The source-confounding diagnostic reports a confounding score of **1.000** for the
`subject` column: `politicsNews` is 100% REAL and `News` is 100% FAKE. Because each
source maps to exactly one class, holding out any source for testing produces a
single-class test set, so an honest out-of-source holdout cannot be run here
(`out_of_source_split_feasible: false`). This is the strongest possible evidence
that the reported accuracy reflects source recognition rather than misinformation
detection. The diagnostic (`outputs/source_confounding_report.json`) runs
automatically; on a source-balanced dataset it will instead execute an
out-of-source holdout via `--eval-out-of-source`.

## Ethical considerations

The model can produce misleading credibility labels. Users may over-trust predictions if the app is presented without caveats. The UI and README should always frame predictions as educational estimates, not truth judgments or fact-check verdicts.

## Recommendations before production use

- Use source-balanced and time-separated datasets
- Evaluate on external publishers not present in training
- Add calibration metrics such as Brier score and reliability plots
- Add confidence/uncertainty warnings
- Keep human review for any real-world use
- Monitor drift over time
