<div align="center">

# Fake News Style-Risk Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LogReg-orange)
![Responsible ML](https://img.shields.io/badge/Responsible%20ML-Leakage%20Analysis-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Fake-News-Detector/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Fake-News-Detector/actions/workflows/ci.yml)

</div>

A responsible machine learning project that turns news text into a **style-risk signal**, using a **TF-IDF + Logistic Regression** pipeline with **honest evaluation**, **dataset leakage analysis**, **leakage-controlled training**, **checksum-verified model loading**, a **Streamlit dashboard**, and **command-line inference**.

> **Important:** This project is a **style-risk detector and educational demo**, not a real-world fact-checker.
>
> The model, thresholds, and reports are designed to demonstrate a professional, honest text-classification workflow. They estimate whether text *looks* stylistically similar to real or fake examples in the training data; they do not verify claims against external evidence and should not be used for fact-checking, moderation, or any high-stakes decision.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training and Evaluation](#training-and-evaluation)
- [Leakage Controls and Honest Evaluation](#leakage-controls-and-honest-evaluation)
- [Model Output](#model-output)
- [Model Artifacts and Loading Safety](#model-artifacts-and-loading-safety)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Fake-news detection is often presented as if a text classifier can decide whether an article is true or false. In reality, a classifier cannot verify factual truth without external evidence. A model score is only useful if it can support a defensible action:

- label text as resembling real or fake examples
- abstain on borderline cases instead of forcing a decision
- expose how much of its performance comes from dataset shortcuts
- communicate uncertainty and limitations clearly

This project demonstrates an end-to-end, honest text-classification workflow on a labeled news dataset. It includes preprocessing, model training, threshold-based decisions with an uncertainty band, dataset leakage analysis, leakage-controlled training, source-confounding diagnostics, visual reports, checksum-verified artifacts, and a Streamlit dashboard.

The goal is to show how a text classifier can be turned into a **responsible decision-support tool**, not just a single accuracy or AUC score.

---

## What This Project Does

This project can:

- Analyze a news headline or article excerpt
- Estimate whether the text resembles real-news or fake-news examples
- Return a `REAL`, `FAKE`, or `UNCERTAIN` label with a probability
- Deduplicate the dataset and generate evaluation metrics
- Generate performance charts and a confidence distribution
- Quantify dataset leakage with two independent diagnostics
- Train a leakage-controlled model that strips source artifacts
- Report whether an out-of-source evaluation is even feasible
- Save model artifacts with checksum sidecars
- Verify artifact integrity on load
- Provide a Streamlit dashboard for interactive triage
- Run automated tests and CI smoke workflows

---

## What This Project Does Not Do

This project does **not**:

- Prove whether a claim is true or false
- Search the web or retrieve supporting evidence
- Replace professional fact-checkers
- Detect all types of misinformation
- Guarantee real-world accuracy on unseen sources
- Make high-stakes moderation or censorship decisions

A real fact-checking system would need claim extraction, evidence retrieval, source-credibility analysis, external databases, and human review.

---

## Key Features

- **TF-IDF vectorization** with word n-grams for text feature extraction
- **Logistic Regression** baseline classifier
- **Pipeline-embedded `TextCleaner`** so training and inference clean text identically (no train/serve skew)
- **REAL / FAKE / UNCERTAIN** output with a configurable uncertainty band
- **Dataset leakage report** with a "contains Reuters" heuristic
- **Source-confounding diagnostic** with a quantified confounding score
- **Leakage-controlled training** via source-artifact stripping
- **Out-of-source evaluation** that runs when the data permits and reports infeasibility when it does not
- **Checksum-verified model loading** with SHA-256 sidecars
- **Streamlit dashboard** for interactive style-risk analysis
- **Command-line inference** with JSON output
- **Unit tests and GitHub Actions CI**
- **Model card, data statement, and security documentation**

---

## System Workflow

```text
Raw news text
        ↓
Text preprocessing (TextCleaner)
        ↓
TF-IDF vectorization
        ↓
Logistic Regression classifier
        ↓
Probability + uncertainty band
        ↓
REAL / FAKE / UNCERTAIN label
        ↓
Leakage and source-confounding diagnostics
        ↓
Charts, metrics, and checksum-verified artifacts
        ↓
Dashboard and command-line review
```

---

## Project Structure

```text
Fake-News-Detector/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── True.csv
│   └── Fake.csv
│
├── docs/
│   ├── data_statement.md
│   ├── model_card.md
│   └── security.md
│
├── outputs/
│   ├── charts/
│   │   ├── class_distribution.png
│   │   ├── confidence_distribution.png
│   │   ├── confusion_matrix.png
│   │   ├── pr_curve.png
│   │   └── roc_curve.png
│   ├── artifact_environment.json
│   ├── data_profile.json
│   ├── leakage_report.json
│   ├── source_confounding_report.json
│   ├── metrics.json
│   ├── holdout_predictions.csv
│   ├── pipeline.joblib
│   ├── pipeline.joblib.sha256
│   ├── vectorizer.joblib
│   └── model.joblib
│
├── src/
│   ├── detect_fake_news.py
│   ├── evaluation.py
│   ├── model_compat.py
│   ├── streamlit_app.py
│   ├── text_clean.py
│   └── train_model.py
│
├── tests/
│   ├── test_cli_predict.py
│   ├── test_evaluation.py
│   ├── test_model_compat.py
│   ├── test_pipeline.py
│   ├── test_text_clean.py
│   └── test_training_smoke_artifacts.py
│
├── README.md
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── requirements-lock.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Fake-News-Detector.git
cd Fake-News-Detector
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development tools (pytest, Ruff):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

Train the model:

```bash
python src/train_model.py
```

Launch the dashboard:

```bash
streamlit run src/streamlit_app.py
```

Classify text from the command line:

```bash
python src/detect_fake_news.py --text "Officials announced a new economic policy today."
```

---

## Training and Evaluation

The training script loads and deduplicates the data, splits it, trains the pipeline, evaluates on a holdout set, writes the leakage and source-confounding diagnostics, generates charts, and saves checksum-verified artifacts.

```bash
python src/train_model.py \
  --real data/True.csv \
  --fake data/Fake.csv \
  --outdir outputs
```

Generated outputs include:

```text
outputs/metrics.json
outputs/data_profile.json
outputs/leakage_report.json
outputs/source_confounding_report.json
outputs/holdout_predictions.csv
outputs/pipeline.joblib
outputs/pipeline.joblib.sha256
outputs/charts/
```

---

## Leakage Controls and Honest Evaluation

The bundled dataset is small (1,998 rows; 1,991 after deduplication: 992 REAL, 999 FAKE) and **not source-balanced**. Almost every REAL article is Reuters wire copy (`politicsNews`), while the FAKE articles come from a different source (`News`). The model can separate the classes by recognizing the source rather than detecting misinformation.

Two diagnostics, written on every run, make this explicit:

<div align="center">

| Diagnostic | Finding |
|---|---|
| `outputs/leakage_report.json` | A one-rule heuristic, "contains Reuters => REAL", reaches about 99.5% accuracy alone |
| `outputs/source_confounding_report.json` | The `subject` column has a **confounding score of 1.000** (`politicsNews` is 100% REAL, `News` is 100% FAKE) |

</div>

To train without the most obvious shortcut, strip Reuters datelines and mentions before vectorizing:

```bash
python src/train_model.py --strip-source-artifacts
```

On this dataset, accuracy moves only from **1.000 to 0.995** — useful evidence that the leakage runs deeper than one token. With Reuters removed, FAKE is still separable from "Getty images" / "featured image" footers and REAL from wire-service style.

An honest out-of-source split (train on some sources, test on entirely held-out ones) is the right robustness check, but it is **impossible on this dataset** because each source maps to exactly one label. The diagnostic reports this automatically; on a source-balanced dataset it runs the real split:

```bash
python src/train_model.py --eval-out-of-source --group-col subject
```

> These diagnostics are transparency tools. High accuracy on this dataset reflects source recognition, not real-world fake-news detection.

---

## Model Output

The model returns one of three labels:

<div align="center">

| Label | Meaning |
|---|---|
| `REAL` | The text resembles real-news examples in the dataset |
| `FAKE` | The text resembles fake-news examples in the dataset |
| `UNCERTAIN` | The probability is too close to the decision boundary |

</div>

With `--json`, the CLI prints a machine-readable result:

```json
{
  "label": "UNCERTAIN",
  "prob_fake": 0.54,
  "threshold": 0.5,
  "uncertainty_margin": 0.1,
  "model_path": "outputs/pipeline.joblib"
}
```

`UNCERTAIN` is returned when the fake-style probability falls inside the band around the threshold (by default 0.45–0.55).

---

## Model Artifacts and Loading Safety

The trained model is stored as a joblib/pickle file. **Deserializing a pickle executes arbitrary code**, so only load artifacts you produced yourself or fully trust — never one downloaded from an untrusted source.

As an integrity safeguard, training writes a checksum sidecar next to the pipeline, and loading verifies it:

```text
outputs/pipeline.joblib
outputs/pipeline.joblib.sha256
```

`model_compat.load_pipeline()` checks the sidecar by default and raises on a mismatch; pass `verify=False` to bypass.

> The checksum guards against accidental corruption or a casual file swap, not a determined attacker. See `docs/security.md` for the full threat model.

---

## Streamlit Dashboard

Launch the app:

```bash
streamlit run src/streamlit_app.py
```

The dashboard helps review:

- the predicted `REAL` / `FAKE` / `UNCERTAIN` label
- the fake-style probability and confidence guidance
- the decision rule and adjustable threshold and uncertainty band
- holdout metrics from the trained model
- a responsible-use warning for short or ambiguous inputs

<div align="center">
<img width="676" alt="Streamlit dashboard" src="https://github.com/user-attachments/assets/cfd823d0-27a3-415b-8d96-7b2bcb274098" />
</div>

---

## Evaluation Metrics

Evaluation uses a stratified train/test split with cross-validation on the training split only.

<div align="center">

| Metric | Why it matters |
|---|---|
| Accuracy | Overall correctness at the default decision rule |
| Macro F1 | Balance across the REAL and FAKE classes |
| Precision / Recall (FAKE) | Quality of FAKE predictions and coverage of FAKE cases |
| ROC-AUC | Ranking quality across thresholds |
| Average precision / PR-AUC | Positive-class ranking quality |
| Confounding score | How strongly the source determines the label (1.0 = fully confounded) |

</div>

Example results from the included run, with and without source stripping:

<div align="center">

| Metric | Leaked (default) | De-leaked (`--strip-source-artifacts`) |
|---|---:|---:|
| Holdout accuracy | 1.000 | 0.995 |
| Holdout macro F1 | 1.000 | 0.995 |
| Holdout ROC-AUC | 1.000 | 1.000 |
| Confounding score | 1.000 | 1.000 |

</div>

> The near-perfect leaked numbers are evidence of source leakage, not real-world performance. Read them alongside the leakage diagnostics.

---

## Visual Reports

### Model evaluation charts

<div align="center">

| Confusion Matrix | ROC Curve |
|---|---|
| ![Confusion matrix](https://github.com/user-attachments/assets/04d9732c-5efb-4d68-b1bc-04d2afa539a8) | ![ROC curve](https://github.com/user-attachments/assets/b96c992f-8d48-4f76-9b43-eab15eeb4438) |
| **Analysis:** The confusion matrix shows correct and incorrect predictions for REAL and FAKE on the holdout set. Accuracy alone can hide weaknesses, so the per-class view matters. | **Analysis:** The ROC curve shows class separation across thresholds. A strong AUC should still be read carefully here, since source leakage inflates separability. |

</div>

### Probability behavior

<div align="center">

| Precision-Recall Curve | Confidence Distribution |
|---|---|
| ![Precision-recall curve](https://github.com/user-attachments/assets/ef755b40-3798-4fe3-936d-287fe72e9a17) | ![Confidence distribution](https://github.com/user-attachments/assets/5c3dcf79-172d-4a28-bb4e-bc07abe836e8) |
| **Analysis:** The precision-recall curve shows how precision and recall trade off for the FAKE class across thresholds. | **Analysis:** The confidence distribution shows how often the model makes strong versus borderline predictions, which connects directly to the `UNCERTAIN` band. |

</div>

<details>
<summary>Additional class distribution chart</summary>

<div align="center">

![Class distribution](https://github.com/user-attachments/assets/d0b7758a-11b7-41da-8c8b-cac99adf469b)

The class distribution chart shows whether the dataset is balanced between REAL and FAKE, which prevents accuracy from being inflated by a dominant class.

</div>

</details>

---

## Testing and CI

Run unit tests locally:

```bash
pytest
```

Compile source files and lint:

```bash
python -m compileall src tests
ruff check src tests
```

The GitHub Actions workflow checks:

- dependency installation
- source compilation
- unit tests
- linting with Ruff
- a default training smoke workflow
- a leakage-controlled (`--strip-source-artifacts`) training smoke workflow
- CLI prediction schema validation
- training artifact and metrics validation
- checksum-sidecar integrity validation

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

The project separates responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `src/text_clean.py` | Text normalization and the pipeline `TextCleaner`, including source-artifact stripping |
| `src/train_model.py` | Trains the pipeline, runs evaluation and diagnostics, writes artifacts |
| `src/evaluation.py` | Source-confounding diagnostic and out-of-source holdout |
| `src/model_compat.py` | Checksum integrity checks and cross-version loading repairs |
| `src/detect_fake_news.py` | Command-line inference with REAL / FAKE / UNCERTAIN output |
| `src/streamlit_app.py` | Interactive dashboard for style-risk analysis |

</div>

Tooling is configured through `pyproject.toml` (Ruff, pytest) and `requirements-dev.txt`.

---

## Limitations

This project has important limitations:

- The dataset is a small educational snapshot, not a production corpus
- The model does not verify factual truth or understand real-world events
- The bundled data separates almost entirely on source, not on misinformation
- High accuracy on this dataset does not imply real-world accuracy
- The model may perform poorly on unseen sources or very short text
- An out-of-source evaluation is impossible on the bundled, single-source data
- Predictions are style-risk estimates, not fact-check verdicts

The project is strongest as a portfolio demonstration of honest, leakage-aware text-classification workflow design.

---

## Responsible Use

This repository is intended for:

- machine learning and NLP education
- demonstrating honest, leakage-aware evaluation
- practicing text-classification workflows
- exploring uncertainty handling and abstention
- responsible-ML documentation practice
- portfolio demonstration

It should not be used as-is for:

- automated fact-checking
- political or news content moderation
- credibility scoring without human review
- legal or journalistic decisions
- any high-stakes classification

Any real deployment would require diverse data, external validation, fairness review, and a human escalation process.

---

## Future Improvements

Potential next improvements:

- Add source-balanced and time-separated datasets
- Add external validation on publishers absent from training
- Extend artifact stripping beyond the Reuters marker
- Add calibration metrics such as Brier score and reliability plots
- Add SHAP or feature-importance explanations
- Add Docker support and deploy the dashboard
- Explore stronger NLP models
- Add out-of-distribution detection

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Streamlit
- joblib
- pytest
- Ruff
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is intended for educational, research, and portfolio purposes.

If you use or modify this project, please keep the responsible-use notes and limitations clear.
