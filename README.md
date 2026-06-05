<div align="center">

# Fake News Style-Risk Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-green)
[![CI](https://github.com/AmirhosseinHonardoust/Fake-News-Detector/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Fake-News-Detector/actions/workflows/ci.yml)

</div>

A professional machine learning project that analyzes news text and predicts whether it stylistically resembles **REAL** or **FAKE** news examples from a labeled dataset.

The project uses a **TF-IDF + Logistic Regression** pipeline and includes a Streamlit dashboard, command-line prediction support, model evaluation, leakage analysis, charts, tests, and responsible machine learning documentation.

> **Important:** This project is a **style-risk detector**, not a real-world fact-checker.  
> It does not verify claims using external evidence. Instead, it estimates whether a text looks stylistically similar to examples labeled as real or fake in the training dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Features](#features)
- [Dashboard Preview](#dashboard-preview)
- [Streamlit App Dashboard](#streamlit-app-dashboard)
- [Charts and Visual Analysis](#charts-and-visual-analysis)
- [How the Model Works](#how-the-model-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the App](#running-the-app)
- [Command-Line Usage](#command-line-usage)
- [Model Output](#model-output)
- [Evaluation](#evaluation)
- [Dataset Leakage Analysis](#dataset-leakage-analysis)
- [Testing](#testing)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Fake news detection is a common natural language processing task. Many beginner projects present fake news classifiers as if they can determine whether a news article is true or false. In reality, a text classifier cannot verify factual truth without external evidence.

This project takes a more responsible approach.

It classifies text based on learned writing patterns from a labeled dataset and clearly communicates that the model is a **style-based risk detector**, not a factual truth engine.

The goal of this project is to demonstrate:

- A clean machine learning workflow
- Responsible evaluation
- Honest model limitations
- Interactive dashboard design
- Professional documentation
- Reproducible training and testing

---

## What This Project Does

This project can:

- Analyze a news headline or article excerpt
- Estimate whether the text resembles real-news or fake-news examples
- Return a probability score
- Mark borderline predictions as `UNCERTAIN`
- Generate evaluation metrics
- Generate model performance charts
- Provide a Streamlit dashboard for interaction
- Save trained model artifacts
- Run automated tests
- Document dataset and model limitations

---

## What This Project Does Not Do

This project does **not**:

- Prove whether a claim is true or false
- Search the web for supporting evidence
- Replace professional fact-checkers
- Detect all types of misinformation
- Guarantee real-world accuracy
- Make high-stakes moderation decisions

A real fact-checking system would require claim extraction, evidence retrieval, source credibility analysis, external databases, and human review.

---

## Features

- **TF-IDF Vectorization** for text feature extraction
- **Logistic Regression** classifier
- **Stratified train/test split**
- **Cross-validation** on training data
- **Saved sklearn pipeline** using `joblib`
- **Streamlit dashboard**
- **Command-line prediction script**
- **REAL / FAKE / UNCERTAIN output**
- **Probability-based confidence score**
- **Confusion matrix chart**
- **ROC curve chart**
- **Class distribution chart**
- **Prediction confidence analysis**
- **Dataset leakage report**
- **Model card**
- **Data statement**
- **Pytest test suite**
- **GitHub Actions CI support**

---

## Dashboard Preview

The project includes an interactive Streamlit dashboard that allows users to paste news text and receive a style-risk prediction.

The dashboard displays:

- Prediction label
- Fake-style probability
- Confidence interpretation
- Responsible-use warning
- Model metrics
- Chart references
- Leakage analysis notes

Example output:

```text
Prediction: UNCERTAIN
Fake-style probability: 54.2%

The model is close to the decision boundary.
Please provide a longer headline or article excerpt for a more reliable prediction.
```

---

## Streamlit App Dashboard

<div align="center">

<img width="676" height="761" alt="Screenshot 2026-05-27 at 01-46-00 Fake News Style-Risk Detector" src="https://github.com/user-attachments/assets/cfd823d0-27a3-415b-8d96-7b2bcb274098" />
</div>

Run the dashboard with:

```bash
streamlit run src/streamlit_app.py
```

The dashboard is designed to be simple, readable, and honest about what the model can and cannot do.

### Dashboard Features

<div align="center">

| Feature | Description |
|---|---|
| Text Input Area | Allows users to paste a headline, paragraph, or article excerpt |
| Prediction Label | Displays `REAL`, `FAKE`, or `UNCERTAIN` |
| Fake-Style Probability | Shows the probability that the text resembles fake-news examples |
| Confidence Guidance | Explains whether the result is strong or borderline |
| Responsible-Use Warning | Reminds users that the model is not a fact-checking system |
| Metrics Section | Displays available evaluation metrics from the trained model |
| Leakage Warning | Explains why high scores should be interpreted carefully |
</div>

### Dashboard Interpretation

The dashboard should be interpreted as a **style-risk analysis tool**.

For example:

```text
Prediction: FAKE
Fake-style probability: 87%
```

This means the text is stylistically similar to fake-news examples in the dataset.  
It does **not** prove that the article is factually false.

Another example:

```text
Prediction: UNCERTAIN
Fake-style probability: 53%
```

This means the model is not confident enough to make a strong classification.  
The input may be too short, too vague, or too different from the training examples.

### Why the Dashboard Includes an UNCERTAIN State

Many fake-news classifiers force every input into either `REAL` or `FAKE`. This can be misleading, especially for short or ambiguous text.

This project uses an uncertainty band:

```text
Low fake-style probability      → REAL
Middle probability range        → UNCERTAIN
High fake-style probability     → FAKE
```

This makes the app more responsible and avoids overconfident predictions.

---

## Charts and Visual Analysis

The project automatically generates visual outputs during training to make model behavior easier to understand.

Generated charts are saved in:

```text
outputs/charts/
```

The main charts include:

<div align="center">

| Chart | Purpose |
|---|---|
| Confusion Matrix | Shows correct and incorrect predictions for REAL and FAKE samples |
| ROC Curve | Shows how well the model separates the two classes across thresholds |
| Precision-Recall Curve | Shows precision/recall trade-offs for the FAKE class |
| Class Distribution | Shows whether the dataset is balanced |
| Confidence Distribution | Helps analyze how confident the model is across predictions |
</div>

### Confusion Matrix

<div align="center">

<img width="520" height="360" alt="confusion_matrix" src="https://github.com/user-attachments/assets/04d9732c-5efb-4d68-b1bc-04d2afa539a8" />
</div>

The confusion matrix helps identify:

- REAL articles correctly classified as REAL
- FAKE articles correctly classified as FAKE
- REAL articles incorrectly classified as FAKE
- FAKE articles incorrectly classified as REAL

This is important because accuracy alone can hide model weaknesses.

### ROC Curve

<div align="center">

<img width="520" height="360" alt="roc_curve" src="https://github.com/user-attachments/assets/b96c992f-8d48-4f76-9b43-eab15eeb4438" />
</div>

The ROC curve shows the trade-off between true positive rate and false positive rate at different classification thresholds.

A strong ROC-AUC score means the model separates the dataset classes well. However, ROC-AUC should still be interpreted carefully because dataset leakage can make results look stronger than they would be in real-world use.

### Precision-Recall Curve

<div align="center">

<img width="520" height="360" alt="pr_curve" src="https://github.com/user-attachments/assets/ef755b40-3798-4fe3-936d-287fe72e9a17" />
</div>

The precision-recall curve is useful for understanding how precision and recall change for the FAKE class across thresholds.

### Class Distribution

<div align="center">

<img width="520" height="350" alt="class_distribution" src="https://github.com/user-attachments/assets/d0b7758a-11b7-41da-8c8b-cac99adf469b" />
</div>

The class distribution chart shows whether the dataset is balanced between REAL and FAKE articles.

Balanced datasets are useful because they prevent accuracy from being inflated by a dominant class.

### Confidence Distribution

<div align="center">

<img width="520" height="350" alt="confidence_distribution" src="https://github.com/user-attachments/assets/5c3dcf79-172d-4a28-bb4e-bc07abe836e8" />
</div>

The confidence distribution helps explain how often the model makes strong predictions versus borderline predictions.

This is directly connected to the `UNCERTAIN` output.

Example:

```text
Fake-style probability: 51%
Output: UNCERTAIN
```

This is more responsible than forcing the output to `REAL` or `FAKE`.

---

## How the Model Works

The model uses a classic supervised machine learning pipeline:

```text
Raw news text
   ↓
Text preprocessing
   ↓
TF-IDF vectorization
   ↓
Logistic Regression classifier
   ↓
Prediction probability
   ↓
REAL / FAKE / UNCERTAIN label
```

### TF-IDF Vectorization

TF-IDF converts text into numerical features by measuring how important words are in a document relative to the full dataset.

### Logistic Regression

Logistic Regression is a strong baseline model for text classification. It is fast, interpretable, and works well with sparse TF-IDF features.

### Uncertainty Handling

Instead of always forcing a binary prediction, this project includes an uncertainty range.

If the model probability is too close to the decision boundary, the final output becomes:

```text
UNCERTAIN
```

This is especially useful for short inputs.

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
│   ├── Fake.csv
│   └── True.csv
│
├── docs/
│   ├── data_statement.md
│   └── model_card.md
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
│   ├── holdout_predictions.csv
│   ├── leakage_report.json
│   ├── metrics.json
│   ├── model.joblib
│   ├── pipeline.joblib
│   └── vectorizer.joblib
│
├── src/
│   ├── detect_fake_news.py
│   ├── model_compat.py
│   ├── streamlit_app.py
│   ├── text_clean.py
│   ├── train_model.py
│   └── utils.py
│
├── tests/
│
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
├── pyproject.toml
└── Makefile
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

For development tools:

```bash
pip install -r requirements-dev.txt
```

---

## Training the Model

Run:

```bash
python src/train_model.py
```

This will:

- Load the dataset
- Preprocess the text
- Split the data into train and test sets
- Train the TF-IDF + Logistic Regression pipeline
- Run evaluation
- Save the trained model
- Save metrics
- Generate charts
- Generate leakage analysis

Generated outputs:

```text
outputs/pipeline.joblib
outputs/metrics.json
outputs/holdout_predictions.csv
outputs/charts/
outputs/leakage_report.json
```

---

## Running the App

Run:

```bash
streamlit run src/streamlit_app.py
```

Then open the local URL shown in your terminal.

The app will load the trained pipeline from:

```text
outputs/pipeline.joblib
```

If the model file does not exist, train the model first:

```bash
python src/train_model.py
```

---

## Command-Line Usage

You can also make predictions directly from the terminal.

```bash
python src/detect_fake_news.py --text "Your news text here"
```

Example:

```bash
python src/detect_fake_news.py --text "Government officials announced a new economic policy today."
```

For JSON output:

```bash
python src/detect_fake_news.py --text "Your news text here" --json
```

---

## Model Output

The model returns one of three labels:

<div align="center">

| Label | Meaning |
|---|---|
| `REAL` | The text resembles real-news examples in the dataset |
| `FAKE` | The text resembles fake-news examples in the dataset |
| `UNCERTAIN` | The model confidence is too close to the decision boundary |
</div>

Example:

```json
{
  "label": "UNCERTAIN",
  "fake_probability": 0.54,
  "confidence": "low"
}
```

---

## Evaluation

The project uses an honest evaluation workflow.

Evaluation includes:

- Stratified train/test split
- Cross-validation on the training set
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve

Metrics are saved to:

```text
outputs/metrics.json
```

Charts are saved to:

```text
outputs/charts/
```

### Why This Matters

A common mistake in beginner ML projects is evaluating the model on the same data used for training. That produces misleadingly high results.

This project avoids that by using a separate test set and reporting metrics more responsibly.

---

## Dataset Leakage Analysis

Fake news datasets can contain hidden shortcuts.

For example, if REAL articles mostly come from one publisher and FAKE articles mostly come from another, the model may learn publisher-specific patterns instead of general fake-news signals.

This project includes a leakage report saved at:

```text
outputs/leakage_report.json
```

The leakage report helps explain whether the model may be relying on dataset-specific artifacts.

This makes the project more transparent and professionally defensible.

---

## Testing

Run the test suite:

```bash
pytest
```

The tests check important project behavior, including:

- Text preprocessing
- Model compatibility
- Prediction output
- Training artifacts
- App-related utility functions

---

## Code Quality

The project includes development tooling through:

```text
pyproject.toml
requirements-dev.txt
```

These files support:

- Automated tests
- Linting configuration
- Cleaner project maintenance
- More professional GitHub presentation

---

## Limitations

This project has important limitations.

The model:

- Does not verify factual truth
- Does not search external sources
- Does not understand real-world events
- May learn dataset-specific patterns
- May perform poorly on unseen news sources
- May be unreliable for very short text
- Should not be used for real-world moderation or censorship

High performance on the included dataset does not guarantee high performance in real-world fake news detection.

---

## Responsible Use

This project is intended for:

- Machine learning education
- NLP portfolio demonstration
- Text classification practice
- Streamlit dashboard development
- Responsible ML documentation practice

It should not be used for:

- Automated fact-checking
- Political content moderation
- Legal or journalistic decisions
- High-stakes classification
- Replacing human review

---

## Future Improvements

Possible future improvements include:

- Add more diverse datasets
- Evaluate on external news datasets
- Add dataset deduplication checks
- Add model comparison experiments
- Add calibration analysis
- Add SHAP or feature importance explanations
- Add Docker support
- Add a full EDA notebook
- Deploy the Streamlit app online
- Add more advanced NLP models
- Improve uncertainty calibration
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
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is intended for educational and portfolio purposes.

If you use or modify this project, please keep the responsible-use notes and limitations clear.
