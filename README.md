# Fake News & Misinformation Detector

Detect fake vs real news articles using Machine Learning, TF-IDF, and Logistic Regression, complete with training scripts, evaluation charts, and an interactive Streamlit web app.
      
---

## Table of Contents
- [Overview](#-overview)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Training the Model](#-training-the-model)
- [Evaluation & Charts](#-evaluation--charts)
- [How It Works](#-how-it-works)
- [Running the Streamlit App](#-running-the-streamlit-app)
- [Code Modules](#-code-modules)
- [Technologies Used](#-technologies-used)
- [License](#-license)
- [Author](#-author)
- [Future Improvements](#-future-improvements)
 
---

## Overview

The **Fake News & Misinformation Detector** is a complete end-to-end **Natural Language Processing (NLP)** project that classifies news headlines and articles as **REAL** or **FAKE**.  
It combines **TF-IDF feature extraction** with a **Logistic Regression classifier**, achieving perfect accuracy on the cleaned dataset.

The project also includes:
- **Model evaluation with visual charts**
- **Interactive Streamlit web app**
- **Reusable and modular code structure**

---

## Demo

### Streamlit Web App

When launched, the app allows you to paste or type any news headline or paragraph and analyze its credibility in real time.

<img width="683" height="490" alt="Screenshot 2025-10-25 at 17-39-13 Fake News Detector" src="https://github.com/user-attachments/assets/d68c9b02-fe94-48d5-85ce-06b292e38dbb" />

- Prediction: *REAL* or *FAKE*  
- Probability bar visualization  
- Adjustable fake-detection threshold  
 
---

## Project Structure

```
fake-news-detector/
│
├── data/
│   ├── True.csv                 # Real news (999 rows)
│   ├── Fake.csv                 # Fake news (999 rows)
│
├── outputs/
│   ├── model.joblib             # Trained Logistic Regression model
│   ├── vectorizer.joblib        # TF-IDF vectorizer
│   ├── pipeline.joblib          # Combined pipeline (optional)
│   ├── metrics.json             # Model performance report
│   ├── confusion_matrix.png     # Confusion Matrix plot
│   ├── roc_curve.png            # ROC curve plot
│   └── pr_curve.png             # Precision-Recall curve plot
│
├── src/
│   ├── text_clean.py            # Text preprocessing utilities
│   ├── utils.py                 # I/O helpers
│   ├── train_model.py           # Training and evaluation script
│   ├── detect_fake_news.py      # CLI prediction script
│   └── streamlit_app.py         # Streamlit web application
│
└── README.md
```

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

---

## Dataset

| File | Type | Rows | Columns |
|------|------|------|----------|
| `True.csv` | Real news | 999 | `title`, `text`, `subject`, `date` |
| `Fake.csv` | Fake news | 999 | `title`, `text`, `subject`, `date` |

> **Dataset Source:**  
> This project uses and modifies the [*Fake and Real News Dataset*](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by **Clément Bisaillon** (Kaggle).  
> Data was cleaned, header-fixed, and **downsampled to 999 REAL and 999 FAKE** news articles for balanced training and clear visualization.  
> Used purely for **educational and research** purposes.

---

## Training the Model

Run the following command from the project root:

```bash
python src/train_model.py --real data/True.csv --fake data/Fake.csv --text-col text --outdir outputs
```

This script will:
1. Load both datasets (real and fake).
2. Clean and merge them using `text_clean.py`.
3. Extract **TF-IDF** features.
4. Train a **Logistic Regression** classifier.
5. Save outputs:
   - `outputs/model.joblib`
   - `outputs/vectorizer.joblib`
   - `outputs/metrics.json`
   - Performance charts (`confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`)

---

## Evaluation & Charts

After training, the model achieves **perfect classification accuracy** on this dataset.

### Confusion Matrix
<img width="1050" height="900" alt="confusion_matrix" src="https://github.com/user-attachments/assets/15735f60-4216-4aae-bf07-e265c034e5ef" />

| True Label | Predicted REAL | Predicted FAKE |
|-------------|----------------|----------------|
| REAL | 999 ✅ | 0 ❌ |
| FAKE | 0 ❌ | 999 ✅ |

The model correctly classified all 1,998 samples.

---

### ROC Curve
<img width="1050" height="900" alt="roc_curve" src="https://github.com/user-attachments/assets/a1b60b15-69f3-4440-ade5-92346731ffa9" />

The ROC curve touches the top-left corner  **AUC = 1.00**  
Perfect separability between classes.

---

### Precision–Recall Curve
<img width="1050" height="900" alt="pr_curve" src="https://github.com/user-attachments/assets/a0786968-55f4-40ad-92ca-6684e0312e50" />

Both precision and recall reach **1.00**, meaning zero false predictions.

---

### Key Metrics
| Metric | Value |
|---------|-------|
| Accuracy | 100 % |
| Precision (FAKE) | 1.00 |
| Recall (FAKE) | 1.00 |
| F1-Score | 1.00 |
| ROC-AUC | 1.00 |

> *Although perfect accuracy is achieved on this dataset, it’s a controlled sample. Real-world news data will naturally introduce noise and uncertainty.*

---

## How It Works

### Pipeline Overview
1. **Text Cleaning** → Remove punctuation, URLs, emails, non-ASCII chars.  
2. **TF-IDF Vectorization** → Convert words into weighted numerical features.  
3. **Logistic Regression** → Predict probability of “FAKE” label.  
4. **Thresholding** → If `p(fake) ≥ 0.5` → FAKE, else REAL.

---

### Example: Command-Line Prediction
```bash
python src/detect_fake_news.py --model outputs/model.joblib   --vectorizer outputs/vectorizer.joblib   --text "It s tough sometimes to imagine that Donald Trump has five children since it s clear from Monday s speech in front of 40,000 Boy Scouts and other attendees at the Boy Scouts Jamboree in West Virginia that he has absolutely no idea what kind of talk is appropriate for children.While most adults would take this opportunity to offer some pearls of adult wisdom or cheerlead the Boy Scouts toward their futures, Trump chose to deliver a tirade of Trumpisms.Like almost any time Trump has tried to string together more than a couple of words at a time, most of his speech was an inarticulate mess which consisted of his trademark whining, a wee bit of swearing and a pointless anecdote about a burned out rich guy at a cocktail party."
```

Output:
```
Label: FAKE | Fake probability: 0.560 | Threshold: 0.40
```

---

## Running the Streamlit App

### Launch the App
```bash
streamlit run src/streamlit_app.py
```

Then open the local web interface:
```
http://localhost:8501
```

### App Features
- Paste any headline or paragraph  
- Analyze with one click  
- Adjust FAKE probability threshold  
- See model file locations and loaded status in sidebar  

---

## Code Modules

| Module | Purpose |
|---------|----------|
| `text_clean.py` | Handles text normalization (lowercasing, regex-based cleaning) |
| `utils.py` | Ensures output directories exist and handles JSON I/O |
| `train_model.py` | Loads data, trains the model, and generates metrics and plots |
| `detect_fake_news.py` | CLI script for predicting individual samples |
| `streamlit_app.py` | Streamlit web app for interactive user testing |

---

## Technologies Used

- **Python 3.10+**
- **scikit-learn** → TF-IDF Vectorizer, Logistic Regression  
- **pandas / numpy** → Data manipulation  
- **matplotlib** → Model visualization  
- **joblib** → Model persistence  
- **Streamlit** → Web interface  

---

## Future Improvements
- Integrate **BERT / DistilBERT** for contextual language understanding  
- Extend dataset for **multi-language** fake news detection  
- Add **Explainable AI** (LIME / SHAP) for model transparency  
- Deploy live on **Streamlit Cloud** or **Hugging Face Spaces**
