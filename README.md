<div align="center">

# Autocurator — Synthetic Data Benchmark

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Metrics%20%2B%20Models-orange)
![Responsible ML](https://img.shields.io/badge/Responsible%20ML-Privacy%20%26%20Fidelity-green)
![Reference Validated](https://img.shields.io/badge/Metrics-Reference%20Validated-brightgreen)
![Status](https://img.shields.io/badge/Status-Portfolio%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Autocurator-Synthetic-Data-Benchmark/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Autocurator-Synthetic-Data-Benchmark/actions/workflows/ci.yml)

</div>

A modular evaluation toolkit that benchmarks **synthetic tabular data** against **real data** across four axes — **fidelity, coverage, privacy, and utility** — with **reference-validated metrics**, a **generator sensitivity benchmark**, **HTML reporting**, **YAML-driven configuration**, and a **reproducible, multi-version CI pipeline**.

> **Important:** Autocurator is an **evaluation and diagnostic tool**, not a certification of privacy or safety.
>
> Its metrics estimate how closely synthetic data *resembles* real data and how much it may *leak*; they are decision-support signals, not guarantees. A low membership-inference score is not a legal privacy guarantee, and high utility does not certify fitness for any downstream use. Read every number alongside the limitations below.

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
- [Running a Benchmark](#running-a-benchmark)
- [Metrics Explained](#metrics-explained)
- [Metric Output](#metric-output)
- [Evaluation Summary](#evaluation-summary)
- [Validation and Sensitivity](#validation-and-sensitivity)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Synthetic data is often presented as if "looks realistic" and "is safe to share" were the same thing. In reality they pull in different directions: data that perfectly reproduces the original is high-fidelity but leaks privacy, while data that is aggressively privatized loses the utility that made it worth generating. A single similarity score cannot capture that trade-off.

Autocurator evaluates a synthetic dataset the way a reviewer would — by asking several independent questions and reporting each honestly:

- how closely the synthetic **marginals and correlations** match the real ones (fidelity)
- whether the synthetic data **covers the real manifold** without collapsing (coverage)
- how much a synthetic record could help **re-identify** a real one (privacy)
- whether a model **trained on synthetic** still works on real data (utility)

The result is a `metrics.json` summary plus an HTML report, backed by metric implementations that are validated against published references and a benchmark showing they actually respond to synthetic-data quality.

---

## What This Project Does

This project can:

- Load real and synthetic tabular CSVs and align their schema
- Score **fidelity** with per-feature JSD, KS, and Wasserstein distances plus a correlation-structure distance
- Score **coverage** with true **PRDC** (precision, recall, density, coverage)
- Score **privacy** with nearest-neighbor distances and a membership-inference AUC
- Run a stronger **holdout membership-inference attack** when a disjoint real set is supplied
- Score **utility** with TSTR / TRTS using a linear or random-forest estimator
- Render an HTML report with PCA, distribution, and correlation figures
- Drive every option from the CLI or a YAML config file
- Benchmark reference generators of known quality on real bundled datasets
- Validate its own metrics against independent reference implementations in CI

---

## What This Project Does Not Do

This project does **not**:

- Certify that synthetic data is legally private or safe to release
- Generate synthetic data itself (it evaluates data you bring)
- Replace a formal differential-privacy accounting or a security review
- Guarantee real-world utility on tasks outside the evaluated target
- Detect every possible privacy attack or failure mode

A production privacy claim would additionally require differential-privacy guarantees, domain-expert review, and validation on realistic attacker models.

---

## Key Features

- **Fidelity metrics** — per-feature Jensen–Shannon, Kolmogorov–Smirnov, and Wasserstein distances, plus a correlation-matrix distance
- **True PRDC coverage** — per-point k-NN hyperspheres (precision, recall, density, coverage), matching the published reference implementation
- **Privacy diagnostics** — synthetic-to-real nearest-neighbor distances and a self-match-corrected membership-inference AUC
- **Holdout membership-inference attack** — optional distance-to-closest-record attack (`--holdout`) where **0.5 means no leakage**
- **Utility (TSTR / TRTS)** — train-on-synthetic-test-on-real and its reverse, with a **linear** or **random-forest** estimator
- **Shared feature space** — one fitted scaler/encoder applied to real, synthetic, and holdout so cross-set distances are consistent
- **HTML reporting** — PCA scatter, per-feature distributions, and correlation heatmaps, with image links that resolve from any report location
- **YAML configuration** — any option settable via `--config`, with CLI overrides
- **Reference-validated metrics** — PRDC and holdout MIA checked against independent implementations to 1e-9 in CI
- **Generator sensitivity benchmark** — reference synthesizers on real bundled datasets, proving the metrics discriminate quality
- **Strict typing and tooling** — Ruff, Black, and strict Mypy, enforced at ≥90% test coverage
- **Reproducible, multi-version CI** — Python 3.10 / 3.11 / 3.12 plus a lockfile-verified install

---

## System Workflow

```text
Real CSV + Synthetic CSV
        ↓
Schema alignment + shared feature space (scaler + one-hot)
        ↓
Fidelity        Coverage        Privacy         Utility
(JSD/KS/WD)     (PRDC)          (NN + MIA)      (TSTR/TRTS)
        ↓
Aggregated metrics.json
        ↓
PCA / distribution / correlation figures
        ↓
HTML report + machine-readable summary
```

---

## Project Structure

```text
Autocurator-Synthetic-Data-Benchmark/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   └── example.yaml
│
├── data/
│   ├── real.csv
│   └── synthetic.csv
│
├── outputs/
│   └── runs/example_run/
│       ├── metrics.json
│       └── plots/
│           ├── pca.png
│           ├── distributions.png
│           └── correlations.png
│
├── reports/
│   └── example_run.html
│
├── scripts/
│   └── benchmark_generators.py
│
├── src/
│   └── autocurator/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── datasets.py
│       ├── generators.py
│       ├── loaders.py
│       ├── preprocess.py
│       ├── report.py
│       ├── viz.py
│       ├── templates/
│       │   └── report.html
│       └── metrics/
│           ├── __init__.py
│           ├── fidelity.py
│           ├── coverage.py
│           ├── privacy.py
│           └── utility.py
│
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_generators_datasets.py
│   ├── test_generator_sensitivity.py
│   ├── test_metrics.py
│   ├── test_pipeline.py
│   └── test_reference_validation.py
│
├── BENCHMARKS.md
├── README.md
├── pyproject.toml
├── requirements.txt
└── requirements.lock
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Autocurator-Synthetic-Data-Benchmark.git
cd Autocurator-Synthetic-Data-Benchmark
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

### 3. Install the Package

The project uses a `src/` layout, so install it (editable) to put `autocurator` on your path:

```bash
pip install -e .
```

For the development tools used by CI (Ruff, Black, Mypy, pytest, prdc):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

Run the bundled example from a config file:

```bash
autocurator --config configs/example.yaml
```

Or pass options explicitly:

```bash
autocurator \
  --real data/real.csv \
  --synthetic data/synthetic.csv \
  --target target \
  --task classification \
  --out_dir outputs/runs/example_run \
  --report reports/example_run.html
```

Then open `reports/example_run.html` in a browser. `python -m autocurator.cli ...` works too once the package is installed.

---

## Running a Benchmark

The CLI aligns schemas, builds one shared feature space, computes all four metric families, renders figures, and writes both a JSON summary and an HTML report.

**Optional flags**

<div align="center">

| Flag | Purpose |
|---|---|
| `--k N` | Neighborhood size for the PRDC coverage metrics (default 5) |
| `--utility_model {linear,rf}` | Utility estimator: linear baseline (default) or random forest |
| `--holdout path.csv` | Disjoint real set enabling the holdout membership-inference attack |
| `--config path.yaml` | YAML defaults for any option; CLI arguments override the file |

</div>

Generated outputs include:

```text
outputs/runs/example_run/metrics.json
outputs/runs/example_run/plots/pca.png
outputs/runs/example_run/plots/distributions.png
outputs/runs/example_run/plots/correlations.png
reports/example_run.html
```

---

## Metrics Explained

### Fidelity

How closely synthetic marginals and correlations match the real data.

- **Jensen–Shannon Distance (JSD):** Distributional difference per feature.
- **Kolmogorov–Smirnov (KS):** Maximum CDF gap per feature.
- **Wasserstein Distance:** Earth-mover distance per feature.
- **Correlation Distance:** Mean absolute difference between the real and synthetic correlation matrices.

### Coverage

PRDC metrics (Naeem et al., 2020) using per-point k-nearest-neighbor hyperspheres.

- **Precision:** Fraction of synthetic points inside a real hypersphere.
- **Recall:** Fraction of real points inside a synthetic hypersphere.
- **Density:** Average number of real hyperspheres each synthetic point falls into, normalized by `k` (robust to outliers).
- **Coverage:** Fraction of real points with at least one synthetic point nearby.

### Privacy

- **Nearest Neighbor Distance:** Larger synthetic-to-real distances indicate less overlap.
- **Membership Inference (MIA):** Distance-based re-identification proxy. An AUC near **0.5** means real and synthetic are indistinguishable by nearest-neighbor distance (low risk). Values far from 0.5 mean they are distinguishable; **below 0.5** indicates synthetic points sit unusually close to real ones (possible memorization). Real points are compared against their nearest *other* real point so self-matches don't make the attack trivially perfect.
- **Holdout MIA (optional):** With `--holdout` (a disjoint real set the generator never saw), `mia_auc_holdout` runs the stronger distance-to-closest-record attack: members vs. non-members scored by proximity to synthetic. **0.5 = no leakage**; above 0.5 means training records are closer to synthetic than holdout records.

### Utility

- **TSTR (Train Synthetic, Test Real)** and **TRTS (Train Real, Test Synthetic)**, reported as AUC for classification or R² for regression, using a linear baseline or a random forest.

---

## Metric Output

A machine-readable summary is written to `metrics.json`:

```json
{
  "fidelity": {"per_feature_mean_jsd": 0.475, "per_feature_mean_ks": 0.12, "correlation_distance": 0.006},
  "coverage": {"precision": 1.0, "recall": 1.0, "density": 1.16, "coverage": 1.0},
  "privacy": {"syn_to_real_mean_nnd": 0.29, "syn_to_real_min_nnd": 0.13, "mia_auc_distance": 0.04},
  "utility": {"TSTR_AUC": 1.0, "TRTS_AUC": 1.0}
}
```

> **Note on the bundled example.** The toy `synthetic.csv` is a lightly perturbed copy of `real.csv`. The corrected membership-inference metric (AUC ≈ 0.04) correctly flags this near-duplication as a privacy risk, and the perfect utility scores reflect that the two sets are almost identical. Treat it as a wiring example, not as a model of good synthetic data.

---

## Evaluation Summary

Example results from the bundled `example_run`:

<div align="center">

| Category | Metric | Value | Interpretation |
|---|---|---|---|
| **Fidelity** | Mean JSD | 0.48 | Moderate per-feature distributional gap |
|  | Correlation Distance | 0.006 | Correlation structure closely preserved |
| **Coverage** | Precision / Recall | 1.00 / 1.00 | Synthetic and real occupy the same region |
|  | Density / Coverage | 1.16 / 1.00 | Real manifold well covered |
| **Privacy** | Mean NN Distance | 0.29 | Average synthetic-to-real distance |
|  | MIA AUC | 0.04 | Far from 0.5 → distinguishable; below 0.5 signals memorization |
| **Utility** | TSTR / TRTS AUC | 1.00 / 1.00 | Models transfer between real and synthetic |

</div>

> The bundled example is deliberately near-duplicative, so its perfect fidelity and utility come *with* a privacy warning. That coupling is the point: fidelity and privacy trade off, and Autocurator reports both.

---

## Validation and Sensitivity

Autocurator's metrics are checked against independent references in CI (`tests/test_reference_validation.py`):

<div align="center">

| Metric | Reference | Agreement |
|---|---|---|
| PRDC | Published `prdc` package (Naeem et al., 2020) | within 1e-9 |
| Holdout MIA | Brute-force scipy + scikit-learn computation | within 1e-9 |

</div>

To confirm the metrics actually *respond* to synthetic-data quality, the repo ships reference generators of known quality (`autocurator.generators`) and real bundled datasets (`autocurator.datasets`: breast_cancer, wine, diabetes). Running every generator against every dataset produces `BENCHMARKS.md`:

```bash
python scripts/benchmark_generators.py > BENCHMARKS.md
```

Representative results on `breast_cancer` (members seen by the generator, holdout unseen):

<div align="center">

| Generator | Precision | Coverage | CorrDist | Holdout MIA | Utility (TSTR) |
|---|---|---|---|---|---|
| `resample` (high quality) | 1.00 | 0.99 | 0.033 | 0.83 | 1.00 |
| `independent` (no correlations) | 0.00 | 0.00 | 0.389 | 0.49 | 0.39 |
| `noise` (wrong distribution) | 0.00 | 0.00 | 0.391 | 0.53 | 0.77 |
| `leaky` (copies real rows) | 1.00 | 1.00 | 0.033 | 0.83 | 1.00 |

</div>

The high-quality generator scores near-perfect fidelity, coverage, and utility; `independent` and `noise` collapse them; and the row-copying `leaky` generator is flagged by the holdout MIA (≈0.83 vs ≈0.50 for the private generators). These directional relationships are asserted in `tests/test_generator_sensitivity.py`.

> The `resample` generator bootstraps real member rows with small jitter, so it too registers a high holdout MIA — high fidelity bought with reduced privacy. The benchmark reports this honestly rather than hiding it.

---

## Visual Reports

### Distribution overlap

<div align="center">

| PCA Projection | Feature Distributions |
|---|---|
| ![PCA projection](https://github.com/user-attachments/assets/528897b6-ab78-4794-98ec-c680f0674287) | ![Feature distributions](https://github.com/user-attachments/assets/b701b7c9-70e0-4223-b09b-e8412f95b21c) |
| **Analysis:** The PCA scatter projects real and synthetic points into two dimensions. Heavy overlap indicates the synthetic data occupies the same region as the real data; separated clusters would signal distributional drift. | **Analysis:** Per-feature histograms compare real and synthetic marginals directly, making it easy to see which features are well matched and which are off — complementing the numeric JSD / KS / Wasserstein scores. |

</div>

### Correlation structure

<div align="center">

![Correlation heatmaps](https://github.com/user-attachments/assets/8253c295-67d2-4db8-8a34-333984a024cb)

**Analysis:** Side-by-side correlation heatmaps show whether the synthetic data preserves cross-feature relationships. A generator can match every marginal yet still destroy correlations — the correlation distance quantifies this at a glance.

</div>

---

## Testing and CI

Run the unit tests locally:

```bash
pytest
```

Run the full quality gate:

```bash
ruff check src tests
black --check src tests
mypy src
pytest
```

The GitHub Actions workflow runs, across Python 3.10 / 3.11 / 3.12:

- dependency installation (editable, with dev tools)
- Ruff linting (E, F, I, B, SIM, UP)
- Black formatting check
- strict Mypy type checking
- the full pytest suite at ≥90% coverage
- reference validation of PRDC and the holdout MIA
- generator sensitivity checks on real datasets
- a separate job that installs from `requirements.lock` and reruns the suite

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
| `src/autocurator/cli.py` | Argument/config parsing, pipeline orchestration, report writing |
| `src/autocurator/config.py` | YAML config loading with DEFAULTS < config < CLI precedence |
| `src/autocurator/loaders.py` | CSV loading and numeric/categorical schema splitting |
| `src/autocurator/preprocess.py` | Shared scaler + one-hot feature space (fit and transform) |
| `src/autocurator/metrics/` | Fidelity, coverage (PRDC), privacy (NN + MIA), utility (TSTR/TRTS) |
| `src/autocurator/viz.py` | PCA, distribution, and correlation figures |
| `src/autocurator/report.py` | Jinja2 HTML rendering from a packaged template |
| `src/autocurator/datasets.py` | Real bundled datasets for benchmarking |
| `src/autocurator/generators.py` | Reference synthesizers of known quality |

</div>

Tooling is configured through `pyproject.toml` (Ruff, Black, Mypy, pytest, coverage).

---

## Reproducibility

- **Version floors** live in `pyproject.toml`; everyday installs resolve compatible versions across the supported Python range.
- **`requirements.lock`** pins an exact, verified dependency set (resolved on Python 3.12). A dedicated CI job installs from it and reruns the suite.
- **CI matrix** exercises Python 3.10, 3.11, and 3.12 so results do not silently drift across environments.

Regenerate the lockfile with:

```bash
pip install . && pip freeze --exclude-editable > requirements.lock
```

---

## Limitations

This project has important limitations:

- The in-repo example is a tiny toy dataset for wiring, not a realistic benchmark
- The no-holdout MIA is a distance **proxy**, not a formal attack; the holdout attack is stronger but optional
- Utility uses a linear or random-forest estimator, not the full model space a downstream task might use
- The metrics are validated for **correctness** against references, but not benchmarked against deep generative baselines (CTGAN, diffusion)
- Privacy scores are diagnostics, not differential-privacy guarantees
- PRDC and nearest-neighbor metrics scale with the product of dataset sizes and can be costly on very large inputs

The project is strongest as a portfolio demonstration of rigorous, honestly-evaluated synthetic-data benchmarking.

---

## Responsible Use

This repository is intended for:

- machine learning education and synthetic-data evaluation
- demonstrating honest, reference-validated metric design
- comparing synthetic-data generators during development
- exploring the fidelity–privacy–utility trade-off
- portfolio demonstration

It should not be used as-is for:

- certifying that a dataset is safe to release publicly
- legal or regulatory privacy compliance
- high-stakes decisions without domain-expert review
- replacing differential privacy or a formal security assessment

Any real deployment would require differential-privacy accounting, realistic attacker models, and human review.

---

## Future Improvements

Potential next improvements:

- Add deep generative baselines (CTGAN, TVAE, diffusion) to the benchmark
- Integrate differential-privacy accounting for formal guarantees
- Add multivariate/joint fidelity metrics beyond per-feature distances
- Support categorical-target utility beyond one-hot encodings
- Add a Streamlit dashboard for interactive report exploration
- Cross-validate scores against an established suite such as SDMetrics

---

## Tech Stack

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib
- seaborn
- Jinja2
- PyYAML
- pytest
- Ruff
- Black
- Mypy
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is released under the MIT License and is intended for educational, research, and portfolio purposes.

If you use or modify this project, please keep the responsible-use notes and limitations clear.
