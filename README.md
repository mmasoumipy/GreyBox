# Stress Risk Assessment System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
<!-- [![Last Updated](https://img.shields.io/badge/Last%20Updated-2025-blue.svg)]() -->

A comprehensive machine learning system for assessing occupational stress risk using ensemble learning, calibrated uncertainty quantification, and explainable AI. Includes a research-ready web interface for conducting two-group studies on AI transparency and user trust.

## 🎯 Features

### Machine Learning
- **5-Model Ensemble** - LightGBM models with different random seeds for robustness
- **Calibrated Uncertainty** - Isotonic regression for accurate probability estimates
- **Conformal Prediction** - 90% coverage prediction intervals
- **SHAP Explanations** - Feature importance and model interpretability
- **OOD Detection** - Isolation Forest for outlier/unusual input detection

### Web Interface
- **Dual-Mode Study Design** - Compare basic assessment vs. full explanations
- **Interactive Questionnaire** - 26-metric health and lifestyle assessment
- **Real-time Risk Assessment** - Instant predictions with uncertainty
- **Personalized Plans** - 7-day stress management recommendations
- **Automatic Logging** - Session data stored in Postgres when `DATABASE_URL` is set (falls back to JSON files)

### Research Framework
- **Two-Group Comparison** - Control (basic) vs. treatment (enhanced) UI
- **Hidden Research Console** - Unlock admin-only tools with an `ADMIN_KEY`
- **Research Dashboard** - Dataset metrics, feature importance, and log review
- **Statistical Analysis** - T-tests, Mann-Whitney U, Cohen's d effect sizes
- **Automated Exports** - Boxplots, distributions, CSVs, and study report saved to `study_outputs/`

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/grey_box.git
cd grey_box/GreyBox

# Install dependencies
pip install -r requirements.txt

# For macOS users (if you get libomp error)
brew install libomp
```

### Setup

```bash
# Create required directories
mkdir -p data artifacts study_logs study_outputs

# Place your dataset
cp your_stress_risk_data.csv data/stress_risk_data.csv
```

### Train Model

```bash
python train.py
```

Expected output:
```
============================================================
STRESS RISK MODEL TRAINING
============================================================
[1/8] Loading dataset...
   Loaded 200 records
...
✅ TRAINING COMPLETE
Model Performance:
  Validation AUC:  0.745
  Test AUC:        0.728
  Conformal Coverage: 90.2%
```

### Run Web Application (Flask UI)

```bash
python flask_app.py
# or
export FLASK_APP=flask_app.py && flask run
```

Open your browser to: `http://127.0.0.1:5000/`

- Assessment and Results are combined on one page; submit the form to see the gauge, uncertainty (G2), drivers, and plan.
- Survey is on a separate page.
- Participants auto-assign and auto-advance (odd → G1, even → G2) after survey submission.

## 🚄 Deploy to Railway

This repo includes a `Procfile` and `railway.json` to run the Flask app with Gunicorn.

### Steps

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub** and select the repo.
3. Set environment variables:
   - `DATABASE_URL` (optional, enables Postgres logging)
   - `ADMIN_KEY` (optional, enables research console)
4. Deploy. Railway provides `PORT` automatically.

### Notes

- Make sure `artifacts/` and `data/stress_risk_data.csv` are present in the repo or otherwise available at runtime.
- If you prefer to train during build, add a Railway build command like `python train.py` (only if you are ok with training on each build).

## 📊 Project Structure

```
GreyBox/
├── train.py                     # Model training pipeline
├── flask_app.py                 # Flask web application (participant flow)
├── streamlit_app.py             # Legacy Streamlit app (optional)
├── analyze_study.py             # Study analysis / visualization pipeline
├── README.md                    # This file
│
├── templates/                   # Flask HTML templates
│   ├── layout.html
│   ├── assessment.html
│   └── survey.html
│
├── data/
│   └── stress_risk_data.csv    # Input dataset (200 samples, 26 features)
│
├── artifacts/                   # Created after training
│   ├── models.pkl              # 5 trained LightGBM models
│   ├── calibrators.pkl         # Calibrated probability models
│   ├── scaler.pkl              # Feature normalizer
│   ├── ood.pkl                 # Out-of-distribution detector
│   ├── conformal.json          # Uncertainty parameters
│   └── feature_meta.json       # Feature metadata
│
├── study_logs/                  # Created by web app
│   ├── participant_001_G1_*.json
│   └── participant_002_G2_*.json
└── study_outputs/               # Generated by analyze_study.py
    ├── survey_responses.csv
    ├── statistical_results.csv
    ├── study_report.txt
    └── *.png visualizations
```

## 📋 Dataset Requirements

The system expects a CSV with 26 features organized into 6 categories:

### Features (26 total)

**Demographics (3)**
- `age` (numeric: 18-100)
- `gender` (categorical)
- `occupation` (categorical)

**Work Metrics (7)**
- `work_hours_per_week`, `job_satisfaction`, `workload_rating`
- `stress_event_count_last_week`, `breaks_per_workday`
- `commute_time_min`, `work_hours_per_week`

**Sleep Data (4)**
- `sleep_quality_rating`, `sleep_duration_hr`
- `time_in_bed_hr`, `motion_variability`

**Activity Data (4)**
- `physical_activity_frequency`, `steps_count`
- `outdoor_time_hr`, `motion_variability`

**Biometrics (3)**
- `avg_heart_rate`, `heart_rate_variability`, `resting_calories_burned`

**Lifestyle (5)**
- `coffee_intake_cups`, `alcohol_intake_per_week`
- `screen_time_hr`, `social_interactions_count`, `screen_unlocks_per_day`

**Target**
- `stress_risk_score` (numeric: 0-1) or `stress_level_category` (Low/Moderate/High)

## 🔬 Study Design

### Two-Group Comparison

**Group 1 (Control - Basic Assessment)**
- Simple risk gauge (0-100%)
- Risk category (Low/Moderate/High)
- No uncertainty information
- No feature explanations

**Group 2 (Treatment - Enhanced Assessment)**
- Risk gauge with uncertainty band
- Uncertainty magnitude visualization
- Aleatoric vs. epistemic uncertainty breakdown
- SHAP feature importance explanations
- Out-of-distribution warnings

### Data Collection

1. Participant selects group (randomly assigned)
2. Enters 26 health/lifestyle metrics (~5 minutes)
3. Views risk assessment (display varies by group)
4. Receives personalized 7-day stress management plan
5. Completes survey:
   - Trust in assessment (1-5)
   - Likelihood to follow recommendations (1-5)
   - Perceived usefulness (1-5)
   - Optional comments

Session automatically saved to Postgres when `DATABASE_URL` is set (falls back to `study_logs/[participant_id]_[group]_[timestamp].json`)
Study analyses (figures, CSVs, report) are written to `study_outputs/` and can be triggered via:

```bash
python analyze_study.py         # CLI version
# or from the Research dashboard -> "Run analyze_study pipeline"
```

## 📊 Model Architecture

### Ensemble Approach
- **Training**: 5 LightGBM models with different random seeds
- **Calibration**: Isotonic regression on validation set
- **Inference**: Average predictions across ensemble

### Uncertainty Quantification
- **Aleatoric Uncertainty**: Data randomness (can't reduce)
- **Epistemic Uncertainty**: Model uncertainty (reduces with more data)
- **Conformal Prediction**: Coverage guarantee for intervals

### Feature Importance
- **SHAP Values**: Local feature contributions per sample
- **Adaptive Explanations**: Top 5 most impactful features
- **Bidirectional**: Shows which factors increase/decrease risk

## 🔧 Configuration

### Model Hyperparameters (train_stress.py)

```python
# LightGBM parameters
n_estimators = 500          # Number of trees (more = better, slower)
learning_rate = 0.05        # Step size (lower = stable, slow)
num_leaves = 31             # Tree complexity
max_depth = 6               # Limit tree depth
subsample = 0.8             # Fraction of data per tree
colsample_bytree = 0.8      # Fraction of features per tree

# Uncertainty parameters
alpha = 0.10                # Confidence level (90% coverage)
early_stopping_rounds = 50  # Stop if no improvement

# OOD detection
contamination = 0.05        # Assume 5% outliers in training
```

### Custom Configuration

Edit the parameters in `train_stress.py` or `streamlit_app_stress.py`:

```python
# Change data path
DATA_PATH = "data/your_dataset.csv"

# Adjust confidence level
alpha = 0.05   # 95% coverage (tighter)
alpha = 0.20   # 80% coverage (wider)

# Add more ensemble models
seeds = [11, 22, 33, 44, 55, 66, 77]  # More models = better, slower
```

## 📈 Expected Performance

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Validation AUC | 0.70-0.80 | Depends on feature relationships |
| Test AUC | 0.65-0.75 | Real-world data is lower than synthetic |
| Conformal Coverage | ~90% | Maintained via conformal prediction |
| Interval Width | 0.25-0.35 | Uncertainty magnitude |
| Training Time | 2-5 min | On standard hardware |

**Note**: Performance varies with dataset quality and size. 200 samples is limited; performance improves with more data.

## 🔍 Analyzing Results

### Automatic Analysis

```bash
python analyze_study.py
```

Generates:
- `results_boxplots.png` - Outcome distributions by group
- `results_barplot.png` - Mean score comparisons
- `results_distributions.png` - Detailed histograms
- `statistical_results.csv` - T-tests, effect sizes
- `study_report.txt` - Comprehensive report

### Manual Analysis

```python
import json
import pandas as pd
from pathlib import Path

# Load all session data
sessions = []
for log_file in Path('study_logs').glob('*.json'):
    with open(log_file) as f:
        sessions.extend(json.load(f))

# Convert to DataFrame
df = pd.DataFrame(sessions)

# Compare groups
print(df.groupby('group')[['trust_score', 'follow_likelihood', 'usefulness']].mean())
```

### Statistical Analysis

```python
from scipy import stats

# T-test
g1_trust = df[df['group'] == 'G1']['trust_score']
g2_trust = df[df['group'] == 'G2']['trust_score']
t_stat, p_value = stats.ttest_ind(g1_trust, g2_trust)

# Cohen's d effect size
mean_diff = g2_trust.mean() - g1_trust.mean()
pooled_std = ((len(g1_trust)-1)*g1_trust.std()**2 + 
              (len(g2_trust)-1)*g2_trust.std()**2) / (len(g1_trust) + len(g2_trust) - 2)
cohens_d = mean_diff / pooled_std**0.5
```

## 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
shap>=0.40.0
streamlit>=1.0.0
plotly>=5.0.0
scipy>=1.7.0
joblib>=1.0.0
```

## 🔄 Model Components

### Training Pipeline (`train_stress.py`)

1. **Data Loading**: Load and validate CSV
2. **Preprocessing**: Handle missing values, normalize features
3. **Split**: Train (60%), Validation (20%), Test (20%)
4. **Training**: 5 LightGBM models with early stopping
5. **Calibration**: Isotonic regression on validation set
6. **Uncertainty**: Compute conformal prediction intervals
7. **Export**: Save models, metadata, statistics

### Web Application (`flask_app.py`)

1. **Auto Group Assignment**: odd IDs → G1 (basic), even IDs → G2 (enhanced)
2. **Assessment & Results**: Single page for inputs, prediction, uncertainty (G2), drivers, and plan generation
3. **Survey**: Separate page; submitting saves JSON log and advances to next participant
4. **Logging**: Session data saved to `study_logs/`

### Analysis Pipeline (`analyze_study.py`)

1. **Data Collection**: Load all session JSONs
2. **Aggregation**: Convert to DataFrame
3. **Comparison**: Group statistics and tests
4. **Visualization**: Generate charts and plots
5. **Report**: Create markdown summary

## 🎓 Key Concepts

### Aleatoric vs. Epistemic Uncertainty

**Aleatoric (Data Uncertainty)**
- Inherent randomness in the problem
- Cannot be reduced with more data
- Example: Natural variation in individual responses

**Epistemic (Model Uncertainty)**
- Model's lack of knowledge
- Reduces with more training data
- Example: Confidence in learned patterns

### SHAP Values

- **Positive (Red)**: Increases stress risk
- **Negative (Green)**: Decreases stress risk
- **Magnitude**: Strength of effect on prediction

### Conformal Prediction

Guarantees coverage regardless of model performance:
- Adapts to actual model difficulty
- Wider intervals = less confident
- ~90% coverage on test set


<!-- ## 📝 Citation

If you use this system in research, please cite:

```bibtex
@software{stress_risk_assessment_2024,
  title={Stress Risk Assessment System: ML-Based Occupational Stress Prediction with Uncertainty Quantification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/stress-risk-assessment}
}
``` -->

<!-- ## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details. -->

<!-- ## 🙏 Acknowledgments

- Built with [LightGBM](https://lightgbm.readthedocs.io/) for ensemble learning
- [SHAP](https://shap.readthedocs.io/) for model explanations
- [Streamlit](https://streamlit.io/) for web interface
- [Plotly](https://plotly.com/) for visualizations -->


<!-- ## 🚀 Getting Started Guide

### For First-Time Users

1. **Clone the repo**: `git clone https://github.com/yourusername/stress-risk-assessment.git`
2. **Install**: `pip install -r requirements.txt`
3. **Train**: `python train_stress.py`
4. **Run**: `streamlit run streamlit_app_stress.py`
5. **Test**: Open `http://localhost:8501`

### For Researchers

1. Collect baseline data with Group 1 (control)
2. Compare with Group 2 (treatment with explanations)
3. Run `python analyze_study.py` for statistical tests
4. Review generated reports and visualizations

### For Developers

1. Review [API Reference](docs/API_REFERENCE.md)
2. Check [Architecture](docs/ARCHITECTURE.md)
3. See [Contributing Guide](CONTRIBUTING.md) -->

---

<!-- **Made with ❤️ for stress research and AI transparency** -->

**Version**: 2.0 |  **Last Updated**: 2026
