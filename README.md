<div align="center">

# 🤖 AutoML Platform

**A free, production-grade AutoML platform deployable on Streamlit Cloud with AI-powered insights, SHAP explainability, and Mistral chat.**

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://automl-platform.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FLAML](https://img.shields.io/badge/AutoML-FLAML-purple)](https://github.com/microsoft/FLAML)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io)
[![Mistral AI](https://img.shields.io/badge/AI-Mistral-orange)](https://mistral.ai)

[🚀 Live Demo](https://automl-platform.streamlit.app/) · [🐛 Report Bug](https://github.com/Pranish-Sapkota/automl-platform/issues) · [💡 Request Feature](https://github.com/Pranish-Sapkota/automl-platform/issues)

<img width="900" src="https://www.automationml.org/wp-content/uploads/2021/02/AutomationML-Logo.svg" alt="Logo"/>

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Getting Started Locally](#-getting-started-locally)
- [Deploying to Streamlit Cloud](#️-deploying-to-streamlit-cloud)
- [Usage Guide](#-usage-guide)
- [Supported Algorithms](#-supported-algorithms)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

**AutoML Platform** is a fully open-source, end-to-end machine learning platform that lets anyone — from data scientists to business analysts — upload a dataset and automatically:

- Detect the ML problem type (classification or regression)
- Profile and analyze data quality
- Preprocess and engineer features automatically
- Train and compare multiple state-of-the-art ML models
- Explain model predictions with SHAP
- Chat with their data using Mistral AI
- Export trained models for production use

All of this runs **completely free** on [Streamlit Cloud](https://streamlit.io/cloud) — no Docker, no Kubernetes, no paid databases, and no vendor lock-in.

---

## 🚀 Live Demo

> **Try it now — no sign-up required:**
>
> ### 👉 [https://automl-platform.streamlit.app/](https://automl-platform.streamlit.app/)

Upload any CSV/Excel dataset and get a full ML pipeline in minutes.  
Sample datasets (Iris, Titanic, Wine Quality) are built-in if you want to explore quickly.

---

## ✨ Features

### 📊 Data Profiling & EDA
- **Dataset Schema** — column types, null counts, unique values, sample data
- **Missing Value Analysis** — heatmaps, per-column counts and percentages
- **Duplicate Detection** — identify and report duplicate rows
- **Outlier Analysis** — IQR and Z-score based detection with box plots
- **Correlation Matrix** — interactive Pearson correlation heatmap
- **Distribution Explorer** — histograms and box plots for any column
- **Target Analysis** — class distribution, imbalance detection, regression stats
- **AI Recommendations** — auto-generated data quality improvement tips

### ⚙️ Auto Preprocessing
- **Smart Imputation** — median/mean for numeric, mode for categorical
- **Categorical Encoding** — One-Hot Encoding or Label Encoding (auto-selected)
- **Feature Scaling** — RobustScaler, StandardScaler, MinMaxScaler
- **Outlier Treatment** — IQR-based clipping at configurable thresholds
- **ID Column Removal** — heuristic detection and exclusion of non-informative columns
- **High-Cardinality Handling** — configurable cardinality threshold

### 🧬 Auto Feature Engineering
- **Date Features** — year, month, day, day-of-week, quarter, is_weekend
- **Interaction Features** — products and ratios between numeric column pairs
- **Row Aggregations** — mean, std, min, max, range, sum across numeric features

### 🤖 AutoML Engine
- **FLAML** — Microsoft's fast, lightweight AutoML framework as primary engine
- **7 algorithms** — Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Logistic Regression, Decision Tree
- **Full leaderboard** — all models ranked with complete metrics
- **Configurable** — time budget, CV folds, test split, random seed

### 📈 Model Explainability (SHAP)
- **Global Importance** — mean |SHAP| bar chart across all features
- **Waterfall Plot** — how each feature pushes a single prediction
- **Bubble Chart** — importance visualised by size and color
- **Local Explanations** — per-sample feature contribution breakdown
- **Auto Explainer Selection** — Tree → Linear → Kernel fallback chain

### 💬 AI Chat Assistant (Mistral AI)
- **Free-form Q&A** — ask anything about your dataset or model
- **EDA Summary** — AI-generated natural-language dataset overview
- **Model Explanation** — plain-English performance analysis
- **Business Insights** — actionable recommendations from your data
- **Streaming responses** — real-time token streaming
- **Full HTTP fallback** — works even without the mistralai SDK installed

### 📦 Model Export
- **Download .pkl** — model file ready for production inference
- **Full Pipeline Bundle** — preprocessor + feature engineer + model in one file
- **Batch Predictions** — upload new data and download predictions as CSV
- **Metrics Report** — full leaderboard exported as CSV or JSON
- **Experiment Summary** — complete run metadata as JSON

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.40+ | UI framework |
| **Charts** | Plotly 5.x | Interactive visualizations |
| **Tables** | StreamLit AgGrid | Sortable/filterable data grids |
| **AutoML** | FLAML 2.x | Hyperparameter search |
| **ML** | scikit-learn 1.5+ | Core algorithms |
| **Boosting** | XGBoost · LightGBM · CatBoost | Gradient boosting models |
| **Explainability** | SHAP 0.46+ | Feature importance |
| **AI** | Mistral AI API | Chat assistant |
| **Data** | Pandas 2.x · NumPy | Data processing |
| **Config** | Pydantic v2 | Settings validation |
| **Storage** | SQLite (built-in) | Experiment metadata |
| **Testing** | pytest | Unit & integration tests |

---

## 🏗️ Architecture

```
automl_platform/
├── app.py                              # Main entry point — navigation & global CSS
├── requirements.txt                    # All Python dependencies
├── LICENSE                             # MIT License
├── pytest.ini                          # Test configuration
│
├── .streamlit/
│   ├── config.toml                     # Theme (dark) + server settings
│   └── secrets.toml.example            # API key template
│
└── src/
    ├── utils/
    │   ├── config.py                   # Pydantic v2 app + ML + Mistral config
    │   ├── logger.py                   # Structured logging (cached per name)
    │   └── helpers.py                  # Utilities: hashing, type inference, ID detection
    │
    ├── services/
    │   └── data_service.py             # Dataset loading (CSV/Excel/Parquet/JSON) + DataProfile
    │
    ├── preprocessing/
    │   └── pipeline.py                 # AutoPreprocessor with fit() / transform() semantics
    │
    ├── feature_engineering/
    │   └── engineer.py                 # Date features · Interactions · Row aggregations
    │
    ├── automl/
    │   └── engine.py                   # AutoMLEngine: model zoo + FLAML + leaderboard builder
    │
    ├── explainability/
    │   └── shap_explainer.py           # SHAPExplainer: auto-selects Tree/Linear/Kernel
    │
    ├── ai/
    │   └── mistral_client.py           # MistralClient: v1 SDK / v0 SDK / pure HTTP fallback
    │
    ├── storage/
    │   └── database.py                 # SQLite: experiments, models, chat history
    │
    ├── visualization/
    │   └── charts.py                   # 14 Plotly chart builders (EDA + model + SHAP)
    │
    ├── components/
    │   └── ui.py                       # Reusable Streamlit components (cards, badges, headers)
    │
    └── pages/
        ├── home.py                     # Dashboard with pipeline overview
        ├── upload.py                   # File upload + sample datasets + target selector
        ├── profiling.py                # Full EDA (7 tabs)
        ├── cleaning.py                 # Preprocessing config + feature engineering
        ├── training.py                 # AutoML training with live progress bar
        ├── leaderboard.py              # Model comparison (6 tabs: table, charts, ROC, CM)
        ├── explainability.py           # SHAP explanations (5 tabs)
        ├── chat.py                     # Mistral AI chat with streaming
        ├── export.py                   # Model download + batch predictions
        └── settings.py                 # API keys + ML config + experiment history
```

---

## 💻 Getting Started Locally

### Prerequisites

- Python **3.10 or higher**
- `pip` (comes with Python)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Pranish-Sapkota/automl-platform.git
cd automl-platform
```

### Step 2 — Create a Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On macOS / Linux:
source .venv/bin/activate

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ First install takes 2–5 minutes due to ML libraries (XGBoost, LightGBM, CatBoost, SHAP).

### Step 4 — Configure Your Mistral API Key (Optional)

The AI Chat Assistant requires a free Mistral API key.  
Get one at [console.mistral.ai](https://console.mistral.ai) — no credit card needed.

```bash
# Copy the example secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Open and add your key
nano .streamlit/secrets.toml   # or use any editor
```

```toml
# .streamlit/secrets.toml
MISTRAL_API_KEY = "your-mistral-api-key-here"
```

Alternatively, set it as an environment variable:

```bash
# macOS / Linux
export MISTRAL_API_KEY="your-mistral-api-key-here"

# Windows (Command Prompt)
set MISTRAL_API_KEY=your-mistral-api-key-here

# Windows (PowerShell)
$env:MISTRAL_API_KEY = "your-mistral-api-key-here"
```

> 💡 You can also enter the API key directly in the app's **Settings** page after launching — no restart needed.

### Step 5 — Run the Application

```bash
streamlit run app.py
```

The app will open automatically at **[http://localhost:8501](http://localhost:8501)**.

### Step 6 — Run the Tests (Optional)

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test module
pytest tests/test_preprocessing.py -v
pytest tests/test_automl_engine.py -v

# Run with coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in your browser
```


---

## 📋 Usage Guide

### Workflow Overview

```
Upload Dataset → Data Profiling → Data Cleaning → AutoML Training → Leaderboard → Explainability → Chat → Export
```

### Step-by-Step

| Step | Page | What to Do |
|---|---|---|
| 1 | **Upload Dataset** | Upload CSV/Excel/Parquet/JSON or load a sample dataset. Select your target column. |
| 2 | **Data Profiling** | Review the automated EDA: missing values, distributions, outliers, correlations. |
| 3 | **Data Cleaning** | Configure scaling, encoding, imputation, and feature engineering options. Click **Apply**. |
| 4 | **AutoML Training** | Set time budget and click **Start AutoML Training**. Watch the live progress bar. |
| 5 | **Model Leaderboard** | Compare all models. View ROC curves, confusion matrices, and metric charts. |
| 6 | **Explainability** | Select a model and explore SHAP global importance, waterfall plots, and local explanations. |
| 7 | **AI Chat Assistant** | Ask questions in natural language. Try the suggested prompts or generate an EDA summary. |
| 8 | **Model Export** | Download your trained model as `.pkl`, export predictions, or download the metrics report. |

### Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | Auto-detects separator |
| Excel | `.xlsx`, `.xls` | First sheet loaded |
| Parquet | `.parquet` | Fastest for large datasets |
| JSON | `.json` | Records or columns orientation |

**Maximum file size:** 200 MB · **Maximum rows:** 500,000

---

## 🧠 Supported Algorithms

| Algorithm | Classification | Regression | Via |
|---|---|---|---|
| Random Forest | ✅ | ✅ | scikit-learn |
| Extra Trees | ✅ | ✅ | scikit-learn |
| Decision Tree | ✅ | ✅ | scikit-learn |
| Logistic Regression | ✅ | — | scikit-learn |
| Ridge Regression | — | ✅ | scikit-learn |
| XGBoost | ✅ | ✅ | xgboost |
| LightGBM | ✅ | ✅ | lightgbm |
| CatBoost | ✅ | ✅ | catboost |
| FLAML AutoML | ✅ | ✅ | flaml (best of all above) |

### Metrics Reported

| Task | Metrics |
|---|---|
| **Binary Classification** | Accuracy · Precision · Recall · F1 · ROC-AUC |
| **Multi-Class Classification** | Accuracy · Weighted Precision · Weighted Recall · Weighted F1 · ROC-AUC |
| **Regression** | RMSE · MAE · R² |

---

## 🔮 Future Enhancements

We have an exciting roadmap planned. Contributions toward any of these are very welcome!

### 🎨 UI / UX Improvements
- [ ] **React-Based Component Architecture** — stateful UI interactions and high-performance rendering switching

- [ ] **Real-time training progress charts** — live loss/metric curves during training
- [ ] **Mobile-responsive layout** — optimised for tablet and phone screens
- [ ] **Customisable dashboard** — drag-and-drop widget arrangement on the Home page
- [ ] **Animated onboarding tour** — step-by-step walkthrough for new users
- [ ] **Split-screen comparison** — view two models side-by-side


### 🤖 ML & AutoML
- [ ] **Time series forecasting** — Prophet, ARIMA, and LSTM support
- [ ] **NLP tasks** — text classification and sentiment analysis (TF-IDF + transformers)
- [ ] **Image classification** — drag-and-drop image dataset support with CNNs
- [ ] **Ensemble builder** — blend top models with configurable weights
- [ ] **Bayesian hyperparameter optimisation** — Optuna integration
- [ ] **Neural network support** — Keras/PyTorch MLPs via FLAML
- [ ] **Semi-supervised learning** — leverage unlabelled data
- [ ] **Anomaly detection** — Isolation Forest, One-Class SVM
- [ ] **Cross-validation visualisation** — per-fold metric charts
- [ ] **Feature selection** — RFE, mutual information, Boruta

### 📊 Data & Analysis
- [ ] **Advanced imputation** — KNN imputer, MICE, iterative imputer
- [ ] **SMOTE / resampling** — built-in class imbalance handling
- [ ] **Data versioning** — track dataset changes across experiments
- [ ] **SQL query interface** — query uploaded datasets with SQL
- [ ] **Automatic report generation** — downloadable PDF/HTML EDA report
- [ ] **Multi-dataset comparison** — profile two datasets side-by-side
- [ ] **Column-level lineage** — track feature transformations end-to-end

### 💬 AI Assistant
- [ ] **GPT / Claude / Gemini support** — pluggable LLM backends
- [ ] **Auto-generated code** — Python code snippets for each preprocessing step
- [ ] **Chart generation from chat** — "show me a bar chart of column X" → rendered chart
- [ ] **Persistent conversation history** — chat history saved across sessions
- [ ] **Voice input** — speak your questions to the assistant
- [ ] **Multi-language support** — responses in user's preferred language

### 🔧 Platform & Infrastructure
- [ ] **User authentication** — Streamlit auth or OAuth (GitHub, Google)
- [ ] **Multi-user experiment sharing** — share experiment links with colleagues
- [ ] **REST API** — expose predictions as an API endpoint (FastAPI integration)
- [ ] **Docker Compose deployment** — one-command self-hosted setup
- [ ] **MLflow integration** — experiment tracking and model registry
- [ ] **Dataset connectors** — load from Google Sheets, S3, BigQuery, Snowflake
- [ ] **Scheduled retraining** — auto-retrain on new data via GitHub Actions
- [ ] **Model monitoring** — drift detection and performance alerts
- [ ] **A/B testing** — compare model performance on production traffic

### 🧪 Testing & Quality
- [ ] **End-to-end Selenium tests** — full UI workflow automation
- [ ] **Performance benchmarks** — track training speed across versions
- [ ] **Property-based testing** — Hypothesis integration for edge case discovery
- [ ] **CI/CD pipeline** — GitHub Actions for automated testing and deployment

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. **Any contribution you make is greatly appreciated.**

### Ways to Contribute

- 🐛 **Report bugs** — open an issue with steps to reproduce
- 💡 **Suggest features** — share your ideas via GitHub Issues
- 🔧 **Submit pull requests** — fix bugs, add features, improve docs
- 📖 **Improve documentation** — fix typos, add examples, clarify explanations
- ⭐ **Star the repo** — help others discover the project

### Development Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/Pranish-Sapkota/automl-platform.git
cd automl-platform

# 2. Create a virtual environment and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Create a new branch for your feature or fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix

# 4. Make your changes, then run the tests
pytest -v

# 5. Check syntax and style
python -m py_compile src/**/*.py

# 6. Commit your changes with a clear message
git add .
git commit -m "feat: add XYZ feature"
# or
git commit -m "fix: resolve ABC bug"

# 7. Push to your fork and open a Pull Request
git push origin feature/your-feature-name
```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use for |
|---|---|
| `feat:` | A new feature |
| `fix:` | A bug fix |
| `docs:` | Documentation changes |
| `style:` | Formatting, no logic change |
| `refactor:` | Code restructuring |
| `test:` | Adding or fixing tests |
| `chore:` | Build process or tooling |

### Pull Request Guidelines

1. **One PR per feature/fix** — keep changes focused and reviewable
2. **Write or update tests** — all new code should have test coverage
3. **Update the README** if you add or change any user-facing functionality
4. **Describe your PR clearly** — what does it do, why is it needed, how was it tested
5. **Link related issues** — use `Closes #123` in the PR description

### Code Style

- Follow **PEP 8** Python style guide
- Use **type hints** throughout (`from __future__ import annotations`)
- Write **docstrings** for all public functions and classes
- Keep functions **small and focused** — single responsibility principle
- Use **Pydantic v2** for all data models and configuration

### Issue Templates

When opening an issue, please include:

**For bugs:**
- Steps to reproduce
- Expected vs actual behaviour
- Python version and OS
- Relevant error messages / screenshots

**For feature requests:**
- What problem does it solve?
- Describe the proposed solution
- Any alternatives considered

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

```
MIT License — Copyright (c) 2025 AutoML Platform Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgements

This project is built on the shoulders of giants. Thank you to:

- **[FLAML](https://github.com/microsoft/FLAML)** (Microsoft) — for the blazing-fast AutoML engine
- **[SHAP](https://github.com/slundberg/shap)** — for making model explainability accessible
- **[Streamlit](https://streamlit.io)** — for making data app development a joy
- **[Mistral AI](https://mistral.ai)** — for the powerful and free AI models
- **[scikit-learn](https://scikit-learn.org)** — the backbone of ML in Python
- **[XGBoost](https://xgboost.ai)**, **[LightGBM](https://lightgbm.readthedocs.io)**, **[CatBoost](https://catboost.ai)** — world-class gradient boosting
- **[Plotly](https://plotly.com)** — for beautiful interactive charts

---

<div align="center">

**Built with ❤️ by Pranish Sapkota using Python, Streamlit, FLAML, SHAP, and Mistral AI**

[⬆️ Back to Top](#-automl-platform)

</div>
