# 🤖 AutoML Platform

> Production-grade AutoML platform deployable **completely free** on Streamlit Cloud.  
> No Docker · No Kubernetes · No paid databases · No vendor lock-in.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

| Category | Capabilities |
|---|---|
| **Data Analysis** | Missing values · Duplicates · Outliers · Correlations · Distributions · Class imbalance |
| **Preprocessing** | Imputation · Label/OHE encoding · RobustScaler · Outlier clipping · ID column removal |
| **Feature Engineering** | Date features · Interaction terms · Row aggregations |
| **AutoML** | FLAML · Random Forest · Extra Trees · XGBoost · LightGBM · CatBoost · Logistic Regression |
| **Explainability** | SHAP global importance · Waterfall plots · Local explanations · Bubble charts |
| **AI Assistant** | Mistral AI chat · EDA summaries · Business insights · Model explanations |
| **Export** | Model download · Full pipeline · Batch predictions · JSON/CSV reports |

---

## 🏗️ Architecture

```
automl_platform/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # All dependencies
├── pytest.ini                      # Test configuration
├── .streamlit/
│   ├── config.toml                 # Theme + server config
│   └── secrets.toml.example        # API key template
├── src/
│   ├── utils/
│   │   ├── config.py               # Pydantic v2 configuration
│   │   ├── logger.py               # Structured logging
│   │   └── helpers.py              # Shared utilities
│   ├── services/
│   │   └── data_service.py         # Dataset loading + profiling
│   ├── preprocessing/
│   │   └── pipeline.py             # AutoPreprocessor (fit/transform)
│   ├── feature_engineering/
│   │   └── engineer.py             # Date, interactions, aggregations
│   ├── automl/
│   │   └── engine.py               # FLAML + model zoo + leaderboard
│   ├── explainability/
│   │   └── shap_explainer.py       # SHAP Tree/Linear/Kernel explainer
│   ├── ai/
│   │   └── mistral_client.py       # Mistral API chat assistant
│   ├── storage/
│   │   └── database.py             # SQLite experiment storage
│   ├── visualization/
│   │   └── charts.py               # All Plotly chart builders
│   ├── components/
│   │   └── ui.py                   # Reusable Streamlit UI components
│   └── pages/
│       ├── home.py                 # Dashboard
│       ├── upload.py               # Dataset upload
│       ├── profiling.py            # EDA + profiling
│       ├── cleaning.py             # Preprocessing config
│       ├── training.py             # AutoML training
│       ├── leaderboard.py          # Model comparison
│       ├── explainability.py       # SHAP explanations
│       ├── chat.py                 # Mistral AI chat
│       ├── export.py               # Model export
│       └── settings.py             # API keys + config
├── data/                           # SQLite DB + uploaded files
├── models/                         # Saved model artifacts
├── reports/                        # Generated reports
└── tests/                          # Pytest test suite
    ├── conftest.py
    ├── test_data_service.py
    ├── test_preprocessing.py
    ├── test_feature_engineering.py
    ├── test_automl_engine.py
    ├── test_storage.py
    ├── test_visualization.py
    └── test_helpers.py
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/automl-platform.git
cd automl-platform
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate.bat     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Mistral API key (optional)

```bash
# Copy example secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Edit and add your key from https://console.mistral.ai
nano .streamlit/secrets.toml
```

Or set as environment variable:
```bash
export MISTRAL_API_KEY="your-key-here"
```

### 5. Run the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/automl-platform.git
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub account
4. Select your repository
5. Set **Main file path** to: `app.py`
6. Click **Deploy**

### Step 3 — Add Secrets (API key)

In Streamlit Cloud dashboard:
1. Click your app → **"⋮" menu** → **"Settings"**
2. Go to **"Secrets"** tab
3. Add:
```toml
MISTRAL_API_KEY = "your-mistral-api-key"
```

That's it — your app is live! 🎉

---

## 🔑 Getting a Free Mistral API Key

1. Go to [console.mistral.ai](https://console.mistral.ai)
2. Create a free account
3. Navigate to **API Keys**
4. Click **"Create new key"**
5. Copy and add to Streamlit Secrets or the Settings page in the app

Free tier includes `open-mistral-7b` model. Paid tiers unlock `mistral-large-latest`.

---

## 🧪 Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/test_automl_engine.py -v
pytest tests/test_preprocessing.py -v
```

---

## 📋 Supported File Formats

| Format | Extension |
|---|---|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| Parquet | `.parquet` |
| JSON | `.json` |

Maximum file size: **200 MB**

---

## 🎯 Supported ML Tasks

| Task | Description | Primary Metric |
|---|---|---|
| Binary Classification | 2-class prediction | F1 / ROC-AUC |
| Multi-Class Classification | 3+ class prediction | Weighted F1 |
| Regression | Continuous value prediction | R² / RMSE |

---

## 🧠 Model Zoo

| Model | Binary | Multi-Class | Regression |
|---|---|---|---|
| Random Forest | ✅ | ✅ | ✅ |
| Extra Trees | ✅ | ✅ | ✅ |
| XGBoost | ✅ | ✅ | ✅ |
| LightGBM | ✅ | ✅ | ✅ |
| CatBoost | ✅ | ✅ | ✅ |
| Logistic Regression | ✅ | ✅ | — |
| Decision Tree | ✅ | ✅ | ✅ |
| FLAML AutoML | ✅ | ✅ | ✅ |

---

## 🔒 Privacy & Security

- **No data leaves your machine** during local use
- On Streamlit Cloud: data is processed ephemerally in the container
- API keys are stored only in session state (browser memory) or Streamlit Secrets
- SQLite database stores only metadata (no raw data)
- Models are saved to container-local storage (resets on redeploy)

---

## ⚡ Performance Notes

- **Max rows:** 500,000 (configurable)
- **FLAML time budget:** 30–600 seconds (configurable)
- **SHAP computation:** auto-sampled to 200 rows for speed
- **Memory:** Streamlit Cloud Free provides ~1 GB RAM

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.40+ |
| Charts | Plotly 5.x |
| ML Engine | FLAML 2.x + scikit-learn |
| Boosting | XGBoost · LightGBM · CatBoost |
| Explainability | SHAP 0.46+ |
| AI Assistant | Mistral AI API |
| Data | Pandas 2.x + NumPy |
| Config | Pydantic v2 |
| Storage | SQLite (built-in) |
| Testing | pytest |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 🙏 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

*Built with ❤️ using Streamlit, FLAML, SHAP, and Mistral AI.*
