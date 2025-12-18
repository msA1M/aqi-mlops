---
title: AQI
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
sdk_version: "0.0.1"
app_file: app.py
pinned: false
---

## Environmental Intelligence Dashboard – AQI & Weather

This Space runs a Docker container that serves:

- A **FastAPI** backend exposing AQI and weather prediction endpoints.
- A **Streamlit** frontend providing an interactive dashboard for AQI prediction, weather forecasts, and data drift monitoring.

The container is built from the `Dockerfile` in this repository and uses pre-trained models stored in the bundled `mlruns` directory via MLflow.

🌍 Environmental Intelligence Platform — End-to-End MLOps Project
An end-to-end Machine Learning Operations (MLOps) system for Air Quality Index (AQI) and Weather Forecasting, designed with realistic industry-grade architecture.
This project demonstrates the full ML lifecycle: live data ingestion, feature engineering, model training, experiment tracking, model registry, explainability, drift monitoring, orchestration, CI/CD, containerization, and interactive dashboards.

🎯 Project Objective
To build a production-style ML system that:
- Predicts AQI for a target city (Brasilia)
- Forecasts weather variables (temperature, humidity, wind speed, pressure) for multiple cities (Brasilia, London, Karachi)
- Monitors live data for data drift
- Produces health-based alerts
- Automates pipelines using Prefect
- Ensures reliability via testing, CI/CD, and Docker

🧠 System Capabilities
✅ Engineering & Machine Learning
- Time-series AQI data ingestion
- Weather data ingestion (live API)
- Feature store with:
  - Lag features (1, 3, 6, 12 hours)
  - Rolling statistics (mean, std, min, max)
  - Time-based features (hour, day, month, cyclical encoding)
  - Interaction features
  - Exponential moving averages
- Multiple ML models:
  - Ridge Regression (with hyperparameter tuning)
  - Random Forest
  - XGBoost
  - Gradient Boosting
- MLflow:
  - Experiment tracking
  - Model comparison
  - Model registry
  - Best-model selection & registration
- SHAP explainability (AQI model)
- AQI category classification & alert logic

🌦️ Weather Forecasting
Predicts:
- Temperature
- Humidity
- Wind Speed
- Pressure
Supported cities:
- Brasilia
- London
- Karachi
Multi-target regression (one model per variable)
Models registered independently in MLflow

🌐 Live Data Integration
- Weather: Open-Meteo API (stable, no API key)
- AQI: Historical + monitored live data
- Live data is used for:
  - Monitoring
  - Drift detection
- ❌ Not blindly used for training

📈 Monitoring & Reliability
- Data drift detection using PSI (Population Stability Index)
- Visual drift plots in Streamlit
- Drift levels:
  - PSI < 0.10 → No drift
  - 0.10–0.20 → Moderate drift
  - > 0.20 → Significant drift
- Automated ML tests:
  - Data validation
  - Model performance regression
  - Inference safety

⚙️ Systems & Ops
| Component | Tool |
|-----------|------|
| API | FastAPI |
| Dashboard | Streamlit |
| Orchestration | Prefect |
| Experiment Tracking | MLflow |
| Drift Detection | PSI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Railway |

🏗️ Architecture Overview
```
            ┌──────────────┐
            │     API      │
            └──────┬───────┘
                   │
           Live Ingestion Pipeline
                   │
        ┌──────────▼──────────┐
        │ Drift Detection &   │
        │ Monitoring          │
        └──────────┬──────────┘
                   │
 Feature Store ──► MLflow ──► Model Registry
                   │
          ┌────────▼────────┐
          │   FastAPI API   │
          └────────┬────────┘
                   │
            ┌──────▼──────┐
            │  Streamlit  │
            │     UI      │
            └─────────────┘
```

🔗 Important Links

### 📊 MLflow UI
**Local Access:** http://localhost:5000

To start MLflow UI locally:
```bash
mlflow ui
```

**View:**
- Experiment runs and metrics
- Model registry
- Model versions
- Compare model performance
- Download models

### 🔄 Prefect Flow Dashboard
**Local Access:** http://localhost:4200

To start Prefect server:
```bash
prefect server start
```

**View:**
- Pipeline runs
- Task execution status
- Flow schedules
- Run history and logs

### 🚂 Railway Deployment
**Deployment URL:** Check your Railway project dashboard

**GitHub Repository:** https://github.com/msA1M/aqi-mlops

**Railway Auto-Deploy:** Railway automatically rebuilds and deploys on every push to `main` branch.

**Railway Configuration:**
- Uses `Dockerfile.railway` for building
- Configured via `railway.json`
- Auto-deploys from GitHub

---

## 🚀 Running Commands Individually

### 📦 Setup & Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 📥 Data Ingestion

#### 1. Ingest Historical AQI Data
```bash
python pipelines/ingest_historical_data.py
```
**What it does:**
- Loads raw AQI data from `data/raw/aqi_historical.csv`
- Cleans and processes data
- Saves to `data/processed/aqi_clean.csv`

#### 2. Ingest Weather Data
```bash
python pipelines/ingest_weather_data.py
```
**What it does:**
- Fetches weather data from Open-Meteo API
- Supports cities: Brasilia, London, Karachi
- Saves timestamped CSV to `data/weather/weather_multi_city_*.csv`

### 🔧 Feature Engineering

#### 3. Build AQI Features
```bash
python feature_store/build_features.py
```
**What it does:**
- Creates time-series features (lags, rolling stats, EMA)
- Adds time-based features (hour, day, month, cyclical)
- Adds interaction features
- Saves to `feature_store/features_v1.csv`

#### 4. Build Weather Features
```bash
python feature_store/build_weather_features.py
```
**What it does:**
- Loads latest weather CSV
- Creates lag features and rolling statistics
- Adds time-based features
- Saves to `feature_store/weather/weather_features.csv`

### 🎯 Model Training

#### 5. Train AQI Models (Brasilia)
```bash
python training/train_regression_models.py
```
**What it does:**
- Trains 4 models: Ridge, Random Forest, XGBoost, Gradient Boosting
- Uses cross-validation for hyperparameter tuning
- Filters data to Brasilia only
- Logs experiments to MLflow
- **Experiment:** `AQI_Regression_Models`

#### 6. Train Weather Models
```bash
python training/train_weather_models.py
```
**What it does:**
- Trains Ridge and Random Forest for each weather variable
- Trains separate models for: temperature, humidity, wind_speed, pressure
- Logs experiments to MLflow
- **Experiment:** `Weather_Forecasting`

#### 7. Register Best AQI Model
```bash
python training/select_and_register_best_model.py
```
**What it does:**
- Finds best model by RMSE from `AQI_Regression_Models` experiment
- Registers model as `AQI_Predictor` in MLflow Model Registry
- Creates new version in registry

#### 8. Register Best Weather Models
```bash
python training/select_and_register_weather_models.py
```
**What it does:**
- Finds best model for each weather variable
- Registers models as:
  - `Weather_Model_temperature`
  - `Weather_Model_humidity`
  - `Weather_Model_wind_speed`
  - `Weather_Model_pressure`

#### 9. Generate SHAP Explanations
```bash
python training/explain_model_shap.py
```
**What it does:**
- Loads registered AQI model
- Computes SHAP values
- Saves summary plot to `shap_summary.png`

### 📊 Monitoring & Drift Detection

#### 10. Compute Data Drift
```bash
python monitoring/data_drift.py
```
**What it does:**
- Calculates PSI (Population Stability Index) for features
- Compares training vs recent data distributions
- Saves drift report to `monitoring/drift_report.csv`

#### 11. Make Retrain Decision
```bash
python monitoring/retrain_decision.py
```
**What it does:**
- Reads drift report
- Determines if retraining is needed (PSI > 0.2)
- Saves retrain signal to `monitoring/retrain_signal.csv`

#### 12. Visualize Drift
```bash
python monitoring/visualize_drift.py
```
**What it does:**
- Generates drift visualization plots
- Saves plots to `monitoring/drift_plots/`

### 🔄 Orchestration (Prefect)

#### 13. Run Full Pipeline (Prefect Flow)
```bash
# Start Prefect server (in one terminal)
prefect server start

# Run pipeline (in another terminal)
python pipelines/prefect_flow.py
```
**What it does:**
- Runs complete pipeline:
  1. Ingests historical AQI data
  2. Ingests weather data
  3. Trains AQI models
  4. Trains weather models
  5. Computes data drift
  6. Makes retrain decision
  7. Conditionally retrains if drift detected

### 🌐 API & UI

#### 14. Start FastAPI Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
**Access:**
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/
- AQI Prediction: POST http://localhost:8000/predict/aqi
- AQI Forecast: GET http://localhost:8000/predict/aqi/forecast?city=Brasilia&steps=24
- Weather Forecast: GET http://localhost:8000/predict/weather?city=London

#### 15. Start Streamlit Dashboard
```bash
streamlit run ui/app.py --server.port 8501
```
**Access:** http://localhost:8501

**Features:**
- Weather Forecasts (Brasilia, London, Karachi)
- AQI Forecasting (Brasilia)
- EDA & Analysis
- Model Explainability (SHAP)
- Data Drift Monitoring

### 🐳 Docker Deployment

#### 16. Build and Run with Docker Compose
```bash
# Build images
docker compose build

# Start services
docker compose up

# Or run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```
**Access:**
- FastAPI: http://localhost:8000/docs
- Streamlit: http://localhost:8501

### 🧪 Testing

#### 17. Run All Tests
```bash
pytest tests/
```

#### 18. Run Specific Test Suites
```bash
# Data quality tests
pytest tests/test_data_quality.py

# Model performance tests
pytest tests/test_model_performance.py

# Inference tests
pytest tests/test_inference.py
```

---

## 🔄 Complete Workflow Example

### First-Time Setup
```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Ingest data
python pipelines/ingest_historical_data.py
python pipelines/ingest_weather_data.py

# 3. Build features
python feature_store/build_features.py
python feature_store/build_weather_features.py

# 4. Train models
python training/train_regression_models.py
python training/train_weather_models.py

# 5. Register best models
python training/select_and_register_best_model.py
python training/select_and_register_weather_models.py

# 6. Start services
uvicorn api.main:app --reload --port 8000 &
streamlit run ui/app.py --server.port 8501
```

### Daily Operations (Using Prefect)
```bash
# Start Prefect server
prefect server start

# Run automated pipeline
python pipelines/prefect_flow.py
```

---

## 📂 Project Structure

```
aqi-mlops/
├── api/                          # FastAPI routes (AQI + Weather)
│   ├── main.py                   # FastAPI app
│   ├── aqi.py                    # AQI prediction endpoints
│   └── weather.py                # Weather prediction endpoints
├── ui/                           # Streamlit dashboard
│   ├── app.py                    # Main Streamlit app
│   └── utils/                    # UI utilities
├── pipelines/                    # Data pipelines
│   ├── ingest_historical_data.py # AQI data ingestion
│   ├── ingest_weather_data.py   # Weather data ingestion
│   └── prefect_flow.py          # Prefect orchestration flow
├── training/                     # Model training & MLflow
│   ├── train_regression_models.py      # AQI model training
│   ├── train_weather_models.py         # Weather model training
│   ├── select_and_register_best_model.py      # Register best AQI model
│   ├── select_and_register_weather_models.py   # Register weather models
│   └── explain_model_shap.py     # SHAP explainability
├── feature_store/                # Engineered features
│   ├── build_features.py         # AQI feature engineering
│   ├── build_weather_features.py # Weather feature engineering
│   ├── features_v1.csv          # AQI features
│   └── weather/                 # Weather features
├── monitoring/                   # Drift detection
│   ├── data_drift.py            # PSI calculation
│   ├── retrain_decision.py      # Retrain logic
│   ├── visualize_drift.py      # Drift visualization
│   └── utils/                   # Drift utilities
├── tests/                        # Automated ML tests
│   ├── test_data_quality.py
│   ├── test_model_performance.py
│   └── test_inference.py
├── data/                         # Data storage
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed data
│   └── weather/                 # Weather data
├── mlruns/                       # MLflow artifacts (local)
├── alerts/                       # Email alert system
├── utils/                        # Shared utilities
├── Dockerfile.api               # API Docker image
├── Dockerfile.ui                # UI Docker image
├── Dockerfile.railway           # Railway deployment
├── docker-compose.yml           # Local development
├── railway.json                 # Railway configuration
├── prefect.yaml                 # Prefect configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔌 API Endpoints

### AQI Prediction
```bash
POST /predict/aqi
Content-Type: application/json

{
  "co": 0.4,
  "co2": 410.0,
  "no2": 20.0,
  "so2": 5.0,
  "o3": 30.0,
  "pm25": 45.0,
  "pm10": 70.0,
  "aqi_lag_1": 85.0,
  "aqi_lag_3": 80.0,
  "aqi_lag_6": 75.0,
  "aqi_roll_mean_3": 82.0,
  "aqi_roll_std_3": 4.1
}
```

### AQI Forecast
```bash
GET /predict/aqi/forecast?city=Brasilia&steps=24
```

### Weather Forecast
```bash
GET /predict/weather?city=London
```

---

## 🚂 Railway Deployment

### Quick Deploy
1. Push to GitHub:
```bash
git add .
git commit -m "Deploy to Railway"
git push origin main
```

2. Railway automatically:
   - Detects the push
   - Builds using `Dockerfile.railway`
   - Deploys your app
   - Provides public URL

### Railway Configuration
- **Dockerfile:** `Dockerfile.railway`
- **Config:** `railway.json`
- **Auto-deploy:** Enabled on push to `main`

### Access Your Deployment
- Check Railway dashboard for your app URL
- Streamlit UI will be available at the Railway URL
- API available at `{railway-url}/docs`

---

## 📊 MLflow & Prefect Access

### MLflow UI
```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```
**View:**
- Experiments: `AQI_Regression_Models`, `Weather_Forecasting`
- Model Registry: `AQI_Predictor`, `Weather_Model_*`
- Compare runs, metrics, parameters

### Prefect Dashboard
```bash
# Start Prefect server
prefect server start

# Access at http://localhost:4200
```
**View:**
- Flow runs: `Environmental Intelligence Pipeline`
- Task execution logs
- Schedule management

---

## 🔄 CI/CD (GitHub Actions)

On every push to `main`:
- ✅ Install dependencies
- ✅ Run tests
- ✅ Build Docker images
- ✅ Validate reproducibility

**Workflow:** `.github/workflows/ci.yml`

---

## 📌 Technologies Used

- **Python 3.9**
- **Scikit-learn** - Machine learning models
- **MLflow** - Experiment tracking & model registry
- **SHAP** - Model explainability
- **FastAPI** - REST API
- **Streamlit** - Interactive dashboard
- **Prefect** - Workflow orchestration
- **Docker & Docker Compose** - Containerization
- **GitHub Actions** - CI/CD
- **Railway** - Cloud deployment
- **XGBoost** - Gradient boosting
- **Pandas & NumPy** - Data processing

---

## 📝 Notes

- **AQI Model:** Trained only for Brasilia (city-specific)
- **Weather Models:** Support Brasilia, London, Karachi
- **Model Updates:** Handled automatically by Prefect flows
- **Drift Detection:** PSI-based, triggers retraining when > 0.2
- **Model Registry:** Uses MLflow Model Registry for versioning

---

## 🆘 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Kill process on port 5000 (MLflow)
lsof -ti:5000 | xargs kill -9
```

### Docker Issues
```bash
# Clean up Docker
docker compose down
docker system prune -f

# Rebuild from scratch
docker compose build --no-cache
docker compose up
```

### Model Loading Issues
- Ensure `mlruns/` directory exists with registered models
- Check MLflow tracking URI is set correctly
- Verify model registry has latest versions

---

## 📚 Additional Resources

- **GitHub Repository:** https://github.com/msA1M/aqi-mlops
- **MLflow Documentation:** https://mlflow.org/docs/latest/index.html
- **Prefect Documentation:** https://docs.prefect.io/
- **Railway Documentation:** https://docs.railway.app/
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **Streamlit Documentation:** https://docs.streamlit.io/
