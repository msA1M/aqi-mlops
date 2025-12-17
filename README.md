🌍 Environmental Intelligence Platform — End-to-End MLOps Project
An end-to-end Machine Learning Operations (MLOps) system for Air Quality Index (AQI) and Weather Forecasting, designed with realistic industry-grade architecture.
This project demonstrates the full ML lifecycle: live data ingestion, feature engineering, model training, experiment tracking, model registry, explainability, drift monitoring, orchestration, CI/CD, containerization, and interactive dashboards.

🎯 Project Objective
To build a production-style ML system that:
Predicts AQI for a target city (Brasilia)
Forecasts weather variables (temperature, humidity, wind speed, pressure) for multiple cities
Monitors live data for data drift
Produces health-based alerts
Automates pipelines using Prefect
Ensures reliability via testing, CI/CD, and Docker

🧠 System Capabilities
✅ Engineering & Machine Learning
Time-series AQI data ingestion
Weather data ingestion (live API)
Feature store with:
Lag features
Rolling statistics
Multiple ML models:
Ridge Regression
Random Forest
MLflow:
Experiment tracking
Model comparison
Model registry
Best-model selection & registration
SHAP explainability (AQI model)
AQI category classification & alert logic

🌦️ Weather Forecasting (New Feature)
Predicts:
Temperature
Humidity
Wind Speed
Pressure
Supported cities:
Brasilia
London
Karachi
Multi-target regression (one model per variable)
Models registered independently in MLflow

🌐 Live Data Integration
Weather: Open-Meteo API (stable, no API key)
AQI: Historical + monitored live data
Live data is used for:
Monitoring
Drift detection
❌ Not blindly used for training
📈 Monitoring & Reliability
Data drift detection using PSI (Population Stability Index)
Visual drift plots in Streamlit
Drift levels:
PSI < 0.10 → No drift
0.10–0.20 → Moderate drift
0.20 → Significant drift
Automated ML tests:
Data validation
Model performance regression
Inference safety

⚙️ Systems & Ops
Component	Tool
API	FastAPI
Dashboard	Streamlit
Orchestration	Prefect
Experiment Tracking	MLflow
Drift Detection	PSI
Containerization	Docker
CI/CD	GitHub Actions

🏗️ Architecture Overview
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

🔁 Orchestration with Prefect (IMPORTANT)
Prefect is the single source of automation.
What Prefect DOES
Ingests AQI & weather data
Updates feature store
Runs training pipelines (when scheduled)
Keeps production data fresh

What Streamlit DOES NOT DO
❌ No training
❌ No retraining
❌ No ingestion
❌ No scheduling
📌 Streamlit is visualization-only
📊 Streamlit Dashboard Behavior
What updates dynamically
Weather predictions (from latest registered models)
AQI predictions (Brasilia)
Drift plots
PSI values
What stays the same
Model parameters (until Prefect retrains)
Reference training distributions
📌 Graphs update when Prefect ingests new data
📌 PSI values change if data distribution shifts

🔌 FastAPI Endpoints
AQI Prediction
POST /predict/aqi
Weather Prediction
GET /predict/weather?city=London
📂 Project Structure

aqi-mlops/
├── api/                 # FastAPI routes (AQI + Weather)
├── ui/                  # Streamlit dashboard
├── pipelines/           # Prefect flows
├── training/            # Model training & MLflow
├── feature_store/       # Engineered features
├── utils/               # Drift & helpers
├── tests/               # Automated ML tests
├── mlruns/              # MLflow artifacts (local)
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
└── README.md
🚀 How to Run (Local)

1️⃣ Create environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2️⃣ Start Prefect
prefect server start
3️⃣ Run pipeline
python pipelines/prefect_flow.py

4️⃣ Start API
uvicorn api.main:app --reload

5️⃣ Run Dashboard
streamlit run ui/app.py

🐳 Docker Deployment
docker compose build
docker compose up
FastAPI → http://localhost:8000/docs
Streamlit → http://localhost:8501
MLflow artifacts are mounted, not baked into images

🔄 CI/CD (GitHub Actions)
On every push to main:
Install dependencies
Run tests
Build Docker images
Validate reproducibility

📌 Technologies Used
Python 3.9
Scikit-learn
MLflow
SHAP
FastAPI
Streamlit
Prefect
Docker & Docker Compose
GitHub Actions
