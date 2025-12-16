🌍 AQI Prediction System — End-to-End MLOps Project
An end-to-end Machine Learning Operations (MLOps) project for Air Quality Index (AQI) prediction, monitoring, and alerting.
The system demonstrates realistic industry-style MLOps practices, including live data ingestion, experiment tracking, model registry, explainability, drift monitoring, orchestration, CI/CD, and containerized deployment.
🎯 Project Objective
To build a production-style ML system that:
Predicts AQI using historical air-quality data
Classifies air quality into health-based categories
Monitors live environmental data for drift
Provides alerts when air quality degrades
Ensures reliability via automation, testing, and CI/CD
🧠 System Capabilities
✅ Engineering & Machine Learning
Historical AQI data ingestion (time-series)
Feature store with lag & rolling statistics
Multiple ML models (Ridge, Random Forest, XGBoost)
MLflow experiment tracking & model registry
Best-model selection and registration
SHAP explainability for model transparency
AQI category classification & alert logic
🌐 Live Data Integration
Live air-quality ingestion from OpenAQ
Robust fallback mechanism when external API is unavailable
Live data used for monitoring & drift detection (not blind prediction)
📈 Monitoring & Reliability
Data drift detection and visualization
Alert triggering based on AQI category
Automated ML tests (data, model performance, inference)
⚙️ Systems & Ops
FastAPI inference service
Streamlit interactive dashboard
Prefect orchestration for end-to-end pipelines
Docker & Docker Compose for deployment
GitHub Actions CI pipeline (tests + Docker builds)
Clean project structure & .gitignore
🏗️ Architecture Overview
            ┌──────────────┐
            │   OpenAQ API │
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
📂 Project Structure
aqi-mlops/
├── api/                    # FastAPI inference service
├── ui/                     # Streamlit dashboard
├── pipelines/              # Ingestion & orchestration
├── training/               # Training, SHAP, model selection
├── feature_store/          # Feature engineering outputs
├── monitoring/             # Drift detection & plots
├── tests/                  # Automated ML tests
├── utils/                  # AQI classification logic
├── mlruns/                 # MLflow artifacts (local)
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
└── README.md
🚀 How to Run (Local)
1️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2️⃣ Train models & register best model
python training/train_regression_models.py
python training/select_and_register_best_model.py
3️⃣ Run FastAPI
uvicorn api.main:app --reload
Open: http://127.0.0.1:8000/docs
4️⃣ Run Streamlit UI
streamlit run ui/app.py
🐳 Docker Deployment
docker compose build
docker compose up
FastAPI → http://localhost:8000/docs
Streamlit → http://localhost:8501
MLflow artifacts are mounted at runtime, not baked into images.
🔁 Orchestration (Prefect)
python pipelines/prefect_flow.py
Or (optional UI):
prefect server start
🧪 Automated ML Tests
pytest tests/
Tests cover:
Data schema & quality
Model performance regression
Inference safety
🔄 CI/CD (GitHub Actions)
On every push to main:
Dependencies installed
ML tests executed
Docker images built
Ensures reproducibility & reliability.
⚠️ Design Decisions (Important)
Live OpenAQ data is used for monitoring, not direct training
Training and inference are decoupled
Retraining is not automatic — triggered by drift signals
Docker images are artifact-agnostic (CI-safe)
🧑‍🏫 How to Explain This Project (One Sentence)
“This project demonstrates a full MLOps lifecycle for AQI prediction, integrating live data ingestion, experiment tracking, explainability, drift monitoring, orchestration, CI/CD, and containerized deployment.”
📌 Technologies Used
Python 3.9
Scikit-learn, XGBoost
MLflow
SHAP
FastAPI
Streamlit
Prefect
Docker & Docker Compose
GitHub Actions
✅ Status
✔ End-to-end MLOps
✔ Realistic & reproducible
✔ Exam-safe and industry-aligned
