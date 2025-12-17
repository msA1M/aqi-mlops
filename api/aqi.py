import mlflow
import os
import pandas as pd
from pydantic import BaseModel
from fastapi import APIRouter
from typing import Any

# Optional email alerts: disable gracefully if alerts package is unavailable
try:  # pragma: no cover
    from alerts.email_alert import send_email_alert  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    def send_email_alert(*args: Any, **kwargs: Any) -> None:
        print("Email alerts disabled: 'alerts' package not available.")

router = APIRouter(prefix="/predict", tags=["AQI"])

# Set MLflow tracking URI if not already set
# Use Docker path only if /app/mlruns exists (Docker), otherwise use local path
if not os.getenv("MLFLOW_TRACKING_URI"):
    if os.path.exists("/app/mlruns"):
        mlflow.set_tracking_uri("file:/app/mlruns")  # Docker
    else:
        mlflow.set_tracking_uri("file:./mlruns")  # Local

# AQI model is registered as AQI_Predictor
# Try registry first (most reliable), then fallback to direct paths if needed
model = None  # Lazy load on first request

# Fallback: Direct model ID from experiment 481652201472430433 (version-3 model)
FALLBACK_MODEL_ID = "m-168880ff890a4a9bb3f7b210da76d290"
FALLBACK_EXPERIMENT_ID = "481652201472430433"
FALLBACK_RUN_ID = "e00c4c8870594a40afa60a0876ddf4e4"


def get_model():
    """Lazy load model on first request - try registry first, then fallback to direct paths"""
    global model
    if model is None:
        # Determine base path for mlruns
        if os.path.exists("/app/mlruns"):
            base_path = "/app/mlruns"  # Docker
        else:
            base_path = "./mlruns"  # Local
        
        # Try registry first (most reliable)
        try:
            model = mlflow.pyfunc.load_model("models:/AQI_Predictor/latest")
            print("✅ AQI model loaded from registry: models:/AQI_Predictor/latest")
        except Exception as e:
            # Fallback 1: try direct filesystem path
            print(f"⚠️ Registry load failed: {e}. Trying direct path...")
            try:
                model_artifacts_path = f"{base_path}/{FALLBACK_EXPERIMENT_ID}/models/{FALLBACK_MODEL_ID}/artifacts"
                
                # Check if artifacts directory exists
                if not os.path.exists(model_artifacts_path):
                    raise FileNotFoundError(f"Model artifacts not found at {model_artifacts_path}")
                
                # Check if MLmodel file exists
                mlmodel_path = os.path.join(model_artifacts_path, "MLmodel")
                if not os.path.exists(mlmodel_path):
                    raise FileNotFoundError(f"MLmodel file not found at {mlmodel_path}")
                
                # Load model directly from artifacts path
                model = mlflow.pyfunc.load_model(model_artifacts_path)
                print(f"✅ AQI model loaded from direct path: {model_artifacts_path}")
            except Exception as e2:
                # Fallback 2: try run URI
                print(f"⚠️ Direct path load failed: {e2}. Trying run URI...")
                try:
                    model = mlflow.pyfunc.load_model(f"runs:/{FALLBACK_RUN_ID}/model")
                    print(f"✅ AQI model loaded from run URI: runs:/{FALLBACK_RUN_ID}/model")
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to load AQI model from all paths. "
                        f"Registry: {e}, Direct path: {e2}, Run URI: {e3}. "
                        f"Please ensure model artifacts exist in mlruns directory."
                    )
    return model


class AQIInput(BaseModel):
    co: float
    co2: float
    no2: float
    so2: float
    o3: float
    pm25: float
    pm10: float
    aqi_lag_1: float
    aqi_lag_3: float
    aqi_lag_6: float
    aqi_roll_mean_3: float
    aqi_roll_std_3: float


def aqi_category(aqi: float):
    if aqi <= 40:
        return "Good", "Safe"
    elif aqi <= 80:
        return "Moderate", "Caution"
    elif aqi <= 120:
        return "Unhealthy (Sensitive)", "Warning"
    elif aqi <= 160:
        return "Unhealthy", "Alert"
    else:
        return "Hazardous", "Critical"


@router.post("/aqi")
def predict_aqi(data: AQIInput):
    model = get_model()  # Load model on first request
    df = pd.DataFrame([data.dict()])
    prediction = float(model.predict(df)[0])

    category, alert_level = aqi_category(prediction)

    send_email_alert(
        aqi=round(prediction, 2),
        category=category,
        alert_level=alert_level
    )

    return {
        "predicted_aqi": round(prediction, 2),
        "category": category,
        "alert_level": alert_level
    }
