import mlflow
import os
import pandas as pd
from pydantic import BaseModel
from fastapi import APIRouter
from alerts.email_alert import send_email_alert

router = APIRouter(prefix="/predict", tags=["AQI"])

# Set MLflow tracking URI if not already set
# Use Docker path only if /app/mlruns exists (Docker), otherwise use local path
if not os.getenv("MLFLOW_TRACKING_URI"):
    if os.path.exists("/app/mlruns"):
        mlflow.set_tracking_uri("file:/app/mlruns")  # Docker
    else:
        mlflow.set_tracking_uri("file:./mlruns")  # Local

# Try to load from registry, fallback to direct run path if registry fails
MODEL_URI_REGISTRY = "models:/AQI_Predictor/latest"
# Fallback: use run ID directly (from version-6 meta.yaml: run_id: 34bafbec55874d9ea44d0baa80ea117e)
MODEL_URI_RUN = "runs:/34bafbec55874d9ea44d0baa80ea117e/model"
model = None  # Lazy load on first request


def get_model():
    """Lazy load model on first request"""
    global model
    if model is None:
        try:
            # Try registry first
            model = mlflow.pyfunc.load_model(MODEL_URI_REGISTRY)
        except (OSError, Exception) as e:
            # Fallback to direct run path if registry fails
            print(f"⚠️ Registry load failed: {e}. Trying direct run path...")
            try:
                model = mlflow.pyfunc.load_model(MODEL_URI_RUN)
                print("✅ Model loaded from run path")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model from both registry and run path. "
                    f"Registry error: {e}. Run path error: {e2}"
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
