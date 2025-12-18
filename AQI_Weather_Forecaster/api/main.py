# api/main.py
from alerts.email_alert import send_email_alert
import mlflow
import os
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from api.aqi import router as aqi_router
from api.weather import router as weather_router

# Set MLflow tracking URI if not already set
# Use Docker path only if /app/mlruns exists (Docker), otherwise use local path
if not os.getenv("MLFLOW_TRACKING_URI"):
    if os.path.exists("/app/mlruns"):
        mlflow.set_tracking_uri("file:/app/mlruns")  # Docker
    else:
        mlflow.set_tracking_uri("file:./mlruns")  # Local

app = FastAPI(title="Environmental Intelligence API")
app.include_router(aqi_router)
app.include_router(weather_router)

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


# IMPORTANT: load latest registered model
MODEL_NAME = "AQI_Predictor"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

# Load model once at startup (lazy load to avoid startup errors)
model = None

def get_model():
    """Lazy load model on first request"""
    global model
    if model is None:
        model = mlflow.pyfunc.load_model(MODEL_URI)
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


@app.get("/")
def root():
    return {"status": "AQI model is running"}


@app.post("/predict")
def predict(data: AQIInput):
    model = get_model()  # Load model on first request
    df = pd.DataFrame([data.dict()])
    prediction = float(model.predict(df)[0])

    category, alert_level = aqi_category(prediction)

    # Send email for Moderate and above
    send_email_alert(
        aqi=round(prediction, 2),
        category=category,
        alert_level=alert_level
    )

    return {
        "predicted_aqi": round(prediction, 2),
        "category": category,
        "alert_level": alert_level,
        "email_sent": alert_level != "Safe"
    }



