# api/main.py
from alerts.email_alert import send_email_alert
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

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

app = FastAPI(title="AQI Prediction API")

# Load model once at startup
model = mlflow.pyfunc.load_model(MODEL_URI)


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



