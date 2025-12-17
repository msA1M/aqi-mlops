from typing import Any

import mlflow
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from api.aqi import router as aqi_router
from api.weather import router as weather_router

# Try to import email alerts; fall back to a no-op in environments (like Spaces)
# where the alerts package might not be available.
try:  # pragma: no cover - optional dependency
    from alerts.email_alert import send_email_alert  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    def send_email_alert(*args: Any, **kwargs: Any) -> None:
        print("Email alerts disabled: 'alerts' package not available.")

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
    # This endpoint is kept for backwards compatibility but delegates
    # to the router-based implementation in `api/aqi.py`.
    from api.aqi import predict_aqi  # local import to avoid circular deps
    return predict_aqi(data)



