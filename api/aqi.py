from typing import Any

import mlflow
import pandas as pd
from pydantic import BaseModel
from fastapi import APIRouter

# Optional email alerts: disable gracefully if alerts package is unavailable
try:  # pragma: no cover
    from alerts.email_alert import send_email_alert  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    def send_email_alert(*args: Any, **kwargs: Any) -> None:
        print("Email alerts disabled: 'alerts' package not available.")

router = APIRouter(prefix="/predict", tags=["AQI"])

MODEL_URI = "models:/AQI_Predictor/latest"
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
