import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/predict", tags=["Weather"])
mlflow.set_tracking_uri("file:./mlruns")

MODELS = {
    "temperature": "models:/Weather_temperature/latest",
    "humidity": "models:/Weather_humidity/latest",
    "wind_speed": "models:/Weather_wind_speed/latest",
    "pressure": "models:/Weather_pressure/latest",
}

@router.get("/weather")
def predict_weather(city: str):
    df = pd.read_csv("feature_store/weather/weather_features.csv")

    city_df = df[df["city"].str.lower() == city.lower()]
    if city_df.empty:
        raise HTTPException(status_code=404, detail="City not found")

    latest = city_df.sort_values("datetime").iloc[-1]
    DROP_COLS = ["datetime", "city", "temperature", "humidity", "wind_speed", "pressure"]
    X = latest.drop(DROP_COLS)

    predictions = {}

    for target, uri in MODELS.items():
        model = mlflow.pyfunc.load_model(uri)
        predictions[target] = float(model.predict(pd.DataFrame([X]))[0])

    return {
        "city": city,
        "predictions": predictions
    }
