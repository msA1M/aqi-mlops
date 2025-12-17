import mlflow
import os
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/predict", tags=["Weather"])

# Set MLflow tracking URI if not already set
# Use Docker path only if /app/mlruns exists (Docker), otherwise use local path
if not os.getenv("MLFLOW_TRACKING_URI"):
    if os.path.exists("/app/mlruns"):
        mlflow.set_tracking_uri("file:/app/mlruns")  # Docker
    else:
        mlflow.set_tracking_uri("file:./mlruns")  # Local

MODELS = {
    "temperature": "models:/Weather_temperature/latest",
    "humidity": "models:/Weather_humidity/latest",
    "wind_speed": "models:/Weather_wind_speed/latest",
    "pressure": "models:/Weather_pressure/latest",
}

# Cache models to avoid reloading on every request
_model_cache = {}

def get_model(uri):
    """Lazy load and cache models"""
    if uri not in _model_cache:
        try:
            _model_cache[uri] = mlflow.pyfunc.load_model(uri)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model loading failed: {str(e)}. Please ensure weather models are registered in MLflow."
            )
    return _model_cache[uri]

@router.get("/weather")
def predict_weather(city: str):
    try:
        df = pd.read_csv("feature_store/weather/weather_features.csv")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Weather features file not found")

    city_df = df[df["city"].str.lower() == city.lower()]
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found in dataset")

    latest = city_df.sort_values("datetime").iloc[-1]
    DROP_COLS = ["datetime", "city", "temperature", "humidity", "wind_speed", "pressure"]
    X = latest.drop(DROP_COLS)

    predictions = {}

    for target, uri in MODELS.items():
        model = get_model(uri)
        predictions[target] = float(model.predict(pd.DataFrame([X]))[0])

    return {
        "city": city,
        "predictions": predictions
    }
