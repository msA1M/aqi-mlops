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

# Weather models are registered as Weather_Model_<target>
# Try registry first (most reliable), then fallback to direct paths if needed
MODEL_REGISTRY_NAMES = {
    "temperature": "Weather_Model_temperature",
    "humidity": "Weather_Model_humidity",
    "wind_speed": "Weather_Model_wind_speed",
    "pressure": "Weather_Model_pressure",
}

# Fallback: Direct model IDs from experiment 931917419341530721 (version-17 models)
# These are the latest registered model IDs as of version-17
WEATHER_MODEL_IDS = {
    "temperature": "m-5d546946af9a4d1fb6f3931fb8b8ad11",
    "humidity": "m-ea1bfddfa17d4197aebb7ea8d942dd70",
    "wind_speed": "m-0fb51f2d25c14992bf71e4839b320ba7",
    "pressure": "m-dbd425c1079c46ea842f8c711b02de89",
}

EXPERIMENT_ID = "931917419341530721"

# Cache models to avoid reloading on every request
_model_cache = {}

def get_model(target):
    """Lazy load and cache models - use direct paths (registry has path resolution issues)"""
    if target not in _model_cache:
        if target not in MODEL_REGISTRY_NAMES:
            raise HTTPException(
                status_code=503,
                detail=f"Unknown weather target: {target}"
            )
        
        # Determine base path for mlruns
        if os.path.exists("/app/mlruns"):
            base_path = "/app/mlruns"  # Docker
        else:
            base_path = "./mlruns"  # Local
        
        # Use direct filesystem path (registry has issues with absolute paths in metadata)
        model_id = WEATHER_MODEL_IDS.get(target)
        if not model_id:
            raise HTTPException(
                status_code=503,
                detail=f"No model ID configured for {target}"
            )
        
        # Direct path to model artifacts
        model_artifacts_path = f"{base_path}/{EXPERIMENT_ID}/models/{model_id}/artifacts"
        
        try:
            # Check if artifacts directory exists
            if not os.path.exists(model_artifacts_path):
                raise FileNotFoundError(f"Model artifacts not found at {model_artifacts_path}")
            
            # Check if MLmodel file exists
            mlmodel_path = os.path.join(model_artifacts_path, "MLmodel")
            if not os.path.exists(mlmodel_path):
                raise FileNotFoundError(f"MLmodel file not found at {mlmodel_path}")
            
            # Load model directly from artifacts path
            _model_cache[target] = mlflow.pyfunc.load_model(model_artifacts_path)
            print(f"✅ Weather model '{target}' loaded from direct path: {model_artifacts_path}")
        except Exception as e:
            # Fallback: try registry (may fail due to absolute paths in metadata)
            print(f"⚠️ Direct path load failed for {target}: {e}. Trying registry...")
            try:
                registry_name = MODEL_REGISTRY_NAMES[target]
                registry_uri = f"models:/{registry_name}/latest"
                _model_cache[target] = mlflow.pyfunc.load_model(registry_uri)
                print(f"✅ Weather model '{target}' loaded from registry: {registry_uri}")
            except Exception as e2:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model loading failed for {target}. Direct path: {e}, Registry: {e2}. "
                           f"Please ensure model artifacts exist at {model_artifacts_path}"
                )
    return _model_cache[target]

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

    for target in ["temperature", "humidity", "wind_speed", "pressure"]:
        model = get_model(target)
        predictions[target] = float(model.predict(pd.DataFrame([X]))[0])

    return {
        "city": city,
        "predictions": predictions
    }
