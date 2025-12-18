import mlflow
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
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
    
    # Get the input data
    input_dict = data.dict()
    
    # Try to load feature store to get default values for missing features
    try:
        features_df = pd.read_csv("feature_store/features_v1.csv", parse_dates=["datetime"])
        brasilia_df = features_df[features_df["city"] == "Brasilia"]
        if len(brasilia_df) > 0:
            # Get median/default values for missing features
            latest_row = brasilia_df.iloc[-1]
            exclude_cols = ["aqi", "city", "datetime"]
            all_feature_cols = [col for col in brasilia_df.columns if col not in exclude_cols]
            
            # Fill missing features with defaults from latest data
            for col in all_feature_cols:
                if col not in input_dict:
                    if pd.notna(latest_row[col]):
                        input_dict[col] = float(latest_row[col])
                    elif "lag" in col or "roll" in col or "diff" in col or "ema" in col:
                        input_dict[col] = 0.0
                    elif "sin" in col or "cos" in col:
                        # Time-based features - calculate from current time
                        if "hour" in col:
                            from datetime import datetime
                            hour = datetime.now().hour
                            input_dict[col] = float(np.sin(2 * np.pi * hour / 24)) if "sin" in col else float(np.cos(2 * np.pi * hour / 24))
                        elif "day" in col:
                            day = datetime.now().weekday()
                            input_dict[col] = float(np.sin(2 * np.pi * day / 7)) if "sin" in col else float(np.cos(2 * np.pi * day / 7))
                        else:
                            input_dict[col] = 0.0
                    else:
                        input_dict[col] = float(brasilia_df[col].median()) if col in brasilia_df.columns else 0.0
    except:
        # If feature store not available, use basic defaults
        pass
    
    df = pd.DataFrame([input_dict])
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


@router.get("/aqi/forecast")
def forecast_aqi(city: str = "Brasilia", steps: int = 24):
    """
    Forecast AQI for a city using historical data.
    Uses the latest available features to predict future AQI values.
    """
    try:
        # Load feature store
        features_df = pd.read_csv("feature_store/features_v1.csv", parse_dates=["datetime"])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Feature store not found. Please run feature engineering first.")
    
    # Filter for the requested city
    city_df = features_df[features_df["city"].str.lower() == city.lower()].copy()
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found in dataset")
    
    # Sort by datetime and get the latest row
    city_df = city_df.sort_values("datetime")
    latest = city_df.iloc[-1]
    
    # Get the model
    model = get_model()
    
    # Prepare features - exclude target and metadata columns
    # Get all columns except target and metadata
    exclude_cols = ["aqi", "city", "datetime"]
    feature_cols = [col for col in city_df.columns if col not in exclude_cols]
    
    # Get latest feature values
    current_features = latest[feature_cols].to_dict()
    
    # Ensure all values are numeric and handle NaN
    for key, value in current_features.items():
        if pd.isna(value):
            # Fill NaN with median or 0 for derived features
            if "lag" in key or "roll" in key or "diff" in key or "ema" in key or "sin" in key or "cos" in key:
                current_features[key] = 0.0
            else:
                current_features[key] = float(city_df[key].median()) if key in city_df.columns else 0.0
        else:
            current_features[key] = float(value)
    
    # For forecasting, we'll use a simple approach:
    # Use the latest features and predict, then update lag features for next predictions
    forecasts = []
    predicted_aqi_values = []
    
    # Use the latest AQI values for lag features
    aqi_history = city_df["aqi"].tail(12).tolist()  # Get last 12 AQI values for lags
    
    for step in range(steps):
        # Create feature vector
        feature_dict = current_features.copy()
        
        # Update AQI lag features based on predicted history
        for lag in [1, 3, 6, 12]:
            lag_key = f"aqi_lag_{lag}"
            if lag_key in feature_dict:
                if step == 0:
                    # First prediction uses actual lags from historical data
                    if len(aqi_history) >= lag:
                        feature_dict[lag_key] = float(aqi_history[-lag])
                    elif lag_key in latest and not pd.isna(latest[lag_key]):
                        feature_dict[lag_key] = float(latest[lag_key])
                    else:
                        feature_dict[lag_key] = float(aqi_history[-1]) if len(aqi_history) > 0 else 0.0
                else:
                    # Subsequent predictions use previous predictions
                    if len(predicted_aqi_values) >= lag:
                        feature_dict[lag_key] = float(predicted_aqi_values[-lag])
                    elif len(aqi_history) >= lag:
                        feature_dict[lag_key] = float(aqi_history[-lag])
                    else:
                        feature_dict[lag_key] = float(predicted_aqi_values[-1]) if len(predicted_aqi_values) > 0 else 0.0
        
        # Update rolling statistics
        for window in [3, 6, 12]:
            mean_key = f"aqi_roll_mean_{window}"
            std_key = f"aqi_roll_std_{window}"
            min_key = f"aqi_roll_min_{window}"
            max_key = f"aqi_roll_max_{window}"
            
            if len(predicted_aqi_values) >= window:
                recent_aqi = predicted_aqi_values[-window:]
                if mean_key in feature_dict:
                    feature_dict[mean_key] = float(pd.Series(recent_aqi).mean())
                if std_key in feature_dict:
                    std_val = float(pd.Series(recent_aqi).std())
                    feature_dict[std_key] = std_val if not pd.isna(std_val) and std_val > 0 else 1.0
                if min_key in feature_dict:
                    feature_dict[min_key] = float(min(recent_aqi))
                if max_key in feature_dict:
                    feature_dict[max_key] = float(max(recent_aqi))
            else:
                # Use actual rolling stats from latest data
                for key in [mean_key, std_key, min_key, max_key]:
                    if key in feature_dict:
                        if key in latest and not pd.isna(latest[key]):
                            feature_dict[key] = float(latest[key])
                        elif "mean" in key:
                            feature_dict[key] = float(latest["aqi"]) if not pd.isna(latest["aqi"]) else 0.0
                        elif "std" in key:
                            feature_dict[key] = 1.0
                        else:
                            feature_dict[key] = float(latest["aqi"]) if not pd.isna(latest["aqi"]) else 0.0
        
        # Update AQI difference features
        if "aqi_diff_1" in feature_dict and len(predicted_aqi_values) > 0:
            feature_dict["aqi_diff_1"] = float(predicted_aqi_values[-1] - predicted_aqi_values[-2]) if len(predicted_aqi_values) >= 2 else 0.0
        if "aqi_diff_3" in feature_dict and len(predicted_aqi_values) >= 3:
            feature_dict["aqi_diff_3"] = float(predicted_aqi_values[-1] - predicted_aqi_values[-3])
        
        # Update EMA features
        for window in [3, 6]:
            ema_key = f"aqi_ema_{window}"
            if ema_key in feature_dict:
                if len(predicted_aqi_values) >= window:
                    # Simple EMA calculation
                    alpha = 2.0 / (window + 1)
                    ema_val = predicted_aqi_values[-1]
                    for val in reversed(predicted_aqi_values[-window:-1]):
                        ema_val = alpha * val + (1 - alpha) * ema_val
                    feature_dict[ema_key] = float(ema_val)
                elif ema_key in latest and not pd.isna(latest[ema_key]):
                    feature_dict[ema_key] = float(latest[ema_key])
                else:
                    feature_dict[ema_key] = float(predicted_aqi_values[-1]) if len(predicted_aqi_values) > 0 else float(latest["aqi"])
        
        # Ensure all features are present and numeric
        for key in feature_dict.keys():
            if pd.isna(feature_dict[key]):
                feature_dict[key] = 0.0
        
        # Predict
        pred_df = pd.DataFrame([feature_dict])
        predicted_aqi = float(model.predict(pred_df)[0])
        predicted_aqi_values.append(predicted_aqi)
        
        category, alert_level = aqi_category(predicted_aqi)
        
        forecasts.append({
            "step": step + 1,
            "predicted_aqi": round(predicted_aqi, 2),
            "category": category,
            "alert_level": alert_level
        })
    
    return {
        "city": city,
        "forecasts": forecasts,
        "current_aqi": float(latest["aqi"]),
        "current_category": aqi_category(latest["aqi"])[0]
    }
