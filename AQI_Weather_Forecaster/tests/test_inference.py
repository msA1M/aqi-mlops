# tests/test_inference.py

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow

MODEL_NAME = "AQI_Predictor"


def load_model():
    # CI environment → train lightweight model
    if os.getenv("CI") == "true":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ])
        return model

    # Local/dev environment → load from MLflow registry
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")


def test_single_prediction():
    model = load_model()

    sample = pd.DataFrame([{
        "co": 0.3,
        "co2": 420,
        "no2": 20,
        "so2": 10,
        "o3": 40,
        "pm25": 30,
        "pm10": 60,
        "aqi_lag_1": 80,
        "aqi_lag_3": 78,
        "aqi_lag_6": 75,
        "aqi_roll_mean_3": 77,
        "aqi_roll_std_3": 2
    }])

    # Fit only if dummy model
    if os.getenv("CI") == "true":
        model.fit(sample, [80])

    pred = model.predict(sample)

    assert len(pred) == 1
    assert not np.isnan(pred[0])
