# tests/test_model_performance.py

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import mlflow

MODEL_NAME = "AQI_Predictor"
FEATURE_PATH = "feature_store/features_v1.csv"

MAX_ACCEPTABLE_RMSE = 200  # sanity check


def load_model():
    if os.getenv("CI") == "true":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ])
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")


def test_model_rmse():
    df = pd.read_csv(FEATURE_PATH)
    X = df.drop(columns=["aqi", "city", "datetime"], errors="ignore")
    y = df["aqi"]

    model = load_model()

    if os.getenv("CI") == "true":
        model.fit(X, y)

    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))

    assert rmse < MAX_ACCEPTABLE_RMSE
