# tests/test_model_performance.py

import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

MODEL_NAME = "AQI_Predictor"
FEATURE_PATH = "feature_store/features_v1.csv"

MAX_ACCEPTABLE_RMSE = 120  # defensible threshold


def test_model_rmse():
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

    df = pd.read_csv(FEATURE_PATH)
    X = df.drop(columns=["aqi", "city", "datetime"], errors="ignore")
    y = df["aqi"]

    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))

    assert rmse < MAX_ACCEPTABLE_RMSE, f"RMSE too high: {rmse}"
