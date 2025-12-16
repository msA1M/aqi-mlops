# tests/test_inference.py

import mlflow
import pandas as pd
import numpy as np

MODEL_NAME = "AQI_Predictor"


def test_single_prediction():
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

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

    pred = model.predict(sample)

    assert len(pred) == 1
    assert not np.isnan(pred[0]), "Prediction is NaN"
