# tests/test_data_quality.py

import pandas as pd

FEATURE_PATH = "feature_store/features_v1.csv"

REQUIRED_COLUMNS = [
    "co", "co2", "no2", "so2", "o3",
    "pm25", "pm10",
    "aqi_lag_1", "aqi_lag_3", "aqi_lag_6",
    "aqi_roll_mean_3", "aqi_roll_std_3",
    "aqi"
]


def test_feature_schema():
    df = pd.read_csv(FEATURE_PATH)

    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_no_missing_values():
    df = pd.read_csv(FEATURE_PATH)
    assert df.isnull().sum().sum() == 0, "Missing values detected"


def test_numeric_features():
    df = pd.read_csv(FEATURE_PATH)
    numeric_df = df.drop(columns=["city", "datetime"], errors="ignore")

    assert numeric_df.applymap(
        lambda x: isinstance(x, (int, float))
    ).all().all(), "Non-numeric values detected"
