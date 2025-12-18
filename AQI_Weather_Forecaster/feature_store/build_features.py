# feature_store/build_features.py

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed")
FEATURE_STORE_PATH = Path("feature_store")

FEATURE_STORE_PATH.mkdir(exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build enhanced time-series features for AQI prediction.
    Includes lag features, rolling statistics, and interaction features.
    """

    df = df.sort_values(["city", "datetime"]).copy()

    # Lag features for AQI
    for lag in [1, 3, 6, 12]:
        df[f"aqi_lag_{lag}"] = df.groupby("city")["aqi"].shift(lag)

    # Rolling statistics for AQI
    for window in [3, 6, 12]:
        df[f"aqi_roll_mean_{window}"] = (
            df.groupby("city")["aqi"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

        df[f"aqi_roll_std_{window}"] = (
            df.groupby("city")["aqi"]
            .shift(1)
            .rolling(window=window)
            .std()
        )
        
        df[f"aqi_roll_min_{window}"] = (
            df.groupby("city")["aqi"]
            .shift(1)
            .rolling(window=window)
            .min()
        )
        
        df[f"aqi_roll_max_{window}"] = (
            df.groupby("city")["aqi"]
            .shift(1)
            .rolling(window=window)
            .max()
        )

    # Lag features for pollutants (important predictors)
    pollutant_cols = ["pm25", "pm10", "no2", "o3", "co"]
    for col in pollutant_cols:
        if col in df.columns:
            for lag in [1, 3]:
                df[f"{col}_lag_{lag}"] = df.groupby("city")[col].shift(lag)
            
            # Rolling mean for pollutants
            df[f"{col}_roll_mean_3"] = (
                df.groupby("city")[col]
                .shift(1)
                .rolling(window=3)
                .mean()
            )

    # Time-based features
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["day_of_month"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Cyclical encoding for time features (better for ML models)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Interaction features (important pollutant combinations)
    if "pm25" in df.columns and "pm10" in df.columns:
        df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1e-8)  # Avoid division by zero
    
    if "no2" in df.columns and "o3" in df.columns:
        df["no2_o3_interaction"] = df["no2"] * df["o3"]
    
    if "pm25" in df.columns and "o3" in df.columns:
        df["pm25_o3_interaction"] = df["pm25"] * df["o3"]

    # Rate of change features
    df["aqi_diff_1"] = df.groupby("city")["aqi"].diff(1)
    df["aqi_diff_3"] = df.groupby("city")["aqi"].diff(3)
    
    # Exponential moving average (gives more weight to recent values)
    for window in [3, 6]:
        df[f"aqi_ema_{window}"] = (
            df.groupby("city")["aqi"]
            .shift(1)
            .ewm(span=window, adjust=False)
            .mean()
        )

    # Drop rows with NaNs from lagging
    df = df.dropna()

    return df


def main():
    input_path = PROCESSED_DATA_PATH / "aqi_clean.csv"
    output_path = FEATURE_STORE_PATH / "features_v1.csv"

    if not input_path.exists():
        raise FileNotFoundError("Run ingestion first")

    df = pd.read_csv(input_path, parse_dates=["datetime"])
    features_df = build_features(df)

    features_df.to_csv(output_path, index=False)

    print("✅ Feature store built successfully")
    print(f"📊 Total features: {len(features_df.columns)}")
    print(f"📊 Total records: {len(features_df)}")
    print("\nFeature columns:")
    print(features_df.columns.tolist())


if __name__ == "__main__":
    main()
