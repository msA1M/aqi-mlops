# feature_store/build_features.py

import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed")
FEATURE_STORE_PATH = Path("feature_store")

FEATURE_STORE_PATH.mkdir(exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build time-series features for AQI prediction.
    """

    df = df.sort_values(["city", "datetime"])

    # Lag features
    for lag in [1, 3, 6]:
        df[f"aqi_lag_{lag}"] = df.groupby("city")["aqi"].shift(lag)

    # Rolling statistics
    df["aqi_roll_mean_3"] = (
        df.groupby("city")["aqi"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["aqi_roll_std_3"] = (
        df.groupby("city")["aqi"]
        .shift(1)
        .rolling(window=3)
        .std()
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
    print("Feature columns:")
    print(features_df.columns.tolist())


if __name__ == "__main__":
    main()
