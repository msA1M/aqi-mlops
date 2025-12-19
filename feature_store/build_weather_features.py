import pandas as pd
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
INPUT_DIR = Path("data/weather")
OUTPUT_DIR = Path("feature_store/weather")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLS = ["temperature", "humidity", "wind_speed", "pressure"]
LAGS = [1, 3, 6]
ROLL_WINDOW = 3


# -----------------------
# LOAD LATEST FILE
# -----------------------
def load_latest_weather_file():
    files = sorted(INPUT_DIR.glob("weather_multi_city_*.csv"))
    if not files:
        raise FileNotFoundError("No weather data files found.")
    return pd.read_csv(files[-1], parse_dates=["datetime"])


# -----------------------
# FEATURE ENGINEERING
# -----------------------
def engineer_features(df):
    df = df.sort_values(["city", "datetime"]).copy()

    # Time-based features
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    for col in BASE_COLS:
        for lag in LAGS:
            df[f"{col}_lag_{lag}"] = df.groupby("city")[col].shift(lag)

        df[f"{col}_roll_mean_{ROLL_WINDOW}"] = (
            df.groupby("city")[col]
            .rolling(ROLL_WINDOW)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{col}_roll_std_{ROLL_WINDOW}"] = (
            df.groupby("city")[col]
            .rolling(ROLL_WINDOW)
            .std()
            .reset_index(level=0, drop=True)
        )

    df.dropna(inplace=True)
    return df


# -----------------------
# SAVE
# -----------------------
def save_features(df):
    path = OUTPUT_DIR / "weather_features.csv"
    df.to_csv(path, index=False)
    print(f"✅ Weather features saved to {path}")


# -----------------------
# MAIN
# -----------------------
def main():
    print("⚙️ Building weather features...")
    df = load_latest_weather_file()
    features = engineer_features(df)
    save_features(features)


if __name__ == "__main__":
    main()
