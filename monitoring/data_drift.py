import pandas as pd
from pathlib import Path
from utils.drift import calculate_psi

# -----------------------
# PATHS
# -----------------------
AQI_FEATURES_PATH = "feature_store/features_v1.csv"
WEATHER_FEATURES_PATH = "feature_store/weather/weather_features.csv"
OUTPUT_PATH = "monitoring/drift_report.csv"

# -----------------------
# CONFIG
# -----------------------
AQI_FEATURES = ["pm25", "pm10", "no2", "o3"]
WEATHER_FEATURES = ["temperature", "humidity", "wind_speed", "pressure"]

# -----------------------
# AQI DRIFT
# -----------------------
def compute_aqi_drift():
    df = pd.read_csv(AQI_FEATURES_PATH)

    records = []

    for feature in AQI_FEATURES:
        train = df[feature].iloc[:-200]
        live = df[feature].iloc[-200:]

        psi = calculate_psi(train, live)

        records.append({
            "domain": "AQI",
            "city": "Brasilia",
            "feature": feature,
            "psi": round(psi, 4)
        })

    return records


# -----------------------
# WEATHER DRIFT
# -----------------------
def compute_weather_drift():
    df = pd.read_csv(WEATHER_FEATURES_PATH)

    records = []

    for city in df["city"].unique():
        city_df = df[df["city"] == city]

        if len(city_df) < 150:
            continue

        for feature in WEATHER_FEATURES:
            train = city_df[feature].iloc[:-100]
            live = city_df[feature].iloc[-100:]

            psi = calculate_psi(train, live)

            records.append({
                "domain": "Weather",
                "city": city,
                "feature": feature,
                "psi": round(psi, 4)
            })

    return records


# -----------------------
# MAIN
# -----------------------
def main():
    print("📉 Running data drift detection...")

    drift_records = []
    drift_records.extend(compute_aqi_drift())
    drift_records.extend(compute_weather_drift())

    drift_df = pd.DataFrame(drift_records)
    drift_df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Drift report saved to {OUTPUT_PATH}")
    print(drift_df)


if __name__ == "__main__":
    main()
