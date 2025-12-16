# monitoring/data_drift.py

import pandas as pd
from pathlib import Path

FEATURE_STORE_PATH = Path("feature_store/features_v1.csv")
LIVE_DATA_PATH = Path("monitoring/live_inputs.csv")

DRIFT_THRESHOLD = 0.25  # 25% change


def compute_drift(train_series, live_series):
    train_mean = train_series.mean()
    live_mean = live_series.mean()

    if train_mean == 0:
        return 0

    return abs(live_mean - train_mean) / abs(train_mean)


def main():
    train_df = pd.read_csv(FEATURE_STORE_PATH)

    if not LIVE_DATA_PATH.exists():
        raise FileNotFoundError(
            "No live data found. Create monitoring/live_inputs.csv"
        )

    live_df = pd.read_csv(LIVE_DATA_PATH)

    feature_cols = [
        c for c in train_df.columns
        if c not in ["aqi", "city", "datetime"]
    ]

    drift_report = {}

    for col in feature_cols:
        drift_score = compute_drift(
            train_df[col],
            live_df[col]
        )

        drift_report[col] = {
            "drift_score": round(drift_score, 3),
            "drift_detected": drift_score > DRIFT_THRESHOLD
        }

    drift_df = pd.DataFrame(drift_report).T
    drift_df.to_csv("monitoring/drift_report.csv")

    print("📈 Drift report generated")
    print(drift_df[drift_df["drift_detected"]])


if __name__ == "__main__":
    main()
