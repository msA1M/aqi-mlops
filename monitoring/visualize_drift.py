# monitoring/visualize_drift.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TRAIN_PATH = Path("feature_store/features_v1.csv")
LIVE_PATH = Path("monitoring/live_inputs.csv")
OUTPUT_DIR = Path("monitoring/drift_plots")

OUTPUT_DIR.mkdir(exist_ok=True)


FEATURES_TO_PLOT = [
    "pm25",
    "pm10",
    "no2",
    "co",
    "aqi_lag_1"
]


def plot_drift(train_df, live_df, feature):
    plt.figure()
    plt.hist(
        train_df[feature],
        bins=40,
        alpha=0.6,
        density=True,
        label="Training"
    )
    plt.hist(
        live_df[feature],
        bins=20,
        alpha=0.6,
        density=True,
        label="Live"
    )

    plt.title(f"Data Drift: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()

    output_path = OUTPUT_DIR / f"{feature}_drift.png"
    plt.savefig(output_path)
    plt.close()


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    live_df = pd.read_csv(LIVE_PATH)

    for feature in FEATURES_TO_PLOT:
        plot_drift(train_df, live_df, feature)

    print("📊 Drift visualizations saved in monitoring/drift_plots/")


if __name__ == "__main__":
    main()
