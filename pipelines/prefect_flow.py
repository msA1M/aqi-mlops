from prefect import flow, task
import subprocess
import pandas as pd
from pathlib import Path

@task
def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

@task
def retrain_if_needed():
    signal_path = Path("monitoring/retrain_signal.csv")

    if not signal_path.exists():
        print("✅ No retraining needed.")
        return

    df = pd.read_csv(signal_path)

    print("🔁 Retraining models due to detected drift...")

    if "AQI" in df["domain"].values:
        print("→ Retraining AQI models")
        run("python training/train_regression_models.py")

    if "Weather" in df["domain"].values:
        print("→ Retraining Weather models")
        run("python training/train_weather_models.py")


@flow(name="Environmental Intelligence Pipeline")
def full_pipeline():
    # Ingestion
    run("python pipelines/ingest_historical_data.py")
    run("python pipelines/ingest_weather_data.py")

    # Training
    run("python training/train_regression_models.py")
    run("python training/train_weather_models.py")

    # Drift
    run("python monitoring/data_drift.py")
    run("python monitoring/retrain_decision.py")

    # Conditional retraining
    retrain_if_needed()


if __name__ == "__main__":
    full_pipeline()
