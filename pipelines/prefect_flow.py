# pipelines/prefect_flow.py

from prefect import flow, task
import subprocess


@task(retries=2, retry_delay_seconds=10)
def ingest_data():
    subprocess.run(
        ["python", "pipelines/ingest_historical_data.py"],
        check=True
    )


@task(retries=2, retry_delay_seconds=10)
def build_features():
    subprocess.run(
        ["python", "feature_store/build_features.py"],
        check=True
    )


@task(retries=2, retry_delay_seconds=10)
def train_models():
    subprocess.run(
        ["python", "training/train_regression_models.py"],
        check=True
    )


@task(retries=2, retry_delay_seconds=10)
def register_best_model():
    subprocess.run(
        ["python", "training/select_and_register_best_model.py"],
        check=True
    )


@task(retries=1, retry_delay_seconds=5)
def run_drift_detection():
    subprocess.run(
        ["python", "monitoring/data_drift.py"],
        check=True
    )


@flow(name="AQI End-to-End MLOps Pipeline")
def aqi_mlops_pipeline():
    ingest_data()
    build_features()
    train_models()
    register_best_model()
    run_drift_detection()


if __name__ == "__main__":
    aqi_mlops_pipeline()
