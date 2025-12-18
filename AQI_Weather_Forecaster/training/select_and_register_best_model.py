# training/select_and_register_best_model.py

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "AQI_Regression_Models"
MODEL_NAME = "AQI_Predictor"


def main():
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError("Experiment not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )

    best_run = runs[0]
    best_rmse = best_run.data.metrics["rmse"]
    best_run_id = best_run.info.run_id

    print(f"🏆 Best model run: {best_run_id}")
    print(f"RMSE: {best_rmse}")

    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(
        f"✅ Model registered as version {result.version}. "
        f"Using latest version for inference."
    )


    print(f"✅ Model registered as Production: {MODEL_NAME} (v{result.version})")


if __name__ == "__main__":
    main()
