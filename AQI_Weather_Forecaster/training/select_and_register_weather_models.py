import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Weather_Forecasting"
MODEL_PREFIX = "Weather"
TARGETS = ["temperature", "humidity", "wind_speed", "pressure"]

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("Weather_Forecasting experiment not found")

    for target in TARGETS:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.target = '{target}'",
            order_by=["metrics.rmse ASC"],
            max_results=1
        )

        if not runs:
            print(f"⚠️ No runs found for {target}")
            continue

        best_run = runs[0]
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_name = f"{MODEL_PREFIX}_{target}"

        print(f"🏆 Best {target} model → Run {run_id}")

        mlflow.register_model(model_uri, model_name)

    print("✅ All weather models registered")

if __name__ == "__main__":
    main()
