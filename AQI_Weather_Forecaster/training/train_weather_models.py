import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------
# CONFIG
# -----------------------
mlflow.set_tracking_uri("file:./mlruns")
FEATURE_PATH = Path("feature_store/weather/weather_features.csv")
EXPERIMENT_NAME = "Weather_Forecasting"

TARGETS = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure"
]

DROP_COLS = ["datetime", "city"]


# -----------------------
# METRICS
# -----------------------

def log_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)



# -----------------------
# TRAIN ONE MODEL
# -----------------------
def train_model(df, target, model_name, model):
    X = df.drop(columns=DROP_COLS + TARGETS)
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Remove low variance features
    variance_threshold = 1e-8
    feature_variances = X.var()
    low_variance_features = feature_variances[feature_variances < variance_threshold].index.tolist()
    if low_variance_features:
        X = X.drop(columns=low_variance_features)
    
    # Clip extreme values
    for col in X.columns:
        q01 = X[col].quantile(0.001)
        q99 = X[col].quantile(0.999)
        X[col] = X[col].clip(lower=q01, upper=q99)
    
    X = X.dropna()
    y = df.loc[X.index, target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    with mlflow.start_run(run_name=f"{model_name}_{target}"):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        log_metrics(y_test, preds)

        mlflow.log_param("target", target)
        mlflow.log_param("model", model_name)

        mlflow.sklearn.log_model(model, artifact_path="model")


# -----------------------
# MAIN
# -----------------------
def main():
    mlflow.set_experiment("Weather_Forecasting")
    print("🌦️ Training weather models...")
    df = pd.read_csv(FEATURE_PATH, parse_dates=["datetime"])

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Baseline: Ridge with better numerical stability
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, solver='auto', max_iter=1000))
    ])

    # Improved: Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    for target in TARGETS:
        print(f"→ Training Ridge for {target}")
        train_model(df, target, "Ridge", ridge)

        print(f"→ Training RandomForest for {target}")
        train_model(df, target, "RandomForest", rf)

    print("✅ Weather model training complete.")


if __name__ == "__main__":
    main()
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Weather_Forecasting"
MODEL_NAME_PREFIX = "Weather_Model"

TARGETS = ["temperature", "humidity", "wind_speed", "pressure"]

def main():
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    if exp is None:
        raise ValueError("Weather_Forecasting experiment not found")

    for target in TARGETS:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"params.target = '{target}'",
            order_by=["metrics.rmse ASC"],
            max_results=1
        )

        best_run = runs[0]
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        model_name = f"{MODEL_NAME_PREFIX}_{target}"

        print(f"🏆 Best {target} model → {run_id}")

        mlflow.register_model(model_uri, model_name)

    print("✅ Weather models registered successfully.")

if __name__ == "__main__":
    main()
