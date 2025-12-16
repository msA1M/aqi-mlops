# training/train_regression_models.py

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


FEATURES_PATH = Path("feature_store/features_v1.csv")

TARGET = "aqi"
DROP_COLS = ["city", "datetime"]


def load_data():
    df = pd.read_csv(FEATURES_PATH)

    # Drop non-numeric + leakage columns
    X = df.drop(columns=[TARGET, "city", "datetime"])

    # Force numeric (safety check)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET], errors="coerce")

    # Drop rows that became NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )



def log_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    return rmse, mae, r2


def train_ridge(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="Ridge_Regression"):
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=1.0))
        ])

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        log_metrics(y_test, preds)
        mlflow.sklearn.log_model(model, "model")


def train_random_forest(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="Random_Forest"):
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        log_metrics(y_test, preds)
        mlflow.sklearn.log_model(model, "model")


def train_xgboost(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="XGBoost"):
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror"
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        log_metrics(y_test, preds)
        mlflow.sklearn.log_model(model, "model")


def main():
    mlflow.set_experiment("AQI_Regression_Models")

    X_train, X_test, y_train, y_test = load_data()

    train_ridge(X_train, X_test, y_train, y_test)
    train_random_forest(X_train, X_test, y_train, y_test)
    train_xgboost(X_train, X_test, y_train, y_test)

    print("✅ Training complete. Check MLflow UI.")


if __name__ == "__main__":
    main()
