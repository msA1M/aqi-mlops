# training/train_regression_models.py

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


FEATURES_PATH = Path("feature_store/features_v1.csv")

TARGET = "aqi"
DROP_COLS = ["city", "datetime"]
TARGET_CITY = "Brasilia"  # Model is only for Brasilia


def load_data():
    """Load and filter data for Brasilia only"""
    df = pd.read_csv(FEATURES_PATH, parse_dates=["datetime"])
    
    # Filter to Brasilia only - this is critical for accuracy
    df = df[df["city"] == TARGET_CITY].copy()
    
    if len(df) == 0:
        raise ValueError(f"No data found for {TARGET_CITY}")
    
    print(f"📊 Loaded {len(df)} records for {TARGET_CITY}")
    
    # Drop non-numeric + leakage columns
    X = df.drop(columns=[TARGET, "city", "datetime"])

    # Force numeric (safety check)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET], errors="coerce")

    # Drop rows that became NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Remove features with zero or near-zero variance (causes numerical issues)
    variance_threshold = 1e-8
    feature_variances = X.var()
    low_variance_features = feature_variances[feature_variances < variance_threshold].index.tolist()
    if low_variance_features:
        print(f"⚠️  Removing {len(low_variance_features)} low-variance features: {low_variance_features[:5]}...")
        X = X.drop(columns=low_variance_features)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Clip extreme values to prevent overflow (keep 99.9% of data)
    for col in X.columns:
        q01 = X[col].quantile(0.001)
        q99 = X[col].quantile(0.999)
        X[col] = X[col].clip(lower=q01, upper=q99)
    
    print(f"✅ After cleaning: {len(X)} records, {len(X.columns)} features")
    print(f"📈 Target statistics - Mean: {y.mean():.2f}, Std: {y.std():.2f}, Range: [{y.min():.2f}, {y.max():.2f}]")

    return train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # No shuffle for time series
    )


def log_metrics(y_true, y_pred, cv_scores=None):
    """Log comprehensive metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    
    if cv_scores is not None:
        mlflow.log_metric("cv_rmse_mean", -cv_scores.mean())
        mlflow.log_metric("cv_rmse_std", cv_scores.std())

    return rmse, mae, r2


def train_ridge(X_train, X_test, y_train, y_test):
    """Train Ridge Regression with improved hyperparameters and numerical stability"""
    with mlflow.start_run(run_name="Ridge_Regression"):
        # Use time series cross-validation
        kf = KFold(n_splits=5, shuffle=False)
        
        # Try multiple alpha values with higher regularization to prevent numerical issues
        best_alpha = 1.0
        best_cv_score = float('inf')
        
        # Suppress warnings during CV search
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            for alpha in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
                try:
                    model = Pipeline([
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("model", Ridge(alpha=alpha, solver='auto', max_iter=1000))
                    ])
                    cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, 
                                                 scoring='neg_mean_squared_error', 
                                                 error_score='raise')
                    if cv_scores.mean() < best_cv_score and not np.isnan(cv_scores.mean()):
                        best_cv_score = cv_scores.mean()
                        best_alpha = alpha
                except (ValueError, RuntimeWarning) as e:
                    continue
        
        # Build final model with best alpha
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=best_alpha, solver='auto', max_iter=1000))
        ])
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, 
                                         scoring='neg_mean_squared_error',
                                         error_score='raise')
        
        log_metrics(y_test, preds, cv_scores)
        mlflow.log_param("alpha", best_alpha)
        mlflow.log_param("model_type", "Ridge")
        mlflow.sklearn.log_model(model, "model")
        
        print(f"  Ridge - RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}, "
              f"R²: {r2_score(y_test, preds):.4f}, Alpha: {best_alpha}")


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with improved hyperparameters"""
    with mlflow.start_run(run_name="Random_Forest"):
        kf = KFold(n_splits=5, shuffle=False)
        
        # Try different configurations
        best_params = {"n_estimators": 300, "max_depth": 15, "min_samples_split": 5}
        best_cv_score = float('inf')
        
        for n_est in [200, 300, 400]:
            for max_d in [10, 15, 20]:
                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, 
                                             scoring='neg_mean_squared_error')
                if cv_scores.mean() < best_cv_score:
                    best_cv_score = cv_scores.mean()
                    best_params = {"n_estimators": n_est, "max_depth": max_d}
        
        model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, 
                                     scoring='neg_mean_squared_error')
        
        log_metrics(y_test, preds, cv_scores)
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.sklearn.log_model(model, "model")
        
        print(f"  RandomForest - RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}, "
              f"R²: {r2_score(y_test, preds):.4f}, "
              f"Params: {best_params}")


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with improved hyperparameters"""
    with mlflow.start_run(run_name="XGBoost"):
        kf = KFold(n_splits=5, shuffle=False)
        
        # Try different configurations
        best_params = {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 7}
        best_cv_score = float('inf')
        
        for lr in [0.01, 0.03, 0.05]:
            for max_d in [5, 7, 9]:
                model = XGBRegressor(
                    n_estimators=400,
                    learning_rate=lr,
                    max_depth=max_d,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    random_state=42,
                    objective="reg:squarederror",
                    n_jobs=-1
                )
                cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, 
                                             scoring='neg_mean_squared_error')
                if cv_scores.mean() < best_cv_score:
                    best_cv_score = cv_scores.mean()
                    best_params = {"learning_rate": lr, "max_depth": max_d}
        
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, 
                                     scoring='neg_mean_squared_error')
        
        log_metrics(y_test, preds, cv_scores)
        mlflow.log_params(best_params)
        mlflow.log_param("n_estimators", 400)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.sklearn.log_model(model, "model")
        
        print(f"  XGBoost - RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}, "
              f"R²: {r2_score(y_test, preds):.4f}, "
              f"Params: {best_params}")


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting Regressor"""
    with mlflow.start_run(run_name="Gradient_Boosting"):
        kf = KFold(n_splits=5, shuffle=False)
        
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, 
                                     scoring='neg_mean_squared_error')
        
        log_metrics(y_test, preds, cv_scores)
        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.sklearn.log_model(model, "model")
        
        print(f"  GradientBoosting - RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}, "
              f"R²: {r2_score(y_test, preds):.4f}")


def main():
    mlflow.set_experiment("AQI_Regression_Models")
    
    print("🚀 Training AQI models for Brasilia...")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"\n📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    print(f"\n🔧 Training models...\n")
    
    train_ridge(X_train, X_test, y_train, y_test)
    train_random_forest(X_train, X_test, y_train, y_test)
    train_xgboost(X_train, X_test, y_train, y_test)
    train_gradient_boosting(X_train, X_test, y_train, y_test)

    print("\n✅ Training complete. Check MLflow UI for detailed metrics.")


if __name__ == "__main__":
    main()
