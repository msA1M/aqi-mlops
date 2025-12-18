# training/explain_model_shap.py

import mlflow
import shap
import pandas as pd
import matplotlib.pyplot as plt

MODEL_NAME = "AQI_Predictor"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

FEATURES_PATH = "feature_store/features_v1.csv"


def main():
    # Load model from MLflow Registry
    model = mlflow.pyfunc.load_model(MODEL_URI)

    # Load feature data
    df = pd.read_csv(FEATURES_PATH)

    X = df.drop(columns=["aqi", "city", "datetime"])

    # Sample data for SHAP (keep it light)
    X_sample = X.sample(200, random_state=42)

    # SHAP explainer (tree or general fallback)
    try:
        explainer = shap.Explainer(model.predict, X_sample)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, X_sample)

    shap_values = explainer(X_sample)

    # Global explanation
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")

    print("✅ SHAP summary plot saved as shap_summary.png")


if __name__ == "__main__":
    main()
