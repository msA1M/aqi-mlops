import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import mlflow
import shap
from utils.drift import calculate_psi
from pathlib import Path

# Use environment variable if set (for docker-compose), otherwise localhost (for single container/HF Spaces)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Environmental Intelligence Dashboard",
    layout="wide"
)

st.title("🌍 Environmental Intelligence Dashboard")
st.caption("AQI & Weather Monitoring System (MLOps-enabled)")

# =======================
# SIDEBAR — CITY & NAVIGATION
# =======================
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Forecasts", "EDA & Analysis", "Model Explainability", "Data Drift Monitoring"]
)

st.sidebar.header("Location")
city = st.sidebar.selectbox(
    "Select City",
    ["Brasilia", "London", "Karachi"]
)

# =======================
# PAGE: FORECASTS
# =======================
if page == "Forecasts":
    # =======================
    # WEATHER PREDICTION
    # =======================
    st.subheader("🌦️ Weather Forecast")
    
    resp = requests.get(
        f"{API_URL}/predict/weather",
        params={"city": city}
    )
    
    if resp.status_code == 200:
        preds = resp.json()["predictions"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature (°C)", round(preds["temperature"], 2))
        c2.metric("Humidity (%)", round(preds["humidity"], 2))
        c3.metric("Wind Speed (m/s)", round(preds["wind_speed"], 2))
        c4.metric("Pressure (hPa)", round(preds["pressure"], 2))
    else:
        st.error(f"Weather API unavailable for {city}. Please ensure weather data is available.")
    
    st.divider()
    
    # =======================
    # AQI FORECASTING
    # =======================
    st.subheader("🌫️ AQI Forecast")
    
    st.info("AQI forecasting is available **for Brasilia**. The model predicts future AQI values using historical data and time-series features.")
    
    forecast_city = "Brasilia" if city != "Brasilia" else city
    if city != "Brasilia":
        st.warning(f"Showing AQI forecast for Brasilia (model trained for Brasilia only)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        forecast_steps = st.slider("Forecast Steps (hours ahead)", 1, 48, 24)
    with col2:
        if st.button("🔮 Forecast AQI"):
            with st.spinner("Generating AQI forecast..."):
                try:
                    resp = requests.get(
                        f"{API_URL}/predict/aqi/forecast",
                        params={"city": forecast_city, "steps": forecast_steps}
                    )
                    
                    if resp.status_code == 200:
                        forecast_data = resp.json()
                        current_aqi = forecast_data["current_aqi"]
                        forecasts = forecast_data["forecasts"]
                        
                        # Display current AQI
                        st.metric("Current AQI", round(current_aqi, 2), 
                                 forecast_data["current_category"])
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame(forecasts)
                        
                        # Plot forecast
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(forecast_df["step"], forecast_df["predicted_aqi"], 
                               marker='o', linewidth=2, label="Forecasted AQI")
                        ax.axhline(y=40, color='green', linestyle='--', alpha=0.5, label='Good')
                        ax.axhline(y=80, color='yellow', linestyle='--', alpha=0.5, label='Moderate')
                        ax.axhline(y=120, color='orange', linestyle='--', alpha=0.5, label='Unhealthy (Sensitive)')
                        ax.axhline(y=160, color='red', linestyle='--', alpha=0.5, label='Unhealthy')
                        ax.set_xlabel("Hours Ahead")
                        ax.set_ylabel("AQI")
                        ax.set_title(f"AQI Forecast for {forecast_city}")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Display forecast table
                        st.subheader("Forecast Details")
                        st.dataframe(forecast_df.style.background_gradient(subset=["predicted_aqi"], 
                                                                          cmap="RdYlGn_r"))
                        
                        # Alert summary
                        alert_counts = forecast_df["alert_level"].value_counts()
                        if len(alert_counts) > 0:
                            st.subheader("Alert Summary")
                            alert_cols = st.columns(len(alert_counts))
                            for idx, (level, count) in enumerate(alert_counts.items()):
                                with alert_cols[idx]:
                                    st.metric(level, count)
                    else:
                        st.error(f"AQI Forecast API error: {resp.text}")
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")

# =======================
# PAGE: EDA & ANALYSIS
# =======================
elif page == "EDA & Analysis":
    st.subheader("📊 Exploratory Data Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["AQI Analysis (Brasilia)", "Weather Analysis (Multi-city)"]
    )
    
    if analysis_type == "AQI Analysis (Brasilia)":
        try:
            df = pd.read_csv("feature_store/features_v1.csv", parse_dates=["datetime"])
            brasilia_df = df[df["city"] == "Brasilia"].copy()
            
            if len(brasilia_df) == 0:
                st.warning("No data available for Brasilia")
            else:
                st.info(f"Analyzing {len(brasilia_df)} records for Brasilia")
                
                # Time series plot
                st.subheader("AQI Time Series")
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(brasilia_df["datetime"], brasilia_df["aqi"], linewidth=1, alpha=0.7)
                ax.set_xlabel("Date")
                ax.set_ylabel("AQI")
                ax.set_title("AQI Over Time - Brasilia")
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Distribution
                st.subheader("AQI Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(brasilia_df["aqi"], bins=50, edgecolor='black', alpha=0.7)
                    ax.set_xlabel("AQI")
                    ax.set_ylabel("Frequency")
                    ax.set_title("AQI Distribution")
                    ax.axvline(brasilia_df["aqi"].mean(), color='r', linestyle='--', 
                              label=f'Mean: {brasilia_df["aqi"].mean():.2f}')
                    ax.legend()
                    st.pyplot(fig)
                
                with col2:
                    st.metric("Mean AQI", round(brasilia_df["aqi"].mean(), 2))
                    st.metric("Median AQI", round(brasilia_df["aqi"].median(), 2))
                    st.metric("Std Dev", round(brasilia_df["aqi"].std(), 2))
                    st.metric("Min AQI", round(brasilia_df["aqi"].min(), 2))
                    st.metric("Max AQI", round(brasilia_df["aqi"].max(), 2))
                
                # Correlation matrix
                st.subheader("Feature Correlations")
                numeric_cols = ["co", "co2", "no2", "so2", "o3", "pm25", "pm10", "aqi"]
                corr_df = brasilia_df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, ax=ax)
                st.pyplot(fig)
                
                # Feature distributions
                st.subheader("Pollutant Distributions")
                pollutants = ["pm25", "pm10", "no2", "o3", "co"]
                n_cols = 3
                n_rows = (len(pollutants) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
                
                for idx, pollutant in enumerate(pollutants):
                    if pollutant in brasilia_df.columns:
                        axes[idx].hist(brasilia_df[pollutant].dropna(), bins=30, 
                                      edgecolor='black', alpha=0.7)
                        axes[idx].set_title(f"{pollutant} Distribution")
                        axes[idx].set_xlabel(pollutant)
                        axes[idx].set_ylabel("Frequency")
                
                # Hide extra subplots
                for idx in range(len(pollutants), len(axes)):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
        except FileNotFoundError:
            st.error("AQI feature store not found. Please run feature engineering first.")
        except Exception as e:
            st.error(f"Error loading AQI data: {str(e)}")
    
    else:  # Weather Analysis
        try:
            df = pd.read_csv("feature_store/weather/weather_features.csv", parse_dates=["datetime"])
            city_df = df[df["city"] == city].copy()
            
            if len(city_df) == 0:
                st.warning(f"No weather data available for {city}")
            else:
                st.info(f"Analyzing {len(city_df)} records for {city}")
                
                # Time series for all weather variables
                st.subheader("Weather Variables Time Series")
                weather_vars = ["temperature", "humidity", "wind_speed", "pressure"]
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                axes = axes.flatten()
                
                for idx, var in enumerate(weather_vars):
                    axes[idx].plot(city_df["datetime"], city_df[var], linewidth=1, alpha=0.7)
                    axes[idx].set_title(f"{var.capitalize()} Over Time")
                    axes[idx].set_xlabel("Date")
                    axes[idx].set_ylabel(var)
                    axes[idx].grid(True, alpha=0.3)
                    plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistics
                st.subheader("Weather Statistics")
                stats_cols = st.columns(4)
                for idx, var in enumerate(weather_vars):
                    with stats_cols[idx]:
                        st.metric(f"{var.capitalize()} Mean", round(city_df[var].mean(), 2))
                        st.metric(f"{var.capitalize()} Std", round(city_df[var].std(), 2))
                        st.metric(f"{var.capitalize()} Min", round(city_df[var].min(), 2))
                        st.metric(f"{var.capitalize()} Max", round(city_df[var].max(), 2))
                
                # Correlation
                st.subheader("Weather Variable Correlations")
                corr_df = city_df[weather_vars].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, ax=ax)
                st.pyplot(fig)
                
        except FileNotFoundError:
            st.error("Weather feature store not found. Please run weather feature engineering first.")
        except Exception as e:
            st.error(f"Error loading weather data: {str(e)}")

# =======================
# PAGE: MODEL EXPLAINABILITY
# =======================
elif page == "Model Explainability":
    st.subheader("🔍 Model Explainability (SHAP)")
    
    st.info("This section shows feature importance and model explanations using SHAP values.")
    
    try:
        # Set MLflow tracking URI
        if os.path.exists("/app/mlruns"):
            mlflow.set_tracking_uri("file:/app/mlruns")
            base_path = "/app/mlruns"
        else:
            mlflow.set_tracking_uri("file:./mlruns")
            base_path = "./mlruns"
        
        # Load model with fallback mechanism (same as API)
        MODEL_NAME = "AQI_Predictor"
        FALLBACK_MODEL_ID = "m-168880ff890a4a9bb3f7b210da76d290"
        FALLBACK_EXPERIMENT_ID = "481652201472430433"
        
        with st.spinner("Loading model and computing SHAP values..."):
            try:
                # Try registry first, but fallback silently if it fails
                model = None
                try:
                    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
                except Exception:
                    # Registry failed, use fallback to direct path
                    pass
                
                # If registry load failed, use direct path
                if model is None:
                    model_artifacts_path = f"{base_path}/{FALLBACK_EXPERIMENT_ID}/models/{FALLBACK_MODEL_ID}/artifacts"
                    if not os.path.exists(model_artifacts_path):
                        raise FileNotFoundError(f"Model artifacts not found at {model_artifacts_path}. Please ensure models are available.")
                    model = mlflow.pyfunc.load_model(model_artifacts_path)
                
                # Load feature data
                df = pd.read_csv("feature_store/features_v1.csv")
                brasilia_df = df[df["city"] == "Brasilia"].copy()
                
                if len(brasilia_df) == 0:
                    st.warning("No data available for Brasilia")
                else:
                    # Prepare features
                    feature_cols = ["co", "co2", "no2", "so2", "o3", "pm25", "pm10",
                                   "aqi_lag_1", "aqi_lag_3", "aqi_lag_6",
                                   "aqi_roll_mean_3", "aqi_roll_std_3"]
                    X = brasilia_df[feature_cols].dropna()
                    
                    # Sample for faster computation
                    sample_size = min(200, len(X))
                    X_sample = X.sample(sample_size, random_state=42)
                    
                    # Create SHAP explainer
                    try:
                        explainer = shap.Explainer(model.predict, X_sample)
                    except:
                        explainer = shap.KernelExplainer(model.predict, X_sample)
                    
                    shap_values = explainer(X_sample)
                    
                    # Summary plot
                    st.subheader("SHAP Summary Plot")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
                    st.pyplot(fig)
                    
                    # Feature importance bar plot
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'feature': X_sample.columns,
                        'importance': np.abs(shap_values.values).mean(0)
                    }).sort_values('importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(feature_importance['feature'], feature_importance['importance'])
                    ax.set_xlabel('Mean |SHAP Value|')
                    ax.set_title('Feature Importance')
                    plt.gca().invert_yaxis()
                    st.pyplot(fig)
                    
                    # Display feature importance table
                    st.dataframe(feature_importance)
                    
                    # Waterfall plot for a single prediction (if supported)
                    try:
                        st.subheader("Single Prediction Explanation")
                        sample_idx = st.slider("Select sample index", 0, len(X_sample)-1, 0)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        if hasattr(shap_values, 'base_values'):
                            base_val = shap_values.base_values[sample_idx] if isinstance(shap_values.base_values, (list, np.ndarray)) else shap_values.base_values
                        else:
                            base_val = 0
                        
                        shap.waterfall_plot(shap.Explanation(
                            values=shap_values.values[sample_idx],
                            base_values=base_val,
                            data=X_sample.iloc[sample_idx].values,
                            feature_names=X_sample.columns
                        ), show=False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"Waterfall plot not available: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error loading model or computing SHAP: {str(e)}")
                st.info("Make sure the model is registered in MLflow and SHAP is installed.")
                import traceback
                st.code(traceback.format_exc())
    
    except FileNotFoundError:
        st.error("Feature store not found. Please run feature engineering first.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# =======================
# PAGE: DATA DRIFT MONITORING
# =======================
elif page == "Data Drift Monitoring":
    st.subheader("📉 Data Drift Monitoring")
    
    st.info(
        "⚙️ Model retraining is handled automatically by Prefect. "
        "This dashboard is for **monitoring only**."
    )
    
    drift_type = st.selectbox(
        "Select Drift Type",
        ["AQI (Brasilia)", "Weather (Multi-city)"]
    )
    
    if drift_type == "AQI (Brasilia)":
        try:
            df = pd.read_csv("feature_store/features_v1.csv", parse_dates=["datetime"])
            brasilia_df = df[df["city"] == "Brasilia"].copy()
            
            if len(brasilia_df) < 200:
                st.warning(f"Insufficient data for drift analysis. Need at least 200 records (currently: {len(brasilia_df)}).")
            else:
                train_df = brasilia_df.iloc[:-200]
                recent_df = brasilia_df.iloc[-200:]
                
                feature = st.selectbox("Feature", ["pm25", "pm10", "no2", "o3", "aqi", "co"])
                
                if feature in train_df.columns and feature in recent_df.columns:
                    psi = calculate_psi(train_df[feature].dropna(), recent_df[feature].dropna())
                    
                    # Enhanced visualization
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(train_df[feature].dropna(), bins=30, alpha=0.6, 
                               label="Training Data", color='blue', edgecolor='black')
                        ax.hist(recent_df[feature].dropna(), bins=30, alpha=0.6, 
                               label="Recent Data", color='red', edgecolor='black')
                        ax.set_xlabel(feature)
                        ax.set_ylabel("Frequency")
                        ax.set_title(f"Data Distribution Comparison - {feature}")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col2:
                        # PSI interpretation
                        if psi < 0.1:
                            drift_status = "✅ No Drift"
                            drift_color = "green"
                        elif psi < 0.2:
                            drift_status = "⚠️ Minor Drift"
                            drift_color = "orange"
                        else:
                            drift_status = "🚨 Significant Drift"
                            drift_color = "red"
                        
                        st.metric("PSI Score", round(psi, 4))
                        st.markdown(f"<h3 style='color: {drift_color}'>{drift_status}</h3>", 
                                   unsafe_allow_html=True)
                        
                        # Statistics comparison
                        st.subheader("Statistics Comparison")
                        st.write("**Training Data:**")
                        st.write(f"Mean: {train_df[feature].mean():.2f}")
                        st.write(f"Std: {train_df[feature].std():.2f}")
                        st.write("**Recent Data:**")
                        st.write(f"Mean: {recent_df[feature].mean():.2f}")
                        st.write(f"Std: {recent_df[feature].std():.2f}")
                
        except FileNotFoundError:
            st.error("AQI feature store not found.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    else:  # Weather Drift
        try:
            df = pd.read_csv("feature_store/weather/weather_features.csv", parse_dates=["datetime"])
            city_df = df[df["city"] == city].copy()
            
            if len(city_df) < 200:
                st.warning(f"Insufficient data for {city}. Need at least 200 records (currently: {len(city_df)}).")
            else:
                train_df = city_df.iloc[:-200]
                recent_df = city_df.iloc[-200:]
                
                feature = st.selectbox(
                    "Feature",
                    ["temperature", "humidity", "wind_speed", "pressure"]
                )
                
                psi = calculate_psi(train_df[feature].dropna(), recent_df[feature].dropna())
                
                # Enhanced visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(train_df[feature].dropna(), bins=30, alpha=0.6, 
                           label="Training Data", color='blue', edgecolor='black')
                    ax.hist(recent_df[feature].dropna(), bins=30, alpha=0.6, 
                           label="Recent Data", color='red', edgecolor='black')
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Data Distribution Comparison - {feature} ({city})")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    # PSI interpretation
                    if psi < 0.1:
                        drift_status = "✅ No Drift"
                        drift_color = "green"
                    elif psi < 0.2:
                        drift_status = "⚠️ Minor Drift"
                        drift_color = "orange"
                    else:
                        drift_status = "🚨 Significant Drift"
                        drift_color = "red"
                    
                    st.metric("PSI Score", round(psi, 4))
                    st.markdown(f"<h3 style='color: {drift_color}'>{drift_status}</h3>", 
                               unsafe_allow_html=True)
                    
                    # Statistics comparison
                    st.subheader("Statistics Comparison")
                    st.write("**Training Data:**")
                    st.write(f"Mean: {train_df[feature].mean():.2f}")
                    st.write(f"Std: {train_df[feature].std():.2f}")
                    st.write("**Recent Data:**")
                    st.write(f"Mean: {recent_df[feature].mean():.2f}")
                    st.write(f"Std: {recent_df[feature].std():.2f}")
        
        except FileNotFoundError:
            st.error("Weather feature store not found.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
