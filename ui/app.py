import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from utils.drift import calculate_psi

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Environmental Intelligence Dashboard",
    layout="wide"
)

st.title("🌍 Environmental Intelligence Dashboard")
st.caption("AQI & Weather Monitoring System (MLOps-enabled)")

# =======================
# SIDEBAR — CITY
# =======================
st.sidebar.header("Location")
city = st.sidebar.selectbox(
    "Select City",
    ["Brasilia", "London", "Karachi"]
)

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
    st.error("Weather API unavailable")

st.divider()

# =======================
# AQI PREDICTION
# =======================
st.subheader("🌫️ AQI Prediction")

st.info("AQI model is trained **only for Brasilia**")

if city != "Brasilia":
    st.warning("AQI prediction shown for Brasilia only")

with st.form("aqi_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        co = st.slider("CO", 0.0, 5.0, 0.4)
        co2 = st.slider("CO₂", 350.0, 600.0, 410.0)
        no2 = st.slider("NO₂", 0.0, 100.0, 20.0)
        so2 = st.slider("SO₂", 0.0, 50.0, 5.0)

    with col2:
        o3 = st.slider("O₃", 0.0, 120.0, 30.0)
        pm25 = st.slider("PM2.5", 0.0, 150.0, 45.0)
        pm10 = st.slider("PM10", 0.0, 200.0, 70.0)

    with col3:
        aqi_lag_1 = st.slider("AQI Lag 1", 0.0, 300.0, 85.0)
        aqi_lag_3 = st.slider("AQI Lag 3", 0.0, 300.0, 80.0)
        aqi_lag_6 = st.slider("AQI Lag 6", 0.0, 300.0, 75.0)
        aqi_roll_mean_3 = st.slider("AQI Rolling Mean (3)", 0.0, 300.0, 82.0)
        aqi_roll_std_3 = st.slider("AQI Rolling Std (3)", 0.0, 50.0, 4.1)

    submitted = st.form_submit_button("Predict AQI")

if submitted:
    payload = {
        "co": co,
        "co2": co2,
        "no2": no2,
        "so2": so2,
        "o3": o3,
        "pm25": pm25,
        "pm10": pm10,
        "aqi_lag_1": aqi_lag_1,
        "aqi_lag_3": aqi_lag_3,
        "aqi_lag_6": aqi_lag_6,
        "aqi_roll_mean_3": aqi_roll_mean_3,
        "aqi_roll_std_3": aqi_roll_std_3,
    }

    res = requests.post(f"{API_URL}/predict/aqi", json=payload)

    if res.status_code == 200:
        out = res.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted AQI", out["predicted_aqi"])
        c2.metric("Category", out["category"])
        c3.metric("Alert Level", out["alert_level"])
    else:
        st.error("AQI API error")

st.divider()

# =======================
# DRIFT MONITORING
# =======================
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
    df = pd.read_csv("feature_store/features_v1.csv")
    train_df = df.iloc[:-200]
    recent_df = df.iloc[-200:]

    feature = st.selectbox("Feature", ["pm25", "pm10", "no2", "o3", "aqi"])

    psi = calculate_psi(train_df[feature], recent_df[feature])

    fig, ax = plt.subplots()
    ax.hist(train_df[feature], bins=30, alpha=0.6, label="Training")
    ax.hist(recent_df[feature], bins=30, alpha=0.6, label="Recent")
    ax.legend()
    st.pyplot(fig)

    st.metric("PSI", round(psi, 4))

else:
    df = pd.read_csv("feature_store/weather/weather_features.csv")
    city_df = df[df["city"] == city]

    train_df = city_df.iloc[:-200]
    recent_df = city_df.iloc[-200:]

    feature = st.selectbox(
        "Feature",
        ["temperature", "humidity", "wind_speed", "pressure"]
    )

    psi = calculate_psi(train_df[feature], recent_df[feature])

    fig, ax = plt.subplots()
    ax.hist(train_df[feature], bins=30, alpha=0.6, label="Training")
    ax.hist(recent_df[feature], bins=30, alpha=0.6, label="Recent")
    ax.legend()
    st.pyplot(fig)

    st.metric("PSI", round(psi, 4))
