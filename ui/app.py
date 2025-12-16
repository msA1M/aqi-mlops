# ui/app.py

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="AQI Monitoring Dashboard", layout="centered")

st.title("🌍 Air Quality Monitoring Dashboard")
st.markdown("Predict AQI, risk category, and alerts using a production ML model.")

st.divider()

st.subheader("Enter Air Quality Features")

with st.form("aqi_form"):
    col1, col2 = st.columns(2)

    with col1:
        co = st.number_input("CO", 0.0, 5.0, 0.3)
        co2 = st.number_input("CO₂", 300.0, 600.0, 420.0)
        no2 = st.number_input("NO₂", 0.0, 200.0, 25.0)
        so2 = st.number_input("SO₂", 0.0, 100.0, 10.0)
        o3 = st.number_input("O₃", 0.0, 200.0, 40.0)
        pm25 = st.number_input("PM2.5", 0.0, 300.0, 35.0)

    with col2:
        pm10 = st.number_input("PM10", 0.0, 400.0, 70.0)
        aqi_lag_1 = st.number_input("AQI (t-1)", 0.0, 300.0, 85.0)
        aqi_lag_3 = st.number_input("AQI (t-3)", 0.0, 300.0, 80.0)
        aqi_lag_6 = st.number_input("AQI (t-6)", 0.0, 300.0, 78.0)
        aqi_roll_mean_3 = st.number_input("AQI Rolling Mean (3)", 0.0, 300.0, 81.0)
        aqi_roll_std_3 = st.number_input("AQI Rolling Std (3)", 0.0, 50.0, 3.0)

    submitted = st.form_submit_button("🔍 Predict AQI")

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

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        st.divider()
        st.subheader("📊 Prediction Result")

        st.metric("Predicted AQI", result["predicted_aqi"])

        category = result["category"]
        alert = result["alert_level"]

        if alert in ["Safe"]:
            st.success(f"Category: {category} | Alert: {alert}")
        elif alert in ["Caution", "Warning"]:
            st.warning(f"Category: {category} | Alert: {alert}")
        else:
            st.error(f"Category: {category} | Alert: {alert}")

    except Exception as e:
        st.error(f"API Error: {e}")
