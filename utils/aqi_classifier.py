# utils/aqi_classifier.py

def classify_aqi(aqi):
    if aqi <= 50:
        return "Good", "Safe"
    elif aqi <= 100:
        return "Moderate", "Caution"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "Warning"
    elif aqi <= 200:
        return "Unhealthy", "Alert"
    else:
        return "Hazardous", "Critical"

