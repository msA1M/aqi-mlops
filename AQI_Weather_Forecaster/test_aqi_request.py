import requests
import json

# Valid payload with all required fields
payload = {
    "co": 0.4,
    "co2": 410.0,
    "no2": 20.0,
    "so2": 5.0,
    "o3": 30.0,
    "pm25": 45.0,
    "pm10": 70.0,
    "aqi_lag_1": 85.0,
    "aqi_lag_3": 80.0,
    "aqi_lag_6": 75.0,
    "aqi_roll_mean_3": 82.0,
    "aqi_roll_std_3": 4.1
}

response = requests.post("http://127.0.0.1:8000/predict/aqi", json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
