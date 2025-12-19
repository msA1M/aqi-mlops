import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
TIMEZONE = "UTC"

CITIES = {
    "Brasilia": {"lat": -15.7939, "lon": -47.8828},
    "Karachi": {"lat": 24.8607, "lon": 67.0011},
    "London": {"lat": 51.5074, "lon": -0.1278},
}

OUTPUT_DIR = Path("data/weather")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# FETCH WEATHER
# -----------------------
def fetch_weather_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "surface_pressure"
        ],
        "timezone": TIMEZONE,
        "past_days": 7
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# -----------------------
# TRANSFORM
# -----------------------
def transform_weather(json_data, city):
    hourly = json_data["hourly"]

    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "pressure": hourly["surface_pressure"],
        "city": city
    })

    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# -----------------------
# MAIN
# -----------------------
def main():
    print("🌦️ Fetching multi-city weather data...")
    all_dfs = []

    for city, coords in CITIES.items():
        print(f"  → {city}")
        raw = fetch_weather_data(coords["lat"], coords["lon"])
        df_city = transform_weather(raw, city)
        all_dfs.append(df_city)

    final_df = pd.concat(all_dfs, ignore_index=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"weather_multi_city_{timestamp}.csv"
    final_df.to_csv(path, index=False)

    print(f"✅ Weather data saved to {path}")


if __name__ == "__main__":
    main()
