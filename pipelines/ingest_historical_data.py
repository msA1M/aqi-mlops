import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")

RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


COLUMN_MAPPING = {
    "Date": "datetime",
    "City": "city",
    "CO": "co",
    "CO2": "co2",
    "NO2": "no2",
    "SO2": "so2",
    "O3": "o3",
    "PM2.5": "pm25",
    "PM10": "pm10",
    "AQI": "aqi",
}


CANONICAL_COLUMNS = list(COLUMN_MAPPING.values())


def ingest_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns to canonical schema
    df = df.rename(columns=COLUMN_MAPPING)

    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    return df[CANONICAL_COLUMNS]


def main():
    input_csv = RAW_DATA_PATH / "aqi_historical.csv"

    if not input_csv.exists():
        raise FileNotFoundError(
            "Place your dataset at data/raw/aqi_historical.csv"
        )

    df = ingest_data(input_csv)
    df.to_csv(PROCESSED_DATA_PATH / "aqi_clean.csv", index=False)

    print("✅ Historical data ingested & normalized")
    print("Schema:", df.columns.tolist())
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
