import requests
import pandas as pd

# Addis Ababa coordinates
LATITUDE = 9.03
LONGITUDE = 38.74

# Open-Meteo historical weather API
URL = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "cloud_cover",
        "shortwave_radiation"
    ],
    "timezone": "Africa/Addis_Ababa"
}

print("Downloading historical weather data...")

response = requests.get(URL, params=params, timeout=60)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data["hourly"])

df.rename(columns={
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "shortwave_radiation": "solar_radiation"
}, inplace=True)

df.to_csv("data/solar_weather_data.csv", index=False)

print("\nDownload successful!")
print("Total records:", len(df))
print("Saved to: data/solar_weather_data.csv")
print(df.head())