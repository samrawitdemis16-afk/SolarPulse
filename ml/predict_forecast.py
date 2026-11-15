import requests
import pandas as pd
import joblib

# Addis Ababa location
LATITUDE = 9.03
LONGITUDE = 38.74

# Load our trained AI model
model = joblib.load("models/solar_forecast_model.joblib")

# Request upcoming weather forecasts
URL = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "cloud_cover",
        "is_day"
    ],
    "forecast_days": 3,
    "timezone": "Africa/Addis_Ababa"
}

print("Fetching upcoming weather forecasts...")

response = requests.get(URL, params=params, timeout=60)
response.raise_for_status()

data = response.json()

# Prepare forecast dataset
df = pd.DataFrame(data["hourly"])

df.rename(columns={
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity"
}, inplace=True)

df["time"] = pd.to_datetime(df["time"])

# Extract time features
df["hour"] = df["time"].dt.hour
df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.day_of_year

# AI input features
features = [
    "temperature",
    "humidity",
    "cloud_cover",
    "hour",
    "month",
    "day_of_year"
]

# Predict solar radiation
df["predicted_radiation"] = model.predict(df[features])

# Remove negative predictions
df["predicted_radiation"] = (
    df["predicted_radiation"].clip(lower=0)
)

# Set nighttime generation to zero
df.loc[df["is_day"] == 0, "predicted_radiation"] = 0

# Estimate electricity generation
PANEL_CAPACITY_KW = 5
SYSTEM_EFFICIENCY = 0.85

df["predicted_generation_kwh"] = (
    df["predicted_radiation"] / 1000
    * PANEL_CAPACITY_KW
    * SYSTEM_EFFICIENCY
)

# Save forecast results
df.to_csv("data/solar_forecast.csv", index=False)

# Daily energy predictions
df["date"] = df["time"].dt.date

daily = df.groupby("date")["predicted_generation_kwh"].sum()

print("\nSOLARPULSE: THREE-DAY FORECAST")
print("--------------------------------")

for date, energy in daily.items():
    print(f"{date}: {energy:.2f} kWh")

print("\nForecast saved successfully!")
print("File: data/solar_forecast.csv")