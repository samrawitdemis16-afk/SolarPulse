import pandas as pd

# Load downloaded weather data
df = pd.read_csv("data/solar_weather_data.csv")

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# Extract useful time features
df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.day_of_year

# Handle missing values
df = df.dropna()

# Estimate solar panel electricity generation
PANEL_CAPACITY_KW = 5
SYSTEM_EFFICIENCY = 0.85

df["solar_generation_kwh"] = (
    df["solar_radiation"] / 1000
    * PANEL_CAPACITY_KW
    * SYSTEM_EFFICIENCY
)

# Save prepared dataset
df.to_csv("data/solar_training_data.csv", index=False)

print("Dataset prepared successfully!")
print("Total records:", len(df))
print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())