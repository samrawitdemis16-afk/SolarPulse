import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Generate 30 days of hourly data
timestamps = pd.date_range(
    end=pd.Timestamp.now().floor("h"),
    periods=30 * 24,
    freq="h"
)

departments = {
    "Computer Lab": (1200, 4500),
    "Classrooms": (400, 1800),
    "Lighting": (250, 1200),
    "Other": (300, 1500)
}

records = []

for timestamp in timestamps:

    hour = timestamp.hour
    weekday = timestamp.weekday()

    # Determine normal usage pattern
    if 7 <= hour <= 17 and weekday < 5:
        usage_factor = np.random.uniform(0.65, 1.0)
        low_activity = False

    elif 18 <= hour <= 21:
        usage_factor = np.random.uniform(0.25, 0.55)
        low_activity = False

    else:
        usage_factor = np.random.uniform(0.03, 0.15)
        low_activity = True

    for department, power_range in departments.items():

        base_power = np.random.uniform(*power_range)

        expected_power = base_power * usage_factor

        # Decide whether an unusual event occurs
        is_anomaly = np.random.random() < 0.05

        anomaly_type = "normal"

        if is_anomaly:

            # Night-time anomalies only occur during low activity
            if low_activity:
                anomaly_type = np.random.choice([
                    "high_usage",
                    "night_usage"
                ])
            else:
                anomaly_type = "high_usage"

            if anomaly_type == "high_usage":

                # Unusually high electricity consumption
                expected_power = (
                    base_power
                    * usage_factor
                    * np.random.uniform(2.5, 3.0)
                )

            elif anomaly_type == "night_usage":

                # Unusually high usage during low-activity hours
                expected_power = (
                    base_power
                    * np.random.uniform(0.75, 1.0)
                )

        # Simulated electrical readings
        voltage = np.random.uniform(220, 240)

        power_factor = np.random.uniform(0.85, 0.98)

        current = expected_power / (voltage * power_factor)

        power = voltage * current * power_factor

        energy_kwh = power / 1000

        records.append({

            "timestamp": timestamp,

            "department": department,

            "hour": hour,

            "weekday": weekday,

            "voltage_v": round(voltage, 2),

            "current_a": round(current, 3),

            "power_factor": round(power_factor, 3),

            "power_w": round(power, 2),

            "energy_kwh": round(energy_kwh, 4),

            "anomaly_type": anomaly_type,

            "is_anomaly": int(is_anomaly)

        })


# Create dataset
df = pd.DataFrame(records)

os.makedirs("data", exist_ok=True)

file_path = "data/energy_consumption_data.csv"

df.to_csv(file_path, index=False)


# Display results
print("SOLARPULSE ENERGY DATA GENERATED SUCCESSFULLY!")

print("----------------------------------------------")

print("Total records:", len(df))

print("Departments:", df["department"].nunique())

print("Anomalies:", df["is_anomaly"].sum())

print("Normal records:", (df["is_anomaly"] == 0).sum())

print("\nAnomaly types:")

print(df["anomaly_type"].value_counts())

print("\nSaved to:", file_path)

print("\nDataset preview:")

print(df.head())