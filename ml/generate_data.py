import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

with open("data/solar_data.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "hour",
        "temperature",
        "cloud_cover",
        "irradiance",
        "solar_generation"
    ])

    start = datetime(2026, 10, 1)

    for day in range(30):
        for hour in range(24):

            temperature = random.uniform(18, 32)
            cloud_cover = random.uniform(0, 100)

            if 6 <= hour <= 18:
                sunlight = math.sin(
                    math.pi * (hour - 6) / 12
                )

                irradiance = (
                    sunlight * 850 *
                    (1 - cloud_cover / 120)
                )

                irradiance = max(0, irradiance)
            else:
                irradiance = 0

            solar_generation = (
                irradiance / 1000 * 5 * 0.85
            )

            writer.writerow([
                hour,
                round(temperature, 2),
                round(cloud_cover, 2),
                round(irradiance, 2),
                round(solar_generation, 3)
            ])

print("Solar dataset created successfully!")