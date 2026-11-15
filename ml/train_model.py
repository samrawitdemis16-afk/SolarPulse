import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load prepared dataset
df = pd.read_csv("data/solar_training_data.csv")

# Features used by the AI
features = [
    "temperature",
    "humidity",
    "cloud_cover",
    "hour",
    "month",
    "day_of_year"
]

# Target: solar radiation
X = df[features]
y = df["solar_radiation"]

# Chronological split: first 80% for training
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Create AI model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("Training SolarPulse AI model...")

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nTraining completed!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 3))

# Save trained model
joblib.dump(model, "models/solar_forecast_model.joblib")

print("\nModel saved successfully!")