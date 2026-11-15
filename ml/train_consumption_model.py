import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


print("SOLARPULSE AI ENERGY CONSUMPTION FORECASTER")
print("------------------------------------------")

# Load dataset
DATA_FILE = "data/energy_consumption_data.csv"
MODEL_FILE = "models/consumption_forecast_model.joblib"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("Total records:", len(df))

# Prepare timestamp features
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday
df["month"] = df["timestamp"].dt.month
df["day_of_month"] = df["timestamp"].dt.day

# Sort records chronologically
df = df.sort_values("timestamp")

# Features available when making future predictions
features = [
    "department",
    "hour",
    "weekday",
    "month",
    "day_of_month"
]

target = "energy_kwh"

X = df[features]
y = df[target]

# Chronological 80/20 split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Encode department names
preprocessor = ColumnTransformer(
    transformers=[
        (
            "department",
            OneHotEncoder(handle_unknown="ignore"),
            ["department"]
        )
    ],
    remainder="passthrough"
)

# Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42
    ))
])

print("\nTraining consumption forecasting model...")

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("------------------------------------------")
print(f"Mean Absolute Error: {mae:.4f} kWh")
print(f"R2 Score: {r2:.4f}")

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("\nMODEL TRAINED SUCCESSFULLY!")
print("Saved:", MODEL_FILE)