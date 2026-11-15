import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/solar_training_data.csv")

features = [
    "temperature",
    "humidity",
    "cloud_cover",
    "hour",
    "month",
    "day_of_year"
]

split = int(len(df) * 0.8)

test = df.iloc[split:].copy()

# Keep daylight records only
test = test[test["solar_radiation"] > 0]

X_test = test[features]
y_test = test["solar_radiation"]

model = joblib.load("models/solar_forecast_model.joblib")

predictions = model.predict(X_test)

print("DAYLIGHT MODEL EVALUATION")
print("-------------------------")
print("Daylight records:", len(test))
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))