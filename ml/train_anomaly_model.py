import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
DATA_FILE = "data/energy_consumption_data.csv"

df = pd.read_csv(DATA_FILE)

print("SOLARPULSE AI ANOMALY DETECTOR")
print("--------------------------------")
print("Dataset loaded successfully!")
print("Total records:", len(df))
print("Anomalies:", df["is_anomaly"].sum())

# Features used by the AI model
features = [
    "department",
    "hour",
    "weekday",
    "voltage_v",
    "current_a",
    "power_factor",
    "power_w",
    "energy_kwh"
]

X = df[features]
y = df["is_anomaly"]

# Separate categorical and numerical features
categorical_features = ["department"]

numerical_features = [
    "hour",
    "weekday",
    "voltage_v",
    "current_a",
    "power_factor",
    "power_w",
    "energy_kwh"
]

# Prepare data
preprocessor = ColumnTransformer(
    transformers=[
        ("department", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),

        ("numbers", "passthrough", numerical_features)
    ]
)

# Create AI model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
print("\nTraining anomaly detection model...")

pipeline.fit(X_train, y_train)

# Evaluate
# Evaluate using a lower anomaly threshold
probabilities = pipeline.predict_proba(X_test)[:, 1]

THRESHOLD = 0.35
predictions = (probabilities >= THRESHOLD).astype(int)

print("\nMODEL EVALUATION")
print("--------------------------------")

print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Anomaly"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
joblib.dump(
    pipeline,
    "models/anomaly_detection_model.joblib"
)

print("\n--------------------------------")
print("MODEL TRAINED SUCCESSFULLY!")
print("Saved: models/anomaly_detection_model.joblib")