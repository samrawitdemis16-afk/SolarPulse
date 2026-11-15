import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# ==========================================
# FILE PATHS
# ==========================================

FORECAST_FILE = "data/solar_forecast.csv"
ENERGY_FILE = "data/energy_consumption_data.csv"

MODEL_FILE = "models/anomaly_detection_model.joblib"

CONSUMPTION_MODEL_FILE = (
    "models/consumption_forecast_model.joblib"
)

ANOMALY_THRESHOLD = 0.35


# ==========================================
# WEBSITE PAGES
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/forecast")
def forecast_page():
    return render_template("forecast.html")


@app.route("/monitoring")
def monitoring_page():
    return render_template("monitoring.html")


@app.route("/detector")
def detector_page():
    return render_template("detector.html")


@app.route("/runway")
def runway_page():
    return render_template("runway.html")


@app.route("/simulator")
def simulator_page():
    return render_template("simulator.html")


@app.route("/recommendations")
def recommendations_page():
    return render_template("recommendations.html")


# NEW: CONSUMPTION FORECAST PAGE

@app.route("/consumption")
def consumption_page():
    return render_template("consumption.html")

@app.route("/energy-intelligence")
def energy_intelligence_page():
    return render_template("energy_intelligence.html")


# ==========================================
# API 1: HOURLY SOLAR FORECAST
# ==========================================

@app.route("/api/forecast")
def get_forecast():

    if not os.path.exists(FORECAST_FILE):
        return jsonify({"error": "Forecast data not found"}), 404

    df = pd.read_csv(FORECAST_FILE)

    df["time"] = pd.to_datetime(df["time"])

    forecast = []

    for _, row in df.iterrows():

        forecast.append({
            "time": row["time"].strftime("%Y-%m-%d %H:%M"),

            "radiation": round(
                float(row["predicted_radiation"]), 2
            ),

            "generation": round(
                float(row["predicted_generation_kwh"]), 3
            )
        })

    return jsonify(forecast)


# ==========================================
# API 2: DAILY SOLAR GENERATION SUMMARY
# ==========================================

@app.route("/api/daily-summary")
def daily_summary():

    if not os.path.exists(FORECAST_FILE):
        return jsonify({"error": "Forecast data not found"}), 404

    df = pd.read_csv(FORECAST_FILE)

    df["time"] = pd.to_datetime(df["time"])

    df["date"] = df["time"].dt.strftime("%Y-%m-%d")

    daily = (
        df.groupby("date")["predicted_generation_kwh"]
        .sum()
        .reset_index()
    )

    daily["predicted_generation_kwh"] = (
        daily["predicted_generation_kwh"].round(2)
    )

    return jsonify(daily.to_dict(orient="records"))


# ==========================================
# API 3: ENERGY CONSUMPTION BY DEPARTMENT
# ==========================================

@app.route("/api/energy-summary")
def energy_summary():

    if not os.path.exists(ENERGY_FILE):
        return jsonify({"error": "Energy data not found"}), 404

    df = pd.read_csv(ENERGY_FILE)

    summary = (
        df.groupby("department")["energy_kwh"]
        .sum()
        .reset_index()
    )

    return jsonify(summary.to_dict(orient="records"))


# ==========================================
# API 4: AI DAILY ENERGY CONSUMPTION FORECAST
# ==========================================

@app.route("/api/daily-consumption-forecast")
def daily_consumption_forecast():

    if not os.path.exists(FORECAST_FILE):
        return jsonify({"error": "Forecast data not found"}), 404

    if not os.path.exists(CONSUMPTION_MODEL_FILE):
        return jsonify({
            "error": "Consumption forecasting model not found"
        }), 404

    try:

        model = joblib.load(CONSUMPTION_MODEL_FILE)

        forecast_df = pd.read_csv(FORECAST_FILE)

        forecast_df["time"] = pd.to_datetime(
            forecast_df["time"]
        )

        departments = [
            "Computer Lab",
            "Classrooms",
            "Lighting",
            "Other"
        ]

        predictions = []

        for _, row in forecast_df.iterrows():

            timestamp = row["time"]

            for department in departments:

                input_data = pd.DataFrame([{
                    "department": department,
                    "hour": timestamp.hour,
                    "weekday": timestamp.dayofweek,
                    "month": timestamp.month,
                    "day_of_month": timestamp.day
                }])

                predicted_energy = float(
                    model.predict(input_data)[0]
                )

                predictions.append({

                    "date": timestamp.strftime("%Y-%m-%d"),

                    "predicted_consumption_kwh": max(
                        0, predicted_energy
                    )

                })

        results_df = pd.DataFrame(predictions)

        daily = (
            results_df.groupby("date")[
                "predicted_consumption_kwh"
            ]
            .sum()
            .reset_index()
        )

        results = []

        for _, row in daily.iterrows():

            results.append({

                "date": row["date"],

                "estimated_consumption_kwh": round(
                    float(row["predicted_consumption_kwh"]), 2
                )

            })

        return jsonify(results)

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# ==========================================
# API 5: AI ENERGY ANOMALY DETECTOR
# ==========================================

@app.route("/api/anomalies")
def detect_anomalies():

    if not os.path.exists(ENERGY_FILE):
        return jsonify({"error": "Energy data not found"}), 404

    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "Anomaly model not found"}), 404

    try:

        model = joblib.load(MODEL_FILE)

        df = pd.read_csv(ENERGY_FILE)

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

        # Recreate the original test split

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        test_df = df.loc[X_test.index].copy()

        # Predict on unseen test records

        probabilities = model.predict_proba(X_test)[:, 1]

        test_df["anomaly_probability"] = probabilities

        anomalies = test_df[
            test_df["anomaly_probability"] >= ANOMALY_THRESHOLD
        ].copy()

        anomalies = anomalies.sort_values(
            "anomaly_probability",
            ascending=False
        )

        results = []

        for _, row in anomalies.iterrows():

            results.append({

                "timestamp": str(row["timestamp"]),

                "department": row["department"],

                "hour": int(row["hour"]),

                "power_w": round(
                    float(row["power_w"]), 2
                ),

                "energy_kwh": round(
                    float(row["energy_kwh"]), 4
                ),

                "anomaly_probability": round(
                    float(row["anomaly_probability"]) * 100, 2
                ),

                "status": "Potential anomaly"

            })

        return jsonify({

            "total_test_records": len(test_df),

            "detected_anomalies": len(results),

            "threshold": ANOMALY_THRESHOLD,

            "data_type": "Simulated energy consumption data",

            "anomalies": results

        })

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# ==========================================
# API 6: AI DEPARTMENT-LEVEL CONSUMPTION FORECAST
# ==========================================

@app.route("/api/department-consumption-forecast")
def department_consumption_forecast():

    if not os.path.exists(FORECAST_FILE):
        return jsonify({"error": "Forecast data not found"}), 404

    if not os.path.exists(CONSUMPTION_MODEL_FILE):
        return jsonify({
            "error": "Consumption forecasting model not found"
        }), 404

    try:

        model = joblib.load(CONSUMPTION_MODEL_FILE)

        forecast_df = pd.read_csv(FORECAST_FILE)

        forecast_df["time"] = pd.to_datetime(
            forecast_df["time"]
        )

        departments = [
            "Computer Lab",
            "Classrooms",
            "Lighting",
            "Other"
        ]

        results = []

        for _, row in forecast_df.iterrows():

            timestamp = row["time"]

            for department in departments:

                input_data = pd.DataFrame([{

                    "department": department,

                    "hour": timestamp.hour,

                    "weekday": timestamp.dayofweek,

                    "month": timestamp.month,

                    "day_of_month": timestamp.day

                }])

                prediction = float(
                    model.predict(input_data)[0]
                )

                results.append({

                    "time": timestamp.strftime(
                        "%Y-%m-%d %H:%M"
                    ),

                    "department": department,

                    "predicted_consumption_kwh": round(
                        max(0, prediction), 4
                    )

                })

        return jsonify({

            "data_type": (
                "AI predictions based on simulated "
                "historical consumption data"
            ),

            "departments": departments,

            "total_predictions": len(results),

            "predictions": results

        })

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)