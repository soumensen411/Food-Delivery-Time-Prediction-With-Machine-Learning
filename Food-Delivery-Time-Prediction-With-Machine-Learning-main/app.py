"""Flask app serving the web UI and a /api/predict endpoint."""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
MODEL_PATH = Path("model/model.joblib")
BUNDLE = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

TRAFFIC = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
OPTIONS = {
    "Type_of_vehicle": ["motorcycle", "scooter", "electric_scooter", "bicycle"],
    "Weatherconditions": ["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"],
    "Type_of_order": ["Snack", "Meal", "Drinks", "Buffet"],
    "City": ["Metropolitian", "Urban", "Semi-Urban"],
    "Festival": ["No", "Yes"],
    "Road_traffic_density": list(TRAFFIC),
}


def build_row(d):
    """Turn form values into the exact column layout used in training."""
    row = {
        "Delivery_person_Age": float(d["age"]),
        "Delivery_person_Ratings": float(d["rating"]),
        "Distance": float(d["distance"]),
        "Order_hour": float(d["hour"]),
        "Road_traffic_density": TRAFFIC[d["traffic"]],
        "Vehicle_condition": float(d["vehicle_condition"]),
        "multiple_deliveries": float(d["multiple_deliveries"]),
    }
    chosen = {"Type_of_vehicle": d["vehicle"], "Weatherconditions": d["weather"], "Festival": d["festival"],
              "Type_of_order": d["order_type"], "City": d["city"]}
    for feat, val in chosen.items():
        row[f"{feat}_{val}"] = 1
    return pd.DataFrame([row]).reindex(columns=BUNDLE["columns"], fill_value=0)


def demo_estimate(d):
    """Rough rule-of-thumb from the notebook's EDA. Only used when no trained model exists."""
    t = 8 + 1.4 * float(d["distance"])
    t += {"Low": 0, "Medium": 4, "High": 5, "Jam": 9}[d["traffic"]]
    t += {"Sunny": -3, "Cloudy": 3, "Fog": 3}.get(d["weather"], 0)
    t += 4 * float(d["multiple_deliveries"])
    t += 15 if d["festival"] == "Yes" else 0
    t += 3 if float(d["hour"]) in (19, 20, 21) else 0
    return t


@app.get("/")
def index():
    return render_template("index.html", options=OPTIONS, live=BUNDLE is not None,
                           meta=(BUNDLE or {}).get("meta"))


@app.post("/api/predict")
def predict():
    d = request.get_json(force=True, silent=True) or {}
    try:
        for key, field in [("vehicle", "Type_of_vehicle"), ("weather", "Weatherconditions"),
                           ("order_type", "Type_of_order"), ("city", "City"),
                           ("festival", "Festival"), ("traffic", "Road_traffic_density")]:
            if d.get(key) not in OPTIONS[field]:
                raise ValueError(f"Invalid {key}")
        if not (15 <= float(d["age"]) <= 70 and 1 <= float(d["rating"]) <= 5 and 0 < float(d["distance"]) <= 100
                and 0 <= float(d["hour"]) <= 23 and 0 <= float(d["vehicle_condition"]) <= 3
                and 0 <= float(d["multiple_deliveries"]) <= 3):
            raise ValueError("A value is out of range")
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=str(e) or "Invalid input"), 400

    if BUNDLE:
        x = BUNDLE["scaler"].transform(build_row(d))
        minutes, margin, mode = float(BUNDLE["model"].predict(x)[0]), BUNDLE["meta"]["rmse"], "model"
    else:
        minutes, margin, mode = demo_estimate(d), 6.0, "demo"
    minutes = max(minutes, 5)
    return jsonify(minutes=round(minutes, 1), low=max(round(minutes - margin), 5),
                   high=round(minutes + margin), mode=mode)


if __name__ == "__main__":
    app.run(debug=True)
