"""Train the delivery-time model (same pipeline as the notebook) and save it for the web app.

Usage: python train.py path/to/Food_delivery.csv
Creates: model/model.joblib
"""
import json, math, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TRAFFIC = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
FEATURES = ["Type_of_order", "Delivery_person_Age", "Delivery_person_Ratings", "Distance", "City",
            "Order_hour", "Weatherconditions", "Road_traffic_density", "Vehicle_condition",
            "Type_of_vehicle", "Festival", "multiple_deliveries"]
DUMMY_COLS = ["Type_of_vehicle", "Weatherconditions", "Festival", "Type_of_order", "City"]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def main(csv_path):
    df = pd.read_csv(csv_path)
    df["Weatherconditions"] = df["Weatherconditions"].str.removeprefix("conditions ")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().replace("NaN", np.nan)
    df["Time_taken(min)"] = df["Time_taken(min)"].str.removeprefix("(min)").str.strip().astype(int)
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(upper=5)
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["Weatherconditions", "City", "Road_traffic_density", "Festival"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["Distance"] = df.apply(lambda r: haversine(r["Restaurant_latitude"], r["Restaurant_longitude"],
                                                  r["Delivery_location_latitude"], r["Delivery_location_longitude"]), axis=1)
    df = df[df["Distance"] < 100].reset_index(drop=True)
    df["Order_hour"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce").dt.hour
    df["Order_hour"] = df["Order_hour"].fillna(df["Order_hour"].median())

    X = df[FEATURES].copy()
    y = df["Time_taken(min)"]
    X["Road_traffic_density"] = X["Road_traffic_density"].map(TRAFFIC)
    X = pd.get_dummies(X, columns=DUMMY_COLS, drop_first=True, dtype=int)

    # Remember which categories exist so the app can build the same columns
    categories = {c: sorted(df[c].unique().tolist()) for c in DUMMY_COLS}

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

    try:
        from xgboost import XGBRegressor
        model, name = XGBRegressor(random_state=42), "XGBoost"
    except ImportError:
        model, name = RandomForestRegressor(random_state=42, n_jobs=-1), "Random Forest"
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    rmse, r2 = float(np.sqrt(mean_squared_error(y_te, pred))), float(r2_score(y_te, pred))
    print(f"{name}: R2={r2:.4f}  RMSE={rmse:.3f} min")

    Path("model").mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "columns": list(X.columns), "categories": categories,
                 "meta": {"name": name, "r2": r2, "rmse": rmse}}, "model/model.joblib")
    print("Saved model/model.joblib")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python train.py path/to/Food_delivery.csv")
    main(sys.argv[1])
