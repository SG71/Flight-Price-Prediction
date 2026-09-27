"""
app.py
------
Flask backend for the Flight Fare Prediction System.

Endpoints:
    GET  /            -> renders the HTML front-end (index.html)
    POST /predict      -> accepts flight details as JSON, returns predicted price
    POST /add_data      -> accepts a new labelled flight record and appends it
                            to data/flights.csv for future retraining
"""

import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.pkl")
DATA_PATH = os.path.join("data", "flights.csv")

# Load the trained pipeline once at start-up (pickle -> no retraining needed)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

REQUIRED_FIELDS = [
    "airline", "source", "departure", "stops", "arrival",
    "destination", "class", "duration", "days_left",
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row = {field: [payload[field]] for field in REQUIRED_FIELDS}
    row["duration"] = [float(payload["duration"])]
    row["days_left"] = [float(payload["days_left"])]
    input_df = pd.DataFrame(row)

    predicted_price = model.predict(input_df)[0]
    return jsonify({"predicted_price": round(float(predicted_price), 2)})


@app.route("/add_data", methods=["POST"])
def add_data():
    payload = request.get_json(force=True)

    required = REQUIRED_FIELDS + ["price"]
    missing = [f for f in required if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    new_row = pd.DataFrame([{field: payload[field] for field in required}])

    if os.path.exists(DATA_PATH):
        new_row.to_csv(DATA_PATH, mode="a", header=False, index=False)
    else:
        new_row.to_csv(DATA_PATH, mode="w", header=True, index=False)

    return jsonify({"message": "Record added successfully. Thank you for contributing data!"})


if __name__ == "__main__":
    app.run(debug=True)
