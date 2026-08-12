"""
Flask app that loads the trained neural network and serves delivery-time
predictions through a small web form.

Run:
    python train_model.py     # once, to produce ./model/
    python app.py              # then start the server on localhost:5000
"""

from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from tensorflow import keras

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------------
model = keras.models.load_model("model/delivery_time_model.keras")
scaler = joblib.load("model/scaler.pkl")
config = joblib.load("model/feature_config.pkl")

NUMERIC_FEATURES = config["numeric_features"]
CATEGORY_COLUMNS = config["category_columns"]
TOP_CATEGORIES = sorted(config["top_categories"])


def build_input_row(form):
    """Turn the submitted form into the exact same feature layout the model
    was trained on (same columns, same order, one-hot categories aligned)."""

    order_time = datetime.strptime(form["order_time"], "%Y-%m-%dT%H:%M")

    onshift = float(form["total_onshift_partners"])
    busy = float(form["total_busy_partners"])
    outstanding = float(form["total_outstanding_orders"])
    min_price = float(form["min_item_price"])
    max_price = float(form["max_item_price"])
    subtotal = float(form["subtotal"])
    num_distinct = float(form["num_distinct_items"])

    row = {
        "market_id": float(form["market_id"]),
        "order_protocol": float(form["order_protocol"]),
        "total_items": float(form["total_items"]),
        "subtotal": subtotal,
        "num_distinct_items": num_distinct,
        "min_item_price": min_price,
        "max_item_price": max_price,
        "total_onshift_partners": onshift,
        "total_busy_partners": busy,
        "total_outstanding_orders": outstanding,
        "order_hour": order_time.hour,
        "order_day_of_week": order_time.weekday(),
        "is_weekend": int(order_time.weekday() in (5, 6)),
        "busy_partner_ratio": (busy / onshift) if onshift else 0.0,
        "orders_per_partner": (outstanding / onshift) if onshift else 0.0,
        "price_range": max_price - min_price,
        "avg_item_price": (subtotal / num_distinct) if num_distinct else subtotal,
    }

    numeric_df = pd.DataFrame([row])[NUMERIC_FEATURES]

    # One-hot the category the same way training did: known top category
    # gets its own column, anything else (including blank) falls into "other".
    category = form.get("store_category", "other")
    if category not in TOP_CATEGORIES:
        category = "other"
    cat_df = pd.DataFrame([{col: 0 for col in CATEGORY_COLUMNS}])
    col_name = f"cat_{category}"
    if col_name in cat_df.columns:
        cat_df[col_name] = 1

    full_row = pd.concat([numeric_df.reset_index(drop=True),
                           cat_df.reset_index(drop=True)], axis=1)
    return full_row


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            X = build_input_row(request.form)
            X_scaled = scaler.transform(X)
            pred_minutes = float(model.predict(X_scaled, verbose=0).flatten()[0])
            prediction = round(max(pred_minutes, 0), 1)
        except Exception as exc:
            error = f"Couldn't generate a prediction: {exc}"

    return render_template(
        "index.html",
        categories=TOP_CATEGORIES,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
