
"""
Porter Delivery Time Prediction
--------------------------------
CS361 - AIMLDL Application, Lab Assignment II

This script builds an end-to-end pipeline that:
1. Loads the raw Porter delivery dataset
2. Engineers features from the raw columns (timestamps, partner load, price stats, etc.)
3. Cleans up missing values and obvious outliers
4. Trains a feed-forward neural network (Keras) to predict delivery time in minutes
5. Saves the trained model + preprocessing objects so app.py can serve predictions

Run this once to produce model/ before starting the Flask app.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
DATA_PATH = "dataset.csv"          # rename your uploaded csv to this, or edit the path
MODEL_DIR = "model"

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Target: minutes between order placed and order delivered
    df["delivery_time_minutes"] = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds() / 60.0

    # A handful of rows have a negative or absurdly large gap (bad timestamps).
    # Keep deliveries between 5 minutes and 2 hours - covers ~99% of real orders.
    df = df[(df["delivery_time_minutes"] >= 5) & (df["delivery_time_minutes"] <= 120)]

    # Time-of-day / day-of-week features from the order timestamp
    df["order_hour"] = df["created_at"].dt.hour
    df["order_day_of_week"] = df["created_at"].dt.dayofweek
    df["is_weekend"] = df["order_day_of_week"].isin([5, 6]).astype(int)

    # Fleet load at the time the order was placed
    df["busy_partner_ratio"] = df["total_busy_partners"] / df["total_onshift_partners"].replace(0, np.nan)
    df["busy_partner_ratio"] = df["busy_partner_ratio"].fillna(0)
    df["orders_per_partner"] = df["total_outstanding_orders"] / df["total_onshift_partners"].replace(0, np.nan)
    df["orders_per_partner"] = df["orders_per_partner"].fillna(0)

    # Price spread within a single order
    df["price_range"] = df["max_item_price"] - df["min_item_price"]
    df["avg_item_price"] = df["subtotal"] / df["num_distinct_items"].replace(0, np.nan)
    df["avg_item_price"] = df["avg_item_price"].fillna(df["subtotal"])

    # store_primary_category has 70+ levels and a long tail - bucket rare
    # categories together so one-hot encoding doesn't explode into noise.
    top_categories = df["store_primary_category"].value_counts().nlargest(15).index
    df["store_category_grouped"] = df["store_primary_category"].where(
        df["store_primary_category"].isin(top_categories), other="other"
    )
    df["store_category_grouped"] = df["store_category_grouped"].fillna("unknown")

    return df


NUMERIC_FEATURES = [
    "market_id",
    "order_protocol",
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_partners",
    "total_busy_partners",
    "total_outstanding_orders",
    "order_hour",
    "order_day_of_week",
    "is_weekend",
    "busy_partner_ratio",
    "orders_per_partner",
    "price_range",
    "avg_item_price",
]
CATEGORICAL_FEATURE = "store_category_grouped"
TARGET = "delivery_time_minutes"


# ---------------------------------------------------------------------------
# 3. Clean up missing values
# ---------------------------------------------------------------------------
def handle_missing_values(df):
    df = df.copy()
    df["market_id"] = df["market_id"].fillna(df["market_id"].mode()[0])
    df["order_protocol"] = df["order_protocol"].fillna(df["order_protocol"].mode()[0])

    # Missing partner-load fields usually mean the raw feed just didn't log
    # them - a median fill is safer here than dropping ~8% of the rows.
    for col in ["total_onshift_partners", "total_busy_partners", "total_outstanding_orders"]:
        df[col] = df[col].fillna(df[col].median())

    return df


# ---------------------------------------------------------------------------
# 4. Build model-ready matrices
# ---------------------------------------------------------------------------
def build_feature_matrix(df, category_columns=None):
    """One-hot encode the grouped category column and return X, y, and the
    final list of one-hot column names (so we can align train/test/inference)."""
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURE], prefix="cat")
    if category_columns is not None:
        dummies = dummies.reindex(columns=category_columns, fill_value=0)

    X = pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    y = df[TARGET].reset_index(drop=True)
    return X, y, list(dummies.columns)


# ---------------------------------------------------------------------------
# 5. Neural network
# ---------------------------------------------------------------------------
def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),   # regression output
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Engineering features...")
    df = engineer_features(df)
    df = handle_missing_values(df)
    print(f"Rows after cleaning: {len(df)}")

    X, y, category_columns = build_feature_matrix(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Scale features - neural nets converge much faster on standardized input.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled.shape[1])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop],
        verbose=2,
    )

    # ---------------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Test set performance ---")
    print(f"MAE  : {mae:.2f} minutes")
    print(f"RMSE : {rmse:.2f} minutes")
    print(f"R2   : {r2:.3f}")

    # ---------------------------------------------------------------------
    # Save everything app.py needs to reproduce this pipeline at inference time
    # ---------------------------------------------------------------------
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(f"{MODEL_DIR}/delivery_time_model.keras")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(
        {
            "numeric_features": NUMERIC_FEATURES,
            "category_columns": category_columns,
            "top_categories": [c.replace("cat_", "") for c in category_columns if c != "cat_other"],
        },
        f"{MODEL_DIR}/feature_config.pkl",
    )

    print(f"\nSaved model + preprocessing artifacts to ./{MODEL_DIR}/")


if __name__ == "__main__":
    main()
