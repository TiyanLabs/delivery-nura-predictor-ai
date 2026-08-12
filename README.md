# Porter Delivery Time Prediction — CS361 Lab Assignment II

End-to-end pipeline: raw order data -> feature engineering -> Keras neural
network regression -> Flask web app that serves live predictions.

## Files

- `train_model.py` — loads `dataset.csv`, engineers features, trains the
  neural network, evaluates it, and saves the model + preprocessing objects
  into `model/`.
- `app.py` — Flask server that loads the saved model and exposes a form for
  entering order details and getting a predicted delivery time.
- `templates/index.html` — the web UI.
- `model/` — created after you run `train_model.py` (holds the `.keras`
  model file, the fitted `StandardScaler`, and the feature config).

## How to run

1. Put the dataset in this folder and rename it `dataset.csv` (or edit
   `DATA_PATH` at the top of `train_model.py`).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train the model:
   ```
   python train_model.py
   ```
   This prints the network architecture, training progress, and final
   test-set MAE / RMSE / R².
4. Start the web app:
   ```
   python app.py
   ```
   Open `http://127.0.0.1:5000` and fill in the order form to get a
   predicted delivery time in minutes.

## What the pipeline does

**Feature engineering**
- Target = minutes between `created_at` and `actual_delivery_time`,
  clipped to a 5–120 minute window to drop bad timestamps.
- Time features: hour of day, day of week, weekend flag.
- Fleet-load features: busy-partner ratio, outstanding orders per partner.
- Price features: price range and average item price within the order.
- `store_primary_category` (70+ raw levels) is bucketed to the top 15
  categories + "other", then one-hot encoded.

**Missing values**
- `market_id`, `order_protocol`: filled with the mode.
- `total_onshift_partners`, `total_busy_partners`, `total_outstanding_orders`:
  filled with the median (about 8% of rows are missing these).

**Model**
- Fully-connected network: 128 → 64 → 32 → 1, ReLU activations, batch
  normalization + dropout for regularization, Adam optimizer, MSE loss.
- Early stopping on validation loss to avoid overfitting.
- Inputs are standardized with `StandardScaler` fit on the training split.

**Deployment**
- The Flask app rebuilds the same feature vector from form input, applies
  the saved scaler, and runs the saved Keras model to return a prediction.
