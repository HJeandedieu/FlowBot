"""
FlowPay - anomaly_detector.py
Trains two models on normal flow readings:
  - Isolation Forest  → flags anomalies (primary check)
  - Prophet           → learns the trend baseline (supporting context)
Saves both to models/ folder so you never retrain from scratch.
Exposes is_anomaly(flow_value) → True/False
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from prophet import Prophet

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH       = os.path.join("data", "readings.csv")
MODEL_DIR      = "models"
IF_MODEL_PATH  = os.path.join(MODEL_DIR, "isolation_forest.pkl")
PR_MODEL_PATH  = os.path.join(MODEL_DIR, "prophet_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load & filter ──────────────────────────────────────────────────────────
def load_normal_readings():
    """Load CSV and keep only normal readings for training."""
    df = pd.read_csv(CSV_PATH)
    normal_df = df[df["is_anomaly"] == 0].copy()
    print(f"[Data] Total rows: {len(df)} | Normal rows used for training: {len(normal_df)}")
    return df, normal_df


# ── 2. Train Isolation Forest ─────────────────────────────────────────────────
def train_isolation_forest(normal_df):
    """
    contamination=0.05 → the model expects ~5% of ALL data to be anomalies.
    We train only on normal data so it learns what 'normal' looks like.
    """
    X = normal_df[["flow_lpm"]].values          # shape: (n_samples, 1)

    model = IsolationForest(
        contamination=0.05,
        random_state=42                          # makes results reproducible
    )
    model.fit(X)

    joblib.dump(model, IF_MODEL_PATH)
    print(f"[Isolation Forest] Trained and saved → {IF_MODEL_PATH}")
    return model


# ── 3. Train Prophet ──────────────────────────────────────────────────────────
def train_prophet(normal_df):
    """
    Prophet requires exactly two columns: ds (timestamp) and y (value).
    It learns the trend and seasonality baseline of normal flow over time.
    """
    prophet_df = normal_df[["timestamp", "flow_lpm"]].copy()
    prophet_df = prophet_df.rename(columns={
        "timestamp": "ds",
        "flow_lpm":  "y"
    })

    # Prophet expects ds as datetime — convert from Unix timestamp
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], unit="s")

    model = Prophet(
        interval_width=0.95,                     # 95% confidence interval
        daily_seasonality=False,
        weekly_seasonality=False
    )
    model.fit(prophet_df)

    joblib.dump(model, PR_MODEL_PATH)
    print(f"[Prophet] Trained and saved → {PR_MODEL_PATH}")
    return model


# ── 4. is_anomaly() ───────────────────────────────────────────────────────────
def is_anomaly(flow_value: float) -> bool:
    """
    Loads the saved Isolation Forest model and checks a single reading.
    Returns True  → anomaly (leak / spike)
    Returns False → normal flow
    Isolation Forest predict() returns:
       -1 → anomaly
        1 → normal
    """
    if not os.path.exists(IF_MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train() first.")

    model = joblib.load(IF_MODEL_PATH)
    prediction = model.predict([[flow_value]])   # expects 2D input
    return prediction[0] == -1


# ── 5. Train pipeline ─────────────────────────────────────────────────────────
def train():
    print("=" * 50)
    print("FlowPay — Training Anomaly Detection Models")
    print("=" * 50)

    df, normal_df = load_normal_readings()
    train_isolation_forest(normal_df)
    train_prophet(normal_df)

    print("\n[Done] Both models saved to models/")


# ── 6. Manual tests ───────────────────────────────────────────────────────────
def run_tests():
    print("\n" + "=" * 50)
    print("Running is_anomaly() tests ...")
    print("=" * 50)

    test_cases = [
        (0.5,  False, "normal tap flow"),
        (0.65, False, "normal tap flow"),
        (7.0,  True,  "burst pipe / leak"),
        (5.5,  True,  "spike reading"),
    ]

    all_passed = True
    for value, expected, description in test_cases:
        result = is_anomaly(value)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] is_anomaly({value}) = {result}  ({description})")

    print()
    if all_passed:
        print("All tests passed — detection engine is ready.")
    else:
        print("Some tests failed — check your training data and retrain.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
    run_tests()