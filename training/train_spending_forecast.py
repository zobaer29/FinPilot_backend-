import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "spending_history.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_spending_forecast():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)

    print(f"Loaded {len(df)} historical spending records.")

    df["budget_ratio"] = df["actual_spent"] / df["budget"]

    X = df[["budget", "budget_ratio"]].values
    y = df["actual_spent"].values

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    predictions = model.predict(X)
    mae = float(mean_absolute_error(y, predictions))
    rmse = float(root_mean_squared_error(y, predictions))
    r2 = float(r2_score(y, predictions))

    print("\n--- Spending Forecast Model Evaluation ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    model_path = os.path.join(MODELS_DIR, "spending_forecast.joblib")
    joblib.dump(model, model_path)
    print(f"Saved forecast model to: {model_path}")

    return {
        "modelName": "Spending Forecast Model",
        "algorithm": "Ridge Regression with Trend Features",
        "datasetSize": len(df),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2Score": round(r2, 4),
        "fallbackMechanism": "Deterministic Linear Projection when records < 3",
    }


if __name__ == "__main__":
    train_spending_forecast()
