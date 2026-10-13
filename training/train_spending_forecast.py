import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "spending_history.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_spending_forecast():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)

    print(f"Loaded {len(df)} historical spending records.")

    # Features: [budget, actual_to_budget_ratio] -> target: next projected spent
    df["budget_ratio"] = df["actual_spent"] / df["budget"]

    X = df[["budget", "budget_ratio"]].values
    y = df["actual_spent"].values

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    rmse = root_mean_squared_error(y, predictions)

    print("\n--- Spending Forecast Model Evaluation ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    model_path = os.path.join(MODELS_DIR, "spending_forecast.joblib")
    joblib.dump(model, model_path)
    print(f"Saved forecast model to: {model_path}")


if __name__ == "__main__":
    train_spending_forecast()
