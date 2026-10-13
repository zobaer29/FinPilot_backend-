import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_transaction_classifier import train_transaction_classifier
from training.train_problem_classifier import train_problem_classifier
from training.train_spending_forecast import train_spending_forecast

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_all():
    print("========================================")
    print("Starting FinPilot AI Full Model Training")
    print("========================================")

    print("\n[1/3] Training Transaction Classifier...")
    tx_eval = train_transaction_classifier()

    print("\n[2/3] Training Financial Problem Classifier...")
    prob_eval = train_problem_classifier()

    print("\n[3/3] Training Spending Forecast Model...")
    forecast_eval = train_spending_forecast()

    results = {
        "generatedAt": "2026-10-13",
        "models": {
            "transactionClassifier": tx_eval,
            "problemClassifier": prob_eval,
            "spendingForecast": forecast_eval,
        },
    }

    eval_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved real evaluation results to: {eval_path}")
    print("\n========================================")
    print("All Models Successfully Trained & Saved!")
    print("========================================")


if __name__ == "__main__":
    train_all()
