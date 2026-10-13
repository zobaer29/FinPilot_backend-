import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_transaction_classifier import train_transaction_classifier
from training.train_problem_classifier import train_problem_classifier
from training.train_spending_forecast import train_spending_forecast

if __name__ == "__main__":
    print("========================================")
    print("Starting FinPilot AI Full Model Training")
    print("========================================")
    
    print("\n[1/3] Training Transaction Classifier...")
    train_transaction_classifier()

    print("\n[2/3] Training Financial Problem Classifier...")
    train_problem_classifier()

    print("\n[3/3] Training Spending Forecast Model...")
    train_spending_forecast()

    print("\n========================================")
    print("All Models Successfully Trained & Saved!")
    print("========================================")
