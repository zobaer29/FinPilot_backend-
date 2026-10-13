import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class MLService:
    def __init__(self):
        self.transaction_vectorizer = self._load_model("transaction_vectorizer.joblib")
        self.transaction_classifier = self._load_model("transaction_classifier.joblib")
        self.problem_vectorizer = self._load_model("problem_vectorizer.joblib")
        self.problem_classifier = self._load_model("problem_classifier.joblib")
        self.spending_forecast_model = self._load_model("spending_forecast.joblib")

    def _load_model(self, filename: str):
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                return None
        return None

    def get_models_status(self) -> dict[str, bool]:
        return {
            "transaction_classifier": self.transaction_classifier is not None,
            "problem_classifier": self.problem_classifier is not None,
            "spending_forecast": self.spending_forecast_model is not None,
        }

    def classify_transaction(self, description: str) -> tuple[str, float, str]:
        cleaned = description.lower().strip()

        # If trained models are loaded, predict with model
        if self.transaction_vectorizer and self.transaction_classifier:
            try:
                vec = self.transaction_vectorizer.transform([cleaned])
                probs = self.transaction_classifier.predict_proba(vec)[0]
                max_idx = int(np.argmax(probs))
                category = str(self.transaction_classifier.classes_[max_idx])
                confidence = float(probs[max_idx])

                # If confidence is reasonably high, return model prediction
                if confidence >= 0.35:
                    return category, round(confidence, 2), "trained-logistic-tfidf"
            except Exception as err:
                print(f"Model prediction error: {err}")

        # Deterministic keyword fallback
        keyword_categories = {
            "Food": ("food", "restaurant", "cafe", "grocery", "foodpanda", "chaldal", "burger", "pizza", "lunch", "dinner", "biryani"),
            "Transport": ("uber", "pathao", "bus", "train", "fuel", "transport", "metro", "cng", "parking", "toll"),
            "Housing": ("rent", "apartment", "mortgage", "housing"),
            "Utilities": ("utility", "electricity", "desco", "dpdc", "water", "wasa", "gas", "internet", "broadband", "recharge"),
            "Shopping": ("shop", "daraz", "clothing", "amazon", "shoe", "bata", "apex", "jacket", "watch"),
            "Entertainment": ("netflix", "spotify", "youtube", "movie", "cineplex", "cinema", "game"),
            "Health": ("medicine", "pharmacy", "hospital", "doctor", "clinic", "dental", "checkup"),
            "Education": ("tuition", "course", "udemy", "coursera", "book", "school", "exam", "ielts"),
            "Income": ("salary", "freelance", "payment", "income", "bonus", "dividend", "upwork", "fiverr"),
        }

        for cat, keywords in keyword_categories.items():
            if any(kw in cleaned for kw in keywords):
                return cat, 0.92, "rule-heuristic-fallback"

        return "Other", 0.50, "default-fallback"

    def classify_problem(self, text: str) -> tuple[str, float, str]:
        cleaned = text.lower().strip()

        if self.problem_vectorizer and self.problem_classifier:
            try:
                vec = self.problem_vectorizer.transform([cleaned])
                probs = self.problem_classifier.predict_proba(vec)[0]
                max_idx = int(np.argmax(probs))
                problem_type = str(self.problem_classifier.classes_[max_idx])
                confidence = float(probs[max_idx])

                if confidence >= 0.25:
                    return problem_type, round(confidence, 2), "trained-problem-tfidf"
            except Exception as err:
                print(f"Problem prediction error: {err}")

        patterns = {
            "GOAL_PLANNING": ("goal", "laptop", "save enough", "deadline", "vacation", "buy a", "afford"),
            "SAVINGS_PROBLEM": ("cannot save", "unable to save", "zero savings", "run out of cash", "balance drops"),
            "OVERSPENDING": ("overspend", "spending too much", "out of control", "leaking away"),
            "VARIABLE_INCOME": ("freelance", "variable income", "changes every month", "fluctuate"),
            "BILL_PROBLEM": ("bill", "broadband", "electricity", "unpaid", "due"),
            "DEBT_PROBLEM": ("debt", "loan", "interest", "installment"),
            "EMERGENCY_FUND": ("emergency fund", "emergency reserve", "cash cushion"),
            "BUDGET_PLANNING": ("budget", "50 30 20", "spending plan", "allocations"),
        }

        for ptype, keywords in patterns.items():
            if any(kw in cleaned for kw in keywords):
                return ptype, 0.90, "rule-heuristic-fallback"

        return "BUDGET_PLANNING", 0.55, "default-fallback"

    def forecast_spending(
        self,
        current_spending: float,
        day_of_month: int,
        days_in_month: int,
        budget: float,
        historical_spending: list[float],
    ) -> tuple[float, float, bool, str]:
        remaining_days = max(days_in_month - day_of_month, 0)
        daily_rate = (current_spending / max(day_of_month, 1))
        linear_forecast = current_spending + (daily_rate * remaining_days)

        if self.spending_forecast_model and budget > 0:
            try:
                ratio = current_spending / budget
                X = np.array([[budget, ratio]])
                ridge_pred = float(self.spending_forecast_model.predict(X)[0])
                predicted_total = (ridge_pred * 0.4) + (linear_forecast * 0.6)
                method = "trained-ridge"
            except Exception:
                predicted_total = linear_forecast
                method = "deterministic-fallback"
        elif len(historical_spending) >= 3:
            recent_avg = sum(historical_spending[-3:]) / 3
            predicted_total = (linear_forecast * 0.7) + (recent_avg * 0.3)
            method = "trend"
        else:
            predicted_total = linear_forecast
            method = "deterministic-fallback"

        predicted_month_end = round(predicted_total, 2)
        predicted_overspend = round(max(0.0, predicted_month_end - budget) if budget > 0 else 0.0, 2)
        is_overspend_risk = predicted_overspend > 0

        return predicted_month_end, predicted_overspend, is_overspend_risk, method


ml_service = MLService()
