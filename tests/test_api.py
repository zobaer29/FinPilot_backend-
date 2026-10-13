import unittest
from app.main import health, classify_transaction, classify_problem, forecast_spending
from app.schemas import TransactionCategoryRequest, ProblemRequest, ForecastRequest


class TestMLService(unittest.TestCase):
    def test_health_check(self):
        res = health()
        self.assertEqual(res.status, "ok")
        self.assertIn("transaction_classifier", res.modelsLoaded)
        self.assertTrue(res.modelsLoaded["transaction_classifier"])

    def test_classify_transaction_food(self):
        req = TransactionCategoryRequest(description="Foodpanda burger delivery")
        res = classify_transaction(req)
        self.assertEqual(res.category, "Food")
        self.assertGreater(res.confidence, 0.3)

    def test_classify_transaction_transport(self):
        req = TransactionCategoryRequest(description="Uber ride to office")
        res = classify_transaction(req)
        self.assertEqual(res.category, "Transport")
        self.assertGreater(res.confidence, 0.3)

    def test_classify_problem_goal(self):
        req = ProblemRequest(text="I want to buy a laptop in 6 months but cannot save enough")
        res = classify_problem(req)
        self.assertEqual(res.problemType, "GOAL_PLANNING")
        self.assertGreater(res.confidence, 0.2)

    def test_spending_forecast(self):
        req = ForecastRequest(
            currentSpending=7500.0,
            budget=10000.0,
            dayOfMonth=15,
            daysInMonth=30,
            historicalSpending=[9500.0, 9800.0, 10200.0],
        )
        res = forecast_spending(req)
        self.assertGreater(res.predictedMonthEnd, 0)
        self.assertIsNotNone(res.isOverspendRisk)


if __name__ == "__main__":
    unittest.main()
