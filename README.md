# FinPilot AI — ML Service (Backend)

FinPilot AI's specialized machine learning and financial prediction microservice. Built with **FastAPI**, **scikit-learn**, **Pandas**, and **NumPy**, this service provides intelligent categorization, financial problem classification, and spending forecasting without relying on external proprietary LLM APIs.

---

## 🎯 Architecture & Models

1. **Transaction Classifier (`Model 1`)**:
   - **Technique**: Text preprocessing + TF-IDF Vectorization + Multinomial Naive Bayes / Logistic Regression.
   - **Input**: Transaction description (e.g., *"Foodpanda order"*).
   - **Output**: Categorized spend label (e.g., `Food`, `Transport`, `Housing`, `Shopping`, `Utilities`, `Income`) with confidence score.

2. **Financial Problem Classifier (`Model 2`)**:
   - **Technique**: Problem text preprocessing + TF-IDF Vectorization + Logistic Regression.
   - **Input**: Natural-language query from the FinPilot Problem Box (e.g., *"I earn 60,000 but cannot save for a laptop within 6 months"*).
   - **Output**: Problem type (e.g., `GOAL_PLANNING`, `SAVINGS_PROBLEM`, `OVERSPENDING`, `VARIABLE_INCOME`, `BILL_PROBLEM`, `DEBT_PROBLEM`).

3. **Spending Forecast Engine (`Model 3`)**:
   - **Technique**: Linear spending rate regression + historical moving average with deterministic fallback.
   - **Input**: Current month-to-date spending, days elapsed, and historical category expenditures.
   - **Output**: Forecasted month-end spending, potential overspend alert, and confidence estimation.

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Train Models

To run the full model training pipeline and generate artifact files in `models/`:

```bash
python -m training.train_all
```

Or train individual models:

```bash
python -m training.train_transaction_classifier
python -m training.train_problem_classifier
python -m training.train_spending_forecast
```

### 3. Run FastAPI Service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The interactive OpenAPI docs are accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Endpoints

- `GET /ml/health` — Health check endpoint.
- `POST /ml/transaction-category` — Classify transaction descriptions.
- `POST /ml/problem-classification` — Classify user-reported financial problems.
- `POST /ml/spending-forecast` — Forecast category spending and overruns.

---

## 🧪 Testing

Run automated tests:

```bash
pytest tests/
```
