from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    CategoryResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ProblemRequest,
    ProblemResponse,
    TransactionCategoryRequest,
)
from app.services.ml_service import ml_service

app = FastAPI(
    title="FinPilot AI ML Service",
    description="Specialized machine learning and financial prediction microservice for FinPilot AI",
    version="0.1.0",
)

# Enable CORS for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ml/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="finpilot-ml-service",
        modelsLoaded=ml_service.get_models_status(),
    )


@app.post("/ml/transaction-category", response_model=CategoryResponse)
def classify_transaction(request: TransactionCategoryRequest) -> CategoryResponse:
    category, confidence, model_used = ml_service.classify_transaction(request.description)
    return CategoryResponse(
        category=category,
        confidence=confidence,
        modelUsed=model_used,
    )


@app.post("/ml/problem-classification", response_model=ProblemResponse)
def classify_problem(request: ProblemRequest) -> ProblemResponse:
    problem_type, confidence, model_used = ml_service.classify_problem(request.text)
    return ProblemResponse(
        problemType=problem_type,
        confidence=confidence,
        modelUsed=model_used,
    )


@app.post("/ml/spending-forecast", response_model=ForecastResponse)
def forecast_spending(request: ForecastRequest) -> ForecastResponse:
    predicted_end, predicted_overspend, risk, method = ml_service.forecast_spending(
        current_spending=request.currentSpending,
        day_of_month=request.dayOfMonth,
        days_in_month=request.daysInMonth,
        budget=request.budget,
        historical_spending=request.historicalSpending,
    )
    return ForecastResponse(
        predictedMonthEnd=predicted_end,
        predictedOverspend=predicted_overspend,
        isOverspendRisk=risk,
        method=method,
    )


@app.get("/ml/evaluation")
def get_evaluation() -> dict:
    import json
    import os
    eval_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "evaluation_results.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            return json.load(f)
    return {"error": "Evaluation results not generated yet"}
