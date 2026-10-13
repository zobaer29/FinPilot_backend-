from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class TransactionCategoryRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=300, example="Foodpanda lunch")


class CategoryResponse(BaseModel):
    category: str
    confidence: float
    modelUsed: str = "trained-svm-logistic"


class ProblemRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, example="I earn 60000 but cannot save for laptop")


class ProblemResponse(BaseModel):
    problemType: str
    confidence: float
    modelUsed: str = "trained-classifier"


class ForecastRequest(BaseModel):
    historicalSpending: list[float] = Field(default_factory=list, max_length=36)
    currentSpending: float = Field(..., ge=0)
    budget: float = Field(default=0, ge=0)
    dayOfMonth: int = Field(..., ge=1, le=31)
    daysInMonth: int = Field(..., ge=1, le=31)


class ForecastResponse(BaseModel):
    predictedMonthEnd: float
    predictedOverspend: float
    isOverspendRisk: bool
    method: Literal["trained-ridge", "trend", "deterministic-fallback"]


class HealthResponse(BaseModel):
    status: str
    service: str
    modelsLoaded: dict[str, bool]
