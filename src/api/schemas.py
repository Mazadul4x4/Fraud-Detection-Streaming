"""Request and response models (validated automatically by FastAPI / Pydantic)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]


class Transaction(BaseModel):
    transaction_id: str = Field(min_length=1, examples=["txn_123"])
    cc_num: str = Field(min_length=1, examples=["4263982640269299"])
    amount: float = Field(gt=0, le=1_000_000, examples=[249.99])
    category: Category = Field(examples=["shopping_net"])
    timestamp: datetime = Field(examples=["2020-07-01T23:15:00"])
    customer_dob: date = Field(examples=["1985-04-12"])
    customer_lat: float = Field(ge=-90, le=90, examples=[40.71])
    customer_long: float = Field(ge=-180, le=180, examples=[-74.0])
    merchant_lat: float = Field(ge=-90, le=90, examples=[40.9])
    merchant_long: float = Field(ge=-180, le=180, examples=[-73.8])
    city_pop: int = Field(ge=0, examples=[8_000_000])


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    decision: Literal["approve", "review", "decline"]
    threshold: float
    model_version: str
    latency_ms: float


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    alias: str
    threshold: float
    decline_threshold: float
    val_pr_auc: float | None
