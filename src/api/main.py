"""FastAPI service that scores card transactions in real time.

Run (from the project root):
    uvicorn src.api.main:app --reload
Then open http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.api.card_history import CardHistoryStore
from src.api.schemas import ModelInfo, ScoreResponse, Transaction
from src.training.preprocess import FEATURES, add_transaction_features

logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detector")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
# Scores at or above this are declined outright; between the model threshold and this
# value the transaction goes to a human analyst ("review").
DECLINE_THRESHOLD = float(os.getenv("DECLINE_THRESHOLD", "0.9"))


@dataclass
class LoadedModel:
    model: Any  # anything with predict_proba(DataFrame)
    version: str
    threshold: float
    val_pr_auc: float | None = None


def load_champion() -> LoadedModel:
    """Load the model behind the registry alias, plus its own threshold tag."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    pr_auc = version.tags.get("val_pr_auc")
    return LoadedModel(
        model=model,
        version=str(version.version),
        threshold=round(float(version.tags["threshold"]), 4),
        val_pr_auc=float(pr_auc) if pr_auc else None,
    )


def decide(score: float, threshold: float) -> str:
    if score >= DECLINE_THRESHOLD:
        return "decline"
    if score >= threshold:
        return "review"
    return "approve"


def build_features(txn: Transaction, history: dict[str, float]) -> pd.DataFrame:
    """One-row feature frame, reusing the TRAINING feature code (no skew)."""
    raw = pd.DataFrame(
        {
            "trans_date_trans_time": [pd.Timestamp(txn.timestamp).tz_localize(None)],
            "amt": [txn.amount],
            "category": [txn.category],
            "dob": [pd.Timestamp(txn.customer_dob)],
            "lat": [txn.customer_lat],
            "long": [txn.customer_long],
            "merch_lat": [txn.merchant_lat],
            "merch_long": [txn.merchant_long],
            "city_pop": [txn.city_pop],
        }
    )
    features = add_transaction_features(raw)
    for name, value in history.items():
        features[name] = value
    return features[FEATURES]


def create_app(loader: Callable[[], LoadedModel] = load_champion) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = loader()
        app.state.history = CardHistoryStore()
        logger.info("Loaded %s v%s", MODEL_NAME, app.state.model.version)
        yield

    app = FastAPI(
        title="Real-Time Fraud Detection API",
        version="1.0.0",
        description="Scores card transactions with the MLflow `@champion` model.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict:
        loaded = getattr(request.app.state, "model", None)
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "ok", "model_version": loaded.version}

    @app.get("/model/info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        loaded: LoadedModel = request.app.state.model
        return ModelInfo(
            model_name=MODEL_NAME,
            model_version=loaded.version,
            alias=MODEL_ALIAS,
            threshold=loaded.threshold,
            decline_threshold=DECLINE_THRESHOLD,
            val_pr_auc=loaded.val_pr_auc,
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(txn: Transaction, request: Request) -> ScoreResponse:
        start = time.perf_counter()
        loaded: LoadedModel = request.app.state.model
        history: CardHistoryStore = request.app.state.history

        ts = pd.Timestamp(txn.timestamp).tz_localize(None).to_pydatetime()
        card_features = history.features(txn.cc_num, ts, txn.amount)
        X = build_features(txn, card_features)
        fraud_score = float(loaded.model.predict_proba(X)[0, 1])
        history.add(txn.cc_num, ts, txn.amount)  # future transactions can now see this one

        return ScoreResponse(
            transaction_id=txn.transaction_id,
            fraud_score=round(fraud_score, 6),
            decision=decide(fraud_score, loaded.threshold),
            threshold=loaded.threshold,
            model_version=loaded.version,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    return app


app = create_app()
