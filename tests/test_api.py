"""API contract tests. A stub model replaces MLflow so tests run anywhere, fast."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import DECLINE_THRESHOLD, LoadedModel, create_app
from src.training.preprocess import FEATURES


class StubModel:
    """Returns a fixed score and records the features it received."""

    def __init__(self, score: float):
        self.score = score
        self.last_X = None

    def predict_proba(self, X):
        self.last_X = X
        return np.array([[1 - self.score, self.score]])


def make_client(score: float = 0.01, threshold: float = 0.2):
    stub = StubModel(score)
    app = create_app(loader=lambda: LoadedModel(stub, version="7", threshold=threshold))
    return TestClient(app), stub


VALID = {
    "transaction_id": "txn_1",
    "cc_num": "4263982640269299",
    "amount": 249.99,
    "category": "shopping_net",
    "timestamp": "2020-07-01T23:15:00",
    "customer_dob": "1985-04-12",
    "customer_lat": 40.71,
    "customer_long": -74.0,
    "merchant_lat": 40.9,
    "merchant_long": -73.8,
    "city_pop": 8000000,
}


def test_health_and_model_info():
    client, _ = make_client()
    with client:
        assert client.get("/health").json() == {"status": "ok", "model_version": "7"}
        info = client.get("/model/info").json()
        assert info["model_version"] == "7" and info["threshold"] == 0.2


@pytest.mark.parametrize(
    "score, expected",
    [(0.01, "approve"), (0.5, "review"), (DECLINE_THRESHOLD, "decline")],
)
def test_score_decisions(score, expected):
    client, _ = make_client(score=score, threshold=0.2)
    with client:
        body = client.post("/score", json=VALID).json()
    assert body["decision"] == expected
    assert body["transaction_id"] == "txn_1" and body["model_version"] == "7"
    assert body["latency_ms"] >= 0


def test_model_receives_all_features_in_training_order():
    client, stub = make_client()
    with client:
        client.post("/score", json=VALID)
    assert list(stub.last_X.columns) == FEATURES
    assert stub.last_X["is_night"].iloc[0] == 1  # 23:15 is a night hour


def test_card_history_accumulates_between_requests():
    client, stub = make_client()
    with client:
        client.post("/score", json=VALID)
        client.post("/score", json=VALID | {"transaction_id": "txn_2", "timestamp": "2020-07-01T23:40:00"})
    assert stub.last_X["card_txn_count_1h"].iloc[0] == 1
    assert stub.last_X["card_amt_sum_1h"].iloc[0] == pytest.approx(249.99)


@pytest.mark.parametrize(
    "field, bad_value",
    [("amount", -5), ("amount", "abc"), ("category", "casino"), ("customer_lat", 200), ("cc_num", "")],
)
def test_invalid_input_is_rejected(field, bad_value):
    client, _ = make_client()
    with client:
        response = client.post("/score", json=VALID | {field: bad_value})
    assert response.status_code == 422


def test_missing_field_is_rejected():
    client, _ = make_client()
    payload = {k: v for k, v in VALID.items() if k != "amount"}
    with client:
        assert client.post("/score", json=payload).status_code == 422
