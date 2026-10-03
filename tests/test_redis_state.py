"""Redis-backed card state: parity with training features, persistence, backfill.

Uses fakeredis (an in-memory Redis) so no Redis server is needed for the tests.
"""

import fakeredis
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import LoadedModel, create_app
from src.feature_store.backfill import build_snapshots
from src.feature_store.redis_state import RedisCardHistory
from src.training.preprocess import add_card_history_features
from tests.test_api import VALID, StubModel
from tests.test_card_history import HISTORY_COLUMNS, random_transactions


def stream_through(store, df: pd.DataFrame) -> pd.DataFrame:
    """Score rows one by one (training order) and collect the features."""
    rows = {}
    for r in df.sort_values(["trans_date_trans_time", "trans_num"]).itertuples():
        rows[r.trans_num] = store.features_and_add(r.cc_num, r.trans_date_trans_time.to_pydatetime(), r.amt)
    return pd.DataFrame.from_dict(rows, orient="index")[HISTORY_COLUMNS].sort_index()


def offline_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_card_history_features(df).set_index("trans_num")[HISTORY_COLUMNS].sort_index()


def test_redis_features_match_training_features():
    df = random_transactions(n=300, seed=5)
    store = RedisCardHistory(fakeredis.FakeRedis())
    pd.testing.assert_frame_equal(
        stream_through(store, df), offline_features(df), check_dtype=False, check_names=False, rtol=1e-9
    )


def test_state_survives_restart():
    """A new client (e.g. a restarted API container) sees the history written before."""
    server = fakeredis.FakeServer()
    t0 = pd.Timestamp("2020-07-01 10:00").to_pydatetime()
    RedisCardHistory(fakeredis.FakeRedis(server=server)).features_and_add("A", t0, 50.0)

    restarted = RedisCardHistory(fakeredis.FakeRedis(server=server))
    f = restarted.features_and_add("A", pd.Timestamp("2020-07-01 10:30").to_pydatetime(), 10.0)
    assert f["card_txn_count_1h"] == 1 and f["card_amt_sum_1h"] == pytest.approx(50.0)


def test_backfill_then_stream_matches_training_features():
    """Backfill history, then stream the rest: features equal training's (no cold start)."""
    df = random_transactions(n=400, seed=7)
    cutoff = df["trans_date_trans_time"].quantile(0.7)
    history, live = df[df["trans_date_trans_time"] <= cutoff], df[df["trans_date_trans_time"] > cutoff]

    store = RedisCardHistory(fakeredis.FakeRedis())
    store.put_snapshots(build_snapshots(history))
    expected = offline_features(df).loc[sorted(live["trans_num"])]
    pd.testing.assert_frame_equal(
        stream_through(store, live), expected, check_dtype=False, check_names=False, rtol=1e-9
    )


def test_api_with_redis_backend_keeps_history_across_restarts():
    server = fakeredis.FakeServer()

    def make_app(stub):
        return create_app(
            loader=lambda: LoadedModel(stub, version="7", threshold=0.2),
            history_factory=lambda: RedisCardHistory(fakeredis.FakeRedis(server=server)),
        )

    with TestClient(make_app(StubModel(0.01))) as client:
        client.post("/score", json=VALID)

    stub = StubModel(0.01)
    with TestClient(make_app(stub)) as client:  # "restarted" API, same Redis
        client.post("/score", json=VALID | {"transaction_id": "txn_2", "timestamp": "2020-07-01T23:40:00"})
    assert stub.last_X["card_txn_count_1h"].iloc[0] == 1
