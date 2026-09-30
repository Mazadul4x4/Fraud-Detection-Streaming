"""Training/serving skew test: online features must equal the training features."""

import numpy as np
import pandas as pd
import pytest

from src.api.card_history import CardHistoryStore
from src.training.preprocess import add_card_history_features

HISTORY_COLUMNS = [
    "card_txn_count_1h", "card_amt_sum_1h", "card_txn_count_24h", "card_amt_sum_24h",
    "seconds_since_last_txn", "amt_vs_card_mean",
]


def random_transactions(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    # Minute resolution -> some exact 1h / 24h boundaries and identical timestamps.
    offsets = np.sort(rng.integers(0, 3 * 24 * 60, n))
    return pd.DataFrame(
        {
            "trans_num": [f"t{i:04d}" for i in range(n)],
            "cc_num": rng.choice(["A", "B", "C"], n),
            "trans_date_trans_time": start + pd.to_timedelta(offsets, unit="min"),
            "amt": rng.gamma(2.0, 40.0, n).round(2) + 1,
        }
    )


def test_online_features_match_training_features():
    df = random_transactions()
    offline = add_card_history_features(df).set_index("trans_num")[HISTORY_COLUMNS]

    # Replay the same transactions one by one, in the order training uses.
    store = CardHistoryStore()
    rows = {}
    for r in df.sort_values(["trans_date_trans_time", "trans_num"]).itertuples():
        t = r.trans_date_trans_time.to_pydatetime()
        rows[r.trans_num] = store.features(r.cc_num, t, r.amt)
        store.add(r.cc_num, t, r.amt)
    online = pd.DataFrame.from_dict(rows, orient="index")[HISTORY_COLUMNS]

    pd.testing.assert_frame_equal(
        online.sort_index(), offline.sort_index(), check_dtype=False, check_names=False, rtol=1e-9
    )


def test_exact_window_boundary_is_included():
    """A transaction exactly 1 hour earlier is inside the 1h window (pandas closed='left')."""
    store = CardHistoryStore()
    t0 = pd.Timestamp("2020-01-01 10:00").to_pydatetime()
    store.add("A", t0, 50.0)
    f = store.features("A", pd.Timestamp("2020-01-01 11:00").to_pydatetime(), 10.0)
    assert f["card_txn_count_1h"] == 1


def test_first_transaction_defaults():
    f = CardHistoryStore().features("new-card", pd.Timestamp("2020-01-01").to_pydatetime(), 99.0)
    assert f["card_txn_count_24h"] == 0 and f["amt_vs_card_mean"] == pytest.approx(1.0)
