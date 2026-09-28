"""Unit tests for feature engineering: correctness and NO data leakage."""

import pandas as pd
import pytest

from src.training.preprocess import (
    FEATURES,
    MAX_GAP_SECONDS,
    add_card_history_features,
    build_dataset,
    clean,
    haversine_km,
)


def make_card_txns() -> pd.DataFrame:
    """One card with 4 transactions at known times and amounts."""
    return pd.DataFrame(
        {
            "trans_num": ["t1", "t2", "t3", "t4"],
            "cc_num": ["111"] * 4,
            "trans_date_trans_time": pd.to_datetime(
                [
                    "2019-01-01 10:00",
                    "2019-01-01 10:30",  # 30 min after t1
                    "2019-01-01 12:00",  # 2 h after t1
                    "2019-01-03 12:00",  # 2 days later
                ]
            ),
            "amt": [10.0, 20.0, 30.0, 100.0],
        }
    )


def test_rolling_windows_exclude_current_transaction():
    out = add_card_history_features(make_card_txns())
    # t1: no history. t2: t1 within 1h. t3: t1,t2 outside 1h, inside 24h. t4: nothing in 24h.
    assert out["card_txn_count_1h"].tolist() == [0, 1, 0, 0]
    assert out["card_amt_sum_1h"].tolist() == [0, 10, 0, 0]
    assert out["card_txn_count_24h"].tolist() == [0, 1, 2, 0]
    assert out["card_amt_sum_24h"].tolist() == [0, 10, 30, 0]


def test_seconds_since_last_txn():
    out = add_card_history_features(make_card_txns())
    assert out["seconds_since_last_txn"].tolist() == [MAX_GAP_SECONDS, 1800, 5400, 172800]


def test_amt_vs_card_mean_uses_only_past():
    out = add_card_history_features(make_card_txns())
    # t4: previous mean = (10+20+30)/3 = 20 -> 100/20 = 5.0
    assert out["amt_vs_card_mean"].tolist() == pytest.approx([1.0, 2.0, 2.0, 5.0])


def test_future_transactions_do_not_change_past_features():
    """Adding a later transaction must not change earlier feature values (no leakage)."""
    base = add_card_history_features(make_card_txns())
    extra = pd.DataFrame(
        {
            "trans_num": ["t5"],
            "cc_num": ["111"],
            "trans_date_trans_time": pd.to_datetime(["2019-01-03 12:10"]),
            "amt": [9999.0],
        }
    )
    with_future = add_card_history_features(pd.concat([make_card_txns(), extra]))
    cols = [c for c in base.columns if c.startswith(("card_", "seconds_", "amt_vs"))]
    pd.testing.assert_frame_equal(base[cols], with_future[cols].iloc[:4])


def test_cards_do_not_share_history():
    df = make_card_txns()
    df.loc[1, "cc_num"] = "222"  # t2 now belongs to another card
    out = add_card_history_features(df).set_index("trans_num")
    assert out.loc["t2", "card_txn_count_24h"] == 0
    assert out.loc["t3", "card_txn_count_24h"] == 1  # only t1 on card 111


def test_clean_strips_prefix_and_drops_pii():
    df = pd.DataFrame(
        {"merchant": ["fraud_Kub and Mann"], "unix_time": [1], "first": ["A"], "gender": ["F"]}
    )
    out = clean(df)
    assert out["merchant"].iloc[0] == "Kub and Mann"
    assert not {"unix_time", "first", "gender"} & set(out.columns)


def test_haversine_known_distance():
    # Paris -> London is about 344 km
    assert haversine_km(48.8566, 2.3522, 51.5074, -0.1278) == pytest.approx(344, abs=5)


def test_build_dataset_split_is_chronological():
    def raw(times, amts):
        n = len(times)
        return pd.DataFrame(
            {
                "trans_date_trans_time": pd.to_datetime(times),
                "cc_num": ["111"] * n,
                "merchant": ["fraud_M"] * n,
                "category": ["misc_net"] * n,
                "amt": amts,
                "first": ["A"] * n, "last": ["B"] * n, "gender": ["F"] * n,
                "street": ["s"] * n, "city": ["c"] * n, "state": ["NY"] * n,
                "zip": ["1"] * n, "job": ["j"] * n,
                "lat": [40.0] * n, "long": [-73.0] * n,
                "city_pop": [1000] * n,
                "dob": pd.to_datetime(["1980-01-01"] * n),
                "trans_num": [f"id{t}" for t in times],
                "unix_time": [0] * n,
                "merch_lat": [40.1] * n, "merch_long": [-73.1] * n,
                "is_fraud": [0] * n,
            }
        )

    train_raw = raw(["2019-06-01", "2020-02-01"], [10.0, 20.0])
    test_raw = raw(["2020-07-01"], [30.0])
    train, val, test = build_dataset(train_raw, test_raw)

    assert len(train) == len(val) == len(test) == 1
    assert train["trans_date_trans_time"].max() < val["trans_date_trans_time"].min()
    assert val["trans_date_trans_time"].max() < test["trans_date_trans_time"].min()
    assert set(FEATURES) <= set(train.columns)
    # The test transaction can see the card's earlier (train/val) history.
    assert test["amt_vs_card_mean"].iloc[0] == pytest.approx(30 / 15)
