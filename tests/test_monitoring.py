"""Drift monitoring: detects a real shift, stays quiet on identical distributions."""

import numpy as np
import pandas as pd

from src.monitoring.drift_report import DRIFT_THRESHOLD, MONITORED, drift_metrics, inject_drift


def make_window(n: int, seed: int, amount_scale: float = 30.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "amt": rng.gamma(2.0, amount_scale, n),
        "hour": rng.integers(0, 24, n),
        "is_night": rng.integers(0, 2, n),
        "category": rng.choice(["grocery_pos", "misc_net", "travel"], n),
        "age_years": rng.normal(45, 15, n),
        "distance_km": rng.gamma(2.0, 30.0, n),
        "card_txn_count_24h": rng.poisson(3, n),
        "card_amt_sum_24h": rng.gamma(2.0, 100.0, n),
        "amt_vs_card_mean": rng.gamma(2.0, 0.5, n),
        "seconds_since_last_txn": rng.gamma(2.0, 20000.0, n),
        "score": rng.beta(0.5, 20, n),
    })
    assert set(MONITORED) <= set(df.columns)
    return df


def test_no_drift_on_same_distribution():
    out, _ = drift_metrics(make_window(5000, seed=1), make_window(3000, seed=2))
    assert out["drifted_features"] == []
    assert out["score_drift"] < DRIFT_THRESHOLD


def test_amount_shift_is_detected():
    out, _ = drift_metrics(make_window(5000, seed=1), make_window(3000, seed=2, amount_scale=60.0))
    assert "amt" in out["drifted_features"]


def test_inject_drift_scales_amount_columns():
    df = pd.DataFrame({"amt": [10.0], "log_amt": [0.0], "card_amt_sum_1h": [5.0], "card_amt_sum_24h": [20.0]})
    out = inject_drift(df, 2.0)
    assert out.loc[0, "amt"] == 20.0 and out.loc[0, "card_amt_sum_24h"] == 40.0
    assert out.loc[0, "log_amt"] == np.log1p(20.0)
    assert df.loc[0, "amt"] == 10.0  # original untouched
