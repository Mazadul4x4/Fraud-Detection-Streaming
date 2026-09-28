"""Data cleaning, feature engineering and time-based splitting.

Turns the raw Kaggle CSV files into model-ready train / validation / test
Parquet files in ``data/processed/``.

Every per-card feature uses ONLY transactions that happened BEFORE the
current one, so the same logic can later run in real time (Phase 7)
without data leakage.

Usage (from the project root):
    python -m src.training.preprocess
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Column names ----------------------------------------------------------
TARGET = "is_fraud"
TIME_COL = "trans_date_trans_time"
CARD_COL = "cc_num"
ID_COL = "trans_num"

# Validation period starts here (inside fraudTrain.csv). fraudTest.csv = test.
VAL_START = pd.Timestamp("2020-01-01")

# Hours with a much higher fraud rate (found in the EDA notebook).
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}

# Per-card rolling windows (EDA: fraud on a card comes in bursts of ~1-3 days).
WINDOWS = ["1h", "24h"]

# Cap for "time since previous transaction" (also used for a card's first transaction).
MAX_GAP_SECONDS = 7 * 24 * 3600

# Personal data we never use as model features (privacy / fairness).
PII_COLUMNS = ["first", "last", "street", "city", "state", "zip", "gender", "job"]

NUMERIC_FEATURES = [
    "amt",
    "log_amt",
    "hour",
    "is_night",
    "day_of_week",
    "age_years",
    "distance_km",
    "log_city_pop",
    "card_txn_count_1h",
    "card_amt_sum_1h",
    "card_txn_count_24h",
    "card_amt_sum_24h",
    "seconds_since_last_txn",
    "amt_vs_card_mean",
]
CATEGORICAL_FEATURES = ["category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --- Loading ---------------------------------------------------------------
def load_raw(filename: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read one raw CSV with correct data types."""
    return pd.read_csv(
        raw_dir / filename,
        index_col=0,
        dtype={CARD_COL: "string", "zip": "string"},
        parse_dates=[TIME_COL, "dob"],
    )


# --- Cleaning --------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data-quality issues found in the EDA and drop personal data."""
    df = df.copy()
    # unix_time is offset by 7 years from trans_date_trans_time -> unreliable.
    df = df.drop(columns=["unix_time"], errors="ignore")
    # Every merchant name starts with "fraud_" -> simulator artifact, no signal.
    df["merchant"] = df["merchant"].str.removeprefix("fraud_")
    return df.drop(columns=PII_COLUMNS, errors="ignore")


# --- Features: one transaction at a time -----------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres between two points (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that only need the current transaction."""
    df = df.copy()
    t = df[TIME_COL]
    df["log_amt"] = np.log1p(df["amt"])
    df["hour"] = t.dt.hour
    df["is_night"] = df["hour"].isin(NIGHT_HOURS).astype("int8")
    df["day_of_week"] = t.dt.dayofweek
    df["age_years"] = (t - df["dob"]).dt.days / 365.25
    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["log_city_pop"] = np.log1p(df["city_pop"])
    return df.drop(columns=["dob", "lat", "long", "merch_lat", "merch_long", "city_pop"])


# --- Features: card history (past transactions only) -----------------------
def add_card_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card behaviour, using ONLY transactions strictly before the current one."""
    df = df.sort_values([CARD_COL, TIME_COL, ID_COL]).reset_index(drop=True)

    # Rolling time windows. closed="left" excludes the current transaction.
    by_card = df.set_index(TIME_COL).groupby(CARD_COL, sort=False)["amt"]
    for window in WINDOWS:
        rolling = by_card.rolling(window, closed="left")
        df[f"card_txn_count_{window}"] = rolling.count().fillna(0).to_numpy()
        df[f"card_amt_sum_{window}"] = rolling.sum().fillna(0).to_numpy()

    # Seconds since the card's previous transaction (capped; first txn -> cap).
    gap = df.groupby(CARD_COL, sort=False)[TIME_COL].diff().dt.total_seconds()
    df["seconds_since_last_txn"] = gap.fillna(MAX_GAP_SECONDS).clip(upper=MAX_GAP_SECONDS)

    # Current amount compared with the card's average of all PREVIOUS amounts.
    grouped = df.groupby(CARD_COL, sort=False)["amt"]
    n_previous = grouped.cumcount()
    previous_sum = grouped.cumsum() - df["amt"]
    previous_mean = previous_sum / n_previous.replace(0, np.nan)
    df["amt_vs_card_mean"] = (df["amt"] / previous_mean).fillna(1.0)

    return df


# --- Splitting ---------------------------------------------------------------
def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split: train < VAL_START <= validation; test = fraudTest.csv."""
    is_test = df["source"] == "test"
    train = df[~is_test & (df[TIME_COL] < VAL_START)]
    val = df[~is_test & (df[TIME_COL] >= VAL_START)]
    test = df[is_test]
    return train, val, test


def build_dataset(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """Full pipeline: combine -> clean -> features -> split -> sort by time."""
    df = pd.concat(
        [train_raw.assign(source="train"), test_raw.assign(source="test")],
        ignore_index=True,
    )
    df = clean(df)
    df = add_transaction_features(df)
    # History features on the COMBINED data: a test transaction may look back at
    # the same card's earlier transactions - exactly what happens in production.
    df = add_card_history_features(df)

    columns = [ID_COL, CARD_COL, TIME_COL, *FEATURES, TARGET]
    splits = time_split(df)
    return tuple(s[columns].sort_values(TIME_COL).reset_index(drop=True) for s in splits)


# --- Entry point -------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    logger.info("Loading raw CSV files from %s", RAW_DIR)
    train_raw = load_raw("fraudTrain.csv")
    test_raw = load_raw("fraudTest.csv")

    logger.info("Cleaning and building features (this can take a minute)...")
    splits = build_dataset(train_raw, test_raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in zip(["train", "val", "test"], splits):
        path = PROCESSED_DIR / f"{name}.parquet"
        split.to_parquet(path, index=False)
        logger.info(
            "%-5s | rows=%9s | fraud rate=%.3f%% | %s -> %s | saved %s",
            name,
            f"{len(split):,}",
            100 * split[TARGET].mean(),
            split[TIME_COL].min().date(),
            split[TIME_COL].max().date(),
            path.name,
        )


if __name__ == "__main__":
    main()
