"""Contract tests: stream events must be valid API requests."""

import pandas as pd

from src.api.schemas import Transaction
from src.ingestion.events import RAW_COLUMNS, to_event, to_label_event


def raw_rows() -> pd.DataFrame:
    """Two rows shaped like data/raw/fraudTest.csv."""
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.to_datetime(["2020-06-21 12:14:25", "2020-06-21 23:59:59"]),
            "cc_num": pd.Series(["2291163933867244", "60416207185"], dtype="string"),
            "category": ["personal_care", "shopping_net"],
            "amt": [2.86, 950.0],
            "lat": [33.9659, 40.3207],
            "long": [-80.9355, -110.436],
            "city_pop": [333497, 302],
            "dob": ["1968-03-19", "1990-01-17"],
            "trans_num": ["2da90c7d74bd46a0caf3777415b3ebd3", "324cc204407e99f51b0d6ca0055005e7"],
            "merch_lat": [33.986391, 39.450498],
            "merch_long": [-81.200714, -109.960431],
            "is_fraud": [0, 1],
        }
    )[RAW_COLUMNS]


def test_every_event_is_a_valid_api_request():
    for row in raw_rows().itertuples(index=False):
        Transaction.model_validate(to_event(row))  # raises if the contract is broken


def test_event_has_no_label():
    """The label must never travel with the transaction (it is unknown at payment time)."""
    for row in raw_rows().itertuples(index=False):
        assert "is_fraud" not in to_event(row)


def test_event_values():
    row = next(raw_rows().itertuples(index=False))
    event = to_event(row)
    assert event["cc_num"] == "2291163933867244"  # card number kept as text, not a float
    assert event["timestamp"] == "2020-06-21T12:14:25"
    assert event["customer_dob"] == "1968-03-19"


def test_label_event():
    rows = list(raw_rows().itertuples(index=False))
    assert to_label_event(rows[1]) == {"transaction_id": "324cc204407e99f51b0d6ca0055005e7", "is_fraud": 1}
