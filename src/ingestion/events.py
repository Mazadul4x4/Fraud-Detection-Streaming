"""Conversion between raw dataset rows and streaming events.

Kept separate from the Kafka code so it can be unit-tested without a broker,
and so the event format is checked against the API's request schema.
"""

from __future__ import annotations

import pandas as pd

TRANSACTIONS_TOPIC = "transactions"
LABELS_TOPIC = "fraud-labels"

RAW_COLUMNS = [
    "trans_date_trans_time", "cc_num", "category", "amt", "lat", "long",
    "city_pop", "dob", "trans_num", "merch_lat", "merch_long", "is_fraud",
]


def to_event(row) -> dict:
    """One raw CSV row -> transaction event (exactly the API's /score request body).

    The fraud label is deliberately NOT included: in reality the payment system
    does not know the label when the transaction happens.
    """
    return {
        "transaction_id": str(row.trans_num),
        "cc_num": str(row.cc_num),
        "amount": float(row.amt),
        "category": str(row.category),
        "timestamp": pd.Timestamp(row.trans_date_trans_time).isoformat(),
        "customer_dob": pd.Timestamp(row.dob).date().isoformat(),
        "customer_lat": float(row.lat),
        "customer_long": float(row.long),
        "merchant_lat": float(row.merch_lat),
        "merchant_long": float(row.merch_long),
        "city_pop": int(row.city_pop),
    }


def to_label_event(row) -> dict:
    """The ground-truth label, published on a separate topic.

    Real fraud labels (chargebacks) arrive days or weeks later; keeping them on
    their own topic lets monitoring join them with predictions later (Phase 9).
    """
    return {"transaction_id": str(row.trans_num), "is_fraud": int(row.is_fraud)}
