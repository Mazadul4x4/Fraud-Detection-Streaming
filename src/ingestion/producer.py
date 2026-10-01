"""Replay the held-out transactions into Kafka/Redpanda as a live stream.

Each transaction becomes one JSON message on the `transactions` topic, keyed by
card number. Kafka keeps messages with the same key in the same partition, in
order, so every card's transactions are consumed in the right sequence - which
the per-card history features depend on.

Usage (from the project root, with `docker compose up -d` running):
    python -m src.ingestion.producer --limit 2000 --rate 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.ingestion.events import (
    LABELS_TOPIC,
    RAW_COLUMNS,
    TRANSACTIONS_TOPIC,
    to_event,
    to_label_event,
)
from src.training.preprocess import RAW_DIR

logger = logging.getLogger(__name__)

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
PARTITIONS = 3


def ensure_topics(bootstrap: str, topics: list[str]) -> None:
    """Create topics if they do not exist yet (idempotent)."""
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = admin.list_topics(timeout=10).topics
    new = [NewTopic(t, num_partitions=PARTITIONS, replication_factor=1) for t in topics if t not in existing]
    if not new:
        logger.info("Topics already exist: %s", ", ".join(topics))
        return
    for topic, future in admin.create_topics(new).items():
        try:
            future.result()
            logger.info("Created topic '%s' (%d partitions)", topic, PARTITIONS)
        except KafkaException as exc:
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise


def load_transactions(source: Path, limit: int | None, skip: int = 0) -> pd.DataFrame:
    """Read `limit` transactions after skipping the first `skip` data rows (file is in time order)."""
    df = pd.read_csv(
        source,
        usecols=RAW_COLUMNS,
        skiprows=range(1, skip + 1),  # row 0 is the header
        nrows=limit,
        dtype={"cc_num": "string"},
    )
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=RAW_DIR / "fraudTest.csv")
    parser.add_argument("--limit", type=int, default=2000, help="number of transactions (0 = all)")
    parser.add_argument("--skip", type=int, default=0, help="skip the first N transactions (continue a replay)")
    parser.add_argument("--rate", type=float, default=50, help="messages per second (0 = as fast as possible)")
    parser.add_argument("--bootstrap", default=BOOTSTRAP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ensure_topics(args.bootstrap, [TRANSACTIONS_TOPIC, LABELS_TOPIC])
    df = load_transactions(args.source, args.limit or None, args.skip)
    logger.info("Loaded %s transactions (%s -> %s), fraud rate %.2f%%",
                f"{len(df):,}", df["trans_date_trans_time"].min(), df["trans_date_trans_time"].max(),
                100 * df["is_fraud"].mean())

    producer = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 5, "acks": "all"})
    failures = 0

    def on_delivery(err, msg):
        nonlocal failures
        if err is not None:
            failures += 1
            logger.error("Delivery failed for key %s: %s", msg.key(), err)

    stop = False

    def handle_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)  # Ctrl+C -> finish cleanly

    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    start = time.perf_counter()
    sent = 0
    for row in df.itertuples(index=False):
        if stop:
            logger.info("Stopping early (Ctrl+C)")
            break
        key = str(row.cc_num).encode()
        producer.produce(TRANSACTIONS_TOPIC, key=key, value=json.dumps(to_event(row)), on_delivery=on_delivery)
        producer.produce(LABELS_TOPIC, key=key, value=json.dumps(to_label_event(row)), on_delivery=on_delivery)
        producer.poll(0)  # serve delivery callbacks
        sent += 1
        if sent % 500 == 0:
            logger.info("sent %s messages (%.0f msg/s)", f"{sent:,}", sent / (time.perf_counter() - start))
        if interval:
            # Pace the stream: sleep until this message's scheduled time.
            delay = start + sent * interval - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    remaining = producer.flush(30)
    elapsed = time.perf_counter() - start
    logger.info(
        "Done: %s transactions in %.1fs (%.0f msg/s), %d delivery failures, %d not flushed",
        f"{sent:,}", elapsed, sent / elapsed if elapsed else 0, failures, remaining,
    )


if __name__ == "__main__":
    main()
