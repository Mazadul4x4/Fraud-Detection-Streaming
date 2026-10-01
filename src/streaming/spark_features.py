"""Real-time per-card features with PySpark Structured Streaming.

    Kafka "transactions"  ->  Spark (stateful, per card)  ->  feature table (Parquet)

For every transaction, Spark computes the card-history features (transaction count
and amount in the last 1h / 24h, time since the previous transaction, amount vs
the card's past average) using ONLY that card's earlier transactions.

The calculation reuses `CardHistoryStore` from the API, which is unit-tested to
match the training features exactly - so training, the API and Spark all share
one definition of each feature (no training/serving skew). Spark keeps each
card's state between micro-batches with `applyInPandasWithState`.

Run inside Docker (see docker-compose.yml, service `spark`):
    docker compose up -d spark
Local test without Kafka (JSON-lines files as the source):
    python -m src.streaming.spark_features --source files --input <dir> --available-now
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd
import pyspark
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout
from pyspark.sql.types import (
    DoubleType, LongType, StringType, StructField, StructType, TimestampType,
)

from src.api.card_history import CardHistoryStore

logger = logging.getLogger(__name__)

TOPIC = "transactions"
KAFKA_PACKAGE = f"org.apache.spark:spark-sql-kafka-0-10_2.13:{pyspark.__version__}"

# Exactly the event produced by src/ingestion/events.py (= the API request body)
EVENT_SCHEMA = StructType([
    StructField("transaction_id", StringType()),
    StructField("cc_num", StringType()),
    StructField("amount", DoubleType()),
    StructField("category", StringType()),
    StructField("timestamp", StringType()),
    StructField("customer_dob", StringType()),
    StructField("customer_lat", DoubleType()),
    StructField("customer_long", DoubleType()),
    StructField("merchant_lat", DoubleType()),
    StructField("merchant_long", DoubleType()),
    StructField("city_pop", LongType()),
])

HISTORY_FEATURES = [
    "card_txn_count_1h", "card_amt_sum_1h", "card_txn_count_24h", "card_amt_sum_24h",
    "seconds_since_last_txn", "amt_vs_card_mean",
]
OUTPUT_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("cc_num", StringType()),
        StructField("event_time", TimestampType()),
    ]
    + [StructField(name, DoubleType()) for name in HISTORY_FEATURES]
)
# Per-card state kept by Spark between micro-batches (times as epoch seconds).
# The recent-transactions list is stored as a JSON string: array columns are not
# supported in applyInPandasWithState state schemas.
STATE_SCHEMA = StructType([
    StructField("recent_json", StringType()),
    StructField("last_time", DoubleType()),
    StructField("total_count", LongType()),
    StructField("total_amount", DoubleType()),
])


def _to_dt(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)


def _to_epoch(dt: datetime) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp()


def compute_card_features(
    key: tuple, batches: Iterator[pd.DataFrame], state: GroupState
) -> Iterator[pd.DataFrame]:
    """Called by Spark once per card per micro-batch, with that card's new events."""
    card = key[0]
    store = CardHistoryStore()
    if state.exists:
        recent_json, last_time, total_count, total_amount = state.get
        recent = json.loads(recent_json)  # [[epoch, amount], ...]
        store.import_state(card, {
            "recent_times": [_to_dt(t) for t, _ in recent],
            "recent_amounts": [a for _, a in recent],
            "last_time": _to_dt(last_time) if last_time is not None else None,
            "total_count": total_count,
            "total_amount": total_amount,
        })

    events = pd.concat(list(batches)).sort_values(["event_epoch", "transaction_id"])
    rows = []
    for e in events.itertuples(index=False):
        t = _to_dt(e.event_epoch)
        features = store.features(card, t, e.amount)
        store.add(card, t, e.amount)
        rows.append({"transaction_id": e.transaction_id, "cc_num": card, "event_time": t, **features})

    snapshot = store.export_state(card)
    recent = [[_to_epoch(t), a] for t, a in zip(snapshot["recent_times"], snapshot["recent_amounts"])]
    state.update((
        json.dumps(recent),
        _to_epoch(snapshot["last_time"]),
        snapshot["total_count"],
        snapshot["total_amount"],
    ))
    yield pd.DataFrame(rows, columns=[f.name for f in OUTPUT_SCHEMA.fields])


def create_spark(source: str) -> SparkSession:
    builder = (
        SparkSession.builder.appName("card-history-features")
        .master(os.getenv("SPARK_MASTER", "local[2]"))
        .config("spark.sql.session.timeZone", "UTC")    # event times are naive local times
        .config("spark.sql.shuffle.partitions", "3")    # default 200 is far too many for a laptop
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "1g"))
        .config("spark.ui.showConsoleProgress", "false")
    )
    if os.getenv("IVY_HOME"):  # cache downloaded jars in a Docker volume
        builder = builder.config("spark.jars.ivy", os.environ["IVY_HOME"])
    if source == "kafka":
        builder = builder.config("spark.jars.packages", KAFKA_PACKAGE)
    return builder.getOrCreate()


def read_events(spark: SparkSession, args) -> DataFrame:
    if args.source == "kafka":
        raw = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", args.bootstrap)
            .option("subscribe", TOPIC)
            .option("startingOffsets", "earliest")
            .option("maxOffsetsPerTrigger", args.max_per_batch)
            .load()
            .select(F.col("value").cast("string").alias("json"))
        )
    else:  # JSON-lines files: used for local testing without Kafka
        raw = (
            spark.readStream.format("text")
            .option("maxFilesPerTrigger", 1)
            .load(args.input)
            .select(F.col("value").alias("json"))
        )
    return (
        raw.select(F.from_json("json", EVENT_SCHEMA).alias("e"))
        .select("e.*")
        .withColumn("event_epoch", F.to_timestamp("timestamp").cast("double"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["kafka", "files"], default="kafka")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092"))
    parser.add_argument("--input", help="folder of JSON-lines files (--source files)")
    parser.add_argument("--output", default=os.getenv("FEATURES_PATH", "data/processed/streaming_features"))
    parser.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH", "checkpoints/card_features"))
    parser.add_argument("--max-per-batch", type=int, default=5000, help="max Kafka messages per micro-batch")
    parser.add_argument("--trigger-seconds", type=int, default=10)
    parser.add_argument("--available-now", action="store_true", help="process what exists, then stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    logging.getLogger("py4j").setLevel(logging.WARNING)  # silence Spark<->Python chatter

    spark = create_spark(args.source)
    spark.sparkContext.setLogLevel("WARN")

    features = (
        read_events(spark, args)
        .select("transaction_id", "cc_num", "amount", "event_epoch")
        .groupBy("cc_num")
        .applyInPandasWithState(
            compute_card_features,
            outputStructType=OUTPUT_SCHEMA,
            stateStructType=STATE_SCHEMA,
            outputMode="append",
            timeoutConf=GroupStateTimeout.NoTimeout,
        )
    )

    def write_batch(batch: DataFrame, batch_id: int) -> None:
        batch.write.mode("append").parquet(args.output)
        n = batch.count()
        if n:
            busiest = batch.orderBy(F.desc("card_txn_count_1h")).first()
            logger.info(
                "batch %d: %d transactions -> features | busiest card ...%s: %d txns in last 1h",
                batch_id, n, busiest["cc_num"][-4:], busiest["card_txn_count_1h"],
            )

    writer = (
        features.writeStream.foreachBatch(write_batch)
        .option("checkpointLocation", args.checkpoint)
        .outputMode("append")
    )
    writer = writer.trigger(availableNow=True) if args.available_now else writer.trigger(
        processingTime=f"{args.trigger_seconds} seconds"
    )
    logger.info("Streaming from %s -> %s", args.source, args.output)
    writer.start().awaitTermination()


if __name__ == "__main__":
    main()
