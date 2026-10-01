"""Consume the transaction stream and score every event with the fraud API.

    Kafka topic "transactions"  ->  this consumer  ->  POST /score  ->  decision

Usage (API running, e.g. `docker compose up -d`):
    python -m src.ingestion.score_stream --max-messages 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter

import httpx
from confluent_kafka import Consumer

from src.ingestion.events import TRANSACTIONS_TOPIC

logger = logging.getLogger(__name__)

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
API_URL = os.getenv("API_URL", "http://localhost:8000")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-messages", type=int, default=2000)
    parser.add_argument("--group", default="fraud-scorer", help="consumer group id")
    parser.add_argument("--idle-timeout", type=float, default=15, help="stop after N seconds without messages")
    parser.add_argument("--bootstrap", default=BOOTSTRAP)
    parser.add_argument("--api", default=API_URL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": args.group,
        "auto.offset.reset": "earliest",  # a new group starts from the beginning of the topic
    })
    consumer.subscribe([TRANSACTIONS_TOPIC])
    client = httpx.Client(base_url=args.api, timeout=10)

    decisions: Counter = Counter()
    api_latencies: list[float] = []
    errors = 0
    processed = 0
    start = last_message = time.perf_counter()

    try:
        while processed < args.max_messages:
            msg = consumer.poll(1.0)
            if msg is None:
                if time.perf_counter() - last_message > args.idle_timeout:
                    logger.info("No new messages for %.0fs, stopping", args.idle_timeout)
                    break
                continue
            if msg.error():
                logger.error("Kafka error: %s", msg.error())
                continue
            last_message = time.perf_counter()

            response = client.post("/score", json=json.loads(msg.value()))
            if response.status_code != 200:
                errors += 1
                logger.warning("API returned %s: %s", response.status_code, response.text[:200])
                continue
            body = response.json()
            decisions[body["decision"]] += 1
            api_latencies.append(body["latency_ms"])
            processed += 1

            if body["decision"] != "approve":
                logger.info("%-7s | score=%.3f | card=...%s | %s",
                            body["decision"].upper(), body["fraud_score"],
                            msg.key().decode()[-4:], body["transaction_id"])
            if processed % 500 == 0:
                logger.info("processed %s | %s", f"{processed:,}", dict(decisions))
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        consumer.close()  # commits offsets, so a restart continues where we stopped
        client.close()

    elapsed = time.perf_counter() - start
    latencies = sorted(api_latencies)
    p50 = latencies[len(latencies) // 2] if latencies else float("nan")
    p99 = latencies[int(len(latencies) * 0.99) - 1] if latencies else float("nan")
    print(
        f"\nScored {processed:,} transactions in {elapsed:.1f}s ({processed / elapsed:.1f}/s), "
        f"{errors} API errors\n"
        f"Decisions: {dict(decisions)}\n"
        f"API latency (server-side): p50={p50:.1f} ms, p99={p99:.1f} ms"
    )


if __name__ == "__main__":
    main()
