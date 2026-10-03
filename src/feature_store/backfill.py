"""Pre-load every card's history into Redis before the live stream starts.

Without this, the API starts the test period with empty card histories, while in
reality each card already has ~18 months of past transactions. The backfill
replays the training + validation periods through the same CardHistoryStore and
writes the resulting state of every card to Redis, so the first live transaction
of each card sees exactly the history the training features saw.

Usage (Redis running, e.g. `docker compose up -d redis`):
    python -m src.feature_store.backfill --reset
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import pandas as pd

from src.api.card_history import CardHistoryStore
from src.feature_store.redis_state import RedisCardHistory
from src.training.preprocess import CARD_COL, ID_COL, PROCESSED_DIR, TIME_COL

logger = logging.getLogger(__name__)


def build_snapshots(history: pd.DataFrame) -> dict[str, dict]:
    """Replay transactions in training order; return each card's final state."""
    history = history.sort_values([CARD_COL, TIME_COL, ID_COL])
    store = CardHistoryStore()
    for card, t, amount in zip(history[CARD_COL], history[TIME_COL], history["amt"]):
        store.add(card, t.to_pydatetime(), float(amount))
    return {card: store.export_state(card) for card in history[CARD_COL].unique()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--reset", action="store_true", help="delete existing card states first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    columns = [ID_COL, CARD_COL, TIME_COL, "amt"]
    history = pd.concat(
        [pd.read_parquet(PROCESSED_DIR / f"{split}.parquet", columns=columns) for split in ("train", "val")],
        ignore_index=True,
    )
    logger.info("Replaying %s historical transactions (%s -> %s)",
                f"{len(history):,}", history[TIME_COL].min(), history[TIME_COL].max())

    start = time.perf_counter()
    snapshots = build_snapshots(history)
    logger.info("Built state for %d cards in %.1fs", len(snapshots), time.perf_counter() - start)

    store = RedisCardHistory.from_url(args.redis_url)
    if args.reset:
        logger.info("Deleted %d existing card states", store.clear())
    store.put_snapshots(snapshots)
    logger.info("Redis now holds %d card states", store.card_count())


if __name__ == "__main__":
    main()
