"""End-to-end parity check on REAL data: online (Redis) features vs training features.

Uses a separate Redis database (default db 15) so the API's live state is untouched:
  1. backfill train + validation history into it,
  2. stream the first N test-period transactions through RedisCardHistory
     (exactly what the API does per request),
  3. compare with the features in data/processed/test.parquet (training pipeline).

Usage (Redis running):
    python -m src.feature_store.verify_parity --n 5000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.feature_store.backfill import build_snapshots
from src.feature_store.redis_state import RedisCardHistory
from src.training.preprocess import CARD_COL, ID_COL, PROCESSED_DIR, TIME_COL

HISTORY_FEATURES = [
    "card_txn_count_1h", "card_amt_sum_1h", "card_txn_count_24h", "card_amt_sum_24h",
    "seconds_since_last_txn", "amt_vs_card_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=5000, help="number of test transactions to check")
    parser.add_argument("--redis-url", default=os.getenv("PARITY_REDIS_URL", "redis://localhost:6379/15"))
    args = parser.parse_args()

    columns = [ID_COL, CARD_COL, TIME_COL, "amt"]
    history = pd.concat(
        [pd.read_parquet(PROCESSED_DIR / f"{s}.parquet", columns=columns) for s in ("train", "val")]
    )
    test = (
        pd.read_parquet(PROCESSED_DIR / "test.parquet", columns=columns + HISTORY_FEATURES)
        .sort_values([TIME_COL, ID_COL])
        .head(args.n)
    )

    store = RedisCardHistory.from_url(args.redis_url)
    store.clear()
    store.put_snapshots(build_snapshots(history))

    online = pd.DataFrame(
        [store.features_and_add(c, t.to_pydatetime(), float(a))
         for c, t, a in zip(test[CARD_COL], test[TIME_COL], test["amt"])],
        index=test.index,
    )
    store.clear()

    print(f"Checked {len(test):,} test transactions "
          f"({test[TIME_COL].min()} -> {test[TIME_COL].max()}), {test[CARD_COL].nunique()} cards\n")
    all_ok = True
    for name in HISTORY_FEATURES:
        diff = np.abs(online[name].to_numpy() - test[name].to_numpy())
        ok = np.allclose(online[name], test[name], rtol=1e-9, atol=1e-9)
        all_ok &= ok
        print(f"  {'OK  ' if ok else 'DIFF'} {name:<24} max |difference| = {diff.max():.2e}")
    print("\nPARITY: online features == training features" if all_ok else "\nPARITY FAILED")


if __name__ == "__main__":
    main()
