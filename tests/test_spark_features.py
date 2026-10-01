"""Training/serving skew test for the Spark streaming job (needs Java + PySpark).

Skipped automatically where PySpark is not installed (e.g. the Windows laptop).
Run it inside the Spark container:
    docker compose run --rm spark python -m pytest tests/test_spark_features.py -q
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from src.training.preprocess import add_card_history_features  # noqa: E402
from tests.test_card_history import HISTORY_COLUMNS, random_transactions  # noqa: E402


def test_spark_features_match_training_features(tmp_path):
    df = random_transactions(n=300, seed=3)
    events = [
        {
            "transaction_id": r.trans_num,
            "cc_num": r.cc_num,
            "amount": r.amt,
            "timestamp": r.trans_date_trans_time.isoformat(),
        }
        for r in df.itertuples()
    ]
    # Four files = four micro-batches: card state must carry over between batches.
    source = tmp_path / "in"
    source.mkdir()
    for i in range(4):
        chunk = events[i * len(events) // 4:(i + 1) * len(events) // 4]
        path = source / f"part{i}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in chunk) + "\n")
        # Spark reads files in modification-time order: make the order explicit,
        # like Kafka's per-partition ordering in production.
        os.utime(path, (1_600_000_000 + i * 60, 1_600_000_000 + i * 60))

    output = tmp_path / "out"
    subprocess.run(
        [sys.executable, "-m", "src.streaming.spark_features", "--source", "files",
         "--input", str(source), "--output", str(output),
         "--checkpoint", str(tmp_path / "ckpt"), "--available-now"],
        check=True,
    )

    spark_features = pd.read_parquet(output).set_index("transaction_id")[HISTORY_COLUMNS].sort_index()
    offline = add_card_history_features(df).set_index("trans_num")[HISTORY_COLUMNS].sort_index()
    assert len(spark_features) == len(offline)
    pd.testing.assert_frame_equal(spark_features, offline, check_dtype=False, check_names=False, rtol=1e-9)
