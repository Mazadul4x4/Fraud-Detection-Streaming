"""Train baseline fraud models and compare them on the VALIDATION set.

The test set is intentionally NOT used here. It stays locked until the final
evaluation, so that model choices are never tuned on it.

Usage (from the project root):
    python -m src.training.train                  # all models
    python -m src.training.train --model xgboost  # one model
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.training.evaluate import evaluate
from src.training.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET,
)

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def load_split(name: str, processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(processed_dir / f"{name}.parquet")
    return df[FEATURES], df[TARGET]


def make_preprocessor(
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Scale numbers and one-hot encode the merchant category."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def make_model(
    name: str,
    y_train: pd.Series,
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Build a full pipeline (preprocessing + classifier).

    `numeric` / `categorical` let experiments train on a subset of features.
    """
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()

    if name == "dummy":
        # Always predicts the base rate: the floor every real model must beat.
        clf = DummyClassifier(strategy="prior")
    elif name == "logreg":
        # class_weight="balanced": a missed fraud costs ~170x more than a false alarm.
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    elif name == "xgboost":
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=n_neg / n_pos,  # same idea as class_weight="balanced"
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", clf)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["dummy", "logreg", "xgboost", "all"], default="all"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    logger.info("train=%s rows | val=%s rows", f"{len(X_train):,}", f"{len(X_val):,}")

    names = ["dummy", "logreg", "xgboost"] if args.model == "all" else [args.model]
    MODELS_DIR.mkdir(exist_ok=True)
    results = {}

    for name in names:
        logger.info("Training %s ...", name)
        start = time.perf_counter()
        pipeline = make_model(name, y_train).fit(X_train, y_train)
        scores = pipeline.predict_proba(X_val)[:, 1]
        results[name] = evaluate(y_val, scores) | {"train_seconds": time.perf_counter() - start}
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")

    table = pd.DataFrame(results).T.round(4)
    print("\nValidation results (higher is better; base_rate = random guessing):\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
