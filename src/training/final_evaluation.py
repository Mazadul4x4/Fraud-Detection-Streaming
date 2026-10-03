"""One-time final evaluation of the champion model on the held-out TEST set.

The test set (Jun-Dec 2020, fraudTest.csv) was never used to choose features,
models, hyperparameters or the decision threshold. This script applies the
champion model AND its validation-chosen threshold to it, unchanged, and records
the result in MLflow. It refuses to run twice on the same model version (use
--force only if you understand why): repeatedly "checking" the test set turns it
into a second validation set and makes the reported numbers optimistic.

Usage (MLflow running, e.g. `docker compose up -d mlflow`):
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.training.final_evaluation
"""

from __future__ import annotations

import argparse
import os
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import confusion_matrix

from src.training.evaluate import evaluate
from src.training.preprocess import FEATURES, PROCESSED_DIR, PROJECT_ROOT, TARGET

MODEL_NAME = "fraud-detector"
FP_COST = 10.0  # same business assumption used to choose the threshold
RESULTS_PATH = PROJECT_ROOT / "docs" / "final_results.md"


def business_cost(y, scores, amounts, threshold: float, fp_cost: float = FP_COST) -> dict:
    flagged = scores >= threshold
    missed = (y == 1) & ~flagged
    false_alarms = (y == 0) & flagged
    return {
        "no_model_loss": float(amounts[y == 1].sum()),
        "missed_fraud_loss": float(amounts[missed].sum()),
        "false_alarm_cost": float(fp_cost * false_alarms.sum()),
        "total_cost": float(amounts[missed].sum() + fp_cost * false_alarms.sum()),
    }


def summarise(df: pd.DataFrame, model, threshold: float) -> dict:
    y = df[TARGET].to_numpy()
    scores = model.predict_proba(df[FEATURES])[:, 1]
    metrics = evaluate(y, scores, threshold=threshold)
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold).ravel()
    cost = business_cost(y, scores, df["amt"].to_numpy(), threshold)
    return metrics | cost | {"rows": len(df), "frauds": int(y.sum()),
                             "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="evaluate again even if already done")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if version.tags.get("test_evaluated") == "true" and not args.force:
        sys.exit(f"{MODEL_NAME} v{version.version} was already evaluated on the test set "
                 f"(test PR-AUC {version.tags.get('test_pr_auc')}). Refusing to look again.")

    threshold = float(version.tags["threshold"])
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    val = summarise(pd.read_parquet(PROCESSED_DIR / "val.parquet"), model, threshold)
    test = summarise(pd.read_parquet(PROCESSED_DIR / "test.parquet"), model, threshold)

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name=f"final-test-evaluation-v{version.version}"):
        mlflow.set_tags({"model_version": version.version, "evaluation": "held-out test set"})
        mlflow.log_params({"threshold": threshold, "fp_cost": FP_COST})
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val.items()})
    for key in ("pr_auc", "recall_at_1pct_fpr", "precision", "recall"):
        client.set_model_version_tag(MODEL_NAME, version.version, f"test_{key}", f"{test[key]:.4f}")
    client.set_model_version_tag(MODEL_NAME, version.version, "test_evaluated", "true")

    rows = [
        ("Transactions", f"{val['rows']:,}", f"{test['rows']:,}"),
        ("Frauds (rate)", f"{val['frauds']:,} ({100 * val['base_rate']:.2f}%)",
         f"{test['frauds']:,} ({100 * test['base_rate']:.2f}%)"),
        ("PR-AUC", f"{val['pr_auc']:.4f}", f"{test['pr_auc']:.4f}"),
        ("PR-AUC lift over random", f"{val['pr_auc_lift']:.0f}x", f"{test['pr_auc_lift']:.0f}x"),
        ("ROC-AUC", f"{val['roc_auc']:.4f}", f"{test['roc_auc']:.4f}"),
        ("Recall @ 1% FPR", f"{val['recall_at_1pct_fpr']:.4f}", f"{test['recall_at_1pct_fpr']:.4f}"),
        (f"Precision @ threshold {threshold:.2f}", f"{val['precision']:.4f}", f"{test['precision']:.4f}"),
        (f"Recall @ threshold {threshold:.2f}", f"{val['recall']:.4f}", f"{test['recall']:.4f}"),
        ("Frauds caught / missed", f"{val['tp']:,} / {val['fn']:,}", f"{test['tp']:,} / {test['fn']:,}"),
        ("False alarms (share of legit)", f"{val['fp']:,} ({val['fp'] / (val['fp'] + val['tn']):.3%})",
         f"{test['fp']:,} ({test['fp'] / (test['fp'] + test['tn']):.3%})"),
        ("Fraud losses with no model", f"${val['no_model_loss']:,.0f}", f"${test['no_model_loss']:,.0f}"),
        ("Total cost with model", f"${val['total_cost']:,.0f}", f"${test['total_cost']:,.0f}"),
        ("Cost reduction", f"{1 - val['total_cost'] / val['no_model_loss']:.1%}",
         f"{1 - test['total_cost'] / test['no_model_loss']:.1%}"),
    ]
    table = pd.DataFrame(rows, columns=["Metric", "Validation (Jan-Jun 2020)", "Test (Jun-Dec 2020, held out)"])
    print(f"\n{MODEL_NAME} v{version.version} (@champion), threshold {threshold:.2f} chosen on validation\n")
    print(table.to_string(index=False))

    RESULTS_PATH.write_text(
        f"# Final results: `{MODEL_NAME}` v{version.version} (@champion)\n\n"
        f"Decision threshold **{threshold:.2f}**, chosen on validation (false alarm = ${FP_COST:.0f}) "
        "and applied unchanged to the held-out test set, which was not used for any modelling decision.\n\n"
        + table.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved to {RESULTS_PATH.relative_to(PROJECT_ROOT)}; logged to MLflow; model version tagged.")


if __name__ == "__main__":
    main()
