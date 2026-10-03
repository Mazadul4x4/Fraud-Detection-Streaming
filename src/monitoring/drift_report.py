"""Weekly model monitoring over the held-out period: data drift, prediction drift, performance.

For every week of the test period (June-December 2020) the champion model scores
the transactions, then we compare that week with the validation period (reference):

  * Data drift        Evidently: share of monitored features whose distribution moved
  * Prediction drift  Evidently: distance between this week's and the reference fraud scores
  * Performance       needs labels (in reality: chargebacks, days later)
                      Retrain trigger = recall @ 1% FPR drops > 5% vs reference.
                      Recall @ FPR is used rather than PR-AUC because PR-AUC also falls
                      when fraud simply becomes rarer (prior shift), without any model decay.

`--inject-drift` simulates a sudden change from a given date (amounts x factor),
to measure how fast the monitoring detects it.

Usage (MLflow running, e.g. `docker compose up -d mlflow`):
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.monitoring.drift_report
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.monitoring.drift_report --inject-drift 2020-10-01
"""

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")  # no usage data sent anywhere

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from evidently import Report  # noqa: E402
from evidently.metrics import ValueDrift  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

from src.training.evaluate import evaluate  # noqa: E402
from src.training.preprocess import FEATURES, PROCESSED_DIR, PROJECT_ROOT, TARGET, TIME_COL  # noqa: E402

logger = logging.getLogger(__name__)

MONITORED = [
    "amt", "hour", "is_night", "category", "age_years", "distance_km",
    "card_txn_count_24h", "card_amt_sum_24h", "amt_vs_card_mean", "seconds_since_last_txn",
]
DRIFT_THRESHOLD = 0.1          # normalised Wasserstein / Jensen-Shannon distance
DATASET_DRIFT_SHARE = 0.3      # alert if >= 30% of monitored features drift
RETRAIN_RECALL_DROP = 0.05     # alert if recall @ 1% FPR drops > 5% (relative)
REFERENCE_SAMPLE, WINDOW_SAMPLE = 20_000, 5_000
REPORTS_DIR = PROJECT_ROOT / "reports"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "monitoring_summary.md"


def load_model(source: str):
    if source == "mlflow":
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        return mlflow.sklearn.load_model("models:/fraud-detector@champion")
    import joblib

    return joblib.load(PROJECT_ROOT / "models" / "xgboost.joblib")


def inject_drift(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Simulated shift: every amount multiplied by `factor` (e.g. sudden inflation / new spending habits)."""
    df = df.copy()
    df["amt"] *= factor
    df["log_amt"] = np.log1p(df["amt"])
    for col in ("card_amt_sum_1h", "card_amt_sum_24h"):
        df[col] *= factor
    return df


def drift_metrics(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[dict, object]:
    report = Report([
        DataDriftPreset(columns=MONITORED, num_method="wasserstein", cat_method="jensenshannon",
                        threshold=DRIFT_THRESHOLD, drift_share=DATASET_DRIFT_SHARE),
        ValueDrift(column="score", method="wasserstein", threshold=DRIFT_THRESHOLD),
    ])
    snapshot = report.run(current, reference)
    out = {"drifted_features": []}
    for m in snapshot.dict()["metrics"]:
        name, value = m["metric_name"], m["value"]
        if name.startswith("DriftedColumnsCount"):
            out["drift_share"] = value["share"]
        elif name.startswith("ValueDrift(column=score"):
            out["score_drift"] = value
        elif name.startswith("ValueDrift(column="):
            column = name.split("column=")[1].split(",")[0]
            out[f"{column}_drift"] = value
            if value >= DRIFT_THRESHOLD:
                out["drifted_features"].append(column)
    return out, snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-source", choices=["mlflow", "local"], default="mlflow")
    parser.add_argument("--inject-drift", metavar="DATE", help="simulate drift from this date, e.g. 2020-10-01")
    parser.add_argument("--drift-factor", type=float, default=1.8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    model = load_model(args.model_source)
    reference = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    if args.inject_drift:
        drifted = test[TIME_COL] >= pd.Timestamp(args.inject_drift)
        test.loc[drifted] = inject_drift(test.loc[drifted], args.drift_factor)
        logger.info("Injected drift (amount x%.1f) into %s transactions from %s",
                    args.drift_factor, f"{drifted.sum():,}", args.inject_drift)

    reference["score"] = model.predict_proba(reference[FEATURES])[:, 1]
    test["score"] = model.predict_proba(test[FEATURES])[:, 1]
    ref_metrics = evaluate(reference[TARGET], reference["score"])
    ref_sample = reference.sample(min(REFERENCE_SAMPLE, len(reference)), random_state=42)
    logger.info("Reference (validation): PR-AUC %.4f, recall@1%%FPR %.4f",
                ref_metrics["pr_auc"], ref_metrics["recall_at_1pct_fpr"])

    REPORTS_DIR.mkdir(exist_ok=True)
    rows = []
    for week_start, week in test.groupby(pd.Grouper(key=TIME_COL, freq="W-MON", label="left")):
        if len(week) < 1000 or week[TARGET].sum() < 5:
            continue  # too small to judge
        perf = evaluate(week[TARGET], week["score"])
        drift, snapshot = drift_metrics(ref_sample, week.sample(min(WINDOW_SAMPLE, len(week)), random_state=42))
        recall_drop = 1 - perf["recall_at_1pct_fpr"] / ref_metrics["recall_at_1pct_fpr"]
        alerts = []
        if drift["drift_share"] >= DATASET_DRIFT_SHARE:
            alerts.append("DATA_DRIFT")
        if drift["score_drift"] >= DRIFT_THRESHOLD:
            alerts.append("PREDICTION_DRIFT")
        if recall_drop > RETRAIN_RECALL_DROP:
            alerts.append("RETRAIN")
        rows.append({
            "week": week_start.date().isoformat(),
            "transactions": len(week),
            "fraud_rate_%": round(100 * week[TARGET].mean(), 2),
            "pr_auc": round(perf["pr_auc"], 4),
            "recall_at_1pct_fpr": round(perf["recall_at_1pct_fpr"], 4),
            "drifted_features_%": round(100 * drift["drift_share"]),
            "amt_drift": round(drift["amt_drift"], 3),
            "score_drift": round(drift["score_drift"], 3),
            "alerts": ", ".join(alerts) or "-",
        })
        if alerts:
            snapshot.save_html(str(REPORTS_DIR / f"drift_{week_start.date()}.html"))

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    alerted = table[table["alerts"] != "-"]
    print(f"\nWeeks with alerts: {len(alerted)} of {len(table)}  (HTML reports in {REPORTS_DIR.name}/)")

    title = f"with simulated drift from {args.inject_drift} (amount x{args.drift_factor})" \
        if args.inject_drift else "(no simulated drift)"
    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(
        f"# Weekly monitoring {title}\n\n"
        f"Reference: validation period, PR-AUC {ref_metrics['pr_auc']:.4f}, "
        f"recall @ 1% FPR {ref_metrics['recall_at_1pct_fpr']:.4f}.\n"
        f"Alerts: DATA_DRIFT if >= {DATASET_DRIFT_SHARE:.0%} of monitored features drift "
        f"(distance >= {DRIFT_THRESHOLD}); PREDICTION_DRIFT if the score distribution distance >= "
        f"{DRIFT_THRESHOLD}; RETRAIN if recall @ 1% FPR drops > {RETRAIN_RECALL_DROP:.0%}.\n\n"
        + table.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Summary written to %s", SUMMARY_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
