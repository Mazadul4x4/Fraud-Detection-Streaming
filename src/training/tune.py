"""Hyperparameter tuning with Optuna + experiment tracking with MLflow.

1. Optuna searches XGBoost hyperparameters, maximising validation PR-AUC.
   Every trial is logged to MLflow as a nested run.
2. The best configuration is retrained on the full training set.
3. A cost-optimal decision threshold is chosen on the validation set.
4. The model is registered in the MLflow Model Registry and promoted to the
   "champion" alias only if it beats the current champion ("challenger" otherwise).

The test set is NOT used here.

Usage (from the project root):
    python -m src.training.tune --n-trials 3 --train-sample 0.2   # quick smoke run
    python -m src.training.tune --n-trials 20                     # real search

View results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db    ->  http://localhost:5000
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import time

import mlflow
import optuna
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.training.evaluate import evaluate, find_cost_optimal_threshold
from src.training.train import RANDOM_STATE, load_split, make_preprocessor

logger = logging.getLogger(__name__)

# Local SQLite store (supports the Model Registry). Override with the env variable,
# e.g. MLFLOW_TRACKING_URI=http://localhost:5000 once MLflow runs in Docker (Phase 5).
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "fraud-detector"
FP_COST = 10.0  # assumed cost of one false alarm (see notebooks/02_model_experiments.ipynb)

# Recent MLflow versions save scikit-learn models with the secure "skops" format,
# which only loads explicitly trusted classes. XGBoost classes must be listed here.
SKOPS_TRUSTED = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


def suggest_params(trial: optuna.Trial) -> dict:
    """The hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def build_pipeline(params: dict, pos_weight: float) -> Pipeline:
    """Preprocessing + XGBoost with class weighting (the Phase 2 decision)."""
    clf = XGBClassifier(
        **params,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", make_preprocessor()), ("model", clf)])


def trusted_types_kwarg() -> dict:
    """Pass `skops_trusted_types` only if this MLflow version supports it."""
    supported = "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters
    return {"skops_trusted_types": SKOPS_TRUSTED} if supported else {}


def promote(client: MlflowClient, version: str, pr_auc: float) -> str:
    """Champion / challenger: promote only if better than the current champion."""
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_pr_auc = float(champion.tags.get("val_pr_auc", "0"))
    except MlflowException:
        champion, champion_pr_auc = None, float("-inf")

    alias = "champion" if pr_auc > champion_pr_auc else "challenger"
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    if champion is None:
        logger.info("No previous champion -> version %s is the new champion", version)
    else:
        logger.info(
            "Current champion v%s PR-AUC=%.4f | new v%s PR-AUC=%.4f -> alias '%s'",
            champion.version, champion_pr_auc, version, pr_auc, alias,
        )
    return alias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--train-sample", type=float, default=1.0,
        help="Fraction of training rows used DURING the search (final model always uses 100%%)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    if args.train_sample < 1.0:
        X_search = X_train.sample(frac=args.train_sample, random_state=RANDOM_STATE)
        y_search = y_train.loc[X_search.index]
    else:
        X_search, y_search = X_train, y_train
    logger.info("Search on %s training rows, %d trials", f"{len(X_search):,}", args.n_trials)

    with mlflow.start_run(run_name="optuna-search") as parent:
        mlflow.log_params({"n_trials": args.n_trials, "train_sample": args.train_sample})

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
                start = time.perf_counter()
                model = build_pipeline(params, pos_weight).fit(X_search, y_search)
                metrics = evaluate(y_val, model.predict_proba(X_val)[:, 1])
                mlflow.log_params(params)
                mlflow.log_metrics(metrics | {"train_seconds": time.perf_counter() - start})
            logger.info("trial %02d | PR-AUC=%.4f", trial.number, metrics["pr_auc"])
            return metrics["pr_auc"]

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(objective, n_trials=args.n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_search_pr_auc", study.best_value)
        logger.info("Best search PR-AUC=%.4f with %s", study.best_value, study.best_params)

    # --- Final model: best params, FULL training data ---------------------------
    with mlflow.start_run(run_name="final-model") as run:
        mlflow.set_tag("optuna_parent_run", parent.info.run_id)
        model = build_pipeline(study.best_params, pos_weight).fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]

        threshold, cost = find_cost_optimal_threshold(y_val, scores, X_val["amt"], fp_cost=FP_COST)
        metrics = evaluate(y_val, scores, threshold=threshold)
        mlflow.log_params(study.best_params | {"scale_pos_weight": pos_weight, "fp_cost": FP_COST})
        mlflow.log_metrics(metrics | {"threshold": threshold, "val_total_cost": cost})

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_val.head(100), scores[:100]),
            input_example=X_val.head(3),
            registered_model_name=MODEL_NAME,
            **trusted_types_kwarg(),
        )

    client = MlflowClient()
    version = str(model_info.registered_model_version)
    for key, value in {
        "val_pr_auc": metrics["pr_auc"],
        "threshold": threshold,
        "fp_cost": FP_COST,
        "run_id": run.info.run_id,
    }.items():
        client.set_model_version_tag(MODEL_NAME, version, key, str(value))
    alias = promote(client, version, metrics["pr_auc"])

    print(
        f"\nFinal model -> {MODEL_NAME} v{version} (@{alias})\n"
        f"  validation PR-AUC   : {metrics['pr_auc']:.4f}\n"
        f"  recall @ 1% FPR     : {metrics['recall_at_1pct_fpr']:.4f}\n"
        f"  threshold (cost-opt): {threshold:.2f}  -> precision {metrics['precision']:.3f}, "
        f"recall {metrics['recall']:.3f}\n"
        f"  validation cost     : ${cost:,.0f}\n"
        f"\nOpen the dashboard:  mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )


if __name__ == "__main__":
    main()
