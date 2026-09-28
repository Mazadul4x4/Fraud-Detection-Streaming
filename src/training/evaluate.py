"""Evaluation metrics for highly imbalanced fraud detection.

Accuracy is deliberately NOT reported: with ~0.5% fraud, a model that never
predicts fraud already scores ~99.5% accuracy (see notebooks/01_eda.ipynb).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y_true, scores, max_fpr: float = 0.01) -> float:
    """Share of frauds caught while flagging at most `max_fpr` of legitimate transactions."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= max_fpr].max())


def evaluate(y_true, scores, threshold: float = 0.5) -> dict[str, float]:
    """All metrics we track, for one set of predicted fraud probabilities."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    base_rate = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, scores))
    return {
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / base_rate,  # how many times better than random guessing
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "base_rate": base_rate,
    }


def find_cost_optimal_threshold(
    y_true,
    scores,
    amounts,
    fp_cost: float = 10.0,
    thresholds=None,
) -> tuple[float, float]:
    """Threshold that minimises total business cost.

    Cost model: a missed fraud (false negative) costs its transaction amount;
    a false alarm (false positive) costs `fp_cost` (review + customer friction).
    Returns (best_threshold, total_cost_at_that_threshold).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    amounts = np.asarray(amounts, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    costs = []
    for t in thresholds:
        flagged = scores >= t
        missed_loss = amounts[(y_true == 1) & ~flagged].sum()
        false_alarms = ((y_true == 0) & flagged).sum()
        costs.append(missed_loss + fp_cost * false_alarms)

    best = int(np.argmin(costs))
    return float(thresholds[best]), float(costs[best])
