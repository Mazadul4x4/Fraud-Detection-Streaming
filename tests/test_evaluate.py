"""Unit tests for the evaluation metrics."""

import numpy as np
import pytest

from src.training.evaluate import evaluate, recall_at_fpr


def test_perfect_model():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.1, 0.3, 0.9, 0.8])
    m = evaluate(y, scores)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["recall_at_1pct_fpr"] == pytest.approx(1.0)
    assert m["precision"] == m["recall"] == pytest.approx(1.0)


def test_random_scores_pr_auc_close_to_base_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.01).astype(int)
    m = evaluate(y, rng.random(len(y)))
    assert m["pr_auc"] == pytest.approx(y.mean(), abs=0.003)
    assert m["pr_auc_lift"] == pytest.approx(1.0, abs=0.3)
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.02)


def test_recall_at_fpr_respects_limit():
    # 100 legit, 10 fraud. Fraud scores overlap with the top 5 legit scores.
    y = np.array([0] * 100 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.95, 100), np.linspace(0.5, 1.0, 10)])
    # With max 1% FPR we may flag at most 1 legit transaction.
    assert 0 < recall_at_fpr(y, scores, 0.01) < 1


def test_cost_optimal_threshold_balances_both_errors():
    from src.training.evaluate import find_cost_optimal_threshold

    # Two frauds worth $500 each, scored 0.3 and 0.9; one legit scored 0.5.
    y = np.array([1, 1, 0])
    scores = np.array([0.3, 0.9, 0.5])
    amounts = np.array([500.0, 500.0, 20.0])

    # Cheap false alarms: catching both frauds (threshold <= 0.3) is optimal.
    t, cost = find_cost_optimal_threshold(y, scores, amounts, fp_cost=10)
    assert t <= 0.3 and cost == 10

    # Very expensive false alarms: better to flag only the 0.9 fraud.
    t, cost = find_cost_optimal_threshold(y, scores, amounts, fp_cost=10_000)
    assert 0.5 < t <= 0.9 and cost == 500
