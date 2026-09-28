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
