"""Business-cost calculation used in the final evaluation."""

import numpy as np
import pytest

from src.training.final_evaluation import business_cost


def test_business_cost():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])   # catches fraud 1, misses fraud 2, 1 false alarm
    amounts = np.array([500.0, 300.0, 20.0, 40.0])
    cost = business_cost(y, scores, amounts, threshold=0.5, fp_cost=10)
    assert cost["no_model_loss"] == pytest.approx(800.0)
    assert cost["missed_fraud_loss"] == pytest.approx(300.0)
    assert cost["false_alarm_cost"] == pytest.approx(10.0)
    assert cost["total_cost"] == pytest.approx(310.0)
