"""Unit tests for model pipelines."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.training.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.training.train import make_model


def make_data(n: int = 2_000, fraud_rate: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(NUMERIC_FEATURES))), columns=NUMERIC_FEATURES)
    X[CATEGORICAL_FEATURES[0]] = rng.choice(["misc_net", "home", "travel"], n)
    y = pd.Series((rng.random(n) < fraud_rate).astype(int))
    return X, y


def test_resampling_changes_training_data_only():
    """Resamplers must run during fit but never drop or add rows at prediction time."""
    X, y = make_data()
    for samplers in ([RandomUnderSampler(sampling_strategy=0.5, random_state=0)],
                     [SMOTE(sampling_strategy=0.5, random_state=0)]):
        model = make_model("xgboost", y, samplers=samplers, class_weighting=False).fit(X, y)
        assert len(model.predict_proba(X)) == len(X)


def test_class_weighting_flag():
    X, y = make_data()
    weighted = make_model("xgboost", y, class_weighting=True)
    unweighted = make_model("xgboost", y, class_weighting=False)
    assert weighted["model"].scale_pos_weight > 1
    assert unweighted["model"].scale_pos_weight == 1.0
