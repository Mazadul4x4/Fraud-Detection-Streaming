# Model Card: `fraud-detector`

## Model details
- **Type:** XGBoost gradient-boosted trees inside a scikit-learn pipeline (scaling + one-hot encoding)
- **Registry:** MLflow Model Registry, model `fraud-detector`, served via the `@champion` alias
- **Hyperparameters:** selected by Optuna (TPE sampler) maximising validation PR-AUC
- **Class imbalance:** class weighting (`scale_pos_weight` = negatives / positives)
- **Decision threshold:** cost-optimal on validation, stored as a model-version tag.
  The threshold is specific to each model version and must never be reused across versions.

## Intended use
- Real-time scoring of card transactions to **flag likely fraud for review or blocking**
- **Not** intended for: credit decisions, customer profiling, or fully automated account closure
  without human review

## Training data
- Simulated card transactions (Kaggle, kartik2112, CC0-1.0), ~1.85M rows, 983 cards
- Chronological split: train = 2019, validation = Jan-Jun 2020, test = Jun-Dec 2020 (held out)
- Fraud rate: 0.56% (train), 0.62% (validation), 0.39% (test)

## Features
- Transaction: amount (raw + log), hour, night flag, day of week, merchant category,
  customer age, customer-merchant distance, city population (log)
- Card history (past transactions only, leakage-tested): count and amount in the last
  1h and 24h, time since previous transaction, amount relative to the card's past average

## Performance (validation set)
| Model | PR-AUC |
|---|---|
| Dummy (base rate) | 0.0061 |
| Logistic regression | 0.4315 |
| XGBoost, hand-picked parameters | 0.9835 |
| **XGBoost, Optuna-tuned (champion)** | **0.9841** |

- Tuning gave a negligible gain: the model is near the ceiling for this data;
  further improvement needs new information (features), not tuning.
- Ablation: removing real-time card-history features roughly halves precision
  (~5.6x more false alarms at similar recall).
- Business cost (missed fraud = its amount, false alarm = $10 assumed):
  about $1.22M of fraud losses without a model vs about $14K with the model on validation.
- **Test-set results:** pending final evaluation.

## Limitations
- **Simulated data:** fraud patterns are unusually clean (distinct amount clusters, no fraud
  above ~$1,376, ~77% of cards compromised). Real-world performance is expected to be lower.
- **Non-calibrated scores:** class weighting inflates scores; outputs are risk scores,
  not true probabilities.
- **Drift:** monthly fraud rate varied ~2.7x in the data; the model requires monitoring
  and periodic retraining.
- The $10 false-alarm cost is an assumption; the optimal threshold is sensitive to it.

## Ethical considerations
- Names, street address, city, state, ZIP code, gender and job are **excluded** from the
  features (privacy / GDPR data minimisation, and to avoid discrimination).
- Customer age is used; its effect should be audited for disparate impact before real deployment.
- Flagged transactions should go to human review rather than trigger irreversible actions.
