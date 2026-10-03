# Model Card: `fraud-detector`

## Model details
- **Type:** XGBoost gradient-boosted trees inside a scikit-learn pipeline (scaling + one-hot encoding)
- **Registry:** MLflow Model Registry, model `fraud-detector`, served via the `@champion` alias
- **Hyperparameters:** selected by Optuna (TPE sampler) maximising validation PR-AUC
- **Class imbalance:** class weighting (`scale_pos_weight` = negatives / positives)
- **Decision threshold:** cost-optimal on validation (0.23 for the current champion), stored as a
  model-version tag. Thresholds are specific to each model version and must never be reused across versions.
- **Promotion rule:** a new version becomes `@champion` only if its validation PR-AUC beats the current one.

## Intended use
- Real-time scoring of card transactions to **approve, send to human review, or decline**
- **Not** intended for: credit decisions, customer profiling, or automated account closure without human review

## Training data
- Simulated card transactions (Kaggle, kartik2112, CC0-1.0), ~1.85M rows, 983 cards
- Chronological split: train = 2019, validation = Jan-Jun 2020, test = Jun-Dec 2020 (held out)
- Fraud rate: 0.56% (train), 0.62% (validation), 0.39% (test)

## Features
- Transaction: amount (raw + log), hour, night flag, day of week, merchant category, customer age,
  customer-merchant distance, city population (log)
- Card history (past transactions only): count and amount in the last 1h and 24h, time since the
  previous transaction, amount relative to the card's past average
- Serving: card history is kept per card in Redis and computed at request time with the same code
  as training; verified identical to the training features on 5,000 real held-out transactions

## Performance
| Metric | Validation | Test (held out, evaluated once) |
|---|---|---|
| PR-AUC | 0.9827 | 0.9691 |
| ROC-AUC | 0.9998 | 0.9992 |
| Recall @ 1% FPR | 0.9948 | 0.9860 |
| Precision / recall @ threshold 0.23 | 0.79 / 0.98 | 0.73 / 0.97 |
| False alarms (share of legitimate) | 0.163% | 0.139% |
| Fraud-related cost reduction vs no model | 98.7% | 97.9% |

- Baselines (validation): logistic regression PR-AUC 0.43, dummy 0.006.
- Ablation: without real-time card-history features, precision falls from 0.82 to 0.44 (~5.6x more false alarms).
- Business cost assumes a missed fraud costs its amount and a false alarm costs $10.

## Monitoring
- Weekly Evidently report: data drift (10 features), prediction drift, and performance on delayed labels.
- Retrain trigger: recall @ 1% FPR drops more than 5% vs the reference (PR-AUC is not used because it
  also falls when fraud becomes rarer).
- Over 28 test weeks recall @ 1% FPR stayed 0.95-1.0; seasonal data drift occurred without model decay.
- A simulated x1.8 amount shift was detected in its first week and cut recall @ 1% FPR to 0.66-0.87.

## Limitations
- **Simulated data:** fraud patterns are unusually clean (distinct amount clusters, no fraud above
  ~$1,376, ~77% of cards compromised). Real-world performance is expected to be lower.
- **Sensitive to amount shifts:** the model relies on absolute amount ranges (shown by the drift experiment).
- **Non-calibrated scores:** class weighting inflates scores; outputs are risk scores, not probabilities.
- The $10 false-alarm cost is an assumption; the optimal threshold is sensitive to it.
- Serving latency on the development laptop: p99 120 ms single user, above the 50 ms target.

## Ethical considerations
- Names, street address, city, state, ZIP code, gender and job are **excluded** from the features
  (privacy / GDPR data minimisation, and to avoid discrimination).
- Customer age is used; its effect should be audited for disparate impact before real deployment.
- Flagged transactions should go to human review rather than trigger irreversible actions.
