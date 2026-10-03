# Final results: `fraud-detector` v1 (@champion)

Decision threshold **0.23**, chosen on validation (false alarm = $10) and applied unchanged to the held-out test set, which was not used for any modelling decision.

| Metric                        | Validation (Jan-Jun 2020)   | Test (Jun-Dec 2020, held out)   |
|:------------------------------|:----------------------------|:--------------------------------|
| Transactions                  | 371,825                     | 555,719                         |
| Frauds (rate)                 | 2,286 (0.61%)               | 2,145 (0.39%)                   |
| PR-AUC                        | 0.9827                      | 0.9691                          |
| PR-AUC lift over random       | 160x                        | 251x                            |
| ROC-AUC                       | 0.9998                      | 0.9992                          |
| Recall @ 1% FPR               | 0.9948                      | 0.9860                          |
| Precision @ threshold 0.23    | 0.7880                      | 0.7284                          |
| Recall @ threshold 0.23       | 0.9773                      | 0.9650                          |
| Frauds caught / missed        | 2,234 / 52                  | 2,070 / 75                      |
| False alarms (share of legit) | 601 (0.163%)                | 772 (0.139%)                    |
| Fraud losses with no model    | $1,220,266                  | $1,133,325                      |
| Total cost with model         | $16,064                     | $24,220                         |
| Cost reduction                | 98.7%                       | 97.9%                           |
