# Baseline valuation models

Four intentionally simple estimators establish the minimum standard for later
econometric and machine-learning models:

1. global median sale price;
2. median by broad property type (falling back to property class);
3. median by assessor neighborhood (falling back to census tract);
4. median price per square foot by neighborhood and property type, multiplied
   by the subject building area.

All medians and fallback values are learned from the training period only.
Previously unseen groups use the global training median. The PPSF estimator uses
the global training PPSF when its segment is unseen and the global price median
when subject square footage is unavailable.

Validation is strictly out of time: by default the latest observed year is the
test period, while `--test-start-year` can reserve several recent years. Metrics
are MAE, median absolute error, RMSE, MAPE, and R². Predictions and complete
fitted lookup tables are retained for reproducibility.

