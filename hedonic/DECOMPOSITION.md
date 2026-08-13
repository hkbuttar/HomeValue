# Property versus neighborhood decomposition

This analysis fits nested log-price models on an identical training sample and compares
them on an identical future holdout:

- A: structural property features and broad property type;
- B: A plus month effects and a linear market trend;
- C: B plus assessor-neighborhood controls;
- D: C plus available accessibility measures.

Reported increments include in-sample R² and adjusted R² changes and out-of-
sample MAE and RMSE improvements. Positive error improvement means the larger
model predicted the future holdout more accurately.

Model D is deliberately marked unavailable when accessibility features have not
yet been constructed. It activates automatically after accessibility processing provides at
least one supported distance or station-count feature; no placeholder values are
fabricated.

This decomposition quantifies incremental predictive information and conditional
association. It does not identify a causal neighborhood or accessibility effect.
