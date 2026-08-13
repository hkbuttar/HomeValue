# Nonlinear valuation models

This analysis trains Random Forest, HistGradientBoosting, and XGBoost on an identical
future-year holdout. All models predict log sale price and use a training-
residual smearing factor for dollar predictions.

The feature matrix uses explicit allowlists for structural, temporal,
neighborhood, accessibility, and prior-only spatial features. It excludes the
target, identifiers, raw latitude/longitude, post-sale flags, other model
predictions, and future/contemporaneous nearby outcomes. Numeric imputation and
categorical encoding are fit only on the training period. High-cardinality
categorical features are reported and omitted rather than exploding the matrix.

The fitted preprocessing and models are stored together. When existing baseline,
hedonic, comparable, or spatial prediction artifacts overlap the same held-out
sales, their metrics are reported beside ML. A winning ML model establishes
predictive superiority on that test sample, not causal or universal superiority.

