# Spatial lag model

Step 16 estimates maximum-likelihood SAR on log sale price using the same
single-year, one-sale-per-PIN cross-section as the autocorrelation audit. The
default spatial weights are row-standardized eight-nearest neighbors in
EPSG:3435. Exact constant and collinear design columns are removed before fit.

The model is compared with hedonic OLS on pseudo/in-sample fit, AIC/BIC, and
residual Moran's I. Spatial impact multipliers distinguish direct, indirect, and
total propagation implied by the estimated rho.

Predictive validation holds out the easternmost of five K-means spatial blocks.
The SAR training model never observes held-out prices. Held-out predictions
solve the SAR system using subject features, held-out spatial topology, and
observed prices from training neighbors; this conditional mode is recorded in
the report and reflects a comparable-sales deployment setting.

Rho indicates conditional spatial dependence, not automatically causal price
interaction. Shared omitted location characteristics can produce the same
pattern, motivating the spatial-error comparison in Step 17.

