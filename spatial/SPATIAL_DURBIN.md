# Conditional Spatial Durbin robustness model

Step 18 is gated. By default SDM is estimated only when the Step 17 report shows
at least one of: significant SAR rho, significant SEM lambda, or positive and
significant OLS residual Moran's I at the configured 5% level. Otherwise the
pipeline writes a `skipped_not_justified` report and stops. `--force` is an
explicit sensitivity-analysis override and is recorded in the status.

The fitted SDM adds one spatial lag of every retained hedonic predictor to SAR.
It reports a nested SAR-versus-SDM likelihood-ratio test, AIC/BIC, rho, residual
Moran's I, impact multipliers, and stability of common SAR/SDM coefficients.

Spatial-block predictions use subject features, spatially lagged features, and
observed training-neighbor prices while solving the held-out SAR system. No
held-out price enters prediction.

SDM can show that neighboring observed characteristics add information, but its
coefficients and impact multipliers are not causal spillovers without stronger
identification assumptions.

