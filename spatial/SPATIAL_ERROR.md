# Spatial error model

This analysis estimates maximum-likelihood SEM on the same log-price cross-section,
hedonic design, and row-standardized KNN weights as OLS and SAR. Lambda measures
spatial correlation in the model's error process, while rho in SAR measures
conditional dependence in neighboring outcomes.

The full-sample comparison reports AIC/BIC, pseudo R², parameter uncertainty,
and residual Moran's I for OLS, SAR, and SEM. Lowest AIC provides a transparent
specification comparison, not proof of the underlying mechanism.

Spatial-block validation holds out the same deterministic eastern block. SEM
prediction uses subject features and observed training-neighbor residuals while
solving the held-out error system; it never uses held-out prices. OLS and SAR
predictions are evaluated on that identical block.

A better SEM fit is consistent with shared omitted local conditions; a better
SAR fit is consistent with direct conditional outcome dependence. Either can
also reflect misspecification, weights choices, or other noncausal processes.

