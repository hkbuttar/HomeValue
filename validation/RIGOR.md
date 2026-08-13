# Statistical rigor audit

This analysis turns the project's validation principles into a machine-readable audit.

```bash
python -m validation.rigor
```

It bootstraps held-out MAE, RMSE, and MdAPE; computes a paired confidence
interval for the leading model's MAE advantage; refits a model across repeated
seeds; checks residual Moran's I under alternative neighbor counts; and compares
full, outlier-trimmed, and available sale-quality samples. Existing robust
standard errors, grouped cross-validation, temporal/spatial holdouts, and
coefficient-stability artifacts are recorded in one checklist. Statistical
significance is never treated as evidence of practical importance by itself.
