# Transaction-filter sensitivity

This analysis uses Cook County's sale-quality indicators to define two auditable
samples. Strict sales require explicitly clean flags, single-family class,
non-multisale status, complete core fields, and a $25,000 floor. Moderate sales
allow missing flag metadata and small multifamily classes, but no explicit
quality failure or multi-parcel sale.

```bash
python -m validation.filter_sensitivity
```

Each sample receives the same temporal hedonic validation, robust coefficient
estimation, and spatial-lag fit. Outputs compare sample size, MAE/RMSE/MdAPE,
coefficients, spatial rho, and CTA/lake accessibility findings. Any direction or
significance change is reported as filter sensitivity, not smoothed away.
