# Calibration and valuation intervals

This analysis builds split-conformal valuation ranges from absolute log residuals in
the validation period. It then measures interval coverage on the untouched
final-test period.

```bash
python -m validation.intervals --nominal-coverage 0.90
```

Log-scale residuals produce positive ranges whose dollar width grows with the
estimated property value. Outputs include sale-level lower and upper bounds,
overall and segment coverage, interval widths, and a 50%/80%/90%/95%
calibration curve. Conformal guarantees are marginal and rely on calibration
and future sales being sufficiently exchangeable, so neighborhood and price
group coverage remain essential diagnostics.
