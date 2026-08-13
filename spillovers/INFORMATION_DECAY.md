# Spatial valuation-information decay

This analysis rebuilds leakage-safe comparable estimates at 0.25, 0.5, 1, and 2
miles. Every comparable predates its target transaction.

```bash
python -m spillovers.decay
```

The decay curve separates coverage from accuracy. Available-sample metrics show
operational performance at each radius; common-target metrics compare the same
properties at every radius. Marginal MAE improvement then measures whether the
next distance band adds useful information rather than merely increasing
coverage. The complete maximum-radius link table remains available for audit.
