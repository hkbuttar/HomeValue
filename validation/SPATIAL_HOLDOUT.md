# Spatial holdout validation

Step 27 compares ordinary shuffled folds, a future-period holdout, and grouped
geographic folds. Geographic folds keep every sale from the same tract,
assessor neighborhood, or municipality on one side of a split.

```bash
python -m validation.spatial_holdout --folds 5
```

Each preprocessing pipeline is fitted inside its training fold. The output
reports MAE increases and ratios relative to random cross-validation for every
valuation model. A substantial geographic penalty is evidence that apparent
random-fold accuracy does not transfer well to less familiar places.
