# Main model benchmark

This analysis assembles median, PPSF, comparable-sales, hedonic, spatial-lag,
spatial-error, and gradient-boosting results into one audit table.

```bash
python -m benchmark.models
```

The table reports MAE, RMSE, median absolute percentage error, residual Moran's
I, and whether temporal and spatial tests exist. Separate temporal and spatial
MAEs prevent unlike validation designs from being silently conflated. The
primary rank is descriptive because older model artifacts do not all share the
same holdout. Prediction and explanation remain separate objectives, and the
report never assumes machine learning wins by definition.
