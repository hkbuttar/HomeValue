# Out-of-time validation

Step 26 uses strictly ordered training, later-validation, and final-test
samples. The final-test sample does not influence preprocessing or model
selection.

```bash
python -m validation.out_of_time
```

By default, the penultimate observed year starts validation and the latest year
starts final testing. All three nonlinear valuation families are selected using
validation MAE, refit on training plus validation sales, and scored on the
untouched final period. Sale-level predictions are retained for auditing.
