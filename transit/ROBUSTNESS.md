# CTA premium robustness

Step 32 estimates a predeclared robustness ladder: property controls, year
effects, neighborhood controls, ACS context, spatial dependence, nonlinear
distance, and alternative geographic subsets.

```bash
python -m transit.robustness
```

The specification table reports the CTA-distance coefficient, robust or spatial
standard error, confidence interval, p-value, and implied effect of moving one
mile closer. Nonlinear terms receive a joint test. The report explicitly says
when an initially detectable association disappears after neighborhood controls.
All estimates remain conditional associations rather than causal transit effects.
