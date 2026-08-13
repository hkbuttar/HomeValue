# Cross-method explainability

This analysis places the project's explanations on a common footing. Hedonic output
retains coefficients, robust confidence intervals, and interpretable marginal
effects. Spatial output retains parameters and direct, indirect, and total
multipliers. ML output includes raw-feature permutation importance, aggregated
SHAP importance, and numeric partial-dependence profiles.

```bash
python -m explainability.report
```

The method-agreement table checks whether living area, bathrooms, neighborhood
income, CTA access, lake distance, and spatial spillovers appear across the
three approaches. Agreement strengthens a descriptive finding; disagreement is
reported rather than hidden. None of these diagnostics alone establishes a
causal effect.
