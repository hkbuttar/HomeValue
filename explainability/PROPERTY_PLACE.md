# Property versus place attribution

Step 31 decomposes an individual fitted-model estimate into a reference-market
baseline plus property, place, and time/market contributions.

```bash
python -m explainability.property_place --sale-id SALE_ID
```

Contributions are Monte Carlo Shapley values calculated from actual fitted-model
counterfactual predictions. They therefore reconcile exactly to the estimate
instead of presenting fabricated dollar coefficients. Structural features form
the property component; neighborhood, accessibility, and prior spatial features
form place; calendar features form time/market. The median/mode reference
profile is saved with every run. Attributions describe model behavior relative
to that reference and are not causal effects.
