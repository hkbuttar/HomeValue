# Local comparable-sales engine

Step 19 searches in projected EPSG:3435 coordinates and requires every
comparable transaction to predate its target strictly. The target PIN itself is
always excluded. Comparables must match broad property type and satisfy explicit
distance, recency, log-square-footage, and building-age constraints.

Three transparent tiers preserve coverage without silently abandoning quality:

- strict: 1 mile, 1 year, log-size difference ≤0.25, age difference ≤15 years;
- relaxed: 3 miles, 3 years, log-size difference ≤0.40, age difference ≤30;
- broad: 5 miles, 5 years, log-size difference ≤0.55, age difference ≤50.

The first tier with at least three candidates is used, capped at the ten highest
weighted comparables. Weights decay exponentially with geographic distance,
recency, size difference, and age difference. The primary estimate is the
weighted recorded price specified in the project plan; a weighted-PPSF estimate
is retained as a companion diagnostic.

Outputs include every target-to-comparable link and normalized weight, effective
comparable count, fallback tier, weighted 10th–90th price interval, coverage,
and latest-year evaluation. No future or same-day sale can enter a prediction.

