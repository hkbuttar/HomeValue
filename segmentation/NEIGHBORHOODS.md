# Neighborhood market segmentation

Step 24 clusters neighborhood markets, not individual homes. Profiles combine
price/PPSF, index appreciation, volatility, transaction velocity, property mix
and size, income, ownership, density, and transit measures when available.
Neighborhoods below the sale-count threshold are excluded rather than heavily
imputed.

Numeric gaps are median-imputed and features are robust-scaled. K-means cluster
counts from 2 through 8 (bounded by sample size) are compared by silhouette
score. The best silhouette count is selected, with fewer clusters breaking an
exact tie.

Stability is measured by repeatedly clustering 80% neighborhood subsamples and
computing adjusted Rand agreement with the fitted solution. Cluster profiles and
membership counts are retained. Descriptive archetype names are generated only
after fitting from relative centroid characteristics; duplicate descriptions
receive neutral numeric suffixes.

These archetypes summarize the chosen features and observation period. They are
not permanent or intrinsic neighborhood identities; Step 25 tests that directly.

