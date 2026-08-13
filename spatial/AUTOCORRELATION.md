# Spatial autocorrelation audit

Step 15 uses a single sale-year cross-section and retains only the latest sale
per PIN in that year. This avoids treating the same property at different dates
as independent spatial neighbors. Very large cross-sections are reproducibly
capped at 25,000 observations for tractable permutation inference.

Coordinates are projected to EPSG:3435 before constructing row-standardized
8-nearest-neighbor and one-mile fixed-distance weights. Optional census-tract
polygons enable Queen-contiguity weights on tract-median outcomes. Islands are
reported rather than hidden.

Moran's I is calculated with 999 seeded permutations for recorded sale price,
price per square foot, and residual log price from a full-cross-section hedonic
model with property, neighborhood, and available accessibility controls. The
report records analytical and permutation p-values, expected I, and permutation
z-scores.

Positive residual Moran's I with permutation p below 0.05 is treated as evidence
that observable controls did not remove all spatial structure. This justifies
testing spatial econometric models; it does not by itself identify the mechanism.

