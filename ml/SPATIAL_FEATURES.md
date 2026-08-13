# Prior-only spatial ML features

Step 21 constructs local market features only from transactions whose recorded
sale date is strictly earlier than the target. The target PIN is excluded even
when it sold previously. Coordinates are projected to EPSG:3435 before a
one-mile default search.

Within a three-year lookback, the layer provides weighted median prior price,
prior sale count, distance/recency-weighted PPSF, most recent prior-sale age, and
weighted mean distance. Weights decay with distance and time.

Neighborhood appreciation compares median prices in the newer and older halves
of a two-year pre-sale window. Each half requires at least three observations;
otherwise appreciation remains missing. This prevents unstable ratios from
being presented as local trends.

Every contributing target/prior-sale link is persisted for auditing. The report
asserts the temporal inequality across all links. The enriched output feeds
directly into the Step 20 `prior_spatial` feature allowlist.

