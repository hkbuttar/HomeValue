# ACS neighborhood feature layer

The ACS layer uses a deliberately limited set of tract-level 5-year estimates:
median household income, poverty, bachelor's-or-higher and graduate education,
owner/renter occupancy, vacancy, median housing age, average household size,
population, housing units, and public-transit/automobile commuting shares.

Rates are calculated from published numerators and denominators, not percentages
copied from unrelated universes. Zero denominators and Census unavailable-value
sentinels become missing. Values outside `[0, 1]` are rejected for rates.

Population and housing-unit density are created only when tract land area in
square miles is supplied. Otherwise the report explicitly labels density as
unavailable; no area is inferred from parcel geometry.

The feature report includes missingness, absolute pairwise correlations above
0.8, and variance inflation factors where the sample permits them. These are
diagnostics for feature selection, not automatic deletion rules. ACS vintage and
lag remain governed by the Step 3 historical-alignment policy.

