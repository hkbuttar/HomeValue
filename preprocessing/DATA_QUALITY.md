# Data-quality audit

The audit is non-destructive: it preserves every canonical sale and appends
`dq_*` flags. It produces HTML and JSON reports plus a flagged Parquet table.

Checks cover duplicate identifiers and transactions, duplicate PIN/year sales,
missing and implausible prices, nonpositive and outlying building area,
impossible ages, missing or out-of-bounds coordinates, failed historical joins,
repeat sales, resales within 365 days, property-class inconsistencies, and
missing structural characteristics.

The hard bounds are explicit in `QualityRules`. Square-footage outliers combine
those bounds with Tukey fences on log building area. Findings are classified as
`error`, `warning`, or `info`; an audit flag is evidence for review and is not
automatically an instruction to delete the sale.

For every retained input feature, the JSON report contains its dtype,
missingness, unique count, distribution or most common values, and coverage by
sale year. It also reports property, parcel, and ACS join-success rates.

