# Analytical population rules

Every raw sale is retained and assigned exactly one `population_status`:
`market`, `ambiguous`, or `excluded`. Exclusion takes precedence over ambiguity.

## Primary population

The default primary population is Cook County single-family detached homes and
townhouses in classes `202`–`210`, `234`, `278`, and `295`. Classes `211` and
`212` (small multi-family/mixed-use) can be enabled explicitly. Condominiums
(`299`), vacant land, commercial property, and all other classes are excluded.

## Exclusions

A row is excluded when any of these conditions holds:

- invalid PIN, sale date, or sale price;
- sale price below $10,000 (configurable);
- class is outside the selected residential population;
- any Cook County sale-quality flag is true:
  `sale_filter_same_sale_within_365`, `sale_filter_less_than_10k`, or
  `sale_filter_deed_type`.

The price threshold intentionally duplicates the official less-than-$10K flag
as a transparent data-quality guard. Both reasons are retained when both apply.

## Ambiguous sales

Otherwise eligible rows are ambiguous when one or more official filter flags
is missing or when the record is part of a multi-parcel sale. These records are
preserved for sensitivity analysis but are not in the primary population.

No filtering decision is inferred from buyer/seller names or deed text. The
output records machine-readable reason lists, the complete input row, and a
manifest containing the exact rules and counts used for each run.

