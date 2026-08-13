# Historical alignment policy

Historical alignment uses the sale's recorded calendar year as the observation
date. Property characteristics, parcel geography, and ACS tract features are
matched to the newest available snapshot whose year is no later than the sale
year. Future snapshots are rejected by default.

Every match records its source year, lag, and one of these statuses:

- `exact`: snapshot year equals sale year;
- `historical`: an earlier snapshot within the permitted lag;
- `unmatched_stale`: history exists but exceeds the permitted lag;
- `unmatched_no_history`: only future records, or no records, exist;
- `unmatched_missing_key`: the sale lacks a matching entity/geography key;
- `current_state_future`: future data was used only when the user explicitly
  enabled `--allow-future-snapshots`.

Default maximum lags are three years for property and parcel data and five years
for ACS. Improvement cards are intentionally retained as multiple linked rows;
their aggregation into one sale row belongs to construction of the core sales
table in the relevant upstream analysis. Only `market` rows from the relevant upstream analysis are aligned when the population
label is present.

