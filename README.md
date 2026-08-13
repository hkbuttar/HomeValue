# HomeValue

Chicago housing valuation and spatial market intelligence built from recorded
Cook County parcel sales.

## Step 1: environment and data acquisition

The repository currently implements the acquisition foundation. Large Cook
County tables are queried through Socrata by year and selected columns, paged in
bounded chunks, and written as partitioned Parquet files. The 50M-row Parcel
Universe is never loaded in full.

### Setup

Python 3.11 or 3.12 is recommended. Geospatial wheels are easiest to install in
a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

An unauthenticated Socrata request works for small pulls. For sustained pulls,
create an app token and export it (never commit it):

```bash
export SOCRATA_APP_TOKEN="..."
# Optional, but avoids the Census API's shared unauthenticated quota:
export CENSUS_API_KEY="..."
```

### Acquire data

Run a small, safe smoke pull first:

```bash
python -m preprocessing.acquire cook-county --dataset sales --years 2023 --limit 1000
python -m preprocessing.acquire cook-county --dataset characteristics --years 2023 --limit 1000
python -m preprocessing.acquire cook-county --dataset parcels --years 2023 --limit 1000
python -m preprocessing.acquire acs --year 2023
python -m preprocessing.acquire cta
```

For a real bounded pull, omit `--limit` and choose only needed years. Parcel
Universe can additionally be constrained to residential PINs saved by the sales
pull:

```bash
python -m preprocessing.acquire cook-county --dataset sales --years 2019:2023
python -m preprocessing.acquire cook-county --dataset parcels --years 2019:2023 \
  --pins-from data/raw/cook_county/sales
```

Outputs live under `data/raw/<source>/`, partitioned by year where applicable.
Each run writes a JSON manifest recording source URL, query, timestamp, row
count, and SHA-256 hashes. Raw data is gitignored; directory marker files remain.

Use `python -m preprocessing.acquire --help` for all options. Tests do not use
the network:

```bash
pytest
```

### Primary data sources

- [Cook County Assessor — Parcel Sales](https://datacatalog.cookcountyil.gov/resource/wvhk-k5uv.json)
- [Cook County Assessor — Single and Multi-Family Improvement Characteristics](https://datacatalog.cookcountyil.gov/resource/x54s-btds.json)
- [Cook County Assessor — Parcel Universe](https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.json)
- [Census ACS 5-year API](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- [CTA GTFS](https://www.transitchicago.com/developers/gtfs/)

PINs are read and stored as strings and normalized to exactly 14 digits.

## Step 2: analytical population

Classify every raw sale without silently dropping records:

```bash
python -m preprocessing.population
```

This writes a year-partitioned table to
`data/processed/residential_sales_population`. Each row is labeled `market`,
`ambiguous`, or `excluded`, with machine-readable reasons. The primary market
population defaults to single-family homes and townhouses. Small multi-family
classes can be included for a separate sensitivity run:

```bash
python -m preprocessing.population --include-small-multifamily \
  --output data/processed/residential_sales_population_with_multifamily
```

See [the complete population rules](preprocessing/POPULATION_RULES.md) for
class definitions, precedence, and every exclusion rule.

## Step 3: historical alignment

Align market sales with contemporaneous or earlier property, parcel, and ACS
snapshots:

```bash
python -m preprocessing.historical
```

The command writes linked sales, improvement-card, parcel, and ACS Parquet
tables plus an alignment report under `data/processed/historical_alignment`.
Every match includes its vintage, lag, and status. Future snapshots are never
used unless `--allow-future-snapshots` is explicitly supplied, in which case
they are labeled `current_state_future`.

See [the historical alignment policy](preprocessing/HISTORICAL_ALIGNMENT.md).

## Step 4: canonical core sales table

Aggregate property cards and join the historically aligned sources into one row
per market sale:

```bash
python -m preprocessing.core_sales
```

This writes `data/processed/core_sales.parquet` and a schema report containing
column types and missingness. Property, location, tract-neighborhood, and basic
calendar-market features are included when present in the acquired sources.
See [the core table construction rules](preprocessing/CORE_SALES_SCHEMA.md).

## Step 5: data-quality audit

Run the formal audit before modeling:

```bash
python -m preprocessing.quality
```

Outputs under `data/processed/quality` include a browsable
`data_quality_report.html`, full JSON metrics, and a Parquet copy of the sales
with non-destructive `dq_*` flags. See [the audit methodology](preprocessing/DATA_QUALITY.md).

## Step 6: exploratory market analysis

Generate grouped summaries, repeat-sale diagnostics, charts, and the spatial
market overview:

```bash
python -m market.exploratory
```

Results are written under `data/processed/exploration`, including
`market_exploration.html`, CSV tables, and PNG figures. See the
[exploration methodology](market/EXPLORATION.md).

## Step 7: baseline valuation models

Train and evaluate the four simple benchmarks with a held-out future period:

```bash
python -m ml.baselines
# Or reserve multiple recent years:
python -m ml.baselines --test-start-year 2022
```

Predictions, metrics, and fitted median lookup tables are written under
`data/processed/baselines`. See [the baseline methodology](ml/BASELINES.md).

## Step 8: hedonic price model

Fit the interpretable log-price regression and evaluate it out of time:

```bash
python -m hedonic.model
```

Predictions, robust coefficient estimates, confidence intervals, model schema,
and evaluation metrics are written under `data/processed/hedonic`. See the
[hedonic model methodology](hedonic/HEDONIC_MODEL.md).

## Step 9: property versus neighborhood decomposition

Fit the nested property, market, neighborhood, and accessibility specifications:

```bash
python -m hedonic.decomposition
```

The output directly reports the incremental R², adjusted R², future-period MAE,
and RMSE contributed by each feature group. Model D remains explicitly
unavailable until accessibility features exist. See the
[decomposition methodology](hedonic/DECOMPOSITION.md).

## Step 10: ACS neighborhood layer

Engineer the limited tract-level socioeconomic, housing, and commuting feature
set and inspect its multicollinearity diagnostics:

```bash
python -m neighborhood.acs
```

Outputs include the ACS feature table, an ACS-enriched core sales table, and a
JSON coverage/correlation/VIF report under `data/processed/acs_neighborhood`.
See [the ACS feature methodology](neighborhood/ACS_FEATURES.md).
