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

## Step 11: CTA rail accessibility

Build projected nearest-station distances, station counts, and nearest-line
features from the acquired CTA GTFS feed:

```bash
python -m accessibility.cta
```

Outputs under `data/processed/cta_accessibility` include the station table,
sale-level features, an enriched core table, and a coverage report. See the
[CTA accessibility methodology](accessibility/CTA_FEATURES.md).

## Step 12: amenity accessibility

Acquire the official Chicago geometry extracts, then add lake, downtown, and
major-park distances to the CTA-enriched table:

```bash
python -m accessibility.amenities acquire
python -m accessibility.amenities build
```

Outputs under `data/processed/accessibility` include sale-level features, the
combined accessibility-enriched core table, and a provenance/coverage report.
See [the amenity accessibility methodology](accessibility/AMENITY_FEATURES.md).

## Step 13: CTA accessibility premium

Compare linear, banded, cubic-spline, and GAM-style CTA distance effects using
the same future holdout:

```bash
python -m transit.premium
```

Outputs under `data/processed/cta_premium` include predictions, robust
coefficients, premium curves with confidence intervals, a chart, and evidence
flags. See [the CTA premium methodology](transit/CTA_PREMIUM.md).

## Step 14: lakefront and downtown gradients

Estimate continuous linear and nonlinear price gradients for Lake Michigan and
downtown proximity:

```bash
python -m accessibility.gradients
```

Outputs under `data/processed/amenity_gradients` include predictions, robust
coefficients, gradient curves, a comparison chart, lakefront decay, and the
joint downtown-distance test. See [the gradient methodology](accessibility/GRADIENTS.md).

## Step 15: spatial autocorrelation

Test spatial clustering in prices, PPSF, and controlled hedonic residuals:

```bash
python -m spatial.autocorrelation
# Optional tract-adjacency alternative:
python -m spatial.autocorrelation --tract-polygons path/to/tracts.geojson
```

Outputs include Moran's I results, the spatial audit sample, scatter plots, and
the residual-dependence conclusion. See the
[spatial autocorrelation methodology](spatial/AUTOCORRELATION.md).

## Step 16: spatial lag model

Estimate the SAR model and compare it with hedonic OLS:

```bash
python -m spatial.lag_model
```

Outputs include rho and impact multipliers, OLS/SAR fit and residual-dependence
comparisons, spatial-block predictions, and future deployment cautions. See the
[spatial lag methodology](spatial/SPATIAL_LAG.md).

## Step 17: spatial error model

Estimate SEM and compare the OLS, SAR, and SEM explanations of spatial
dependence:

```bash
python -m spatial.error_model
```

Outputs include lambda and rho inference, common fit and residual diagnostics,
spatial-block predictions for all three models, and a cautious mechanism
assessment. See [the spatial error methodology](spatial/SPATIAL_ERROR.md).

## Step 18: conditional Spatial Durbin robustness model

Run the diagnostics-gated SDM extension:

```bash
python -m spatial.durbin_model
# Explicit sensitivity override:
python -m spatial.durbin_model --force
```

The command skips cleanly when earlier spatial evidence is insufficient. When
justified, it writes SAR/SDM comparisons, WX coefficients, impact multipliers,
and spatial-block predictions. See [the SDM methodology](spatial/SPATIAL_DURBIN.md).

## Step 20: nonlinear valuation models

Train Random Forest, HistGradientBoosting, and XGBoost on the shared future
holdout:

```bash
python -m ml.valuation
```

Outputs include held-out predictions, serialized preprocessing/models, metrics,
approved feature groups, and comparisons with available earlier model artifacts.
See [the ML valuation methodology](ml/VALUATION_MODELS.md).

## Step 21: prior-only spatial ML features

Build local price, PPSF, volume, and neighborhood-appreciation features from
strictly earlier transactions:

```bash
python -m ml.spatial_features
```

Outputs include the feature table, every contributing prior-sale link, an
enriched core table, and temporal-leakage validation. See the
[spatial feature methodology](ml/SPATIAL_FEATURES.md).

## Step 22: repeat-sales analysis

Analyze consecutive same-property transactions and build the robustness index:

```bash
python -m market.repeat_sales
```

Outputs include repeat pairs, annual and neighborhood appreciation summaries, a
simplified repeat-sales index and chart, and model-residual persistence when
predictions are available. See [the repeat-sales methodology](market/REPEAT_SALES.md).

## Step 23: neighborhood price indices

Build median-PPSF, property-adjusted, and repeat-sales neighborhood indices:

```bash
python -m neighborhood.price_index
```

Outputs include the neighborhood-year panel, growth co-movement correlations,
divergence rankings, a trajectory chart, and coverage diagnostics. See the
[neighborhood index methodology](neighborhood/PRICE_INDEX.md).

## Step 24: neighborhood market segmentation

Cluster neighborhood market profiles and validate the archetypes:

```bash
python -m segmentation.neighborhoods
```

Outputs include neighborhood assignments, cluster profiles, silhouette model
selection, bootstrap stability, post-fit archetype names, and a profile chart.
See [the segmentation methodology](segmentation/NEIGHBORHOODS.md).

## Step 25: segment stability over time

Re-estimate neighborhood regimes over historical periods and quantify their
longitudinal consistency:

```bash
python -m segmentation.stability
```

Outputs include aligned neighborhood segment histories, conditional transition
matrices, persistence rates, and consecutive-period adjusted Rand indices. See
[the stability methodology](segmentation/STABILITY.md).

## Step 26: out-of-time validation

Select models on later sales and evaluate them on an untouched most-recent
period:

```bash
python -m validation.out_of_time
```

Outputs include validation predictions, final-test predictions, fitted models,
and separate metrics for all three nonlinear valuation families. See the
[out-of-time validation methodology](validation/OUT_OF_TIME.md).

## Step 27: spatial holdout validation

Compare random, future-period, and geographically grouped validation:

```bash
python -m validation.spatial_holdout
```

The benchmark creates grouped folds for tracts, assessor neighborhoods, and
municipalities when available, then reports each model's error penalty relative
to random folds. See the [spatial holdout methodology](validation/SPATIAL_HOLDOUT.md).

## Step 28: error by market segment

Audit held-out errors across property and market slices:

```bash
python -m validation.error_segments
```

Outputs include segment-level MAE, RMSE, median APE, signed percentage error,
small-group reliability flags, and ranked failure modes. See the
[segment error methodology](validation/ERROR_SEGMENTS.md).

## Step 29: calibration and valuation intervals

Turn point predictions into calibrated valuation ranges and test their actual
coverage:

```bash
python -m validation.intervals
```

Outputs include sale-level conformal ranges, interval widths, calibration
curves, and coverage by price group and neighborhood. See the
[valuation interval methodology](validation/INTERVALS.md).

## Step 30: explainability

Compare what hedonic, spatial, and machine-learning models say drives value:

```bash
python -m explainability.report
```

Outputs include coefficient marginal effects, spatial parameters and
multipliers, permutation and SHAP importance, partial dependence, and a
cross-method agreement table. See the
[explainability methodology](explainability/EXPLAINABILITY.md).

## Step 19: local comparable-sales engine

Build leakage-safe traditional comparable valuations:

```bash
python -m spillovers.comps
```

Outputs include sale-level predictions, every selected comparable and weight,
coverage/fallback diagnostics, and latest-year metrics. See the
[comparable-sales methodology](spillovers/COMPARABLES.md).
