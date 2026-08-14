# HomeValue — Chicago Housing Valuation & Spatial Market Intelligence

An explainable valuation research system built from recorded Cook County home
sales, historical property records, neighborhood context, accessibility, and
spatial market structure.

**[Explore the live demo](https://home-value-beta.vercel.app/)**

> What determines the value of a home—the structure, the neighborhood,
> accessibility, nearby properties, or the broader market?

HomeValue answers that question with classical hedonic models, spatial
econometrics, comparable sales, machine learning, calibrated uncertainty, and
an interactive Next.js application backed by FastAPI.

## Why HomeValue?

Ordinary house-price prediction treats observations as independent. Housing is
not: nearby homes share amenities, land markets, school access, transit, local
expectations, and unobserved neighborhood conditions. Ignoring that dependence
can leave spatially clustered errors, overstate confidence, and obscure whether
a valuation comes from the building or its location.

HomeValue therefore treats predictive accuracy and spatial explanation as
separate but connected goals. Results are labeled **Robust**, **Suggestive**,
**Exploratory**, or **Data-limited** rather than being presented as equally
settled findings.

## Research questions

1. How much sale-price variation can property characteristics alone explain?
2. How much additional information comes from neighborhood characteristics?
3. Does CTA accessibility materially relate to value after controlling for location?
4. How large and localized is the lakefront price gradient?
5. Are hedonic-model residuals spatially autocorrelated?
6. Do spatial econometric models materially improve on conventional hedonic OLS?
7. Does machine learning outperform explicit spatial models?
8. Does that advantage survive out-of-time testing?
9. Does it survive geographic holdout testing?
10. How far away can a comparable sale be before its information deteriorates?
11. Which property types and neighborhoods are hardest to value?
12. How stable are neighborhood housing-market archetypes over time?
13. How much of an individual valuation is attributable to property versus place?

Generate the evidence-synthesis report with:

```bash
python -m reporting.results
```

## Data

The pipeline combines:

- Cook County Assessor Parcel Sales
- Single- and Multi-Family Improvement Characteristics
- Cook County Parcel Universe
- Census ACS five-year estimates
- CTA GTFS stations and service geography
- Chicago parks, shoreline, downtown, and auxiliary spatial layers

Every PIN is stored as a string and normalized to 14 digits. Acquisition runs
write manifests containing source URLs, queries, timestamps, row counts, and
SHA-256 hashes. Raw data is excluded from version control and production images.

Primary sources: [Cook County Parcel Sales](https://datacatalog.cookcountyil.gov/resource/wvhk-k5uv.json),
[property characteristics](https://datacatalog.cookcountyil.gov/resource/x54s-btds.json),
[Parcel Universe](https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.json),
[Census ACS](https://www.census.gov/data/developers/data-sets/acs-5year.html), and
[CTA GTFS](https://www.transitchicago.com/developers/gtfs/).

## Data engineering

```text
Sale
  × Historical Parcel
  × Property
  × Census Tract
  × Transit
  × Amenities
  × Market
```

Sales are classified as market, ambiguous, or excluded without silently
dropping records. Historical joins use contemporaneous or earlier snapshots to
prevent future information leakage. The canonical analytical table contains one
row per market sale and records match vintages, lags, and data-quality flags.

Core pipeline commands:

```bash
python -m preprocessing.acquire --help
python -m preprocessing.population
python -m preprocessing.historical
python -m preprocessing.core_sales
python -m preprocessing.quality
```

Detailed policies live in [population rules](preprocessing/POPULATION_RULES.md),
[historical alignment](preprocessing/HISTORICAL_ALIGNMENT.md), and the
[core-sales schema](preprocessing/CORE_SALES_SCHEMA.md).

## Chicago housing market

The exploratory layer profiles sale price, price per square foot, transaction
volume, housing stock, repeat sales, and geographic variation. Annual
neighborhood indices distinguish citywide movement from local trajectories.
Run `python -m market.exploratory` and see the
[market methodology](market/EXPLORATION.md).

## Property model

The interpretable baseline is an OLS hedonic model of log recorded sale price,
with structural characteristics, property type, time, and controlled location
features. Robust coefficient intervals and future-period errors keep
explanation separate from in-sample fit. See the
[hedonic methodology](hedonic/HEDONIC_MODEL.md).

## Property versus place

Nested models add market timing and neighborhood information to an identical
property-only sample. Their held-out improvement measures incremental
information; fitted-model attribution then reconciles each estimate into
property and place contributions. See [model decomposition](hedonic/DECOMPOSITION.md)
and [valuation attribution](explainability/PROPERTY_PLACE.md).

## Spatial diagnostics

Moran's I, permutation inference, multiple weight definitions, and mapped
residuals test whether conventional-model errors cluster geographically. A
significant pattern indicates omitted spatial structure, not its cause. See
[spatial autocorrelation](spatial/AUTOCORRELATION.md).

## Spatial econometrics

Spatial autoregressive and spatial error models are compared with OLS on common
samples. A conditional Spatial Durbin specification probes neighboring-feature
spillovers only when diagnostics support it. See [SAR](spatial/SPATIAL_LAG.md),
[SEM](spatial/SPATIAL_ERROR.md), and [SDM](spatial/SPATIAL_DURBIN.md).

## Comparable sales

The comps engine searches projected Chicago coordinates, uses only transactions
recorded before the target sale, and combines distance, recency, and property
similarity weights. Radius experiments quantify how spatial valuation
information decays. See [comparable sales](spillovers/COMPARABLES.md) and
[information decay](spillovers/INFORMATION_DECAY.md).

## Machine learning

Random Forest, histogram gradient boosting, and XGBoost benchmark nonlinear
relationships. Prior-only local market features prevent future-sale leakage.
Models use ordered validation and untouched final periods, plus geographic
holdouts. See [valuation models](ml/VALUATION_MODELS.md) and
[spatial features](ml/SPATIAL_FEATURES.md).

## Accessibility

CTA proximity and nonlinear lake, downtown, and park gradients are estimated
with progressively richer controls and spatial robustness checks. These are
conditional associations: accessibility can proxy for unobserved neighborhood
attributes. See [CTA analysis](transit/CTA_PREMIUM.md) and
[amenity gradients](accessibility/GRADIENTS.md).

## Neighborhood dynamics

Annual price indices and market profiles describe price level, appreciation,
housing stock, accessibility, transaction activity, and model error.
Longitudinal re-estimation measures whether data-driven market archetypes persist
or transition over time. See [neighborhood segmentation](segmentation/NEIGHBORHOODS.md)
and [stability analysis](segmentation/STABILITY.md).

## Valuation uncertainty

Split-conformal intervals transform held-out absolute log residuals into likely
valuation ranges. Coverage is audited overall and by relevant market segments;
the intervals describe model uncertainty, not every source of transaction risk.
See [interval methodology](validation/INTERVALS.md).

## Model benchmark

Median, PPSF, comps, hedonic, spatial, and ML results are assembled with their
sample and validation design. The report refuses to declare a universal winner
when metrics come from incompatible holdouts. See the
[benchmark methodology](benchmark/MODELS.md).

## HomeValue application

```text
Next.js interface
       │
       ▼
FastAPI research API
       │
       ├── Unified valuation engine
       ├── Trained model artifacts
       └── Precomputed Parquet and JSON outputs
```

The interface includes market, valuation, neighborhood, spatial, accessibility,
and model-research views. Normal web requests never estimate spatial models or
run large joins. Start development services with:

```bash
uvicorn api.app:app --reload
cd frontend && npm install && npm run dev
```

API documentation is available at `/docs`. See [API usage](api/API.md),
[frontend setup](frontend/README.md), and the [engine design](engine/ENGINE.md).

## Reproducibility and validation

Python 3.11 or 3.12 and Node.js 20.9 or newer are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
MPLCONFIGDIR=/tmp/homevalue-mpl LOKY_MAX_CPU_COUNT=8 python -m pytest -q
```

Tests cover identifiers, transaction filters, historical alignment, coordinate
math, nearest neighbors, spatial weights, leakage, temporal and geographic
splits, interval coverage, and API schemas. See the
[validation matrix](validation/TESTING.md).

## Deployment

The repository includes non-root FastAPI and Next.js images and a health-checked
Compose stack. Mount only curated processed artifacts into production:

```bash
docker compose up --build
```

See the [deployment guide](DEPLOYMENT.md) for environment variables, artifact
requirements, and payload limits.

## Limitations

- Recorded price does not reveal every private transaction condition.
- Property records can contain reporting and historical inconsistencies.
- Some characteristics may not be perfectly contemporaneous with a sale.
- ACS estimates contain sampling uncertainty.
- Census tracts are imperfect neighborhood definitions.
- Geographic associations are not causal effects.
- Transit access may proxy for unobserved neighborhood attributes.
- Spatial-weight definitions are modeling choices.
- Cook County results may not generalize to other housing markets.
- HomeValue estimates are research outputs, not appraisals.

## Future work

- Building permits and renovation histories
- School and richer transportation accessibility
- Commercial land-use exposure
- Mortgage-rate regime analysis
- Assessment-equity research
- Multi-city validation
- Expanded repeat-sales indices

## License and intended use

This project is intended for research and educational use. Confirm source-data
licenses and deployment terms before redistributing derived datasets or using
the estimates in operational decisions.
