# Testing and validation

Step 39 maintains small, deterministic tests for the full analytical contract:

- PIN normalization, duplicate handling, positive sale prices, and market-sale filters;
- historical property/ACS alignment without future leakage;
- projected coordinates, exact distances, and nearest CTA stations;
- historical comparable selection and prior-only spatial features;
- known-neighbor KNN construction, row-standardized weights, and Moran's I;
- conformal interval coverage and strictly ordered temporal splits;
- geographically disjoint grouped folds; and
- strict Pydantic request and response schemas for the forthcoming API.

Run the complete gate with:

```bash
MPLCONFIGDIR=/tmp/homevalue-mpl LOKY_MAX_CPU_COUNT=8 python -m pytest -q
```

Synthetic geometry tests use coordinates with analytically known nearest
neighbors. API models reject unknown fields, impossible numeric values,
unpaired coordinates, malformed PINs, and valuation ranges that do not contain
the estimate.
