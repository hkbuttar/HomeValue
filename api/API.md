# FastAPI backend

This analysis exposes the unified valuation engine and read-only research artifacts
through typed endpoints:

```bash
uvicorn api.app:app --reload
```

The API includes valuation, citywide and neighborhood markets, comparables,
model performance and errors, spatial results, CTA/lake research, and market
segments. Pagination is bounded to 500 records. CORS defaults to the Next.js
development origin and can be configured with `HOMEVALUE_CORS_ORIGINS`.

`HOMEVALUE_DATA_ROOT` changes the processed-artifact root. Missing analytical
artifacts produce explicit 404 responses; missing model bundles produce a 503
only for valuation, leaving available research endpoints operational. OpenAPI
documentation is available at `/docs` and `/openapi.json`.
