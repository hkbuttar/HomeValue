# HomeValue deployment

HomeValue deploys as two stateless containers: FastAPI serves precomputed
research artifacts and model inference, while Next.js serves the web interface.
Normal requests never fit models or execute spatial joins.

## Required production artifacts

Populate `data/processed` before deployment. The API reads the following
outputs at runtime:

- `validation/out_of_time/final_models.joblib`
- `validation/intervals/interval_results.json`
- `accessibility/core_sales_with_accessibility.parquet`
- `neighborhood_indices/neighborhood_price_indices.parquet`
- `segmentation/neighborhood_segments.parquet`
- `comparables/comparable_predictions.parquet`
- `comparables/comparable_links.parquet`
- the JSON and CSV reports exposed by the research endpoints

Keep these artifacts in a read-only persistent volume or copy a curated bundle
into the backend image in the deployment pipeline. Do not ship `data/raw`.
Regenerate artifacts offline when models or source data change.

## Local production smoke test

```bash
docker compose up --build
curl http://localhost:8000/health
```

Open `http://localhost:3000`. Stop the services with `docker compose down`.

## Hosted configuration

Backend environment:

- `HOMEVALUE_DATA_ROOT`: absolute artifact directory, default
  `/app/data/processed` in the image
- `HOMEVALUE_CORS_ORIGINS`: comma-separated deployed frontend origins

Frontend build argument:

- `NEXT_PUBLIC_API_URL`: public HTTPS URL of the FastAPI service. This value is
  embedded into the browser bundle during `npm run build`.

Terminate TLS at the hosting platform, keep the API and artifact store in the
same region, and run at least the `/health` probe. The backend image uses two
workers; adjust that count only after measuring artifact memory use.

## Payload policy

Production routes expose summaries and paginated records with server-enforced
limits. Any future geometry route should return simplified neighborhood
aggregates or a bounded sample, never the full historical transaction table.
