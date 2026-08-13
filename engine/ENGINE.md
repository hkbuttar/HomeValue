# Unified HomeValue engine

This analysis provides one reusable inference interface over the fitted analytical
pipeline. It combines structural inputs, calendar context, optional neighborhood
profiles, current CTA accessibility, the validation-selected nonlinear model,
split-conformal uncertainty, fitted-model value attribution, and strictly prior
local comparable sales.

The engine loads the serialized preprocessing/model bundle and its retained
Duan smearing factor, ensuring production dollar estimates match validation.
Every response contains the reference-market baseline, property/place/time
contributions, calibrated bounds, ranked feature drivers, and nearby historical
comparables. The components reconcile exactly to the point estimate.

The engine is intentionally framework-independent; the relevant upstream analysis exposes it through
FastAPI without duplicating valuation logic.
