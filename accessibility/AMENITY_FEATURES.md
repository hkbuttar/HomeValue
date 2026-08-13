# Amenity accessibility

Step 12 uses two official City of Chicago geometry layers: Chicago Park District
properties (`ejsh-fztr`) and Hydro (`knfe-65pw`). Downloads are bounded GeoJSON
extracts with a timestamp, URL, row count, and SHA-256 manifest. Downtown is an
explicit point reference at Chicago City Hall (`41.8837, -87.6325`) because the
City's legacy Central Business District map asset does not expose geometry via
its current API or GeoJSON export.

Only parks of at least 10 acres are considered “major” by default. The nearest
park distance is measured to its polygon, so a property inside a park has zero
distance. Lake Michigan is identified as the largest polygon in the Hydro layer;
its extent is orders of magnitude larger than the river and inland-water
features. Downtown distance is measured to the City Hall reference.

All geometries and properties are transformed to EPSG:3435 before distances are
calculated and converted from feet to miles. Missing coordinates remain missing.
The resulting features are:

- `lake_distance_miles`;
- `downtown_distance_miles`;
- `park_distance_miles` and `nearest_major_park`.

These City datasets describe current or periodically updated geometry, not a
complete historical amenity network. Rows are labeled `current_geometry_snapshot`.
