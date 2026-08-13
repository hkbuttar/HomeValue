# CTA rail accessibility

CTA rail stations are identified from GTFS routes with `route_type=1`. Platform
stops are collapsed to their parent station so multi-platform complexes count
once. Served route names are retained for the nearest-line feature.

Property and station coordinates are transformed from WGS84 to EPSG:3435
(NAD83 / Illinois East, US survey feet). Distances are computed in that projected
coordinate system and converted to miles. The layer contains:

- distance to the nearest CTA rail station;
- number of distinct parent stations within 0.5 and 1 mile;
- nearest station ID/name and its served line or lines.

Missing property coordinates remain missing rather than receiving a sentinel
distance. The standard CTA feed represents the currently published network, not
a historical station snapshot. Every row is therefore labeled
`current_network_snapshot`; older-sale analyses must treat this as current-state
accessibility or acquire archived GTFS feeds separately.

