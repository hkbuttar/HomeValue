# Error by market segment

This analysis decomposes held-out valuation errors by price decile, property type,
neighborhood, municipality, building age, transit distance, time, market
archetype, and urban/suburban context whenever those fields are available.

```bash
python -m validation.error_segments
```

Absolute percentage error is calculated against positive observed sale prices.
The report emphasizes median APE because percentage errors can be skewed, keeps
small-group results visible, and excludes groups below the configured sample
minimum from worst-segment rankings. This makes weak performance on unusual or
underserved market slices visible instead of hiding it inside overall MAE.
