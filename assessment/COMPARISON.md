# Assessor comparison (optional extension)

Step 37 matches sales to Cook County assessed values by normalized 14-digit PIN
and exact year. It prioritizes Board of Review totals, then certified totals,
then mailed totals.

```bash
python -m assessment.comparison --assessments PATH_TO_ASSESSED_VALUES.parquet
```

The source fields are assessed values, not market values. The workflow divides
them by a row-level `level_of_assessment` when present or by the explicitly
configured residential ratio (default 10%). It reports match coverage,
assessment error by price and geography, and a same-transaction comparison with
available HomeValue predictions. The chosen stage and ratio remain on every
matched record for audit.

Dataset: <https://datacatalog.cookcountyil.gov/d/uzyt-m557>
