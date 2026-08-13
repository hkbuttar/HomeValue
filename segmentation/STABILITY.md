# Segment stability over time

This analysis re-estimates neighborhood market segments in non-overlapping historical
periods. Because cluster numbers have no inherent meaning, each period's labels
are aligned to the preceding period by minimum centroid distance before changes
are interpreted.

Run:

```bash
python -m segmentation.stability --period-years 3 --clusters 4
```

The workflow writes neighborhood-period assignments, conditional transition
probabilities, neighborhood persistence rates, and a JSON report with
consecutive-period adjusted Rand indices. Only neighborhoods meeting the sales
threshold in a period enter that period's fit. Transitions describe movement
between relative market regimes and should not be read as causal effects.
