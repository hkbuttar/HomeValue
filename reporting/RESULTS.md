# Results and honest comparison

This analysis synthesizes the prior analysis artifacts into explicit answers to all
thirteen research questions.

```bash
python -m reporting.results
```

Every answer is classified as Robust, Suggestive, Exploratory, or Data-limited.
The generator does not invent missing results or compare incompatible holdouts
as if they were equivalent. It writes machine-readable JSON, a Markdown report,
and a valid Jupyter notebook containing the same evidence-based conclusions.
