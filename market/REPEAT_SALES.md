# Repeat-sales analysis

This analysis orders valid transactions within PIN and pairs each sale with its
immediately preceding sale. Same-PIN same-day records are removed because their
order is indeterminate. Outputs report holding period, total log/percentage
change, and continuously compounded annualized appreciation.

Appreciation is summarized by current sale year and assessor neighborhood (or
census tract). These differences describe the repeat-sale sample and can reflect
both local markets and changing property condition.

The simplified annual repeat-sales index regresses each pair's log price change
on the difference between purchase- and resale-year indicators, with the first
year normalized to 100 and HC3 inference. It is a robustness index rather than a
production-grade Case-Shiller implementation; sparse years and changing sample
composition remain important limitations.

When sale-level hedonic predictions or residuals are present, the report
correlates residuals across consecutive sales. Persistence can reveal stable
unobserved property quality or omitted local information, but is not itself a
causal diagnosis.

