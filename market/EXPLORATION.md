# Exploratory market analysis

The pipeline analyzes total sale price and price per square foot by year,
municipality, property type/class, building age and size, assessor neighborhood,
and census tract. PPSF is used only as a companion diagnostic.

Annual outputs include median-price growth, transaction-volume growth, and a
descriptive market phase. A year is labeled `boom` above 10% median growth,
`bust` below -10%, and `stable` otherwise; these are descriptive thresholds,
not causal claims.

Repeat-sale outputs contain the previous sale date and price, elapsed days, and
price change. Rapid resales are those within 365 days. Neighborhood dispersion
is measured with the interquartile range of prices by tract and year.

The spatial figure uses a hexagonal grid over observed coordinates and colors
each cell by its median sale price. Prices are clipped only for visualization at
the 1st and 99th percentiles; grouped CSV outputs retain untrimmed prices.

