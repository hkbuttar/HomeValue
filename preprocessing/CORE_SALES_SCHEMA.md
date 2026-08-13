# Core sales table construction

`core_sales.parquet` contains exactly one row per market sale. It retains source
identifiers and historical-alignment metadata alongside engineered features.

## Improvement-card aggregation

Cook County characteristics are improvement-level, so a parcel may have several
cards in the selected year. Cards are aggregated per sale as follows:

- building area, bedrooms, rooms, bathrooms, and garage spaces: sum;
- land area: maximum, because parcel land is repeated across cards;
- year built: oldest improvement;
- stories: maximum parsed story count;
- basement: true when any improvement has one;
- categorical construction fields: first non-null card value;
- `property_card_count`: number of matched cards.

The schema report records every output column, dtype, missing count, and missing
rate. Features absent from the source are left missing or omitted; accessibility
and market-index features are not fabricated before their dedicated pipelines run.
