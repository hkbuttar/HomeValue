"""Generate exploratory Chicago/Cook County housing-market analysis."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUPINGS = {
    "year": "year",
    "municipality": "municipality",
    "property_type": "residence_type",
    "property_class": "class",
    "assessor_neighborhood": "nbhd",
    "census_tract": "census_tract",
    "building_age_band": "building_age_band",
    "building_size_band": "building_size_band",
}


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def prepare_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Add transparent EDA-only derived fields without removing observations."""
    required = {"sale_id", "pin", "sale_date", "sale_price"}
    if missing := sorted(required.difference(sales.columns)):
        raise ValueError(f"sales table is missing: {', '.join(missing)}")
    frame = sales.copy()
    frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
    frame["sale_price"] = pd.to_numeric(frame["sale_price"], errors="coerce")
    frame["year"] = frame["sale_date"].dt.year.astype("Int64")
    sqft = _numeric_column(frame, "building_sqft")
    frame["price_per_sqft"] = frame["sale_price"].where(
        frame["sale_price"].gt(0) & sqft.gt(0)
    ) / sqft.where(sqft.gt(0))
    age = _numeric_column(frame, "building_age")
    frame["building_age_band"] = pd.cut(
        age, [-np.inf, 9, 24, 49, 74, 99, np.inf],
        labels=["0-9", "10-24", "25-49", "50-74", "75-99", "100+"],
    )
    frame["building_size_band"] = pd.cut(
        sqft, [0, 999, 1499, 1999, 2999, np.inf],
        labels=["<1,000", "1,000-1,499", "1,500-1,999", "2,000-2,999", "3,000+"],
    )
    return frame


def summarize_group(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarize both total value and PPSF so PPSF is never the sole metric."""
    if column not in frame:
        return pd.DataFrame()
    valid = frame.loc[frame[column].notna() & frame["sale_price"].gt(0)].copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(column, observed=True, dropna=False)
        .agg(
            transaction_count=("sale_id", "size"),
            median_sale_price=("sale_price", "median"),
            mean_sale_price=("sale_price", "mean"),
            sale_price_p25=("sale_price", lambda values: values.quantile(0.25)),
            sale_price_p75=("sale_price", lambda values: values.quantile(0.75)),
            median_price_per_sqft=("price_per_sqft", "median"),
        )
        .reset_index()
        .sort_values(column)
    )


def repeat_sale_analysis(frame: pd.DataFrame, rapid_days: int = 365) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.loc[frame["sale_date"].notna() & frame["sale_price"].gt(0)].sort_values(
        ["pin", "sale_date"]
    ).copy()
    ordered["previous_sale_date"] = ordered.groupby("pin")["sale_date"].shift()
    ordered["previous_sale_price"] = ordered.groupby("pin")["sale_price"].shift()
    ordered["days_since_previous_sale"] = (
        ordered["sale_date"] - ordered["previous_sale_date"]
    ).dt.days
    ordered["price_change_rate"] = (
        ordered["sale_price"] / ordered["previous_sale_price"] - 1
    )
    repeated = ordered.loc[ordered["previous_sale_date"].notna()].copy()
    rapid = repeated.loc[repeated["days_since_previous_sale"].between(0, rapid_days)].copy()
    return repeated, rapid


def market_cycles(yearly: pd.DataFrame) -> pd.DataFrame:
    if yearly.empty:
        return yearly.copy()
    result = yearly.sort_values("year").copy()
    result["median_price_growth"] = result["median_sale_price"].pct_change()
    result["volume_growth"] = result["transaction_count"].pct_change()
    result["market_phase"] = np.select(
        [result["median_price_growth"].ge(0.10), result["median_price_growth"].le(-0.10)],
        ["boom", "bust"],
        default="stable",
    )
    result.loc[result["median_price_growth"].isna(), "market_phase"] = "baseline"
    return result


def neighborhood_dispersion(frame: pd.DataFrame) -> pd.DataFrame:
    geography = "census_tract" if "census_tract" in frame else "nbhd"
    if geography not in frame:
        return pd.DataFrame()
    valid = frame.loc[frame[geography].notna() & frame["sale_price"].gt(0)]
    result = valid.groupby(["year", geography], observed=True).agg(
        transaction_count=("sale_id", "size"),
        median_sale_price=("sale_price", "median"),
        price_p25=("sale_price", lambda values: values.quantile(0.25)),
        price_p75=("sale_price", lambda values: values.quantile(0.75)),
    ).reset_index()
    result["price_iqr"] = result["price_p75"] - result["price_p25"]
    return result


def _plot_yearly(cycles: pd.DataFrame, output: Path) -> None:
    if cycles.empty:
        return
    fig, first = plt.subplots(figsize=(10, 5))
    first.plot(cycles["year"], cycles["median_sale_price"], marker="o", color="#155e75")
    first.set_ylabel("Median sale price ($)", color="#155e75")
    first.set_xlabel("Sale year")
    second = first.twinx()
    second.bar(cycles["year"], cycles["transaction_count"], alpha=0.2, color="#ea580c")
    second.set_ylabel("Transactions", color="#9a3412")
    first.set_title("Median sale price and transaction volume")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_spatial(frame: pd.DataFrame, output: Path) -> bool:
    needed = {"longitude", "latitude", "sale_price"}
    if not needed.issubset(frame):
        return False
    spatial = frame.dropna(subset=list(needed)).copy()
    spatial = spatial.loc[spatial["sale_price"].gt(0)]
    if spatial.empty:
        return False
    lower, upper = spatial["sale_price"].quantile([0.01, 0.99])
    spatial["mapped_price"] = spatial["sale_price"].clip(lower, upper)
    fig, axis = plt.subplots(figsize=(8, 8))
    plotted = axis.hexbin(
        spatial["longitude"], spatial["latitude"], C=spatial["mapped_price"],
        reduce_C_function=np.median, gridsize=55, mincnt=1, cmap="viridis",
    )
    fig.colorbar(plotted, ax=axis, label="Median sale price ($), clipped 1st–99th percentile")
    axis.set(title="Spatial structure of recorded sale prices", xlabel="Longitude", ylabel="Latitude")
    axis.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def _plot_property_types(summary: pd.DataFrame, output: Path) -> bool:
    if summary.empty:
        return False
    shown = summary.nlargest(12, "transaction_count").sort_values("median_sale_price")
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.barh(shown.iloc[:, 0].astype(str), shown["median_sale_price"], color="#0f766e")
    axis.set(title="Median sale price by property type", xlabel="Median sale price ($)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _table_html(frame: pd.DataFrame, rows: int = 15) -> str:
    if frame.empty:
        return "<p>Not available from the retained source columns.</p>"
    display = frame.head(rows).copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:,.3g}")
    return display.to_html(index=False, escape=True, border=0)


def _render_report(metrics: dict, cycles: pd.DataFrame, municipality: pd.DataFrame,
                   repeats: pd.DataFrame, images: list[str]) -> str:
    image_html = "".join(
        f'<figure><img src="{html.escape(name)}" alt="Market analysis chart"></figure>' for name in images
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>HomeValue Market Exploration</title>
<style>body{{font:14px system-ui;max-width:1200px;margin:2rem}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.4rem;border:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}figure{{margin:2rem 0}}</style></head><body>
<h1>Chicago and Cook County Housing Market Exploration</h1><p>Generated {metrics['created_at']} from {metrics['valid_price_sales']:,} valid-price sales. PPSF is reported as a companion measure, not as the valuation target.</p>
{image_html}<h2>Market cycles</h2>{_table_html(cycles)}<h2>Municipality summary</h2>{_table_html(municipality.sort_values('transaction_count', ascending=False) if not municipality.empty else municipality)}
<h2>Repeat-sale sample</h2>{_table_html(repeats)}<p>Complete grouped tables are available as CSV files beside this report.</p></body></html>"""


def build_exploration(input_path: Path, output_dir: Path) -> dict:
    frame = prepare_sales(pd.read_parquet(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, column in GROUPINGS.items():
        summary = summarize_group(frame, column)
        summaries[name] = summary
        summary.to_csv(output_dir / f"summary_{name}.csv", index=False)
    cycles = market_cycles(summaries["year"])
    cycles.to_csv(output_dir / "market_cycles.csv", index=False)
    repeats, rapid = repeat_sale_analysis(frame)
    repeats.to_csv(output_dir / "repeat_sales.csv", index=False)
    rapid.to_csv(output_dir / "rapid_resales.csv", index=False)
    dispersion = neighborhood_dispersion(frame)
    dispersion.to_csv(output_dir / "neighborhood_dispersion.csv", index=False)

    images = []
    yearly_image = output_dir / "price_and_volume_by_year.png"
    _plot_yearly(cycles, yearly_image)
    if yearly_image.exists():
        images.append(yearly_image.name)
    spatial_image = output_dir / "spatial_median_price.png"
    if _plot_spatial(frame, spatial_image):
        images.append(spatial_image.name)
    type_image = output_dir / "price_by_property_type.png"
    if _plot_property_types(summaries["property_type"], type_image):
        images.append(type_image.name)

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "rows": len(frame),
        "valid_price_sales": int(frame["sale_price"].gt(0).sum()),
        "valid_ppsf_sales": int(frame["price_per_sqft"].notna().sum()),
        "repeat_sales": len(repeats),
        "rapid_resales": len(rapid),
        "years": sorted(frame["year"].dropna().astype(int).unique().tolist()),
        "outputs": sorted({
            *(path.name for path in output_dir.iterdir()),
            "exploration_metrics.json",
            "market_exploration.html",
        }),
    }
    (output_dir / "exploration_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _render_report(metrics, cycles, summaries["municipality"], repeats, images)
    (output_dir / "market_exploration.html").write_text(report, encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_sales.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/exploration"))
    args = parser.parse_args()
    metrics = build_exploration(args.input, args.output)
    print(f"Analyzed {metrics['rows']} sales across {len(metrics['years'])} years")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
