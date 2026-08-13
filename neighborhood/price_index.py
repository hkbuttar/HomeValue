"""Construct neighborhood housing price indices and co-movement diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hedonic.model import HedonicConfig, HedonicModel
from market.repeat_sales import construct_repeat_pairs, repeat_sales_index


@dataclass(frozen=True)
class NeighborhoodIndexConfig:
    minimum_sales_per_year: int = 5
    minimum_repeat_pairs: int = 5
    minimum_overlap_years: int = 3
    maximum_plotted_neighborhoods: int = 12


def _geography_column(frame: pd.DataFrame) -> str:
    for column in ("nbhd", "census_tract", "community_area", "municipality"):
        if column in frame:
            return column
    raise ValueError("neighborhood index requires a neighborhood geography column")


def _normalize_group(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty or valid.iloc[0] <= 0:
        return pd.Series(np.nan, index=values.index)
    return 100 * values / valid.iloc[0]


def _prepare(sales: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required = {"sale_id", "pin", "sale_date", "sale_price", "building_sqft"}
    if missing := sorted(required.difference(sales.columns)):
        raise ValueError(f"neighborhood index input is missing: {', '.join(missing)}")
    frame = sales.copy()
    frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
    frame["sale_price"] = pd.to_numeric(frame["sale_price"], errors="coerce")
    frame["building_sqft"] = pd.to_numeric(frame["building_sqft"], errors="coerce")
    geography = _geography_column(frame)
    frame[geography] = frame[geography].astype("string")
    frame = frame.loc[
        frame["sale_date"].notna() & frame["sale_price"].gt(0)
        & frame["building_sqft"].gt(0) & frame[geography].notna()
    ].copy()
    frame["year"] = frame["sale_date"].dt.year.astype(int)
    frame["price_per_sqft"] = frame["sale_price"] / frame["building_sqft"]
    return frame, geography


def _ppsf_index(frame: pd.DataFrame, geography: str, minimum_sales: int) -> pd.DataFrame:
    panel = frame.groupby([geography, "year"], observed=True).agg(
        transaction_count=("sale_id", "size"),
        median_sale_price=("sale_price", "median"),
        median_ppsf=("price_per_sqft", "median"),
    ).reset_index()
    panel = panel.loc[panel["transaction_count"].ge(minimum_sales)].copy()
    panel["median_ppsf_index"] = panel.groupby(geography, observed=True)["median_ppsf"].transform(
        _normalize_group
    )
    return panel


def _hedonic_adjusted_index(frame: pd.DataFrame, geography: str) -> pd.DataFrame:
    # Property-only residualization keeps the remaining neighborhood/time signal visible.
    model = HedonicModel(HedonicConfig(
        minimum_category_count=5,
        include_time=False,
        include_property_type=True,
        include_neighborhood=False,
        include_accessibility=False,
    )).fit(frame)
    frame = frame.copy()
    frame["property_adjusted_log_price"] = (
        np.log(frame["sale_price"]) - model.predict_log(frame)
    )
    adjusted = frame.groupby([geography, "year"], observed=True).agg(
        hedonic_adjusted_log_level=("property_adjusted_log_price", "median")
    ).reset_index()
    adjusted["hedonic_adjusted_level"] = np.exp(adjusted["hedonic_adjusted_log_level"])
    adjusted["hedonic_adjusted_index"] = adjusted.groupby(
        geography, observed=True
    )["hedonic_adjusted_level"].transform(_normalize_group)
    return adjusted


def _neighborhood_repeat_indices(
    frame: pd.DataFrame,
    geography: str,
    minimum_pairs: int,
) -> tuple[pd.DataFrame, dict]:
    pairs = construct_repeat_pairs(frame)
    if "repeat_geography" not in pairs:
        return pd.DataFrame(), {}
    pairs = pairs.loc[pairs["geography_consistent"].fillna(False)].copy()
    outputs, coverage = [], {}
    for neighborhood, group in pairs.groupby("repeat_geography", observed=True):
        coverage[str(neighborhood)] = len(group)
        if len(group) < minimum_pairs:
            continue
        index, diagnostics = repeat_sales_index(group)
        if index.empty:
            continue
        index[geography] = neighborhood
        index = index.rename(columns={"repeat_sales_index": "neighborhood_repeat_sales_index"})
        index["repeat_index_pairs"] = diagnostics["pairs"]
        outputs.append(index)
    return (pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()), coverage


def _comovement(panel: pd.DataFrame, geography: str, minimum_overlap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = panel.pivot(index="year", columns=geography, values="hedonic_adjusted_index")
    growth = np.log(wide).diff()
    correlation = growth.corr(min_periods=minimum_overlap)
    rows = []
    columns = correlation.columns.tolist()
    for left_position, left in enumerate(columns):
        for right in columns[left_position + 1:]:
            overlap = int(growth[[left, right]].dropna().shape[0])
            value = correlation.loc[left, right]
            if pd.notna(value):
                rows.append({
                    "neighborhood_a": left, "neighborhood_b": right,
                    "growth_correlation": float(value), "overlap_years": overlap,
                })
    pairs = pd.DataFrame(rows)
    divergence = panel.sort_values([geography, "year"]).copy()
    divergence["cumulative_change"] = divergence["hedonic_adjusted_index"] / 100 - 1
    latest = divergence.groupby(geography, observed=True).tail(1)[
        [geography, "year", "cumulative_change"]
    ].sort_values("cumulative_change", ascending=False)
    return pairs, latest


def _plot(panel: pd.DataFrame, geography: str, maximum: int, output: Path) -> None:
    volume = panel.groupby(geography, observed=True)["transaction_count"].sum().nlargest(maximum)
    shown = panel.loc[panel[geography].isin(volume.index)]
    figure, axis = plt.subplots(figsize=(11, 6))
    for neighborhood, group in shown.groupby(geography, observed=True):
        axis.plot(group["year"], group["hedonic_adjusted_index"], marker="o", label=str(neighborhood))
    axis.axhline(100, color="black", linewidth=0.7)
    axis.set(
        xlabel="Sale year", ylabel="Property-adjusted index (base = 100)",
        title="Neighborhood housing-market trajectories",
    )
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def build_neighborhood_indices(
    input_path: Path,
    output_dir: Path,
    config: NeighborhoodIndexConfig | None = None,
) -> dict:
    config = config or NeighborhoodIndexConfig()
    frame, geography = _prepare(pd.read_parquet(input_path))
    ppsf = _ppsf_index(frame, geography, config.minimum_sales_per_year)
    adjusted = _hedonic_adjusted_index(frame, geography)
    panel = ppsf.merge(adjusted, on=[geography, "year"], how="left", validate="one_to_one")
    repeat_index, repeat_coverage = _neighborhood_repeat_indices(
        frame, geography, config.minimum_repeat_pairs
    )
    if not repeat_index.empty:
        panel = panel.merge(
            repeat_index[[geography, "year", "neighborhood_repeat_sales_index", "repeat_index_pairs"]],
            on=[geography, "year"], how="left", validate="one_to_one",
        )
    comovement, divergence = _comovement(panel, geography, config.minimum_overlap_years)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_dir / "neighborhood_price_indices.parquet", index=False)
    comovement.to_csv(output_dir / "neighborhood_growth_correlations.csv", index=False)
    divergence.to_csv(output_dir / "neighborhood_divergence.csv", index=False)
    _plot(panel, geography, config.maximum_plotted_neighborhoods, output_dir / "neighborhood_price_indices.png")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path), "geography": geography,
        "config": asdict(config), "sales": len(frame),
        "neighborhoods": int(panel[geography].nunique()),
        "panel_rows": len(panel),
        "repeat_index_neighborhoods": int(
            panel.loc[panel.get("neighborhood_repeat_sales_index", pd.Series(index=panel.index, dtype=float)).notna(), geography].nunique()
        ),
        "repeat_pair_coverage": repeat_coverage,
        "co_movement_pairs": len(comovement),
        "most_appreciated": divergence.head(10).to_dict(orient="records"),
        "most_diverged_downward": divergence.tail(10).sort_values("cumulative_change").to_dict(orient="records"),
        "caution": "Indices are descriptive and inherit transaction mix, sparse-market, and repeat-sale selection limitations.",
    }
    (output_dir / "neighborhood_index_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/neighborhood_indices"))
    parser.add_argument("--minimum-sales-per-year", type=int, default=5)
    parser.add_argument("--minimum-repeat-pairs", type=int, default=5)
    args = parser.parse_args()
    config = NeighborhoodIndexConfig(
        minimum_sales_per_year=args.minimum_sales_per_year,
        minimum_repeat_pairs=args.minimum_repeat_pairs,
    )
    report = build_neighborhood_indices(args.input, args.output, config)
    print(f"Built indices for {report['neighborhoods']} neighborhoods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

