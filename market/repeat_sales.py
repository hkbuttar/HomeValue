"""Analyze repeat-property appreciation and construct a repeat-sales index."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def construct_repeat_pairs(sales: pd.DataFrame) -> pd.DataFrame:
    """Pair each valid sale with the immediately preceding sale of its PIN."""
    required = {"sale_id", "pin", "sale_date", "sale_price"}
    if missing := sorted(required.difference(sales.columns)):
        raise ValueError(f"repeat-sales input is missing: {', '.join(missing)}")
    frame = sales.copy()
    frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
    frame["sale_price"] = pd.to_numeric(frame["sale_price"], errors="coerce")
    frame = frame.loc[
        frame["pin"].notna() & frame["sale_date"].notna() & frame["sale_price"].gt(0)
    ].sort_values(["pin", "sale_date", "sale_id"]).copy()
    if frame.duplicated(["pin", "sale_date"]).any():
        # Same-PIN same-day records cannot be temporally ordered reliably.
        frame = frame.drop_duplicates(["pin", "sale_date"], keep=False)
    group = frame.groupby("pin", sort=False)
    previous_columns = [
        column for column in (
            "sale_id", "sale_date", "sale_price", "nbhd", "census_tract",
            "residence_type", "class", "prediction_hedonic", "hedonic_residual",
        ) if column in frame
    ]
    for column in previous_columns:
        frame[f"previous_{column}"] = group[column].shift()
    pairs = frame.loc[frame["previous_sale_id"].notna()].copy()
    pairs = pairs.rename(columns={
        "sale_id": "current_sale_id",
        "sale_date": "current_sale_date",
        "sale_price": "current_sale_price",
    })
    pairs["holding_period_days"] = (
        pairs["current_sale_date"] - pairs["previous_sale_date"]
    ).dt.days
    pairs = pairs.loc[pairs["holding_period_days"].gt(0)].copy()
    pairs["holding_period_years"] = pairs["holding_period_days"] / 365.2425
    pairs["log_price_change"] = np.log(
        pairs["current_sale_price"] / pairs["previous_sale_price"]
    )
    pairs["total_appreciation"] = (
        pairs["current_sale_price"] / pairs["previous_sale_price"] - 1
    )
    pairs["annualized_appreciation"] = (
        np.exp(pairs["log_price_change"] / pairs["holding_period_years"]) - 1
    )
    pairs["previous_year"] = pairs["previous_sale_date"].dt.year.astype("Int64")
    pairs["current_year"] = pairs["current_sale_date"].dt.year.astype("Int64")
    geography = next((column for column in ("nbhd", "census_tract") if column in pairs), None)
    if geography:
        pairs["repeat_geography"] = pairs[geography].astype("string")
        previous = f"previous_{geography}"
        pairs["geography_consistent"] = (
            pairs[previous].astype("string").eq(pairs[geography].astype("string"))
            if previous in pairs else True
        )
    else:
        pairs["repeat_geography"] = pd.NA
        pairs["geography_consistent"] = pd.NA
    if "prediction_hedonic" in pairs:
        current_residual = np.log(pairs["current_sale_price"]) - np.log(
            pd.to_numeric(pairs["prediction_hedonic"], errors="coerce")
        )
        previous_residual = np.log(pairs["previous_sale_price"]) - np.log(
            pd.to_numeric(pairs["previous_prediction_hedonic"], errors="coerce")
        )
        pairs["current_model_residual"] = current_residual
        pairs["previous_model_residual"] = previous_residual
    elif "hedonic_residual" in pairs:
        pairs["current_model_residual"] = pd.to_numeric(
            pairs["hedonic_residual"], errors="coerce"
        )
        pairs["previous_model_residual"] = pd.to_numeric(
            pairs["previous_hedonic_residual"], errors="coerce"
        )
    return pairs.reset_index(drop=True)


def repeat_sales_index(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Estimate a simplified Bailey-Muth-Nourse annual repeat-sales index."""
    valid = pairs.loc[
        pairs["previous_year"].notna() & pairs["current_year"].notna()
        & pairs["log_price_change"].notna()
        & pairs["current_year"].gt(pairs["previous_year"])
    ].copy()
    years = sorted(set(valid["previous_year"].astype(int)) | set(valid["current_year"].astype(int)))
    if len(years) < 2 or valid.empty:
        return pd.DataFrame(columns=["year", "repeat_sales_index"]), {
            "pairs": 0, "base_year": years[0] if years else None, "r_squared": None
        }
    base_year = years[0]
    estimated_years = years[1:]
    design = np.zeros((len(valid), len(estimated_years)))
    positions = {year: index for index, year in enumerate(estimated_years)}
    for row, (start, end) in enumerate(
        valid[["previous_year", "current_year"]].astype(int).itertuples(index=False)
    ):
        if start != base_year:
            design[row, positions[start]] = -1
        design[row, positions[end]] = 1
    model = sm.OLS(valid["log_price_change"].to_numpy(float), design).fit(cov_type="HC3")
    log_levels = np.r_[0.0, model.params]
    index = 100 * np.exp(log_levels)
    result = pd.DataFrame({
        "year": years,
        "repeat_sales_index": index,
    })
    result["annual_index_growth"] = result["repeat_sales_index"].pct_change()
    diagnostics = {
        "pairs": len(valid), "base_year": base_year,
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "condition_number": float(model.condition_number),
    }
    return result, diagnostics


def _summaries(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    yearly = pairs.groupby("current_year", dropna=False).agg(
        repeat_pair_count=("current_sale_id", "size"),
        median_total_appreciation=("total_appreciation", "median"),
        median_annualized_appreciation=("annualized_appreciation", "median"),
        median_holding_period_years=("holding_period_years", "median"),
    ).reset_index()
    geography = pairs.loc[pairs["repeat_geography"].notna()].groupby(
        "repeat_geography", observed=True
    ).agg(
        repeat_pair_count=("current_sale_id", "size"),
        median_annualized_appreciation=("annualized_appreciation", "median"),
        annualized_appreciation_iqr=(
            "annualized_appreciation", lambda values: values.quantile(0.75) - values.quantile(0.25)
        ),
    ).reset_index()
    return {"year": yearly, "neighborhood": geography}


def _plot_index(index: pd.DataFrame, output: Path) -> None:
    if index.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(index["year"], index["repeat_sales_index"], marker="o", color="#155e75")
    axis.axhline(100, color="black", linewidth=0.7)
    axis.set(
        xlabel="Sale year", ylabel="Repeat-sales index (base = 100)",
        title="Simplified annual repeat-sales price index",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def build_repeat_sales_analysis(input_path: Path, output_dir: Path) -> dict:
    pairs = construct_repeat_pairs(pd.read_parquet(input_path))
    index, index_diagnostics = repeat_sales_index(pairs)
    summaries = _summaries(pairs)
    residual_pairs = pairs.dropna(
        subset=["previous_model_residual", "current_model_residual"]
    ) if {"previous_model_residual", "current_model_residual"}.issubset(pairs) else pd.DataFrame()
    residual_persistence = (
        float(residual_pairs["previous_model_residual"].corr(residual_pairs["current_model_residual"]))
        if len(residual_pairs) >= 3 else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output_dir / "repeat_sale_pairs.parquet", index=False)
    index.to_csv(output_dir / "repeat_sales_index.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(output_dir / f"repeat_sales_by_{name}.csv", index=False)
    _plot_index(index, output_dir / "repeat_sales_index.png")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path), "repeat_pairs": len(pairs),
        "unique_repeat_properties": int(pairs["pin"].nunique()) if len(pairs) else 0,
        "median_holding_period_years": (
            float(pairs["holding_period_years"].median()) if len(pairs) else None
        ),
        "median_annualized_appreciation": (
            float(pairs["annualized_appreciation"].median()) if len(pairs) else None
        ),
        "index": index_diagnostics,
        "residual_persistence_pairs": len(residual_pairs),
        "residual_persistence_correlation": residual_persistence,
        "residual_interpretation": (
            "Positive repeat-sale residual correlation suggests persistent property-specific or locally omitted information."
            if residual_persistence is not None and residual_persistence > 0 else
            "Residual persistence was unavailable or not positive in the retained repeat-sale sample."
        ),
        "caution": (
            "Repeat-sale appreciation combines market change with renovations, deterioration, and transaction-specific effects."
        ),
    }
    (output_dir / "repeat_sales_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/repeat_sales"))
    args = parser.parse_args()
    report = build_repeat_sales_analysis(args.input, args.output)
    print(f"Analyzed {report['repeat_pairs']} consecutive repeat-sale pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

