"""Classify raw Cook County sales into the analytical population."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from preprocessing.acquire import normalize_pin

QUALITY_FILTERS = (
    "sale_filter_same_sale_within_365",
    "sale_filter_less_than_10k",
    "sale_filter_deed_type",
)


@dataclass(frozen=True)
class PopulationRules:
    """Auditable rules for selecting residential market transactions."""

    single_family_classes: tuple[str, ...] = (
        "202", "203", "204", "205", "206", "207", "208", "209",
        "210", "234", "278", "295",
    )
    small_multifamily_classes: tuple[str, ...] = ("211", "212")
    include_small_multifamily: bool = False
    minimum_price: float = 10_000.0
    ambiguous_multisales: bool = True

    @property
    def target_classes(self) -> set[str]:
        classes = set(self.single_family_classes)
        if self.include_small_multifamily:
            classes.update(self.small_multifamily_classes)
        return classes


def _boolean(series: pd.Series) -> pd.Series:
    """Parse Socrata booleans without treating missing values as false."""
    mapping = {
        True: True, False: False, 1: True, 0: False,
        "true": True, "false": False, "t": True, "f": False,
        "yes": True, "no": False, "1": True, "0": False,
    }
    return series.map(lambda value: mapping.get(value.lower(), pd.NA) if isinstance(value, str)
                      else mapping.get(value, pd.NA)).astype("boolean")


def classify_sales(sales: pd.DataFrame, rules: PopulationRules | None = None) -> pd.DataFrame:
    """Return all input rows with deterministic population labels and reasons.

    Precedence is excluded, ambiguous, then market. No row is silently dropped.
    """
    rules = rules or PopulationRules()
    required = {"pin", "class", "sale_date", "sale_price", *QUALITY_FILTERS}
    missing = sorted(required.difference(sales.columns))
    if missing:
        raise ValueError(f"sales input is missing required columns: {', '.join(missing)}")

    result = sales.copy()
    result["pin"] = result["pin"].map(normalize_pin).astype("string")
    result["class"] = result["class"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    result["sale_date"] = pd.to_datetime(result["sale_date"], errors="coerce")
    result["sale_price"] = pd.to_numeric(result["sale_price"], errors="coerce")
    for column in QUALITY_FILTERS:
        result[column] = _boolean(result[column])
    if "is_multisale" in result:
        result["is_multisale"] = _boolean(result["is_multisale"])

    reasons: list[list[str]] = [[] for _ in range(len(result))]

    def mark(mask: pd.Series, reason: str) -> None:
        for position in mask.fillna(False).to_numpy().nonzero()[0]:
            reasons[position].append(reason)

    mark(result["pin"].str.len().ne(14) | result["pin"].isna(), "invalid_pin")
    mark(result["sale_date"].isna(), "missing_or_invalid_sale_date")
    mark(result["sale_price"].isna(), "missing_or_invalid_sale_price")
    mark(result["sale_price"].lt(rules.minimum_price), "price_below_minimum")
    mark(~result["class"].isin(rules.target_classes), "non_target_property_class")
    for column in QUALITY_FILTERS:
        mark(result[column].eq(True), column)

    excluded = pd.Series([bool(items) for items in reasons], index=result.index)
    uncertain = result[list(QUALITY_FILTERS)].isna().any(axis=1)
    ambiguous_reasons: list[list[str]] = [[] for _ in range(len(result))]
    for position in uncertain.to_numpy().nonzero()[0]:
        ambiguous_reasons[position].append("missing_sale_filter_metadata")
    if rules.ambiguous_multisales and "is_multisale" in result:
        for position in result["is_multisale"].eq(True).fillna(False).to_numpy().nonzero()[0]:
            ambiguous_reasons[position].append("multi_parcel_sale")

    ambiguous = pd.Series([bool(items) for items in ambiguous_reasons], index=result.index)
    result["population_status"] = "market"
    result.loc[ambiguous, "population_status"] = "ambiguous"
    result.loc[excluded, "population_status"] = "excluded"
    result["exclusion_reasons"] = [json.dumps(items) for items in reasons]
    result["ambiguity_reasons"] = [json.dumps(items) for items in ambiguous_reasons]
    result["is_primary_population"] = result["population_status"].eq("market")
    return result


def _input_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {path}")
    return files


def build_population(
    input_path: Path,
    output_path: Path,
    rules: PopulationRules | None = None,
) -> dict:
    """Classify Parquet inputs and write a partitioned analytical population."""
    rules = rules or PopulationRules()
    frames = [pd.read_parquet(path) for path in _input_files(input_path)]
    classified = classify_sales(pd.concat(frames, ignore_index=True), rules)
    classified["sale_year"] = classified["sale_date"].dt.year.astype("Int64")
    output_path.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for year, frame in classified.groupby("sale_year", dropna=False):
        label = "unknown" if pd.isna(year) else str(int(year))
        directory = output_path / f"sale_year={label}"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "part-00000.parquet"
        frame.to_parquet(destination, index=False)
        files.append(destination)

    counts = classified["population_status"].value_counts().reindex(
        ["market", "ambiguous", "excluded"], fill_value=0
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "rules": asdict(rules),
        "rows": len(classified),
        "status_counts": {key: int(value) for key, value in counts.items()},
        "files": {
            str(path.relative_to(output_path)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (output_path / "population_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/cook_county/sales"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/residential_sales_population")
    )
    parser.add_argument("--include-small-multifamily", action="store_true")
    parser.add_argument("--minimum-price", type=float, default=10_000)
    args = parser.parse_args(argv)
    rules = PopulationRules(
        include_small_multifamily=args.include_small_multifamily,
        minimum_price=args.minimum_price,
    )
    manifest = build_population(args.input, args.output, rules)
    print(json.dumps(manifest["status_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
