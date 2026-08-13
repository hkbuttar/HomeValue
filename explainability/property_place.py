"""Decompose individual valuations into property, place, and time contributions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.valuation import FEATURE_GROUPS


@dataclass(frozen=True)
class AttributionConfig:
    permutations: int = 64
    maximum_properties: int = 25
    random_seed: int = 42


GROUP_LABELS = {
    "structural": "property", "temporal": "time_market",
    "neighborhood": "place", "accessibility": "place", "prior_spatial": "place",
}


def _feature_groups(features: list[str]) -> dict[str, str]:
    lookup = {
        feature: GROUP_LABELS[group]
        for group, columns in FEATURE_GROUPS.items()
        for feature in columns
    }
    return {feature: lookup.get(feature, "other") for feature in features}


def _reference_row(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    values = {}
    for feature in features:
        series = frame[feature] if feature in frame else pd.Series(dtype=float)
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            values[feature] = pd.to_numeric(series, errors="coerce").median()
        else:
            mode = series.dropna().mode()
            values[feature] = mode.iloc[0] if len(mode) else pd.NA
    return pd.DataFrame([values], columns=features)


def _predict_dollars(preprocessor, model, frame: pd.DataFrame, smearing_factor: float = 1.0) -> float:
    log_value = float(model.predict(preprocessor.transform(frame))[0])
    return float(np.exp(np.clip(log_value, 0, 50)) * smearing_factor)


def _attributions(target: pd.DataFrame, reference: pd.DataFrame, features: list[str],
                  preprocessor, model, permutations: int, rng,
                  smearing_factor: float = 1.0) -> tuple[float, float, dict[str, float]]:
    if permutations < 1:
        raise ValueError("permutations must be positive")
    baseline = _predict_dollars(preprocessor, model, reference, smearing_factor)
    estimated = _predict_dollars(preprocessor, model, target, smearing_factor)
    contributions = {feature: 0.0 for feature in features}
    for _ in range(permutations):
        current = reference.copy()
        previous_value = baseline
        for position in rng.permutation(len(features)):
            feature = features[int(position)]
            current[feature] = target.iloc[0][feature]
            next_value = _predict_dollars(preprocessor, model, current, smearing_factor)
            contributions[feature] += (next_value - previous_value) / permutations
            previous_value = next_value
    # Floating-point reconciliation preserves the exact baseline + contributions identity.
    residual = estimated - baseline - sum(contributions.values())
    if features:
        lead = max(features, key=lambda feature: abs(contributions[feature]))
        contributions[lead] += residual
    return baseline, estimated, contributions


def decompose_property_values(
    artifact_path: Path,
    properties_path: Path,
    reference_path: Path,
    output_dir: Path,
    sale_ids: list[str] | None = None,
    config: AttributionConfig | None = None,
) -> dict:
    config = config or AttributionConfig()
    artifact = joblib.load(artifact_path)
    features, preprocessor = artifact["features"], artifact["preprocessor"]
    model_name = artifact.get("selected_model") or next(iter(artifact["models"]))
    model = artifact["models"][model_name]
    smearing_factor = artifact.get("smearing_factors", {}).get(model_name, 1.0)
    properties, reference_data = pd.read_parquet(properties_path), pd.read_parquet(reference_path)
    missing = sorted(set(features).difference(properties.columns))
    if missing:
        raise ValueError(f"property data is missing model features: {', '.join(missing)}")
    if sale_ids:
        if "sale_id" not in properties:
            raise ValueError("sale_id filtering requires sale_id in property data")
        properties = properties.loc[properties["sale_id"].astype(str).isin(set(sale_ids))]
    if properties.empty:
        raise ValueError("no properties were selected for attribution")
    properties = properties.head(config.maximum_properties).copy()
    if "year" in properties and "year" in reference_data:
        first_year = pd.to_numeric(properties["year"], errors="coerce").min()
        earlier = reference_data.loc[pd.to_numeric(reference_data["year"], errors="coerce").lt(first_year)]
        if len(earlier):
            reference_data = earlier
    reference = _reference_row(reference_data, features)
    groups = _feature_groups(features)
    rng = np.random.default_rng(config.random_seed)
    rows, summaries = [], []
    for index, property_row in properties.iterrows():
        target = property_row[features].to_frame().T
        baseline, estimated, contributions = _attributions(
            target, reference, features, preprocessor, model, config.permutations, rng,
            smearing_factor,
        )
        property_id = str(property_row.get("sale_id", index))
        for feature, contribution in contributions.items():
            reference_value = reference.iloc[0][feature]
            property_value = property_row[feature]
            rows.append({
                "property_id": property_id, "model": model_name, "component": groups[feature],
                "feature": feature,
                "reference_value": None if pd.isna(reference_value) else str(reference_value),
                "property_value": None if pd.isna(property_value) else str(property_value),
                "dollar_contribution": contribution,
            })
        grouped = pd.Series(contributions).groupby(pd.Series(groups)).sum().to_dict()
        summaries.append({
            "property_id": property_id, "model": model_name, "baseline_market_value": baseline,
            "property_contribution": float(grouped.get("property", 0)),
            "place_contribution": float(grouped.get("place", 0)),
            "time_market_contribution": float(grouped.get("time_market", 0)),
            "other_contribution": float(grouped.get("other", 0)),
            "estimated_value": estimated,
            "reconciliation_error": float(estimated - baseline - sum(contributions.values())),
        })
    detail, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(output_dir / "property_value_attributions.parquet", index=False)
    summary.to_csv(output_dir / "property_value_decompositions.csv", index=False)
    reference.to_json(output_dir / "reference_market_profile.json", orient="records", indent=2)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "artifact": str(artifact_path), "properties_input": str(properties_path),
        "reference_input": str(reference_path), "model": model_name,
        "properties_explained": len(summary), "features_attributed": len(features),
        "maximum_absolute_reconciliation_error": float(summary["reconciliation_error"].abs().max()),
        "method": "Monte Carlo Shapley values from fitted-model counterfactual predictions relative to a median/mode reference market profile.",
        "interpretation_caution": "Contributions explain this model's estimate relative to the stated reference profile; they are not causal price effects.",
    }
    (output_dir / "property_place_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("data/processed/validation/out_of_time/final_models.joblib"))
    parser.add_argument("--properties", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--reference", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/explainability/property_place"))
    parser.add_argument("--sale-id", action="append", dest="sale_ids")
    parser.add_argument("--permutations", type=int, default=64)
    parser.add_argument("--maximum-properties", type=int, default=25)
    args = parser.parse_args()
    report = decompose_property_values(
        args.artifact, args.properties, args.reference, args.output, args.sale_ids,
        AttributionConfig(permutations=args.permutations, maximum_properties=args.maximum_properties),
    )
    print(f"Explained {report['properties_explained']} properties with {report['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
