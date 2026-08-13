"""Create comparable hedonic, spatial, and machine-learning explanations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExplainabilityConfig:
    sample_size: int = 500
    permutation_repeats: int = 5
    random_seed: int = 42
    partial_dependence_points: int = 7


CONCEPT_PATTERNS = {
    "Living Area": ("building_sqft",),
    "Bathrooms": ("bathrooms",),
    "Neighborhood Income": ("median_household_income",),
    "CTA Accessibility": ("cta_distance", "cta_stations"),
    "Lake Distance": ("lake_distance",),
    "Spatial Spillover": ("rho", "lambda", "W_"),
}


def _hedonic_effects(path: Path | None) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame(columns=["term", "coefficient", "ci_lower", "ci_upper", "marginal_effect"])
    table = pd.read_csv(path)
    coefficient = pd.to_numeric(table["coefficient"], errors="coerce")
    table["marginal_effect"] = np.expm1(coefficient)
    log_size = table["term"].eq("log_building_sqft")
    table.loc[log_size, "marginal_effect"] = coefficient.loc[log_size]
    table["marginal_effect_interpretation"] = np.where(
        log_size, "price elasticity", "proportional price change for a one-unit increase"
    )
    return table


def _spatial_effects(coefficients_path: Path | None, results_path: Path | None) -> pd.DataFrame:
    rows = []
    if coefficients_path and coefficients_path.exists():
        coefficients = pd.read_csv(coefficients_path)
        for row in coefficients.to_dict(orient="records"):
            rows.append({"effect_type": "coefficient", **row})
    if results_path and results_path.exists():
        report = json.loads(results_path.read_text(encoding="utf-8"))
        comparison = report.get("full_sample_comparison", {})
        for model, values in comparison.items():
            if not isinstance(values, dict):
                continue
            for parameter in ("rho", "lambda", "rho_p_value", "lambda_p_value"):
                if parameter in values:
                    rows.append({
                        "effect_type": "spatial_parameter", "model": model,
                        "term": parameter, "coefficient": values[parameter],
                    })
            multipliers = values.get("spatial_multipliers", {})
            for term, effects in multipliers.items():
                if isinstance(effects, dict):
                    for effect, value in effects.items():
                        rows.append({
                            "effect_type": "spatial_multiplier", "model": model,
                            "term": term, "effect": effect, "coefficient": value,
                        })
    return pd.DataFrame(rows)


def _sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if "year" in frame:
        latest = pd.to_numeric(frame["year"], errors="coerce").max()
        recent = frame.loc[pd.to_numeric(frame["year"], errors="coerce").eq(latest)]
        if len(recent):
            frame = recent
    return frame.sample(min(size, len(frame)), random_state=seed).copy()


def _ml_explanations(artifact_path: Path | None, data_path: Path | None,
                     config: ExplainabilityConfig):
    empty = pd.DataFrame()
    if not artifact_path or not artifact_path.exists() or not data_path or not data_path.exists():
        return empty, empty, empty, {"status": "not_available"}
    artifact, frame = joblib.load(artifact_path), pd.read_parquet(data_path)
    features = artifact["features"]
    frame = frame.loc[pd.to_numeric(frame["sale_price"], errors="coerce").gt(0)].dropna(subset=features, how="all")
    sample = _sample(frame, config.sample_size, config.random_seed)
    preprocessor = artifact["preprocessor"]
    model_name = artifact.get("selected_model") or next(iter(artifact["models"]))
    model = artifact["models"][model_name]
    transformed = preprocessor.transform(sample[features])
    actual_log = np.log(pd.to_numeric(sample["sale_price"], errors="coerce").to_numpy())
    baseline = np.mean(np.abs(model.predict(transformed) - actual_log))
    rng = np.random.default_rng(config.random_seed)
    permutation_rows = []
    for feature in features:
        increases = []
        for _ in range(config.permutation_repeats):
            shuffled = sample[features].copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            error = np.mean(np.abs(model.predict(preprocessor.transform(shuffled)) - actual_log))
            increases.append(error - baseline)
        permutation_rows.append({
            "model": model_name, "feature": feature,
            "importance_mean_log_mae_increase": float(np.mean(increases)),
            "importance_std": float(np.std(increases)),
        })
    permutation = pd.DataFrame(permutation_rows).sort_values(
        "importance_mean_log_mae_increase", ascending=False
    )
    transformed_names = preprocessor.get_feature_names_out()
    try:
        import shap

        values = shap.TreeExplainer(model).shap_values(transformed)
        if isinstance(values, list):
            values = values[0]
        shap_rows = pd.DataFrame({
            "transformed_feature": transformed_names,
            "mean_absolute_shap": np.abs(np.asarray(values)).mean(axis=0),
        })
        shap_rows["feature"] = shap_rows["transformed_feature"].map(
            lambda name: next((feature for feature in features if name.endswith(f"__{feature}") or f"__{feature}_" in name), name)
        )
        shap_summary = shap_rows.groupby("feature", observed=True)["mean_absolute_shap"].sum().reset_index().sort_values(
            "mean_absolute_shap", ascending=False
        )
        shap_status = "calculated"
    except Exception as error:
        shap_summary = empty
        shap_status = f"unavailable: {type(error).__name__}"
    partial_rows = []
    for feature in features:
        values = pd.to_numeric(sample[feature], errors="coerce").astype(float)
        if values.notna().sum() < 2 or values.nunique() < 2:
            continue
        grid = np.unique(np.quantile(values.dropna(), np.linspace(.05, .95, config.partial_dependence_points)))
        for value in grid:
            varied = sample[features].copy()
            varied[feature] = value
            partial_rows.append({
                "model": model_name, "feature": feature, "feature_value": float(value),
                "mean_prediction_log_price": float(model.predict(preprocessor.transform(varied)).mean()),
            })
    return permutation, shap_summary, pd.DataFrame(partial_rows), {
        "status": "calculated", "model": model_name, "rows": len(sample),
        "baseline_log_mae": float(baseline), "shap_status": shap_status,
    }


def _agreement(hedonic: pd.DataFrame, spatial: pd.DataFrame,
               permutation: pd.DataFrame, shap_summary: pd.DataFrame) -> pd.DataFrame:
    hedonic_terms = hedonic.get("term", pd.Series(dtype=str)).astype(str).tolist()
    spatial_terms = spatial.get("term", pd.Series(dtype=str)).astype(str).tolist()
    ml_features = set(permutation.get("feature", pd.Series(dtype=str)).astype(str)) | set(
        shap_summary.get("feature", pd.Series(dtype=str)).astype(str)
    )
    rows = []
    for concept, patterns in CONCEPT_PATTERNS.items():
        hedonic_evidence = any(any(pattern in term for pattern in patterns) for term in hedonic_terms)
        spatial_evidence = any(any(pattern in term for pattern in patterns) for term in spatial_terms)
        ml_evidence = any(any(pattern in feature for pattern in patterns) for feature in ml_features)
        count = sum((hedonic_evidence, spatial_evidence, ml_evidence))
        rows.append({
            "concept": concept, "hedonic": hedonic_evidence, "spatial": spatial_evidence,
            "machine_learning": ml_evidence, "methods_with_evidence": count,
            "agreement": "broad" if count == 3 else "partial" if count >= 1 else "not_observed",
        })
    return pd.DataFrame(rows)


def build_explainability_report(
    output_dir: Path,
    hedonic_coefficients_path: Path | None = None,
    spatial_coefficients_path: Path | None = None,
    spatial_results_path: Path | None = None,
    ml_artifact_path: Path | None = None,
    data_path: Path | None = None,
    config: ExplainabilityConfig | None = None,
) -> dict:
    config = config or ExplainabilityConfig()
    hedonic = _hedonic_effects(hedonic_coefficients_path)
    spatial = _spatial_effects(spatial_coefficients_path, spatial_results_path)
    permutation, shap_summary, partial, ml_status = _ml_explanations(
        ml_artifact_path, data_path, config
    )
    agreement = _agreement(hedonic, spatial, permutation, shap_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    hedonic.to_csv(output_dir / "hedonic_marginal_effects.csv", index=False)
    spatial.to_csv(output_dir / "spatial_effects.csv", index=False)
    permutation.to_csv(output_dir / "ml_permutation_importance.csv", index=False)
    shap_summary.to_csv(output_dir / "ml_shap_importance.csv", index=False)
    partial.to_csv(output_dir / "ml_partial_dependence.csv", index=False)
    agreement.to_csv(output_dir / "method_agreement.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "hedonic_terms": len(hedonic), "spatial_effect_rows": len(spatial),
        "ml": ml_status, "method_agreement": agreement.to_dict(orient="records"),
        "interpretation_caution": "Importance and partial dependence describe model behavior; coefficients and spatial effects remain associations unless identification supports causal claims.",
    }
    (output_dir / "explainability_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hedonic-coefficients", type=Path, default=Path("data/processed/hedonic/hedonic_coefficients.csv"))
    parser.add_argument("--spatial-coefficients", type=Path, default=Path("data/processed/spatial_durbin/spatial_durbin_coefficients.csv"))
    parser.add_argument("--spatial-results", type=Path, default=Path("data/processed/spatial_durbin/spatial_durbin_results.json"))
    parser.add_argument("--ml-artifact", type=Path, default=Path("data/processed/validation/out_of_time/final_models.joblib"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/explainability"))
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()
    report = build_explainability_report(
        args.output, args.hedonic_coefficients, args.spatial_coefficients,
        args.spatial_results, args.ml_artifact, args.data,
        ExplainabilityConfig(sample_size=args.sample_size),
    )
    print(f"Built explanation report; ML status: {report['ml']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
