"""Verify the complete seed-39 legacy-4G grid and freeze candidate cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml

from analysis.aggregate_realpdebench_legacy_4g import (
    extract_scale_fields,
    finite_float,
    resolve_input_artifact,
    sha256_file,
)


EXPECTED_CONDITIONS = {
    "mse",
    "lad",
    "orpinn_q29",
    "pinn_ebm_weight_1",
    "pinn_ebm_weight_50",
    "napinn_mse",
    "napinn_l1",
    "napinn_q29",
}
NON_NAPINN = {
    "mse",
    "lad",
    "orpinn_q29",
    "pinn_ebm_weight_1",
    "pinn_ebm_weight_50",
}
NAPINN_MATCHED = {
    "napinn_mse": "mse",
    "napinn_l1": "lad",
    "napinn_q29": "orpinn_q29",
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def condition_name(record: dict[str, Any], config: dict[str, Any]) -> str:
    method = str(record.get("method", ""))
    aliases = {
        "mse": "mse",
        "lad": "lad",
        "orpinn_q29": "orpinn_q29",
        "napinn": "napinn_mse",
        "napinn_lad": "napinn_l1",
        "napinn_q29": "napinn_q29",
    }
    if method in aliases:
        return aliases[method]
    if method != "pinn_ebm":
        raise ValueError(f"Unexpected method {method!r}")
    training = config.get("training", {})
    weight = float(
        training.get("joint_pde_weight", training.get("pde_weight", math.nan))
    )
    if math.isclose(weight, 1.0):
        return "pinn_ebm_weight_1"
    if math.isclose(weight, 50.0):
        return "pinn_ebm_weight_50"
    raise ValueError(f"Unexpected direct PINN-EBM joint PDE weight {weight}")


def validate_record(
    metrics_path: Path, artifact_hash_cache: dict[Path, str]
) -> dict[str, Any]:
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise ValueError(f"Incomplete status: {metrics_path}")
    if record.get("evidence_status") != "full_run_complete_unaggregated":
        raise ValueError(f"Non-full evidence status: {metrics_path}")
    if bool(record.get("smoke_test", True)):
        raise ValueError(f"Smoke result in development grid: {metrics_path}")
    if int(record.get("seed", -1)) != 39:
        raise ValueError(f"Non-development optimizer seed: {metrics_path}")
    if (
        record.get("corruption_positive_label") != "gross_outlier"
        or record.get("corruption_negative_label") != "background_only"
    ):
        raise ValueError(f"Incorrect label semantics: {metrics_path}")
    scales = extract_scale_fields(record, metrics_path)
    if scales["corruption_seed"] != 39:
        raise ValueError(f"Non-development corruption seed: {metrics_path}")

    config_path = metrics_path.parent / "config.yaml"
    metadata_path = metrics_path.parent / "run_metadata.json"
    checkpoint_path = metrics_path.parent / "final.pt"
    if not config_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Missing config/run metadata beside {metrics_path}")
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty final checkpoint: {metrics_path}")
    config = load_yaml(config_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("evidence_status") != record["evidence_status"]:
        raise ValueError(f"Run/metric evidence status mismatch: {metrics_path}")

    condition = condition_name(record, config)
    staged = condition.startswith("napinn_") or condition.startswith("pinn_ebm_")
    if int(record.get("pinn_update_steps", -1)) != 30000:
        raise ValueError(f"Incorrect PINN budget: {metrics_path}")
    expected_estimator = 5000 if staged else 0
    if int(record.get("estimator_init_steps", -1)) != expected_estimator:
        raise ValueError(f"Incorrect estimator budget: {metrics_path}")

    artifact = resolve_input_artifact(record, metrics_path)
    if not artifact.is_file():
        raise ValueError(f"Missing input artifact: {artifact}")
    actual_hash = artifact_hash_cache.get(artifact)
    if actual_hash is None:
        actual_hash = sha256_file(artifact)
        artifact_hash_cache[artifact] = actual_hash
    if actual_hash != record.get("input_artifact_sha256"):
        raise ValueError(f"Input artifact checksum mismatch: {metrics_path}")

    required_metrics = (
        "rMAE",
        "rMSE",
        "pde_momentum_rms",
        "continuity_rms",
    )
    values = {
        key: finite_float(record.get(key), label=f"{metrics_path}: {key}")
        for key in required_metrics
    }
    if any(value < 0.0 for value in values.values()):
        raise ValueError(f"Negative error/residual metric: {metrics_path}")
    if bool(record.get("learned_reynolds", False)):
        raise ValueError(f"Scale selector expects fixed Reynolds: {metrics_path}")
    if condition.startswith("napinn_"):
        for key in (
            "gross_outlier_rejection_rate",
            "background_only_rejection_rate",
        ):
            values[key] = finite_float(
                record.get(key), label=f"{metrics_path}: {key}"
            )
            if not 0.0 <= values[key] <= 1.0:
                raise ValueError(f"Invalid rejection rate: {metrics_path}")
    return {
        "condition": condition,
        "ratio": scales["ratio"],
        "background_scale_multiplier": scales[
            "background_scale_multiplier"
        ],
        "gross_scale_multiplier": scales["gross_scale_multiplier"],
        "input_artifact_path": str(artifact),
        "input_artifact_sha256": actual_hash,
        "metrics_path": str(metrics_path),
        **values,
    }


def load_complete_grid(root: Path, plan: dict[str, Any]) -> dict:
    artifact_hash_cache: dict[Path, str] = {}
    records = [
        validate_record(path, artifact_hash_cache)
        for path in sorted(root.rglob("metrics.json"))
    ]
    expected_count = int(plan["development"]["required_complete_cells"])
    if len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} complete development cells, "
            f"found {len(records)}"
        )
    cells: dict[tuple[float, float, float], dict[str, dict]] = {}
    for record in records:
        key = (
            record["ratio"],
            record["background_scale_multiplier"],
            record["gross_scale_multiplier"],
        )
        condition = record["condition"]
        bucket = cells.setdefault(key, {})
        if condition in bucket:
            raise ValueError(f"Duplicate condition {condition} in cell {key}")
        bucket[condition] = record

    expected_cells = {
        (float(ratio), float(base), float(gross))
        for ratio in plan["development"]["ratios"]
        for base in plan["development"]["background_scale_multipliers"]
        for gross in plan["development"]["gross_scale_multipliers"]
    }
    if set(cells) != expected_cells:
        raise ValueError(
            f"Scale-cell mismatch: missing={expected_cells-set(cells)}, "
            f"unexpected={set(cells)-expected_cells}"
        )
    for key, bucket in cells.items():
        if set(bucket) != EXPECTED_CONDITIONS:
            raise ValueError(
                f"Condition mismatch in {key}: "
                f"missing={EXPECTED_CONDITIONS-set(bucket)}, "
                f"unexpected={set(bucket)-EXPECTED_CONDITIONS}"
            )
        hashes = {record["input_artifact_sha256"] for record in bucket.values()}
        if len(hashes) != 1:
            raise ValueError(f"Unpaired input artifacts in {key}: {hashes}")
    return cells


def evaluate_napinn_variant(
    *,
    variant: str,
    bucket: dict[str, dict],
    plan: dict[str, Any],
) -> dict[str, Any]:
    record = bucket[variant]
    matched = bucket[NAPINN_MATCHED[variant]]
    best_non_na_rmae = min(bucket[name]["rMAE"] for name in NON_NAPINN)
    best_non_na_rmse = min(bucket[name]["rMSE"] for name in NON_NAPINN)
    max_physics_factor = float(
        plan["eligibility"]["physics_maximum_matched_degradation_factor"]
    )
    rejection_min = float(
        plan["eligibility"]["gross_outlier_rejection_minimum"]
    )
    background_max = float(
        plan["eligibility"]["background_only_rejection_maximum"]
    )
    checks = {
        "beats_best_non_napinn_rMAE": record["rMAE"] < best_non_na_rmae,
        "beats_best_non_napinn_rMSE": record["rMSE"] < best_non_na_rmse,
        "beats_matched_ungated_rMAE": record["rMAE"] < matched["rMAE"],
        "beats_matched_ungated_rMSE": record["rMSE"] < matched["rMSE"],
        "momentum_not_catastrophic": (
            record["pde_momentum_rms"]
            <= max_physics_factor * matched["pde_momentum_rms"]
        ),
        "continuity_not_catastrophic": (
            record["continuity_rms"]
            <= max_physics_factor * matched["continuity_rms"]
        ),
        "gross_rejection_at_least_minimum": (
            record["gross_outlier_rejection_rate"] >= rejection_min
        ),
        "background_rejection_at_most_maximum": (
            record["background_only_rejection_rate"] <= background_max
        ),
    }
    gain_rmae = (best_non_na_rmae - record["rMAE"]) / best_non_na_rmae
    gain_rmse = (best_non_na_rmse - record["rMSE"]) / best_non_na_rmse
    return {
        "variant": variant,
        "matched_ungated": NAPINN_MATCHED[variant],
        "eligible": all(checks.values()),
        "checks": checks,
        "ineligible_reasons": [
            name for name, passed in checks.items() if not passed
        ],
        "best_non_napinn_rMAE": best_non_na_rmae,
        "best_non_napinn_rMSE": best_non_na_rmse,
        "relative_rMAE_gain": gain_rmae,
        "relative_rMSE_gain": gain_rmse,
        "ranking_score": min(gain_rmae, gain_rmse),
        "ranking_gain_sum": gain_rmae + gain_rmse,
        "record": record,
    }


def ranking_key(candidate: dict[str, Any]) -> tuple:
    return (
        -candidate["ranking_score"],
        -candidate["ranking_gain_sum"],
        candidate["record"]["rMAE"],
        candidate["record"]["rMSE"],
        candidate.get("ratio", 0.0),
        candidate.get("background_scale_multiplier", 0.0),
        candidate.get("gross_scale_multiplier", 0.0),
        candidate["variant"],
    )


def select(cells: dict, plan: dict[str, Any]) -> dict[str, Any]:
    center = plan["development"]["exact_center"]
    heatmap = []
    eligible_candidates = []
    for (ratio, base, gross), bucket in sorted(cells.items()):
        evaluations = [
            evaluate_napinn_variant(
                variant=variant, bucket=bucket, plan=plan
            )
            for variant in sorted(NAPINN_MATCHED)
        ]
        eligible = [item for item in evaluations if item["eligible"]]
        best = min(eligible, key=ranking_key) if eligible else None
        is_center = (
            math.isclose(
                base, float(center["background_scale_multiplier"])
            )
            and math.isclose(
                gross, float(center["gross_scale_multiplier"])
            )
        )
        cell = {
            "ratio": ratio,
            "background_scale_multiplier": base,
            "gross_scale_multiplier": gross,
            "is_exact_center": is_center,
            "input_artifact_sha256": next(
                iter(bucket.values())
            )["input_artifact_sha256"],
            "conditions": bucket,
            "napinn_variant_evaluations": evaluations,
            "best_eligible_napinn_variant": best,
        }
        heatmap.append(cell)
        if best is not None and not is_center:
            eligible_candidates.append(
                {
                    **best,
                    "ratio": ratio,
                    "background_scale_multiplier": base,
                    "gross_scale_multiplier": gross,
                }
            )
    eligible_candidates.sort(key=ranking_key)
    maximum = int(plan["ranking"]["maximum_candidates"])
    selected = eligible_candidates[:maximum]
    return {
        "complete_heatmap": heatmap,
        "eligible_noncenter_candidates_ranked": eligible_candidates,
        "selected_candidates": selected,
        "selection_count": len(selected),
        "no_eligible_candidate": not eligible_candidates,
    }


def main(args: argparse.Namespace) -> None:
    plan = load_yaml(args.plan)
    if plan.get("status") != "frozen_before_seed39_scale_training":
        raise ValueError("Selection plan is not marked frozen before training")
    cells = load_complete_grid(args.input_root, plan)
    result = select(cells, plan)
    payload = {
        "schema_version": 1,
        "generated_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "input_root": str(args.input_root.resolve()),
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "policy": plan,
        **result,
    }
    selection_core = {
        "plan_sha256": payload["plan_sha256"],
        "selected": [
            {
                key: item[key]
                for key in (
                    "ratio",
                    "background_scale_multiplier",
                    "gross_scale_multiplier",
                    "variant",
                )
            }
            for item in payload["selected_candidates"]
        ],
    }
    payload["selection_id"] = hashlib.sha256(
        json.dumps(selection_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Verified {len(cells)} scale cells and selected "
        f"{payload['selection_count']} candidate(s)."
    )
    print(f"Selection ID: {payload['selection_id']}")
    print(f"Output: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde_legacy_4g_scale/"
            "cylinder_real_piv_legacy4g_scale"
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("rebuttal/legacy4g_scale_selection_plan.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde_legacy_4g_scale/"
            "seed39_scale_selection.json"
        ),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
