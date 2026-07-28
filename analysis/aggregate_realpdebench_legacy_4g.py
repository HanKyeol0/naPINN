"""Aggregate complete legacy-4G Cylinder-PIV rebuttal runs.

Only full, non-smoke reporting seeds 40--42 are eligible.  The aggregator
keeps every observed method/scale outcome, verifies that methods in each
paired block used the same corruption artifact, and reports signed paired
differences without selecting favorable cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


REPORTING_SEEDS = (40, 41, 42)
FULL_EVIDENCE_STATUS = "full_run_complete_unaggregated"
NAPINN_METHODS = {"napinn", "napinn_lad", "napinn_q29"}
STAGED_METHODS = {"pinn_ebm", *NAPINN_METHODS}
DEFAULT_EXPECTED_METHODS = (
    "mse",
    "lad",
    "orpinn_q29",
    "pinn_ebm",
    "napinn",
    "napinn_lad",
    "napinn_q29",
)
LOWER_IS_BETTER_FIELD_METRICS = ("rMAE", "rMSE")
METRICS = (
    "rMAE",
    "rMSE",
    "pde_momentum_rms",
    "continuity_rms",
    "gross_outlier_detection_auroc",
    "estimator_gross_outlier_detection_auroc",
    "estimator_mean_negative_log_density",
    "estimator_gross_outlier_mean_negative_log_density",
    "estimator_background_only_mean_negative_log_density",
    "gross_outlier_rejection_rate",
    "background_only_rejection_rate",
    "gross_outlier_mean_gate_weight",
    "background_only_mean_gate_weight",
    "mean_gate_weight",
    "retained_fraction",
    "metadata_reynolds",
    "effective_reynolds",
    "effective_reynolds_relative_error",
    "warmup_sec",
    "estimator_init_sec",
    "joint_sec",
    "training_wall_sec",
    "evaluation_wall_sec",
    "end_to_end_wall_sec",
    "pinn_update_steps",
    "estimator_init_steps",
    "gpu_peak_memory_allocated_bytes",
    "gpu_peak_memory_reserved_bytes",
    "gpu_peak_memory_mb",
)
CORE_METRICS = (
    "rMAE",
    "rMSE",
    "pde_momentum_rms",
    "continuity_rms",
    "metadata_reynolds",
    "effective_reynolds",
    "effective_reynolds_relative_error",
    "warmup_sec",
    "estimator_init_sec",
    "joint_sec",
    "training_wall_sec",
    "evaluation_wall_sec",
    "end_to_end_wall_sec",
    "pinn_update_steps",
    "estimator_init_steps",
    "gpu_peak_memory_allocated_bytes",
    "gpu_peak_memory_reserved_bytes",
)
BASE_INVARIANT_FIELDS = (
    "benchmark",
    "sim_id",
    "source_sha256",
    "sensor_seed",
    "n_spatial_sensors",
    "n_frames",
    "corruption_kind",
    "corruption_positive_label",
    "corruption_negative_label",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def nested_value(mapping: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = mapping
        for part in path:
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            return value
    return None


def consistent_nested_value(
    mapping: dict[str, Any],
    *paths: tuple[str, ...],
    label: str,
    numeric: bool = True,
) -> Any:
    observed = []
    for path in paths:
        value: Any = mapping
        for part in path:
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            observed.append((path, value))
    if not observed:
        raise ValueError(f"Missing {label}")
    normalized = [
        (
            path,
            canonical_float(value, label=f"{label} at {'.'.join(path)}")
            if numeric
            else value,
        )
        for path, value in observed
    ]
    reference = normalized[0][1]
    if any(value != reference for _, value in normalized[1:]):
        raise ValueError(f"Inconsistent {label} aliases: {normalized}")
    return reference


def finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def canonical_float(value: Any, *, label: str) -> float:
    result = finite_float(value, label=label)
    return float(f"{result:.12g}")


def summarize_per_seed(values: dict[int, float]) -> dict[str, Any]:
    ordered = {str(seed): values[seed] for seed in sorted(values)}
    sequence = list(ordered.values())
    return {
        "n": len(sequence),
        "mean": mean(sequence),
        "sample_std": stdev(sequence) if len(sequence) > 1 else 0.0,
        "per_seed": ordered,
    }


def is_legacy_4g(record: dict[str, Any]) -> bool:
    kind = str(record.get("corruption_kind", "")).lower()
    labels_are_explicit = (
        record.get("corruption_positive_label") == "gross_outlier"
        and record.get("corruption_negative_label") == "background_only"
    )
    tokens = ("legacy_4g", "legacy4g", "four_gaussian", "four-gaussian")
    return labels_are_explicit or any(token in kind for token in tokens)


def extract_scale_fields(record: dict[str, Any], path: Path) -> dict[str, Any]:
    corruption = record.get("corruption_metadata")
    if not isinstance(corruption, dict):
        raise ValueError(f"Legacy-4G run lacks corruption_metadata: {path}")

    ratio = consistent_nested_value(
        record,
        ("gross_outlier_row_ratio_requested",),
        ("corruption_metadata", "gross_row_ratio_requested"),
        ("corruption_metadata", "gross_outlier_row_ratio_requested"),
        label=f"{path}: gross-outlier row ratio",
    )
    background_scale = consistent_nested_value(
        record,
        ("background_scale_multiplier",),
        ("corruption_metadata", "base_scale_multiplier"),
        ("corruption_metadata", "background_scale_multiplier"),
        label=f"{path}: background scale multiplier",
    )
    gross_scale = consistent_nested_value(
        record,
        ("gross_scale_multiplier",),
        ("corruption_metadata", "gross_scale_multiplier"),
        label=f"{path}: gross scale multiplier",
    )
    corruption_seed = consistent_nested_value(
        record,
        ("corruption_seed",),
        ("corruption_metadata", "seed"),
        ("corruption_metadata", "corruption_seed"),
        label=f"{path}: corruption seed",
    )
    if not float(corruption_seed).is_integer():
        raise ValueError(f"{path}: corruption seed must be an integer")
    return {
        "ratio": ratio,
        "background_scale_multiplier": background_scale,
        "gross_scale_multiplier": gross_scale,
        "corruption_seed": int(corruption_seed),
    }


def direct_pinn_ebm_weight(
    method: str, config: dict[str, Any], path: Path
) -> float | None:
    if method != "pinn_ebm":
        return None
    weight = nested_value(
        config,
        ("training", "joint_pde_weight"),
        ("training", "pde_weight"),
    )
    if weight is None:
        raise ValueError(
            f"Direct PINN-EBM run lacks joint/base PDE weight in {path}"
        )
    return canonical_float(weight, label=f"{path}: direct PINN-EBM weight")


def group_id(key: tuple[float, float, float, str, float | None]) -> str:
    ratio, background, gross, method, weight = key
    weight_text = "none" if weight is None else f"{weight:g}"
    return (
        f"ratio={ratio:g}|b={background:g}|o={gross:g}|method={method}"
        f"|direct_pinn_ebm_joint_pde_weight={weight_text}"
    )


def resolve_input_artifact(record: dict[str, Any], metrics_path: Path) -> Path:
    raw_path = record.get("input_artifact_path")
    if not raw_path:
        raise ValueError(f"Missing input_artifact_path: {metrics_path}")
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = path.resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    return (metrics_path.parent / path).resolve()


def validate_complete_record(
    record: dict[str, Any],
    metrics_path: Path,
    artifact_hash_cache: dict[Path, str],
) -> dict[str, Any]:
    run_dir = metrics_path.parent
    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "run_metadata.json"
    checkpoint_path = run_dir / "final.pt"
    for required_path in (config_path, metadata_path, checkpoint_path):
        if not required_path.is_file() or required_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Complete run lacks non-empty {required_path.name}: "
                f"{metrics_path}"
            )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config is not a YAML mapping: {config_path}")
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if run_metadata.get("status") != "complete":
        raise ValueError(f"run_metadata status is not complete: {metadata_path}")
    if run_metadata.get("evidence_status") != FULL_EVIDENCE_STATUS:
        raise ValueError(
            f"run_metadata evidence status is not full: {metadata_path}"
        )
    if run_metadata.get("smoke_test", True):
        raise ValueError(f"Complete evidence is marked smoke: {metadata_path}")

    seed = int(record["seed"])
    method = str(record["method"]).lower()
    if int(run_metadata.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch between sidecars: {metrics_path}")
    if str(run_metadata.get("method", "")).lower() != method:
        raise ValueError(f"Method mismatch between sidecars: {metrics_path}")

    if record.get("corruption_positive_label") != "gross_outlier":
        raise ValueError(
            f"Legacy-4G positive label must be gross_outlier: {metrics_path}"
        )
    if record.get("corruption_negative_label") != "background_only":
        raise ValueError(
            f"Legacy-4G negative label must be background_only: {metrics_path}"
        )
    nested_semantics = nested_value(
        record, ("corruption_metadata", "label_semantics")
    )
    if isinstance(nested_semantics, dict):
        nested_positive = nested_semantics.get(
            "positive", nested_semantics.get("true")
        )
        nested_negative = nested_semantics.get(
            "negative", nested_semantics.get("false")
        )
        if nested_positive != "gross_outlier" or nested_negative != (
            "background_only"
        ):
            raise ValueError(
                "Legacy-4G corruption metadata must use gross-outlier/"
                f"background-only labels: {metrics_path}"
            )

    for field in BASE_INVARIANT_FIELDS:
        if record.get(field) is None:
            raise ValueError(
                f"Complete run lacks comparison invariant {field}: "
                f"{metrics_path}"
            )

    for metric in CORE_METRICS:
        if metric not in record:
            raise ValueError(f"Complete run lacks {metric}: {metrics_path}")
        finite_float(record[metric], label=f"{metrics_path}: {metric}")
    if int(record["pinn_update_steps"]) != 30000:
        raise ValueError(
            f"Full run must contain 30,000 PINN updates: {metrics_path}"
        )
    expected_estimator_steps = 5000 if method in STAGED_METHODS else 0
    if int(record["estimator_init_steps"]) != expected_estimator_steps:
        raise ValueError(
            f"{method} requires {expected_estimator_steps} estimator-only "
            f"updates: {metrics_path}"
        )

    if method in NAPINN_METHODS:
        for metric in (
            "gross_outlier_detection_auroc",
            "estimator_gross_outlier_detection_auroc",
            "gross_outlier_rejection_rate",
            "background_only_rejection_rate",
            "retained_fraction",
        ):
            if metric not in record:
                raise ValueError(
                    f"{method} lacks legacy-4G metric {metric}: {metrics_path}"
                )
            finite_float(record[metric], label=f"{metrics_path}: {metric}")
    elif method == "pinn_ebm":
        metric = "estimator_gross_outlier_detection_auroc"
        if metric not in record:
            raise ValueError(
                f"Direct PINN-EBM lacks legacy-4G metric {metric}: "
                f"{metrics_path}"
            )
        finite_float(record[metric], label=f"{metrics_path}: {metric}")

    artifact_path = resolve_input_artifact(record, metrics_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Recorded input artifact does not exist: {artifact_path}"
        )
    recorded_hash = str(record.get("input_artifact_sha256", "")).lower()
    if len(recorded_hash) != 64 or any(
        character not in "0123456789abcdef" for character in recorded_hash
    ):
        raise ValueError(f"Invalid input artifact SHA-256: {metrics_path}")
    actual_hash = artifact_hash_cache.get(artifact_path)
    if actual_hash is None:
        actual_hash = sha256_file(artifact_path)
        artifact_hash_cache[artifact_path] = actual_hash
    if actual_hash != recorded_hash:
        raise ValueError(
            f"Input artifact SHA-256 mismatch for {artifact_path}: "
            f"recorded={recorded_hash}, actual={actual_hash}"
        )
    if str(run_metadata.get("input_artifact_sha256", "")).lower() != recorded_hash:
        raise ValueError(f"Input SHA mismatch in run_metadata: {metrics_path}")
    configured_hash = nested_value(
        config, ("data", "input_artifact_sha256")
    )
    if configured_hash is not None and str(configured_hash).lower() != recorded_hash:
        raise ValueError(f"Input SHA mismatch in config: {metrics_path}")

    scale = extract_scale_fields(record, metrics_path)
    direct_weight = direct_pinn_ebm_weight(method, config, config_path)
    normalized = dict(record)
    normalized.update(scale)
    normalized["_metrics_path"] = str(metrics_path.resolve())
    normalized["_input_artifact_path"] = str(artifact_path)
    normalized["_direct_pinn_ebm_joint_pde_weight"] = direct_weight
    normalized["_corruption_metadata_fingerprint"] = metadata_fingerprint(
        record["corruption_metadata"]
    )
    return normalized


def load_eligible_records(root: Path) -> tuple[list[dict[str, Any]], dict]:
    records: list[dict[str, Any]] = []
    exclusions: dict[str, list[str]] = defaultdict(list)
    artifact_hash_cache: dict[Path, str] = {}
    for path in sorted(root.rglob("metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        source = str(path.resolve())
        if record.get("smoke_test", True):
            exclusions["smoke"].append(source)
            continue
        if record.get("status") != "complete":
            exclusions["incomplete_status"].append(source)
            continue
        if record.get("evidence_status") != FULL_EVIDENCE_STATUS:
            exclusions["not_full_evidence"].append(source)
            continue
        if int(record.get("seed", -1)) not in REPORTING_SEEDS:
            exclusions["nonreporting_seed"].append(source)
            continue
        if not is_legacy_4g(record):
            exclusions["non_legacy_4g"].append(source)
            continue
        records.append(
            validate_complete_record(record, path, artifact_hash_cache)
        )
    if not records:
        raise RuntimeError(f"No eligible legacy-4G runs found below {root}")
    return records, {
        reason: {"count": len(paths), "source_metrics": paths}
        for reason, paths in sorted(exclusions.items())
    }


def make_group_key(
    record: dict[str, Any],
) -> tuple[float, float, float, str, float | None]:
    return (
        record["ratio"],
        record["background_scale_multiplier"],
        record["gross_scale_multiplier"],
        str(record["method"]).lower(),
        record["_direct_pinn_ebm_joint_pde_weight"],
    )


def validate_group_invariants(
    key: tuple[float, float, float, str, float | None],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    seeds = [int(run["seed"]) for run in runs]
    if len(seeds) != len(set(seeds)):
        paths = [run["_metrics_path"] for run in runs]
        raise ValueError(
            f"Duplicate reporting seed in {group_id(key)}: "
            f"seeds={seeds}, paths={paths}"
        )
    reference = {field: runs[0].get(field) for field in BASE_INVARIANT_FIELDS}
    for run in runs[1:]:
        observed = {field: run.get(field) for field in BASE_INVARIANT_FIELDS}
        if observed != reference:
            raise ValueError(
                f"Non-comparable base invariants in {group_id(key)}: "
                f"reference={reference}, observed={observed}, "
                f"path={run['_metrics_path']}"
            )
    return reference


def aggregate_groups(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[make_group_key(record)].append(record)

    output: list[dict[str, Any]] = []
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key in sorted(grouped, key=lambda item: tuple(map(str, item))):
        runs = grouped[key]
        invariants = validate_group_invariants(key, runs)
        seeds = sorted(int(run["seed"]) for run in runs)
        record: dict[str, Any] = {
            "group_id": group_id(key),
            "ratio": key[0],
            "background_scale_multiplier": key[1],
            "gross_scale_multiplier": key[2],
            "method": key[3],
            "direct_pinn_ebm_joint_pde_weight": key[4],
            "n_runs": len(runs),
            "seeds": seeds,
            "complete_for_reporting_seeds": seeds == list(REPORTING_SEEDS),
            "invariants": invariants,
            "source_metrics": [
                run["_metrics_path"]
                for run in sorted(runs, key=lambda value: int(value["seed"]))
            ],
            "input_artifacts_by_seed": {
                str(run["seed"]): {
                    "path": run["_input_artifact_path"],
                    "sha256": run["input_artifact_sha256"],
                    "corruption_seed": run["corruption_seed"],
                    "corruption_metadata_sha256": run[
                        "_corruption_metadata_fingerprint"
                    ],
                }
                for run in sorted(runs, key=lambda value: int(value["seed"]))
            },
            "metrics": {},
        }
        for metric in METRICS:
            values = {
                int(run["seed"]): finite_float(
                    run[metric], label=f"{run['_metrics_path']}: {metric}"
                )
                for run in runs
                if run.get(metric) is not None
            }
            if values:
                record["metrics"][metric] = summarize_per_seed(values)
        output.append(record)
        lookup[key] = record
    return output, lookup


def validate_paired_blocks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: dict[tuple[float, float, float, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for record in records:
        key = (
            record["ratio"],
            record["background_scale_multiplier"],
            record["gross_scale_multiplier"],
            int(record["seed"]),
        )
        blocks[key].append(record)

    output = []
    for key in sorted(blocks):
        runs = blocks[key]
        artifact_hashes = {run["input_artifact_sha256"] for run in runs}
        artifact_paths = {run["_input_artifact_path"] for run in runs}
        corruption_fingerprints = {
            run["_corruption_metadata_fingerprint"] for run in runs
        }
        corruption_seeds = {run["corruption_seed"] for run in runs}
        if len(artifact_hashes) != 1:
            raise ValueError(
                "Unpaired input artifacts in block "
                f"ratio={key[0]:g}, b={key[1]:g}, o={key[2]:g}, "
                f"seed={key[3]}: hashes={sorted(artifact_hashes)}, "
                f"paths={sorted(artifact_paths)}"
            )
        if len(corruption_fingerprints) != 1 or len(corruption_seeds) != 1:
            raise ValueError(
                "Corruption metadata differs within paired block "
                f"ratio={key[0]:g}, b={key[1]:g}, o={key[2]:g}, "
                f"seed={key[3]}"
            )
        method_groups = [group_id(make_group_key(run)) for run in runs]
        if len(method_groups) != len(set(method_groups)):
            raise ValueError(
                f"Duplicate method/weight variant in paired block {key}: "
                f"{method_groups}"
            )
        output.append(
            {
                "ratio": key[0],
                "background_scale_multiplier": key[1],
                "gross_scale_multiplier": key[2],
                "optimizer_seed": key[3],
                "corruption_seed": next(iter(corruption_seeds)),
                "input_artifact_paths": sorted(artifact_paths),
                "input_artifact_sha256": next(iter(artifact_hashes)),
                "corruption_metadata_sha256": next(
                    iter(corruption_fingerprints)
                ),
                "method_groups": sorted(method_groups),
                "paired_input_verified": True,
            }
        )
    return output


def paired_differences(
    group_lookup: dict[tuple[Any, ...], dict[str, Any]]
) -> list[dict[str, Any]]:
    by_scale: dict[tuple[float, float, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for key, group in group_lookup.items():
        by_scale[key[:3]].append(group)

    output = []
    for scale_key in sorted(by_scale):
        groups = sorted(
            by_scale[scale_key], key=lambda value: value["group_id"]
        )
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                metric_differences = {}
                for metric in METRICS:
                    left_metric = left["metrics"].get(metric)
                    right_metric = right["metrics"].get(metric)
                    if left_metric is None or right_metric is None:
                        continue
                    common_seeds = sorted(
                        set(left_metric["per_seed"])
                        & set(right_metric["per_seed"])
                    )
                    if not common_seeds:
                        continue
                    values = {
                        int(seed): (
                            float(left_metric["per_seed"][seed])
                            - float(right_metric["per_seed"][seed])
                        )
                        for seed in common_seeds
                    }
                    metric_differences[metric] = summarize_per_seed(values)

                comparison: dict[str, Any] = {
                    "ratio": scale_key[0],
                    "background_scale_multiplier": scale_key[1],
                    "gross_scale_multiplier": scale_key[2],
                    "left_group_id": left["group_id"],
                    "right_group_id": right["group_id"],
                    "difference_convention": "left_minus_right",
                    "metrics": metric_differences,
                }
                methods = {left["method"], right["method"]}
                napinn_methods = methods & NAPINN_METHODS
                if len(napinn_methods) == 1 and all(
                    metric in metric_differences
                    for metric in LOWER_IS_BETTER_FIELD_METRICS
                ):
                    napinn_method = next(iter(napinn_methods))
                    napinn_is_left = left["method"] == napinn_method
                    napinn_differences = {
                        metric: (
                            metric_differences[metric]["mean"]
                            if napinn_is_left
                            else -metric_differences[metric]["mean"]
                        )
                        for metric in LOWER_IS_BETTER_FIELD_METRICS
                    }
                    if all(value < 0.0 for value in napinn_differences.values()):
                        outcome = "positive_for_napinn"
                    elif all(
                        value > 0.0 for value in napinn_differences.values()
                    ):
                        outcome = "adverse_for_napinn"
                    else:
                        outcome = "mixed_for_napinn"
                    comparison["napinn_field_outcome"] = {
                        "napinn_group_id": (
                            left["group_id"] if napinn_is_left else right["group_id"]
                        ),
                        "comparator_group_id": (
                            right["group_id"] if napinn_is_left else left["group_id"]
                        ),
                        "difference_convention": "napinn_minus_comparator",
                        "mean_differences": napinn_differences,
                        "classification": outcome,
                    }
                output.append(comparison)
    return output


def expected_group_keys(args: argparse.Namespace) -> set[tuple[Any, ...]]:
    expected = set()
    for ratio in args.expected_ratios:
        for background in args.expected_background_scales:
            for gross in args.expected_gross_scales:
                for method in args.expected_methods:
                    if method == "pinn_ebm":
                        for weight in args.expected_direct_pinn_ebm_weights:
                            expected.add(
                                (
                                    canonical_float(ratio, label="expected ratio"),
                                    canonical_float(
                                        background,
                                        label="expected background scale",
                                    ),
                                    canonical_float(
                                        gross, label="expected gross scale"
                                    ),
                                    method,
                                    canonical_float(
                                        weight,
                                        label="expected direct PINN-EBM weight",
                                    ),
                                )
                            )
                    else:
                        expected.add(
                            (
                                canonical_float(ratio, label="expected ratio"),
                                canonical_float(
                                    background,
                                    label="expected background scale",
                                ),
                                canonical_float(
                                    gross, label="expected gross scale"
                                ),
                                method,
                                None,
                            )
                        )
    return expected


def require_complete(
    args: argparse.Namespace,
    group_lookup: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    expected = expected_group_keys(args)
    missing = []
    incomplete = []
    for key in sorted(expected, key=lambda item: tuple(map(str, item))):
        group = group_lookup.get(key)
        if group is None:
            missing.append(group_id(key))
        elif group["seeds"] != list(REPORTING_SEEDS):
            incomplete.append(
                {
                    "group_id": group_id(key),
                    "observed_seeds": group["seeds"],
                    "required_seeds": list(REPORTING_SEEDS),
                }
            )
    if missing or incomplete:
        raise RuntimeError(
            "Legacy-4G completeness check failed: "
            f"missing_groups={missing}, incomplete_groups={incomplete}"
        )
    return {
        "enforced": True,
        "required_reporting_seeds": list(REPORTING_SEEDS),
        "expected_group_count": len(expected),
        "expected_groups": sorted(group_id(key) for key in expected),
        "all_required_groups_complete": True,
    }


def write_csv(
    path: Path,
    groups: list[dict[str, Any]],
    differences: list[dict[str, Any]],
) -> None:
    fieldnames = (
        "record_type",
        "ratio",
        "background_scale_multiplier",
        "gross_scale_multiplier",
        "group_id",
        "left_group_id",
        "right_group_id",
        "method",
        "direct_pinn_ebm_joint_pde_weight",
        "metric",
        "n",
        "mean",
        "sample_std",
        "per_seed_json",
        "napinn_field_outcome",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            for metric, summary in group["metrics"].items():
                writer.writerow(
                    {
                        "record_type": "group_metric",
                        "ratio": group["ratio"],
                        "background_scale_multiplier": group[
                            "background_scale_multiplier"
                        ],
                        "gross_scale_multiplier": group[
                            "gross_scale_multiplier"
                        ],
                        "group_id": group["group_id"],
                        "method": group["method"],
                        "direct_pinn_ebm_joint_pde_weight": group[
                            "direct_pinn_ebm_joint_pde_weight"
                        ],
                        "metric": metric,
                        "n": summary["n"],
                        "mean": summary["mean"],
                        "sample_std": summary["sample_std"],
                        "per_seed_json": json.dumps(
                            summary["per_seed"], sort_keys=True
                        ),
                    }
                )
        for comparison in differences:
            outcome = comparison.get("napinn_field_outcome", {}).get(
                "classification"
            )
            for metric, summary in comparison["metrics"].items():
                writer.writerow(
                    {
                        "record_type": "paired_difference",
                        "ratio": comparison["ratio"],
                        "background_scale_multiplier": comparison[
                            "background_scale_multiplier"
                        ],
                        "gross_scale_multiplier": comparison[
                            "gross_scale_multiplier"
                        ],
                        "left_group_id": comparison["left_group_id"],
                        "right_group_id": comparison["right_group_id"],
                        "metric": metric,
                        "n": summary["n"],
                        "mean": summary["mean"],
                        "sample_std": summary["sample_std"],
                        "per_seed_json": json.dumps(
                            summary["per_seed"], sort_keys=True
                        ),
                        "napinn_field_outcome": outcome,
                    }
                )


def main(args: argparse.Namespace) -> None:
    records, exclusions = load_eligible_records(args.input_root)
    groups, group_lookup = aggregate_groups(records)
    paired_blocks = validate_paired_blocks(records)
    differences = paired_differences(group_lookup)
    if args.require_complete:
        completeness = require_complete(args, group_lookup)
    else:
        completeness = {
            "enforced": False,
            "required_reporting_seeds": list(REPORTING_SEEDS),
            "note": (
                "Use --require-complete to fail on missing exact-center or "
                "explicitly requested grid groups."
            ),
        }

    payload = {
        "schema_version": 1,
        "input_root": str(args.input_root.resolve()),
        "policy": {
            "legacy_four_gaussian_only": True,
            "natural_piv_excluded": True,
            "reporting_seeds": list(REPORTING_SEEDS),
            "seed_39_excluded": True,
            "smoke_tests_excluded": True,
            "partial_and_failed_runs_excluded": True,
            "label_terminology": {
                "positive": "gross_outlier",
                "negative": "background_only",
                "prohibited_for_this_aggregate": ["failed", "clean"],
            },
            "paired_difference_convention": "left_group_minus_right_group",
            "outcome_retention": (
                "All positive, mixed, and adverse eligible cells are retained."
            ),
        },
        "completeness": completeness,
        "excluded_runs": exclusions,
        "n_eligible_runs": len(records),
        "n_groups": len(groups),
        "paired_input_blocks": paired_blocks,
        "groups": groups,
        "paired_differences": differences,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, groups, differences)
    print(
        f"Aggregated {len(records)} complete reporting runs into "
        f"{len(groups)} groups."
    )
    print(f"Verified {len(paired_blocks)} paired input-artifact blocks.")
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde_legacy_4g/"
            "cylinder_real_piv_legacy4g"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde_legacy_4g/"
            "aggregation.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde_legacy_4g/"
            "aggregation.csv"
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail unless every explicitly expected method/scale group has "
            "exactly reporting seeds 40, 41, and 42."
        ),
    )
    parser.add_argument(
        "--expected-ratios",
        type=float,
        nargs="+",
        default=[0.10, 0.15],
    )
    parser.add_argument(
        "--expected-background-scales",
        type=float,
        nargs="+",
        default=[1.0],
        metavar="B",
    )
    parser.add_argument(
        "--expected-gross-scales",
        type=float,
        nargs="+",
        default=[1.0],
        metavar="O",
    )
    parser.add_argument(
        "--expected-methods",
        nargs="+",
        default=list(DEFAULT_EXPECTED_METHODS),
    )
    parser.add_argument(
        "--expected-direct-pinn-ebm-weights",
        type=float,
        nargs="+",
        default=[1.0, 50.0],
        metavar="WEIGHT",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
