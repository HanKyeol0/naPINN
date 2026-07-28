"""Aggregate completed synthetic rebuttal runs without filling missing cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


GROUP_FIELDS = (
    "experiment_name",
    "noise_kind",
    "outlier_ratio",
    "method",
    "pde_loss_weight",
    "data_loss_weight",
    "ema_momentum",
    "rejection_cost",
)
METRICS = (
    "field_rMAE",
    "field_rMSE",
    "pde_parameter_learned",
    "pde_parameter_absolute_error",
    "training_seconds",
    "pinn_update_steps",
    "estimator_only_steps",
    "gpu_peak_memory_allocated_bytes",
)
STAGED_METHODS = {"napinn", "napinn_lad", "napinn_q29", "pinn_ebm"}


def validate_finite_numbers(value: Any, *, location: str) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {location}: {value}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite_numbers(
                child, location=f"{location}.{key}"
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            validate_finite_numbers(
                child, location=f"{location}[{index}]"
            )


def validate_strict_run(
    metrics_path: Path,
    payload: dict[str, Any],
    resolved: dict[str, Any],
    *,
    expected_pinn_steps: int,
) -> None:
    final_path = metrics_path.parent / "final.pt"
    if not final_path.is_file():
        raise FileNotFoundError(
            f"Completed run lacks final checkpoint: {final_path}"
        )
    method = str(payload["method"])
    if method == "mad_pinn":
        run_config = resolved.get("stage1_resolved_config", {}).get("run")
    else:
        run_config = resolved.get("run")
    if not isinstance(run_config, dict):
        raise ValueError(f"Resolved config lacks run mapping: {metrics_path}")
    matching_fields = (
        ("seed", "seed"),
        ("noise_kind", "noise_kind"),
        ("outlier_ratio", "outlier_ratio"),
        ("pde_loss_weight", "pde_weight"),
        ("data_loss_weight", "data_weight"),
    )
    if method != "mad_pinn":
        matching_fields = (("method", "method"), *matching_fields)
    mismatches = []
    for metric_field, config_field in matching_fields:
        if payload.get(metric_field) != run_config.get(config_field):
            mismatches.append(f"{metric_field}!={config_field}")
    if mismatches:
        raise ValueError(
            f"Metrics/config mismatch in {metrics_path}: {mismatches}"
        )
    expected_estimator_steps = 5000 if method in STAGED_METHODS else 0
    if int(payload.get("pinn_update_steps", -1)) != expected_pinn_steps:
        raise ValueError(
            f"Invalid PINN update budget in {metrics_path}: "
            f"{payload.get('pinn_update_steps')} "
            f"(expected {expected_pinn_steps})"
        )
    if int(payload.get("estimator_only_steps", -1)) != expected_estimator_steps:
        raise ValueError(
            f"Invalid estimator-only budget in {metrics_path}: "
            f"{payload.get('estimator_only_steps')} "
            f"(expected {expected_estimator_steps})"
        )
    if method == "mad_pinn":
        mad_checks = {
            "stage1_updates": int(
                payload.get("mad_stage1_pinn_updates", -1)
            )
            == 30000,
            "stage2_updates": int(
                payload.get("mad_stage2_pinn_updates", -1)
            )
            == 30000,
            "total_updates": expected_pinn_steps == 60000,
            "screening_artifact": (
                metrics_path.parent / "mad_screening.npz"
            ).is_file(),
            "stage1_checkpoint": Path(
                str(payload.get("stage1_checkpoint_path", ""))
            ).is_file(),
            "stage1_metrics": Path(
                str(payload.get("stage1_metrics_path", ""))
            ).is_file(),
        }
        failed = sorted(key for key, value in mad_checks.items() if not value)
        if failed:
            raise ValueError(
                f"Invalid MAD-PINN provenance in {metrics_path}: {failed}"
            )
    validate_finite_numbers(payload, location=str(metrics_path))


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main(args):
    expected_pinn_steps = int(
        getattr(args, "expected_pinn_steps", 30000)
    )
    if len(set(args.required_seeds)) != len(args.required_seeds):
        raise ValueError("--required-seeds must not contain duplicates")
    if args.strict:
        if args.expected_runs is None or args.expected_groups is None:
            raise ValueError(
                "--strict requires --expected-runs and --expected-groups"
            )
        expected_from_groups = args.expected_groups * len(args.required_seeds)
        if args.expected_runs != expected_from_groups:
            raise ValueError(
                "--expected-runs must equal --expected-groups times the "
                f"required-seed count ({expected_from_groups})"
            )
        if args.output_json.exists() or args.output_csv.exists():
            raise FileExistsError(
                "Strict aggregation refuses to overwrite existing outputs: "
                f"{args.output_json}, {args.output_csv}"
            )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    source_paths: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    required_seed_set = set(args.required_seeds)
    excluded_seed_paths: list[str] = []
    metrics_paths = sorted(args.input_root.rglob("metrics.json"))
    incomplete_metric_paths: list[str] = []
    for path in metrics_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            incomplete_metric_paths.append(str(path.resolve()))
            continue
        seed = int(payload["seed"])
        if seed not in required_seed_set:
            excluded_seed_paths.append(str(path.resolve()))
            continue
        config_path = path.parent / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Completed run lacks immutable config: {config_path}"
            )
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(resolved, dict):
            raise ValueError(f"Expected YAML mapping in {config_path}")
        if args.strict:
            validate_strict_run(
                path,
                payload,
                resolved,
                expected_pinn_steps=expected_pinn_steps,
            )
        if payload.get("method") == "mad_pinn":
            run_config = resolved.get("stage1_resolved_config", {}).get(
                "run", {}
            )
        else:
            run_config = resolved.get("run", {})
        payload["ema_momentum"] = run_config.get("ema_momentum")
        payload["rejection_cost"] = run_config.get("rejection_cost")
        key = tuple(payload.get(field) for field in GROUP_FIELDS)
        groups[key].append(payload)
        source_paths[key].append(str(path.resolve()))

    if args.strict:
        included_run_count = sum(len(runs) for runs in groups.values())
        if incomplete_metric_paths:
            raise RuntimeError(
                "Strict aggregation found non-complete metrics: "
                f"{incomplete_metric_paths[:10]}"
            )
        if excluded_seed_paths:
            raise RuntimeError(
                "Strict aggregation found completed non-required seeds: "
                f"{excluded_seed_paths[:10]}"
            )
        if included_run_count != args.expected_runs:
            raise RuntimeError(
                "Refusing incomplete strict aggregation: found "
                f"{included_run_count}/{args.expected_runs} required runs"
            )
        if len(groups) != args.expected_groups:
            raise RuntimeError(
                "Refusing strict aggregation with unexpected group count: "
                f"{len(groups)}/{args.expected_groups}"
            )
        required_sorted = sorted(args.required_seeds)
        for key, runs in groups.items():
            seeds = sorted(int(run["seed"]) for run in runs)
            if seeds != required_sorted:
                raise RuntimeError(
                    f"Incomplete or duplicate seed group {key}: {seeds}, "
                    f"expected {required_sorted}"
                )

    output = {
        "status": (
            "strict_complete" if args.strict else "partial_allowed"
        ),
        "evidence_status": (
            "complete_runs_strictly_aggregated"
            if args.strict
            else "diagnostic_aggregation_may_be_partial"
        ),
        "strict": args.strict,
        "input_root": str(args.input_root.resolve()),
        "required_seeds": args.required_seeds,
        "expected_runs": args.expected_runs,
        "expected_groups": args.expected_groups,
        "expected_pinn_steps_per_included_run": expected_pinn_steps,
        "discovered_metrics_files": len(metrics_paths),
        "included_run_count": sum(len(runs) for runs in groups.values()),
        "excluded_nonrequired_seed_runs": {
            "count": len(excluded_seed_paths),
            "source_metrics": excluded_seed_paths,
        },
        "groups": [],
    }
    flat_rows = []
    for key in sorted(groups, key=lambda item: tuple(map(str, item))):
        runs = groups[key]
        seeds = sorted(int(run["seed"]) for run in runs)
        record: dict[str, Any] = dict(zip(GROUP_FIELDS, key))
        record.update(
            {
                "seeds": seeds,
                "complete_for_required_seeds": (
                    seeds == sorted(args.required_seeds)
                ),
                "source_metrics": source_paths[key],
                "metrics": {},
            }
        )
        for metric in METRICS:
            values = [
                float(run[metric])
                for run in runs
                if run.get(metric) is not None
            ]
            if values:
                record["metrics"][metric] = summarize(values)
        phase_names = sorted(
            {
                phase_name
                for run in runs
                for phase_name in run.get("phase_times", {})
                if isinstance(run["phase_times"][phase_name], (int, float))
            }
        )
        for phase_name in phase_names:
            values = [
                float(run["phase_times"][phase_name])
                for run in runs
                if phase_name in run.get("phase_times", {})
            ]
            record["metrics"][f"phase_times.{phase_name}"] = summarize(
                values
            )
        gate_names = sorted(
            {
                gate_name
                for run in runs
                for gate_name in run.get("gate", {})
                if isinstance(run["gate"][gate_name], (int, float))
            }
        )
        for gate_name in gate_names:
            values = [
                float(run["gate"][gate_name])
                for run in runs
                if gate_name in run.get("gate", {})
            ]
            record["metrics"][f"gate.{gate_name}"] = summarize(values)
        estimator_names = sorted(
            {
                metric_name
                for run in runs
                for metric_name in run.get("estimator_scores", {})
                if isinstance(
                    run["estimator_scores"][metric_name], (int, float)
                )
            }
        )
        for metric_name in estimator_names:
            values = [
                float(run["estimator_scores"][metric_name])
                for run in runs
                if metric_name in run.get("estimator_scores", {})
            ]
            record["metrics"][
                f"estimator_scores.{metric_name}"
            ] = summarize(values)
        screening_names = sorted(
            {
                metric_name
                for run in runs
                for metric_name in run.get("screening", {})
                if isinstance(run["screening"][metric_name], (int, float))
            }
        )
        for metric_name in screening_names:
            values = [
                float(run["screening"][metric_name])
                for run in runs
                if metric_name in run.get("screening", {})
            ]
            record["metrics"][f"screening.{metric_name}"] = summarize(
                values
            )
        output["groups"].append(record)
        row = {
            **{field: record[field] for field in GROUP_FIELDS},
            "seeds": ",".join(map(str, seeds)),
            "complete_for_required_seeds": record[
                "complete_for_required_seeds"
            ],
        }
        for metric_name, summary in record["metrics"].items():
            row[f"{metric_name}_mean"] = summary["mean"]
            row[f"{metric_name}_sample_std"] = summary["sample_std"]
        flat_rows.append(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in flat_rows for key in row})
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(
        f"Aggregated {sum(len(runs) for runs in groups.values())} runs "
        f"into {len(groups)} groups."
    )
    print(
        "Excluded "
        f"{len(excluded_seed_paths)} completed runs outside required seeds."
    )
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("analysis/results/runs/rebuttal_synthetic"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_synthetic_aggregation.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_synthetic_aggregation.csv"
        ),
    )
    parser.add_argument(
        "--required-seeds",
        type=int,
        nargs="+",
        default=[40, 41, 42],
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require exact complete run/group counts, one run per required "
            "seed, final checkpoints, matched configs, fixed update budgets, "
            "finite metrics, and non-overwriting outputs."
        ),
    )
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--expected-groups", type=int)
    parser.add_argument(
        "--expected-pinn-steps",
        type=int,
        default=30000,
        help=(
            "Exact per-record PINN-update count required in strict mode. "
            "Use 35000 for conservative compute-reference baselines and "
            "60000 for the two-stage MAD-PINN pipeline."
        ),
    )
    main(parser.parse_args())
