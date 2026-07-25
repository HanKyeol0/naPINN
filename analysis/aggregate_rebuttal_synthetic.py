"""Aggregate completed synthetic rebuttal runs without filling missing cells."""

from __future__ import annotations

import argparse
import csv
import json
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


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main(args):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    source_paths: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    required_seed_set = set(args.required_seeds)
    excluded_seed_paths: list[str] = []
    for path in sorted(args.input_root.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
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
        run_config = resolved.get("run", {})
        payload["ema_momentum"] = run_config.get("ema_momentum")
        payload["rejection_cost"] = run_config.get("rejection_cost")
        key = tuple(payload.get(field) for field in GROUP_FIELDS)
        groups[key].append(payload)
        source_paths[key].append(str(path.resolve()))

    output = {
        "input_root": str(args.input_root.resolve()),
        "required_seeds": args.required_seeds,
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
    main(parser.parse_args())
