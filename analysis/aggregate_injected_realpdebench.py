"""Aggregate held-out injected-corruption RealPDEBench PIV runs.

Natural-PIV and calibration-seed results are excluded by construction.  Only
complete, non-smoke seeds 40--42 with an explicit corruption label schema are
eligible for the rebuttal table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HELD_OUT_SEEDS = {40, 41, 42}
METRICS = (
    "rMAE",
    "rMSE",
    "pde_momentum_rms",
    "continuity_rms",
    "failure_detection_auroc",
    "estimator_failure_detection_auroc",
    "estimator_mean_negative_log_density",
    "estimator_failed_mean_negative_log_density",
    "estimator_clean_mean_negative_log_density",
    "failed_rejection_rate",
    "clean_rejection_rate",
    "mean_gate_weight",
    "retained_fraction",
    "mad_known_failed_rejection_rate",
    "mad_known_clean_rejection_rate",
    "mad_retained_fraction",
    "training_wall_sec",
    "end_to_end_wall_sec",
    "mad_pipeline_training_wall_sec",
    "mad_total_pipeline_pinn_update_steps",
    "warmup_sec",
    "estimator_init_sec",
    "joint_sec",
    "evaluation_wall_sec",
    "metadata_reynolds",
    "effective_reynolds",
    "effective_reynolds_relative_error",
    "gpu_peak_memory_mb",
)


def eligible_records(root: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(root.rglob("metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("smoke_test", True):
            continue
        if record.get("evidence_status") != "full_run_complete_unaggregated":
            continue
        if int(record.get("seed", -1)) not in HELD_OUT_SEEDS:
            continue
        if not record.get("corruption_kind"):
            continue
        record["_path"] = str(path)
        records.append(record)
    if not records:
        raise RuntimeError(f"No eligible injected-PIV runs found below {root}")
    return records


def aggregate(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["tag"])].append(record)
    output: dict[str, dict] = {}
    for tag, group in sorted(grouped.items()):
        seeds = [int(record["seed"]) for record in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate held-out seed in {tag}: {seeds}")
        invariant_keys = (
            "benchmark",
            "sim_id",
            "source_sha256",
            "sensor_seed",
            "n_spatial_sensors",
            "n_frames",
            "n_heldout_measurements",
            "corruption_kind",
        )
        reference = {key: group[0].get(key) for key in invariant_keys}
        for record in group[1:]:
            observed = {key: record.get(key) for key in invariant_keys}
            if observed != reference:
                raise ValueError(
                    f"Non-comparable records in {tag}: "
                    f"{reference=} {observed=} path={record['_path']}"
                )
        summary = {
            "method": group[0]["method"],
            "n_runs": len(group),
            "seeds": sorted(seeds),
            "complete_for_held_out_seeds": set(seeds) == HELD_OUT_SEEDS,
            "invariants": reference,
            "paths": [record["_path"] for record in group],
        }
        for metric in METRICS:
            values = [
                float(record[metric])
                for record in group
                if record.get(metric) is not None
            ]
            if values:
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "sample_std": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "values": values,
                }
        output[tag] = summary
    return output


def write_csv(path: Path, summary: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "tag",
                "method",
                "n_runs",
                "seeds",
                "complete_for_held_out_seeds",
                "corruption_kind",
                "metric",
                "mean",
                "sample_std",
            )
        )
        for tag, values in summary.items():
            for metric in METRICS:
                if metric not in values:
                    continue
                writer.writerow(
                    (
                        tag,
                        values["method"],
                        values["n_runs"],
                        ",".join(map(str, values["seeds"])),
                        values["complete_for_held_out_seeds"],
                        values["invariants"]["corruption_kind"],
                        metric,
                        values[metric]["mean"],
                        values[metric]["sample_std"],
                    )
                )


def main(args) -> None:
    records = eligible_records(args.root)
    summary = aggregate(records)
    payload = {
        "policy": {
            "natural_piv_excluded": True,
            "calibration_seed_39_excluded": True,
            "held_out_seeds": sorted(HELD_OUT_SEEDS),
            "smoke_tests_excluded": True,
            "incomplete_runs_excluded": True,
        },
        "groups": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output.with_suffix(".csv"), summary)
    print(f"Aggregated {len(records)} runs into {len(summary)} groups.")
    print(f"JSON: {args.output}")
    print(f"CSV: {args.output.with_suffix('.csv')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde/cylinder_real_piv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde/"
            "injected_piv_aggregation.json"
        ),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
