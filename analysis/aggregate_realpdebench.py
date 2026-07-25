"""Aggregate completed RealPDEBench Cylinder PIV rebuttal runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "rMAE",
    "rMSE",
    "pde_momentum_rms",
    "continuity_rms",
    "retained_fraction",
    "mean_gate_weight",
    "failure_detection_auroc",
    "failed_rejection_rate",
    "clean_rejection_rate",
    "failed_mean_gate_weight",
    "clean_mean_gate_weight",
    "mad_retained_fraction",
    "mad_known_failed_rejection_rate",
    "mad_known_clean_rejection_rate",
    "training_wall_sec",
    "end_to_end_wall_sec",
    "mad_pipeline_training_wall_sec",
    "mad_total_pipeline_pinn_update_steps",
    "gpu_peak_memory_mb",
)


def load_runs(root: Path, include_smoke: bool):
    records = []
    for path in sorted(root.rglob("metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("smoke_test", False) and not include_smoke:
            continue
        record["_path"] = str(path)
        records.append(record)
    if not records:
        qualifier = " (smoke tests excluded)" if not include_smoke else ""
        raise RuntimeError(f"No metrics.json files found under {root}{qualifier}")
    return records


def validate_comparability(records):
    invariant_keys = (
        "sim_id",
        "source_sha256",
        "sensor_seed",
        "n_spatial_sensors",
        "n_frames",
        "n_heldout_measurements",
    )
    reference = {key: records[0].get(key) for key in invariant_keys}
    for record in records[1:]:
        observed = {key: record.get(key) for key in invariant_keys}
        if observed != reference:
            raise ValueError(
                "Runs do not share the same dataset/split:\n"
                f"reference={reference}\nobserved={observed}\n"
                f"path={record['_path']}"
            )
    return reference


def aggregate(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["tag"]].append(record)
    output = {}
    for tag, method_records in sorted(grouped.items()):
        summary = {
            "method": method_records[0]["method"],
            "n_runs": len(method_records),
            "seeds": sorted(record["seed"] for record in method_records),
            "paths": [record["_path"] for record in method_records],
        }
        for metric in METRICS:
            values = [
                float(record[metric])
                for record in method_records
                if record.get(metric) is not None
            ]
            if not values:
                continue
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": values,
            }
        output[tag] = summary
    return output


def write_csv(path: Path, summary):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("tag", "method", "n_runs", "seeds", "metric", "mean", "std")
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
                        metric,
                        values[metric]["mean"],
                        values[metric]["std"],
                    )
                )


def main(args):
    records = load_runs(args.root, args.include_smoke)
    invariants = validate_comparability(records)
    summary = aggregate(records)
    payload = {
        "smoke_tests_included": args.include_smoke,
        "comparability_invariants": invariants,
        "methods": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = args.output.with_suffix(".csv")
    write_csv(csv_path, summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"JSON: {args.output}")
    print(f"CSV: {csv_path}")


def build_parser():
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
            "analysis/results/runs/rebuttal_realpde/aggregation.json"
        ),
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Explicitly include non-evidentiary smoke-test runs.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
