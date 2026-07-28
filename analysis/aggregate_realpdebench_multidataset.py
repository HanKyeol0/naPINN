#!/usr/bin/env python3
"""Strict complete-only aggregation for the frozen 144-run PIV matrix.

The aggregator addresses only the exact paths in the frozen config manifest.
It refuses to write an aggregate unless all 144 non-smoke runs pass the same
input-hash, config, checkpoint, update-budget, and completion checks used by
the queue.  It additionally requires eight method records per paired
dataset/seed/ratio input and three seeds per reported group.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.run_realpdebench_multidataset_queue import (
    MANIFEST,
    Job,
    build_jobs,
    load_protocol,
    run_directory,
    validate_completed,
)


DEFAULT_ROOT = Path(
    "outputs/rebuttal/realpdebench_multidataset/runs"
)
DEFAULT_OUTPUT = Path(
    "outputs/rebuttal/realpdebench_multidataset/aggregation.json"
)
METRICS = (
    "rMAE",
    "rMSE",
    "pde_momentum_rms",
    "continuity_rms",
    "gross_outlier_detection_auroc",
    "gross_outlier_rejection_rate",
    "background_only_rejection_rate",
    "retained_fraction",
    "estimator_gross_outlier_detection_auroc",
    "training_wall_sec",
    "evaluation_wall_sec",
    "end_to_end_wall_sec",
    "gpu_peak_memory_mb",
)


def read_metrics(root: Path, job: Job) -> dict[str, Any]:
    return json.loads(
        (run_directory(root, job) / "metrics.json").read_text(
            encoding="utf-8"
        )
    )


def finite_numeric_metrics(
    payload: dict[str, Any], path: Path
) -> None:
    for key, value in payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isfinite(float(value)):
                raise ValueError(f"Non-finite metric {key} in {path}")


def aggregate(
    manifest_path: Path,
    root: Path,
    output: Path,
) -> dict[str, Any]:
    protocol = load_protocol(manifest_path)
    jobs = build_jobs(protocol)
    validation_failures = []
    records = []
    for job in jobs:
        valid, reason = validate_completed(protocol, job, root)
        if not valid:
            validation_failures.append({"key": job.key, "reason": reason})
            continue
        metrics_path = run_directory(root, job) / "metrics.json"
        metrics = read_metrics(root, job)
        finite_numeric_metrics(metrics, metrics_path)
        records.append(
            {
                "dataset": job.dataset,
                "seed": job.seed,
                "ratio_percent": job.ratio_percent,
                "method": job.method,
                "input_artifact_sha256": job.input_sha256,
                "metrics_path": str(metrics_path.resolve()),
                **{key: metrics[key] for key in METRICS if key in metrics},
            }
        )
    if validation_failures:
        sample = validation_failures[:10]
        raise RuntimeError(
            f"Refusing incomplete aggregation: {len(validation_failures)}/144 "
            f"required runs invalid or absent. First failures: {sample}"
        )
    if len(records) != 144:
        raise AssertionError("Strict validation did not yield exactly 144 records")

    paired: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        paired[
            (
                record["dataset"],
                record["seed"],
                record["ratio_percent"],
            )
        ].append(record)
    if len(paired) != 18:
        raise AssertionError("Expected exactly 18 paired input blocks")
    pairing_audit = []
    for key, block in sorted(paired.items()):
        methods = {record["method"] for record in block}
        hashes = {record["input_artifact_sha256"] for record in block}
        if methods != set(protocol["methods"]) or len(block) != 8:
            raise AssertionError(f"Incomplete method pairing for {key}")
        if len(hashes) != 1:
            raise AssertionError(f"Methods use different inputs for {key}")
        pairing_audit.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "ratio_percent": key[2],
                "input_artifact_sha256": next(iter(hashes)),
                "methods": sorted(methods),
            }
        )

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["dataset"],
                record["ratio_percent"],
                record["method"],
            )
        ].append(record)
    if len(grouped) != 48:
        raise AssertionError("Expected exactly 48 three-seed groups")
    groups = []
    for (dataset, ratio_percent, method), block in sorted(grouped.items()):
        if sorted(record["seed"] for record in block) != [40, 41, 42]:
            raise AssertionError(
                f"Group {(dataset, ratio_percent, method)} lacks seeds 40--42"
            )
        summary: dict[str, Any] = {
            "dataset": dataset,
            "ratio_percent": ratio_percent,
            "method": method,
            "seeds": [40, 41, 42],
            "n": 3,
        }
        for metric in METRICS:
            values = [record[metric] for record in block if metric in record]
            if not values:
                continue
            if len(values) != 3:
                raise AssertionError(
                    f"Metric {metric} is missing within group "
                    f"{(dataset, ratio_percent, method)}"
                )
            array = np.asarray(values, dtype=np.float64)
            summary[metric] = {
                "mean": float(array.mean()),
                "sample_std": float(array.std(ddof=1)),
                "values_by_seed": {
                    str(record["seed"]): float(record[metric])
                    for record in sorted(block, key=lambda item: item["seed"])
                },
            }
        groups.append(summary)

    payload = {
        "schema_version": 1,
        "status": "strict_complete",
        "evidence_status": "144_full_runs_complete_aggregated",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": (
            __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
        ),
        "run_root": str(root.resolve()),
        "required_run_count": 144,
        "included_run_count": 144,
        "paired_input_block_count": 18,
        "three_seed_group_count": 48,
        "smoke_runs_included": 0,
        "selection_performed": False,
        "claim_boundary": (
            "Held-out PIV is independent real measurement, not noise-free "
            "ground truth. The nominal 2-D PDE has explicit model discrepancy."
        ),
        "pairing_audit": pairing_audit,
        "groups": groups,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError(
            f"Refusing to overwrite aggregate at {output} or its CSV"
        )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output.with_suffix(".csv")
    rows = []
    for group in groups:
        row = {
            "dataset": group["dataset"],
            "ratio_percent": group["ratio_percent"],
            "method": group["method"],
            "n": group["n"],
        }
        for metric in METRICS:
            if metric in group:
                row[f"{metric}_mean"] = group[metric]["mean"]
                row[f"{metric}_sample_std"] = group[metric]["sample_std"]
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = aggregate(args.manifest, args.run_root, args.output)
    print(
        yaml.safe_dump(
            {
                "status": result["status"],
                "included_run_count": result["included_run_count"],
                "three_seed_group_count": result["three_seed_group_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
