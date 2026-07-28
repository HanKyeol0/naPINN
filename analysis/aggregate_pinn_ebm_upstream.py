#!/usr/bin/env python3
"""Strictly validate and compare the two faithful PINN-EBM reproductions.

The caller must name the exact completed A and B run directories. This avoids
silently selecting interrupted predecessors or component smokes from the
shared campaign root. No output is written unless both official executions
and their five sequential upstream runs pass every check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.run_pinn_ebm_upstream import (
    expected_source_status,
    load_yaml,
    sha256_file,
    validate_config,
    verify_dataset,
)


EXPECTED_VARIANTS = {
    "A": "A_upstream_active_0b74f6f",
    "B": "B_paper_spec_8x20",
}
EXPECTED_EVIDENCE = "complete_full_upstream_execution"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping in {path}")
    return payload


def validate_finite_numbers(value: Any, *, location: str) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite number at {location}: {value}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite_numbers(child, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            validate_finite_numbers(child, location=f"{location}[{index}]")


def resolve_inside(run_dir: Path, relative: str) -> Path:
    candidate = (run_dir / relative).resolve()
    if not candidate.is_relative_to(run_dir.resolve()):
        raise ValueError(f"Official result escapes run directory: {relative}")
    return candidate


def validate_aggregate(
    records: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    metric_names = set(records[0]) - {"run_index"}
    if set(aggregate) != metric_names:
        raise ValueError(
            "Aggregate keys do not exactly match per-run metrics: "
            f"{sorted(set(aggregate) ^ metric_names)}"
        )
    for name in sorted(metric_names):
        values = np.asarray([record[name] for record in records], dtype=np.float64)
        expected_mean = float(values.mean())
        expected_std = float(values.std(ddof=1))
        summary = aggregate[name]
        if not isinstance(summary, dict):
            raise ValueError(f"Aggregate metric {name} is not a mapping")
        if not np.isclose(
            float(summary.get("mean", math.nan)),
            expected_mean,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(f"Incorrect aggregate mean for {name}")
        if not np.isclose(
            float(summary.get("std_sample", math.nan)),
            expected_std,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(f"Incorrect aggregate sample std for {name}")


def validate_run(run_dir: Path, expected_variant: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    required = {
        "config": run_dir / "config.yaml",
        "metadata": run_dir / "run_metadata.json",
        "metrics": run_dir / "metrics.json",
        "log": run_dir / "stdout_stderr.log",
        "source": run_dir / "source",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete upstream run {run_dir}; missing {missing}"
        )

    config = load_yaml(required["config"])
    validate_config(config, required["config"])
    if config["variant_id"] != expected_variant:
        raise ValueError(
            f"Expected {expected_variant}, got {config['variant_id']}"
        )
    metadata = read_json(required["metadata"])
    metrics = read_json(required["metrics"])
    for label, payload in (("metadata", metadata), ("metrics", metrics)):
        if payload.get("evidence_status") != EXPECTED_EVIDENCE:
            raise ValueError(
                f"{label} is not complete full-run evidence in {run_dir}"
            )
        if payload.get("variant_id") != expected_variant:
            raise ValueError(f"{label} variant mismatch in {run_dir}")
    if metadata.get("status") != "complete":
        raise ValueError(f"Run metadata is not complete in {run_dir}")

    execution = config["execution"]
    if (
        int(metrics.get("seed", -1)) != int(execution["seed"])
        or int(execution["seed"]) != 0
    ):
        raise ValueError(f"Unexpected seed in {run_dir}")
    if (
        int(metrics.get("nrun", -1)) != int(execution["nrun"])
        or int(execution["nrun"]) != 5
    ):
        raise ValueError(f"Unexpected Nrun in {run_dir}")
    if int(metrics.get("model_index", -1)) != int(
        execution["pinn_ebm_model_index"]
    ):
        raise ValueError(f"Unexpected PINN-EBM model index in {run_dir}")

    records = metrics.get("records")
    aggregate = metrics.get("aggregate")
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError(f"Expected five official run records in {run_dir}")
    if [int(record.get("run_index", -1)) for record in records] != list(range(5)):
        raise ValueError(f"Official run indices are incomplete in {run_dir}")
    if not isinstance(aggregate, dict):
        raise ValueError(f"Missing metric aggregate in {run_dir}")
    validate_finite_numbers(records, location=f"{run_dir}.records")
    validate_finite_numbers(aggregate, location=f"{run_dir}.aggregate")
    validate_aggregate(records, aggregate)

    result_info = metrics.get("official_result")
    if not isinstance(result_info, dict):
        raise ValueError(f"Missing official-result provenance in {run_dir}")
    if metadata.get("official_result") != result_info:
        raise ValueError(f"Metadata/metrics result provenance differs in {run_dir}")
    result_path = resolve_inside(run_dir, str(result_info["path"]))
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing official result pickle: {result_path}")
    if result_path.stat().st_size != int(result_info["size_bytes"]):
        raise ValueError(f"Official result size mismatch in {run_dir}")
    if sha256_file(result_path) != result_info["sha256"]:
        raise ValueError(f"Official result hash mismatch in {run_dir}")

    dataset_path = Path(metadata["dataset"]["path"])
    dataset = verify_dataset(dataset_path, config["dataset"])
    if metadata["dataset"] != dataset:
        raise ValueError(f"Dataset provenance mismatch in {run_dir}")
    source = expected_source_status(config, required["source"])

    return {
        "variant_id": expected_variant,
        "run_dir": str(run_dir),
        "config_path": str(required["config"]),
        "metrics_path": str(required["metrics"]),
        "log_path": str(required["log"]),
        "official_result": result_info,
        "dataset": dataset,
        "archived_source": source,
        "seed": 0,
        "nrun": 5,
        "model_index": int(metrics["model_index"]),
        "records": records,
        "aggregate": aggregate,
        "metric_name_warning": metrics.get("metric_name_warning"),
    }


def build_comparison(
    variants: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    a = variants["A"]["aggregate"]
    b = variants["B"]["aggregate"]
    if set(a) != set(b):
        raise ValueError("A/B aggregate metric sets differ")
    comparison = {}
    for name in sorted(a):
        a_mean = float(a[name]["mean"])
        b_mean = float(b[name]["mean"])
        comparison[name] = {
            "A_mean": a_mean,
            "B_mean": b_mean,
            "B_minus_A": b_mean - a_mean,
        }
    return comparison


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    variants = {
        "A": validate_run(args.run_dir_a, EXPECTED_VARIANTS["A"]),
        "B": validate_run(args.run_dir_b, EXPECTED_VARIANTS["B"]),
    }
    payload = {
        "schema_version": 1,
        "status": "strict_complete",
        "evidence_status": "two_faithful_full_upstream_executions_aggregated",
        "variant_count": 2,
        "official_sequential_runs_per_variant": 5,
        "selection_performed": False,
        "variants": variants,
        "comparison": build_comparison(variants),
        "claim_boundary": (
            "Variant A reproduces the active upstream 4x30 source. Variant B "
            "changes only the declared 8x20 paper architecture. They are "
            "reported separately and are not naPINN compute-matched runs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir-a", type=Path, required=True)
    parser.add_argument("--run-dir-b", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/rebuttal/pinn_ebm_upstream/aggregation_strict.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = aggregate(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "variant_count": payload["variant_count"],
                "official_sequential_runs_per_variant": payload[
                    "official_sequential_runs_per_variant"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
