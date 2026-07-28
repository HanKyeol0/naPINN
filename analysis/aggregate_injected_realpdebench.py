"""Strictly aggregate held-out injected-corruption RealPDEBench PIV runs.

Natural-PIV and calibration-seed results are excluded by construction.  Only
complete, non-smoke seeds 40--42 with an explicit corruption label schema are
eligible for the rebuttal table.  The output is written only after the frozen
102-run reviewer-evidence matrix is complete and internally consistent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HELD_OUT_SEEDS = {40, 41, 42}
VARIANTS = (
    "sensor_drift30",
    "sensor_drift20",
    "sensor_drift10",
    "ar1",
    "spatial_burst",
)
METHOD_TAGS = {
    "mse": "mse",
    "lad": "lad",
    "orpinn_q29": "orpinn_q29",
    "pinn_ebm": "pinn_ebm_pilar",
    "napinn": "napinn_rejection_01",
}
EXPECTED_TAG_METHOD = {
    **{
        f"{tag}_{variant}": method
        for variant in VARIANTS
        for method, tag in METHOD_TAGS.items()
    },
    **{
        f"{METHOD_TAGS[method]}_inverse_re_sensor_drift30": method
        for method in ("mse", "lad", "pinn_ebm", "napinn")
    },
    **{f"mad_pinn_{variant}": "mad_pinn" for variant in VARIANTS},
}
EXPECTED_GROUPS = 34
EXPECTED_RUNS = 102
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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_finite(value: Any, *, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {location}: {value}")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_finite(nested, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_finite(nested, location=f"{location}[{index}]")


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
        record["_path"] = str(path.resolve())
        record["_run_dir"] = str(path.parent.resolve())
        records.append(record)
    if not records:
        raise RuntimeError(f"No eligible injected-PIV runs found below {root}")
    return records


def validate_strict_records(root: Path, records: list[dict]) -> None:
    if len(EXPECTED_TAG_METHOD) != EXPECTED_GROUPS:
        raise AssertionError("Internal expected-tag manifest is inconsistent")
    if len(records) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Strict aggregation requires exactly {EXPECTED_RUNS} eligible "
            f"runs; found {len(records)} below {root}"
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("tag"))].append(record)
    observed_tags = set(grouped)
    expected_tags = set(EXPECTED_TAG_METHOD)
    if observed_tags != expected_tags:
        raise RuntimeError(
            "Strict tag coverage mismatch: "
            f"missing={sorted(expected_tags - observed_tags)} "
            f"unexpected={sorted(observed_tags - expected_tags)}"
        )

    hash_cache: dict[Path, str] = {}
    for tag, group in grouped.items():
        seeds = [int(record.get("seed", -1)) for record in group]
        if len(group) != 3 or set(seeds) != HELD_OUT_SEEDS:
            raise RuntimeError(
                f"Strict seed coverage mismatch for {tag}: {sorted(seeds)}"
            )
        expected_method = EXPECTED_TAG_METHOD[tag]
        for record in group:
            run_dir = Path(record["_run_dir"])
            seed = int(record["seed"])
            if run_dir.name != f"heldout_seed_{seed}":
                raise ValueError(
                    f"Unexpected run directory name for {tag}, seed {seed}: "
                    f"{run_dir}"
                )
            required = {
                "metrics": run_dir / "metrics.json",
                "config": run_dir / "config.yaml",
                "metadata": run_dir / "run_metadata.json",
                "checkpoint": run_dir / "final.pt",
            }
            missing = [
                name for name, path in required.items() if not path.is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"Missing required artifacts for {tag}, seed {seed}: "
                    f"{missing}"
                )
            config = yaml.safe_load(
                required["config"].read_text(encoding="utf-8")
            )
            metadata = json.loads(
                required["metadata"].read_text(encoding="utf-8")
            )
            if not isinstance(config, dict):
                raise ValueError(f"Invalid config mapping: {required['config']}")

            expected_estimator_steps = (
                5000 if expected_method in {"pinn_ebm", "napinn"} else 0
            )
            checks = {
                "metrics_status": record.get("status") == "complete",
                "metrics_evidence": (
                    record.get("evidence_status")
                    == "full_run_complete_unaggregated"
                ),
                "metrics_not_smoke": not bool(record.get("smoke_test", True)),
                "metrics_method": record.get("method") == expected_method,
                "metrics_tag": record.get("tag") == tag,
                "metrics_pinn_budget": int(
                    record.get("pinn_update_steps", -1)
                )
                == 30000,
                "metrics_estimator_budget": int(
                    record.get("estimator_init_steps", -1)
                )
                == expected_estimator_steps,
                "metadata_status": metadata.get("status") == "complete",
                "metadata_evidence": (
                    metadata.get("evidence_status")
                    == "full_run_complete_unaggregated"
                ),
                "metadata_not_smoke": not bool(
                    metadata.get("smoke_test", True)
                ),
                "metadata_seed": int(metadata.get("seed", -1)) == seed,
                "config_seed": int(config.get("seed", -1)) == seed,
                "config_tag": config.get("tag") == tag,
                "config_method": (
                    config.get("method", {}).get("kind") == expected_method
                ),
                "config_pinn_budget": int(
                    config.get("effective_schedule", {}).get(
                        "pinn_update_steps", -1
                    )
                )
                == 30000,
                "config_estimator_budget": int(
                    config.get("effective_schedule", {}).get(
                        "estimator_init_steps", -1
                    )
                )
                == expected_estimator_steps,
            }
            failed = sorted(key for key, value in checks.items() if not value)
            if failed:
                raise ValueError(
                    f"Strict validation failed for {tag}, seed {seed}: {failed}"
                )

            artifact_path = Path(
                str(record.get("input_artifact_path", ""))
            ).resolve()
            artifact_hash = str(record.get("input_artifact_sha256", ""))
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Missing input artifact for {tag}, seed {seed}: "
                    f"{artifact_path}"
                )
            if artifact_path not in hash_cache:
                hash_cache[artifact_path] = sha256_file(artifact_path)
            provenance_checks = {
                "input_file_hash": hash_cache[artifact_path] == artifact_hash,
                "metadata_input_path": (
                    Path(
                        str(metadata.get("input_artifact_path", ""))
                    ).resolve()
                    == artifact_path
                ),
                "metadata_input_hash": (
                    metadata.get("input_artifact_sha256") == artifact_hash
                ),
                "config_input_path": (
                    Path(str(config.get("data", {}).get("path", ""))).resolve()
                    == artifact_path
                ),
                "config_input_hash": (
                    config.get("data", {}).get("input_artifact_sha256")
                    == artifact_hash
                ),
            }
            failed = sorted(
                key for key, value in provenance_checks.items() if not value
            )
            if failed:
                raise ValueError(
                    f"Input provenance failed for {tag}, seed {seed}: {failed}"
                )

            if expected_method == "mad_pinn":
                mad_checks = {
                    "stage1_budget": int(
                        record.get("mad_stage1_pinn_update_steps", -1)
                    )
                    == 30000,
                    "stage2_budget": int(
                        record.get("mad_stage2_pinn_update_steps", -1)
                    )
                    == 30000,
                    "total_budget": int(
                        record.get("mad_total_pipeline_pinn_update_steps", -1)
                    )
                    == 60000,
                    "screening_artifact": (
                        run_dir / "mad_screening.npz"
                    ).is_file(),
                }
                failed = sorted(
                    key for key, value in mad_checks.items() if not value
                )
                if failed:
                    raise ValueError(
                        f"MAD provenance failed for {tag}, seed {seed}: "
                        f"{failed}"
                    )
            _assert_finite(record, location=f"{tag}.seed{seed}")


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
            "input_artifact_path",
            "input_artifact_sha256",
            "learned_reynolds",
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
    if args.output.exists() or args.output.with_suffix(".csv").exists():
        raise FileExistsError(
            "Refusing to overwrite strict aggregation outputs: "
            f"{args.output} / {args.output.with_suffix('.csv')}"
        )
    records = eligible_records(args.root)
    validate_strict_records(args.root, records)
    summary = aggregate(records)
    if len(summary) != EXPECTED_GROUPS:
        raise AssertionError(
            f"Expected {EXPECTED_GROUPS} groups, got {len(summary)}"
        )
    payload = {
        "schema_version": 1,
        "status": "strict_complete",
        "evidence_status": "102_full_runs_complete_strictly_aggregated",
        "included_run_count": len(records),
        "three_seed_group_count": len(summary),
        "policy": {
            "natural_piv_excluded": True,
            "calibration_seed_39_excluded": True,
            "held_out_seeds": sorted(HELD_OUT_SEEDS),
            "smoke_tests_excluded": True,
            "partial_aggregation_permitted": False,
            "expected_runs": EXPECTED_RUNS,
            "expected_groups": EXPECTED_GROUPS,
            "exact_tag_and_seed_coverage_required": True,
            "checkpoint_and_input_hash_validation_required": True,
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
        default=Path("outputs/rebuttal/realpde"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/rebuttal/realpde/injected_piv_aggregation_strict.json"
        ),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
