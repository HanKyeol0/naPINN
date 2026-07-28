"""Strict aggregation for the nine-job natural-PIV naPINN loss matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


EXPECTED_INPUT_SHA256 = (
    "e2513cccae4bdd75da1bf33a85bf23bb66a2dc8350c3e951a5c3e95ebb94b1bd"
)
SEEDS = (40, 41, 42)
METHODS: dict[str, dict[str, Any]] = {
    "napinn": {"tag": "napinn", "loss": "mse", "q": None},
    "napinn_lad": {"tag": "napinn_l1", "loss": "l1", "q": None},
    "napinn_q29": {
        "tag": "napinn_q29",
        "loss": "q_gaussian",
        "q": 2.9,
    },
}
METRICS = (
    "rMAE",
    "rMSE",
    "pde_momentum_rms",
    "continuity_rms",
    "retained_fraction",
    "mean_gate_weight",
    "estimator_mean_negative_log_density",
    "effective_reynolds",
    "effective_reynolds_relative_error",
    "warmup_sec",
    "estimator_init_sec",
    "joint_sec",
    "training_wall_sec",
    "evaluation_wall_sec",
    "end_to_end_wall_sec",
    "gpu_peak_memory_allocated_bytes",
    "gpu_peak_memory_reserved_bytes",
)
FORBIDDEN_LABELED_OUTLIER_TOKENS = (
    "auroc",
    "known_failed",
    "known_clean",
    "gross_outlier",
    "background_only",
    "failed_rejection",
    "clean_rejection",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def expected_run_dir(root: Path, method: str, seed: int) -> Path:
    spec = METHODS[method]
    return (
        root
        / "runs"
        / "cylinder_real_piv"
        / spec["tag"]
        / f"natural_piv_{method}_seed_{seed}"
    )


def validate_history(path: Path) -> dict[str, int]:
    phase_counts = {"mse_warmup": 0, "joint": 0}
    observed_steps = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            phase = record.get("phase")
            if phase not in phase_counts:
                raise ValueError(
                    f"Unexpected phase {phase!r} in {path}:{line_number}"
                )
            phase_counts[phase] += 1
            observed_steps.append(int(record["pinn_step"]))
    if phase_counts != {"mse_warmup": 5000, "joint": 25000}:
        raise ValueError(f"Incorrect phase counts in {path}: {phase_counts}")
    if observed_steps != list(range(1, 30001)):
        raise ValueError(f"Non-contiguous PINN steps in {path}")
    return phase_counts


def validate_run(
    root: Path,
    *,
    method: str,
    seed: int,
) -> dict[str, Any]:
    spec = METHODS[method]
    run_dir = expected_run_dir(root, method, seed)
    required = (
        "config.yaml",
        "run_metadata.json",
        "train_history.jsonl",
        "final.pt",
        "metrics.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} lacks {missing}")
    for step in range(5000, 30001, 5000):
        if not (run_dir / f"step_{step}.pt").is_file():
            raise FileNotFoundError(f"{run_dir} lacks step_{step}.pt")

    metrics = read_json(run_dir / "metrics.json")
    metadata = read_json(run_dir / "run_metadata.json")
    config = read_yaml(run_dir / "config.yaml")
    expected_fields = {
        "status": "complete",
        "evidence_status": "full_run_complete_unaggregated",
        "smoke_test": False,
        "method": method,
        "tag": spec["tag"],
        "seed": seed,
        "input_artifact_sha256": EXPECTED_INPUT_SHA256,
        "pinn_update_steps": 30000,
        "estimator_init_steps": 5000,
        "learned_reynolds": False,
    }
    for key, expected in expected_fields.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"{run_dir}: metrics[{key!r}]={metrics.get(key)!r}, "
                f"expected {expected!r}"
            )
    if metadata.get("status") != "complete" or metadata.get("smoke_test"):
        raise ValueError(f"{run_dir}: incomplete or smoke run metadata")
    if metadata.get("input_artifact_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError(f"{run_dir}: metadata input checksum mismatch")
    for key in metrics:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_LABELED_OUTLIER_TOKENS):
            raise ValueError(
                f"{run_dir}: natural PIV must not expose labeled-outlier "
                f"metric {key!r}"
            )

    resolved = config.get("resolved_data_loss")
    if resolved != {"kind": spec["loss"], "q": spec["q"]}:
        raise ValueError(
            f"{run_dir}: resolved data loss {resolved!r}, expected {spec!r}"
        )
    schedule = config.get("effective_schedule")
    expected_schedule = {
        "warmup_steps": 5000,
        "estimator_init_steps": 5000,
        "joint_steps": 25000,
        "pinn_update_steps": 30000,
        "smoke_test": False,
    }
    if schedule != expected_schedule:
        raise ValueError(f"{run_dir}: schedule mismatch {schedule!r}")
    if config.get("pde", {}).get("learn_reynolds") is not False:
        raise ValueError(f"{run_dir}: natural-PIV Re must remain fixed")
    if config.get("batch") != {"n_f": 8192, "n_data": 2048}:
        raise ValueError(f"{run_dir}: batch protocol mismatch")
    model = config.get("model", {})
    if (
        int(model.get("hidden_layers", -1)) != 5
        or int(model.get("hidden_width", -1)) != 80
        or str(model.get("activation", "")).lower() != "tanh"
    ):
        raise ValueError(f"{run_dir}: model protocol mismatch {model!r}")

    artifact_path = Path(metrics["input_artifact_path"])
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"{run_dir}: input artifact is unavailable at {artifact_path}"
        )
    if sha256_file(artifact_path) != EXPECTED_INPUT_SHA256:
        raise ValueError(f"{run_dir}: current input artifact checksum mismatch")
    phase_counts = validate_history(run_dir / "train_history.jsonl")
    for metric in METRICS:
        finite_float(metrics.get(metric), label=f"{run_dir}:{metric}")
    for required_metadata in ("command", "git", "hardware", "software"):
        if not metadata.get(required_metadata):
            raise ValueError(
                f"{run_dir}: run metadata lacks {required_metadata}"
            )

    return {
        **metrics,
        "_metrics_path": str((run_dir / "metrics.json").resolve()),
        "_checkpoint_path": str((run_dir / "final.pt").resolve()),
        "_config_path": str((run_dir / "config.yaml").resolve()),
        "_metadata_path": str((run_dir / "run_metadata.json").resolve()),
        "_history_path": str((run_dir / "train_history.jsonl").resolve()),
        "_history_phase_counts": phase_counts,
        "_final_checkpoint_sha256": sha256_file(run_dir / "final.pt"),
        "_config_sha256": sha256_file(run_dir / "config.yaml"),
        "_metrics_sha256": sha256_file(run_dir / "metrics.json"),
        "_metadata_sha256": sha256_file(run_dir / "run_metadata.json"),
    }


def summarize(values: dict[int, float]) -> dict[str, Any]:
    ordered = {str(seed): values[seed] for seed in SEEDS}
    sequence = list(ordered.values())
    return {
        "n": len(sequence),
        "mean": mean(sequence),
        "std": stdev(sequence),
        "per_seed": ordered,
    }


def aggregate(records: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method, spec in METHODS.items():
        method_records = {seed: records[(method, seed)] for seed in SEEDS}
        methods[method] = {
            "tag": spec["tag"],
            "reconstruction_loss": spec["loss"],
            "q": spec["q"],
            "seeds": list(SEEDS),
            "metrics": {
                metric: summarize(
                    {
                        seed: finite_float(
                            method_records[seed][metric],
                            label=f"{method}/seed{seed}/{metric}",
                        )
                        for seed in SEEDS
                    }
                )
                for metric in METRICS
            },
            "artifacts": {
                str(seed): {
                    key.removeprefix("_"): value
                    for key, value in method_records[seed].items()
                    if key.startswith("_") and key != "_history_phase_counts"
                }
                for seed in SEEDS
            },
        }

    paired_differences: dict[str, Any] = {}
    for method in ("napinn_lad", "napinn_q29"):
        paired_differences[f"{method}_minus_napinn"] = {
            "sign_convention": (
                "candidate minus naPINN-MSE; negative favors candidate for "
                "lower-is-better metrics"
            ),
            "metrics": {
                metric: summarize(
                    {
                        seed: finite_float(
                            records[(method, seed)][metric],
                            label=f"{method}/seed{seed}/{metric}",
                        )
                        - finite_float(
                            records[("napinn", seed)][metric],
                            label=f"napinn/seed{seed}/{metric}",
                        )
                        for seed in SEEDS
                    }
                )
                for metric in (
                    "rMAE",
                    "rMSE",
                    "pde_momentum_rms",
                    "continuity_rms",
                )
            },
        }
    first = records[("napinn", SEEDS[0])]
    return {
        "status": "complete",
        "protocol": "natural_piv_napinn_robust_loss",
        "labeled_outlier_metrics_applicable": False,
        "interpretation": (
            "Unmodified real PIV has no controlled outlier labels. Gate "
            "retention is descriptive; no outlier AUROC or rejection-rate "
            "claim is valid."
        ),
        "expected_jobs": 9,
        "eligible_jobs": len(records),
        "seeds": list(SEEDS),
        "input_artifact_path": first["input_artifact_path"],
        "input_artifact_sha256": EXPECTED_INPUT_SHA256,
        "comparability_invariants": {
            key: first[key]
            for key in (
                "benchmark",
                "sim_id",
                "source_sha256",
                "sensor_seed",
                "n_spatial_sensors",
                "n_frames",
                "n_heldout_spatial",
                "n_heldout_measurements",
                "metadata_reynolds",
                "learned_reynolds",
            )
        },
        "methods": methods,
        "paired_differences": paired_differences,
    }


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "method",
                "tag",
                "reconstruction_loss",
                "q",
                "metric",
                "seed",
                "value",
                "mean",
                "std",
            )
        )
        for method, method_payload in payload["methods"].items():
            for metric, summary in method_payload["metrics"].items():
                for seed, value in summary["per_seed"].items():
                    writer.writerow(
                        (
                            method,
                            method_payload["tag"],
                            method_payload["reconstruction_loss"],
                            method_payload["q"],
                            metric,
                            seed,
                            value,
                            summary["mean"],
                            summary["std"],
                        )
                    )


def main(args: argparse.Namespace) -> None:
    records = {
        (method, seed): validate_run(args.root, method=method, seed=seed)
        for method in METHODS
        for seed in SEEDS
    }
    input_hashes = {
        record["input_artifact_sha256"] for record in records.values()
    }
    if input_hashes != {EXPECTED_INPUT_SHA256}:
        raise ValueError(f"Natural-PIV input hashes differ: {input_hashes}")
    for invariant in (
        "benchmark",
        "sim_id",
        "source_sha256",
        "sensor_seed",
        "n_spatial_sensors",
        "n_frames",
        "n_heldout_spatial",
        "n_heldout_measurements",
        "metadata_reynolds",
        "learned_reynolds",
    ):
        values = {record[invariant] for record in records.values()}
        if len(values) != 1:
            raise ValueError(f"Invariant {invariant} differs: {values}")

    payload = aggregate(records)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/rebuttal/natural_piv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/rebuttal/natural_piv/aggregation.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/rebuttal/natural_piv/aggregation.csv"),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
