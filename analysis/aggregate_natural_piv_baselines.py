"""Strict aggregation for natural-PIV baselines and the full NLL weight grid."""

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
    "mse": {
        "tag": "mse",
        "runner_method": "mse",
        "loss": "mse",
        "q": None,
        "joint_pde_weight": None,
    },
    "lad": {
        "tag": "lad",
        "runner_method": "lad",
        "loss": "l1",
        "q": None,
        "joint_pde_weight": None,
    },
    "orpinn_q29": {
        "tag": "orpinn_q29",
        "runner_method": "orpinn_q29",
        "loss": "q_gaussian",
        "q": 2.9,
        "joint_pde_weight": None,
    },
    "pinn_ebm_w1": {
        "tag": "pinn_ebm",
        "runner_method": "pinn_ebm",
        "loss": "ebm_nll",
        "q": None,
        "joint_pde_weight": 1.0,
    },
    "pinn_ebm_w10": {
        "tag": "pinn_ebm_weight10",
        "runner_method": "pinn_ebm",
        "loss": "ebm_nll",
        "q": None,
        "joint_pde_weight": 10.0,
    },
    "pinn_ebm_w50": {
        "tag": "pinn_ebm_pilar",
        "runner_method": "pinn_ebm",
        "loss": "ebm_nll",
        "q": None,
        "joint_pde_weight": 50.0,
    },
}
COMMON_METRICS = (
    "rMAE",
    "rMSE",
    "pde_momentum_rms",
    "continuity_rms",
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


def run_name(campaign_method: str, seed: int) -> str:
    return f"natural_piv_baseline_{campaign_method}_seed_{seed}"


def run_dir(root: Path, campaign_method: str, seed: int) -> Path:
    return (
        root
        / "runs"
        / "cylinder_real_piv"
        / METHODS[campaign_method]["tag"]
        / run_name(campaign_method, seed)
    )


def validate_history(
    path: Path, *, runner_method: str
) -> dict[str, int]:
    expected = (
        {"mse_warmup": 5000, "joint": 25000}
        if runner_method == "pinn_ebm"
        else {runner_method: 30000}
    )
    observed = {phase: 0 for phase in expected}
    steps = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            phase = record.get("phase")
            if phase not in observed:
                raise ValueError(
                    f"Unexpected phase {phase!r} in {path}:{line_number}"
                )
            observed[phase] += 1
            steps.append(int(record["pinn_step"]))
    if observed != expected:
        raise ValueError(f"Incorrect phase counts in {path}: {observed}")
    if steps != list(range(1, 30001)):
        raise ValueError(f"Non-contiguous PINN steps in {path}")
    return observed


def validate_run(
    root: Path, *, campaign_method: str, seed: int
) -> dict[str, Any]:
    spec = METHODS[campaign_method]
    directory = run_dir(root, campaign_method, seed)
    required = (
        "config.yaml",
        "run_metadata.json",
        "train_history.jsonl",
        "final.pt",
        "metrics.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} lacks {missing}")
    for step in range(5000, 30001, 5000):
        if not (directory / f"step_{step}.pt").is_file():
            raise FileNotFoundError(f"{directory} lacks step_{step}.pt")

    metrics = read_json(directory / "metrics.json")
    metadata = read_json(directory / "run_metadata.json")
    config = read_yaml(directory / "config.yaml")
    estimator_steps = 5000 if spec["runner_method"] == "pinn_ebm" else 0
    expected_fields = {
        "status": "complete",
        "evidence_status": "full_run_complete_unaggregated",
        "smoke_test": False,
        "method": spec["runner_method"],
        "tag": spec["tag"],
        "seed": seed,
        "input_artifact_sha256": EXPECTED_INPUT_SHA256,
        "pinn_update_steps": 30000,
        "estimator_init_steps": estimator_steps,
        "learned_reynolds": False,
    }
    for key, expected in expected_fields.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"{directory}: metrics[{key!r}]={metrics.get(key)!r}, "
                f"expected {expected!r}"
            )
    if metadata.get("status") != "complete" or metadata.get("smoke_test"):
        raise ValueError(f"{directory}: incomplete or smoke run metadata")
    if metadata.get("input_artifact_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError(f"{directory}: metadata input checksum mismatch")
    for key in metrics:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_LABELED_OUTLIER_TOKENS):
            raise ValueError(
                f"{directory}: invalid labeled-outlier metric {key!r}"
            )

    resolved = config.get("resolved_data_loss")
    if resolved != {"kind": spec["loss"], "q": spec["q"]}:
        raise ValueError(
            f"{directory}: resolved data loss mismatch {resolved!r}"
        )
    staged = spec["runner_method"] == "pinn_ebm"
    expected_schedule = {
        "warmup_steps": 5000 if staged else 0,
        "estimator_init_steps": 5000 if staged else 0,
        "joint_steps": 25000 if staged else 30000,
        "pinn_update_steps": 30000,
        "smoke_test": False,
    }
    if config.get("effective_schedule") != expected_schedule:
        raise ValueError(f"{directory}: effective schedule mismatch")
    observed_weight = config.get("training", {}).get(
        "joint_pde_weight",
        config.get("training", {}).get("pde_weight"),
    )
    if staged:
        if float(observed_weight) != float(spec["joint_pde_weight"]):
            raise ValueError(f"{directory}: direct-NLL weight mismatch")
    if config.get("pde", {}).get("learn_reynolds") is not False:
        raise ValueError(f"{directory}: Reynolds number must be fixed")
    if config.get("batch") != {"n_f": 8192, "n_data": 2048}:
        raise ValueError(f"{directory}: batch protocol mismatch")
    model = config.get("model", {})
    if (
        int(model.get("hidden_layers", -1)) != 5
        or int(model.get("hidden_width", -1)) != 80
        or str(model.get("activation", "")).lower() != "tanh"
    ):
        raise ValueError(f"{directory}: model protocol mismatch")
    artifact_path = Path(metrics["input_artifact_path"])
    if (
        not artifact_path.is_file()
        or sha256_file(artifact_path) != EXPECTED_INPUT_SHA256
    ):
        raise ValueError(f"{directory}: current input artifact mismatch")
    history_counts = validate_history(
        directory / "train_history.jsonl",
        runner_method=spec["runner_method"],
    )
    for metric in COMMON_METRICS:
        finite_float(metrics.get(metric), label=f"{directory}:{metric}")
    if staged:
        finite_float(
            metrics.get("estimator_mean_negative_log_density"),
            label=f"{directory}:estimator_mean_negative_log_density",
        )
    elif "estimator_mean_negative_log_density" in metrics:
        raise ValueError(f"{directory}: non-EBM baseline exposes EBM metric")
    for required_metadata in ("command", "git", "hardware", "software"):
        if not metadata.get(required_metadata):
            raise ValueError(
                f"{directory}: run metadata lacks {required_metadata}"
            )
    return {
        **metrics,
        "_campaign_method": campaign_method,
        "_joint_pde_weight": spec["joint_pde_weight"],
        "_metrics_path": str((directory / "metrics.json").resolve()),
        "_checkpoint_path": str((directory / "final.pt").resolve()),
        "_config_path": str((directory / "config.yaml").resolve()),
        "_metadata_path": str((directory / "run_metadata.json").resolve()),
        "_history_path": str(
            (directory / "train_history.jsonl").resolve()
        ),
        "_history_phase_counts": history_counts,
        "_final_checkpoint_sha256": sha256_file(directory / "final.pt"),
        "_config_sha256": sha256_file(directory / "config.yaml"),
        "_metrics_sha256": sha256_file(directory / "metrics.json"),
        "_metadata_sha256": sha256_file(directory / "run_metadata.json"),
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
    method_payloads: dict[str, Any] = {}
    for campaign_method, spec in METHODS.items():
        metrics = list(COMMON_METRICS)
        if spec["runner_method"] == "pinn_ebm":
            metrics.append("estimator_mean_negative_log_density")
        method_payloads[campaign_method] = {
            "runner_method": spec["runner_method"],
            "tag": spec["tag"],
            "reconstruction_loss": spec["loss"],
            "q": spec["q"],
            "joint_pde_weight": spec["joint_pde_weight"],
            "seeds": list(SEEDS),
            "metrics": {
                metric: summarize(
                    {
                        seed: finite_float(
                            records[(campaign_method, seed)][metric],
                            label=f"{campaign_method}/seed{seed}/{metric}",
                        )
                        for seed in SEEDS
                    }
                )
                for metric in metrics
            },
            "artifacts": {
                str(seed): {
                    key.removeprefix("_"): value
                    for key, value in records[(campaign_method, seed)].items()
                    if key.startswith("_")
                    and key
                    not in {
                        "_campaign_method",
                        "_joint_pde_weight",
                        "_history_phase_counts",
                    }
                }
                for seed in SEEDS
            },
        }

    comparisons = (
        ("lad", "mse"),
        ("orpinn_q29", "mse"),
        *tuple(
            (direct, baseline)
            for direct in ("pinn_ebm_w1", "pinn_ebm_w10", "pinn_ebm_w50")
            for baseline in ("mse", "lad", "orpinn_q29")
        ),
    )
    paired = {}
    for candidate, reference in comparisons:
        paired[f"{candidate}_minus_{reference}"] = {
            "sign_convention": (
                "candidate minus reference; negative favors candidate for "
                "lower-is-better metrics"
            ),
            "metrics": {
                metric: summarize(
                    {
                        seed: finite_float(
                            records[(candidate, seed)][metric],
                            label=f"{candidate}/seed{seed}/{metric}",
                        )
                        - finite_float(
                            records[(reference, seed)][metric],
                            label=f"{reference}/seed{seed}/{metric}",
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
    first = records[("mse", SEEDS[0])]
    return {
        "status": "complete",
        "protocol": "natural_piv_matched_baselines_and_direct_nll_grid",
        "selection_performed": False,
        "predeclared_direct_nll_pde_weights": [1.0, 10.0, 50.0],
        "labeled_outlier_metrics_applicable": False,
        "interpretation": (
            "All three direct-NLL PDE weights are retained. This aggregate "
            "does not select or label a favorable weight. Natural PIV has no "
            "controlled outlier labels, so no AUROC claim is valid."
        ),
        "expected_jobs": 18,
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
        "methods": method_payloads,
        "paired_differences": paired,
    }


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "campaign_method",
                "runner_method",
                "tag",
                "reconstruction_loss",
                "q",
                "joint_pde_weight",
                "metric",
                "seed",
                "value",
                "mean",
                "std",
            )
        )
        for method, values in payload["methods"].items():
            for metric, summary in values["metrics"].items():
                for seed, value in summary["per_seed"].items():
                    writer.writerow(
                        (
                            method,
                            values["runner_method"],
                            values["tag"],
                            values["reconstruction_loss"],
                            values["q"],
                            values["joint_pde_weight"],
                            metric,
                            seed,
                            value,
                            summary["mean"],
                            summary["std"],
                        )
                    )


def main(args: argparse.Namespace) -> None:
    records = {
        (method, seed): validate_run(
            args.root,
            campaign_method=method,
            seed=seed,
        )
        for method in METHODS
        for seed in SEEDS
    }
    for invariant in (
        "input_artifact_sha256",
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
        default=Path("outputs/rebuttal/natural_piv_baselines"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/rebuttal/natural_piv_baselines/aggregation.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/rebuttal/natural_piv_baselines/aggregation.csv"
        ),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
