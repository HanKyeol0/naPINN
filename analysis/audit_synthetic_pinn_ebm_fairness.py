#!/usr/bin/env python3
"""Audit pairing and corruption geometry in the synthetic rebuttal benchmark.

This script does not train a model.  It reconstructs the frozen observations
for MSE, direct PINN-EBM, and naPINN and checks that every method receives the
same coordinates, noisy values, and gross-outlier indices.  It also records
the empirical background-only and gross-corrupted error distributions so the
closest-prior comparison can be interpreted from generated evidence rather
than configuration names alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pinnlab.registry import get_experiment
from pinnlab.utils.seed import seed_everything
from scripts.rebuttal.run_synthetic import configure_experiment, load_yaml


EXPERIMENT_CONFIGS = {
    "allencahn2d": Path("configs/rebuttal/allencahn2d.yaml"),
    "burgers2d": Path("configs/rebuttal/burgers2d.yaml"),
    "lambdaomega2d": Path("configs/rebuttal/lambdaomega2d.yaml"),
}
METHODS = ("mse", "pinn_ebm", "napinn")
GENERATED_DATASETS = {
    "burgers2d": {
        "path": Path(
            "pinnlab/simulation/simulation_result/Burgers2D_3-5/data.npz"
        ),
        "handoff_container_sha256": (
            "ba83755bfd03115838341d757784e85d7f91854e457d5be856de26fd773d0d27"
        ),
    },
    "lambdaomega2d": {
        "path": Path(
            "pinnlab/simulation/simulation_result/"
            "LambdaOmega_Spiral_2/data.npz"
        ),
        "handoff_container_sha256": (
            "d046ba0f2c0c46e54c301b2c2a1c241da956d09bde70c4c9b11ee3f416bf92f6"
        ),
    },
}


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npz_semantic_sha256(path: Path) -> str:
    """Hash array names, dtypes, shapes, and values, ignoring ZIP metadata."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def distribution_summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {"count": 0}
    probabilities = torch.tensor(
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0],
        dtype=flat.dtype,
    )
    quantiles = torch.quantile(flat, probabilities)
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=True)) if flat.numel() > 1 else 0.0,
        "positive_fraction": float((flat > 0.0).float().mean()),
        "minimum": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "maximum": float(quantiles[6]),
    }


def build_observation(
    experiment_name: str,
    *,
    method: str,
    ratio: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    config = configure_experiment(
        load_yaml(EXPERIMENT_CONFIGS[experiment_name]),
        experiment_name=experiment_name,
        method=method,
        device=device,
        outlier_ratio=ratio,
        noise_kind="4G",
        ema_momentum=0.05,
        rejection_cost=0.5,
        estimator_init_steps=5000,
    )
    seed_everything(seed)
    experiment = get_experiment(experiment_name)(config, torch.device(device))

    coordinates = experiment.X_data.detach()
    clean = experiment.y_clean.detach()
    noisy = experiment.y_data.detach()
    error = noisy - clean
    indices = np.asarray(experiment.outlier_indices, dtype=np.int64)
    row_mask = torch.zeros(
        error.shape[0],
        dtype=torch.bool,
        device=error.device,
    )
    if indices.size:
        row_mask[torch.from_numpy(indices).to(error.device)] = True
    scalar_mask = row_mask[:, None].expand_as(error)

    result = {
        "method": method,
        "coordinate_sha256": tensor_sha256(coordinates),
        "clean_value_sha256": tensor_sha256(clean),
        "noisy_value_sha256": tensor_sha256(noisy),
        "gross_row_indices_sha256": array_sha256(indices),
        "measurement_rows": int(error.shape[0]),
        "measurement_components": int(error.shape[1]),
        "gross_rows": int(row_mask.sum()),
        "gross_row_fraction": float(row_mask.float().mean()),
        "gross_scalars": int(scalar_mask.sum()),
        "gross_scalar_fraction": float(scalar_mask.float().mean()),
        "background_only_error": distribution_summary(error[~scalar_mask]),
        "gross_row_total_error": distribution_summary(error[scalar_mask]),
        "configured_four_gaussian_components": config["noise"]["par_list"],
        "configured_background_scale": float(config["noise"]["scale"]),
        "configured_gross_factor_range": [
            float(config["noise"]["extra_noise"]["scale_min"]),
            float(config["noise"]["extra_noise"]["scale_max"]),
        ],
    }
    del experiment
    if device.startswith("cuda:"):
        torch.cuda.empty_cache()
    return result


def main(args: argparse.Namespace) -> None:
    if args.device != "cpu" and not args.device.startswith("cuda:"):
        raise ValueError("--device must be 'cpu' or an explicit CUDA device")
    if args.device.startswith("cuda:") and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA audit device is unavailable.")

    output: dict[str, Any] = {
        "status": "complete",
        "purpose": (
            "Input-pairing and corruption-distribution audit only; no model "
            "training or performance result."
        ),
        "seed": args.seed,
        "device": args.device,
        "conditions": {},
        "checks": {},
        "dataset_provenance": {},
    }
    for experiment_name, dataset in GENERATED_DATASETS.items():
        path = dataset["path"]
        assert isinstance(path, Path)
        current_sha = file_sha256(path)
        historical_sha = str(dataset["handoff_container_sha256"])
        output["dataset_provenance"][experiment_name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "current_container_sha256": current_sha,
            "semantic_array_sha256": npz_semantic_sha256(path),
            "handoff_container_sha256": historical_sha,
            "container_sha_matches_handoff": current_sha == historical_sha,
            "interpretation": (
                "A container-hash mismatch means these destination-server "
                "runs are a regenerated diagnostic dataset and must not be "
                "described as a byte-identical regeneration of the prior "
                "submitted or handoff results."
            ),
        }
    pairing_checks: list[bool] = []
    nesting_checks: list[bool] = []

    for experiment_name in EXPERIMENT_CONFIGS:
        output["conditions"][experiment_name] = {}
        reference_indices: dict[float, np.ndarray] = {}
        for ratio in args.ratios:
            key = f"{int(round(100 * ratio)):02d}pct"
            method_results = [
                build_observation(
                    experiment_name,
                    method=method,
                    ratio=ratio,
                    seed=args.seed,
                    device=args.device,
                )
                for method in METHODS
            ]
            hashes = {
                field: {result[field] for result in method_results}
                for field in (
                    "coordinate_sha256",
                    "clean_value_sha256",
                    "noisy_value_sha256",
                    "gross_row_indices_sha256",
                )
            }
            paired = all(len(values) == 1 for values in hashes.values())
            pairing_checks.append(paired)
            output["conditions"][experiment_name][key] = {
                "methods": method_results,
                "paired_across_methods": paired,
                "unique_hash_counts": {
                    field: len(values) for field, values in hashes.items()
                },
            }

            config = configure_experiment(
                load_yaml(EXPERIMENT_CONFIGS[experiment_name]),
                experiment_name=experiment_name,
                method="mse",
                device=args.device,
                outlier_ratio=ratio,
                noise_kind="4G",
                ema_momentum=0.05,
                rejection_cost=0.5,
                estimator_init_steps=5000,
            )
            seed_everything(args.seed)
            experiment = get_experiment(experiment_name)(
                config,
                torch.device(args.device),
            )
            reference_indices[ratio] = np.asarray(
                experiment.outlier_indices,
                dtype=np.int64,
            )
            del experiment
            if args.device.startswith("cuda:"):
                torch.cuda.empty_cache()

        sorted_ratios = sorted(reference_indices)
        for lower, upper in zip(sorted_ratios, sorted_ratios[1:]):
            lower_indices = reference_indices[lower]
            upper_indices = reference_indices[upper]
            nested = (
                len(lower_indices) <= len(upper_indices)
                and np.array_equal(
                    lower_indices,
                    upper_indices[: len(lower_indices)],
                )
            )
            nesting_checks.append(nested)
            output["conditions"][experiment_name][
                f"nested_{int(100 * lower):02d}_in_{int(100 * upper):02d}"
            ] = nested

    output["checks"] = {
        "all_methods_receive_identical_observations": all(pairing_checks),
        "all_lower_ratio_gross_sets_are_prefixes_of_higher_ratio_sets": all(
            nesting_checks
        ),
        "pairing_check_count": len(pairing_checks),
        "nesting_check_count": len(nesting_checks),
    }
    if not all(pairing_checks):
        output["status"] = "failed"
    if nesting_checks and not all(nesting_checks):
        output["status"] = "failed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["checks"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    if output["status"] != "complete":
        raise AssertionError("Synthetic fairness audit failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/audits/synthetic_pinn_ebm_fairness/seed_40.json"
        ),
    )
    main(parser.parse_args())
