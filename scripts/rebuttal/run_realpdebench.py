#!/usr/bin/env python3
"""Canonical RealPDEBench real fluid-PIV rebuttal runner.

This dedicated runner keeps the Pilar--Wahlström PINN-EBM objective explicit:
MSE warm-up, estimator-only initialization, then one joint NLL backward pass
through both the coordinate PINN and EBM. It never calls ``EBM.train_step``
during joint training, preventing a second estimator update.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pinnlab.registry import get_experiment, get_model
from pinnlab.utils.data_loss import data_loss_q_gaussian
from pinnlab.utils.density import create_density_estimator
from pinnlab.utils.ebm import TrainableLikelihoodGate
from pinnlab.utils.seed import seed_everything

NAPINN_METHODS = {"napinn", "napinn_lad", "napinn_q29"}
STAGED_METHODS = {"pinn_ebm", *NAPINN_METHODS}
BASELINE_METHODS = {"mse", "lad", "orpinn_q19", "orpinn_q29"}
MAD_METHOD = "mad_pinn"
ALL_METHODS = STAGED_METHODS | BASELINE_METHODS | {MAD_METHOD}
METHOD_DATA_LOSSES = {
    "mse": ("mse", None),
    "lad": ("l1", None),
    "orpinn_q19": ("q_gaussian", 1.9),
    "orpinn_q29": ("q_gaussian", 2.9),
    "pinn_ebm": ("ebm_nll", None),
    "napinn": ("mse", None),
    "napinn_lad": ("l1", None),
    "napinn_q29": ("q_gaussian", 2.9),
    MAD_METHOD: ("masked_mse", None),
}
DATA_LOSS_ALIASES = {
    "lad": "l1",
    "l1": "l1",
    "mse": "mse",
    "q_gaussian": "q_gaussian",
    "q-gaussian": "q_gaussian",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return content


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_metadata() -> dict[str, str | None]:
    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "pyyaml": "PyYAML",
        "pandas": "pandas",
        "imageio": "imageio",
        "imageio_ffmpeg": "imageio-ffmpeg",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "wandb": "wandb",
    }
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    for key, distribution in distributions.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def git_metadata() -> dict[str, Any]:
    def run(*args):
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    payload = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "packages": package_metadata(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        payload.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_compute_capability": [
                    properties.major,
                    properties.minor,
                ],
            }
        )
    return payload


def resolve_data_loss(
    config: dict[str, Any], method: str
) -> dict[str, str | float | None]:
    expected_kind, expected_q = METHOD_DATA_LOSSES[method]
    configured = config.get("data_loss", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise ValueError("data_loss must be a YAML mapping")

    raw_kind = configured.get("kind")
    if raw_kind is None:
        resolved_kind = expected_kind
    else:
        normalized_kind = str(raw_kind).strip().lower()
        resolved_kind = DATA_LOSS_ALIASES.get(normalized_kind, normalized_kind)
    if resolved_kind != expected_kind:
        raise ValueError(
            f"Method {method!r} requires data loss {expected_kind!r}, "
            f"got {resolved_kind!r}"
        )

    raw_q = configured.get("q", expected_q)
    resolved_q = None if raw_q is None else float(raw_q)
    if expected_q is None and resolved_q is not None:
        raise ValueError(f"Method {method!r} does not use a q value")
    if expected_q is not None and (
        resolved_q is None or not np.isclose(resolved_q, expected_q)
    ):
        raise ValueError(f"Method {method!r} requires q={expected_q}, got {resolved_q}")
    return {"kind": resolved_kind, "q": resolved_q}


def pointwise_data_loss(
    residual: torch.Tensor,
    resolved_data_loss: dict[str, str | float | None],
) -> torch.Tensor:
    kind = str(resolved_data_loss["kind"])
    if kind == "mse":
        return residual.square()
    if kind == "l1":
        return residual.abs()
    if kind == "q_gaussian":
        q = resolved_data_loss["q"]
        if q is None:
            raise ValueError("q_gaussian data loss requires a resolved q value")
        return data_loss_q_gaussian(residual, q=float(q))
    raise ValueError(f"Unsupported reconstruction data loss: {kind!r}")


def corruption_label_semantics(metadata: dict[str, Any]) -> dict[str, str]:
    corruption = metadata.get("corruption")
    if not isinstance(corruption, dict):
        return {
            "positive": "failed",
            "negative": "clean",
            "auroc": "failure_detection_auroc",
        }
    explicit = corruption.get("label_semantics")
    kind = str(corruption.get("kind", "")).lower()
    legacy_four_gaussian = (
        "four_gaussian" in kind
        or "four-gaussian" in kind
        or "legacy4g" in kind
        or "legacy_4g" in kind
    )
    if isinstance(explicit, dict):
        positive = str(explicit.get("positive", explicit.get("true", ""))).lower()
        negative = str(explicit.get("negative", explicit.get("false", ""))).lower()
        legacy_four_gaussian = legacy_four_gaussian or (
            "gross" in positive and "background" in negative
        )
    if legacy_four_gaussian:
        return {
            "positive": "gross_outlier",
            "negative": "background_only",
            "auroc": "gross_outlier_detection_auroc",
        }
    return {
        "positive": "failed",
        "negative": "clean",
        "auroc": "failure_detection_auroc",
    }


def corruption_metrics_from_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    corruption = metadata.get("corruption")
    if not isinstance(corruption, dict):
        return {}
    result: dict[str, Any] = {"corruption_metadata": corruption}
    kind = corruption.get("kind")
    seed = corruption.get("seed", corruption.get("corruption_seed"))
    if kind is not None:
        result["corruption_kind"] = str(kind)
    if seed is not None:
        result["corruption_seed"] = int(seed)

    semantics = corruption_label_semantics(metadata)
    result["corruption_positive_label"] = semantics["positive"]
    result["corruption_negative_label"] = semantics["negative"]
    if semantics["positive"] == "gross_outlier":
        scalar_fields = {
            "gross_outlier_row_ratio_requested": "gross_row_ratio_requested",
            "gross_outlier_row_ratio_realized": "gross_row_ratio_realized",
            "gross_outlier_rows": "n_gross_rows",
            "training_rows": "n_training_rows",
            "background_scale_multiplier": "base_scale_multiplier",
            "gross_scale_multiplier": "gross_scale_multiplier",
        }
        for output_key, metadata_key in scalar_fields.items():
            if metadata_key in corruption:
                result[output_key] = corruption[metadata_key]
    else:
        if "n_failed_spatial_sensors" in corruption:
            result["failed_spatial_sensors"] = int(
                corruption["n_failed_spatial_sensors"]
            )
        if "failure_fraction_realized" in corruption:
            result["sensor_failure_fraction"] = float(
                corruption["failure_fraction_realized"]
            )
    return result


def update_running_std(
    running_std: torch.Tensor,
    residual: torch.Tensor,
    momentum: float,
) -> torch.Tensor:
    with torch.no_grad():
        batch_std = residual.detach().std(unbiased=True)
        batch_std = torch.clamp(
            batch_std,
            min=1.0e-6,
            max=torch.clamp(running_std * 10.0, min=1.0e-5),
        )
        running_std.mul_(1.0 - momentum).add_(momentum * batch_std)
    return running_std


def finite_or_raise(name: str, value: torch.Tensor, step: int) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite {name} at step {step}: {value}")


def train_step(
    *,
    method: str,
    model,
    experiment,
    optimizer,
    batch,
    pde_weight: float,
    data_weight: float,
    estimator=None,
    gate=None,
    running_std=None,
    std_momentum: float = 0.05,
    estimator_weight: float = 1.0,
    resolved_data_loss: dict[str, str | float | None] | None = None,
    step: int,
) -> dict[str, float]:
    model.train()
    pde_loss = experiment.pde_residual_loss(model, batch)
    residual = experiment.measurement_residual(model, batch)
    flat_residual = residual.reshape(-1, 1)
    rejection_loss = torch.zeros((), device=experiment.device)
    estimator_loss = torch.zeros((), device=experiment.device)

    if method == "mse":
        data_loss = flat_residual.square().mean()
    elif method == "lad":
        data_loss = flat_residual.abs().mean()
    elif method == "orpinn_q19":
        data_loss = data_loss_q_gaussian(flat_residual, q=1.9).mean()
    elif method == "orpinn_q29":
        data_loss = data_loss_q_gaussian(flat_residual, q=2.9).mean()
    elif method == MAD_METHOD:
        scalar_mask = batch["data_scalar_mask"]
        if scalar_mask.shape != residual.shape:
            raise ValueError(
                f"MAD scalar mask {scalar_mask.shape} does not match "
                f"residual {residual.shape}"
            )
        if not bool(scalar_mask.any()):
            raise ValueError("MAD batch contains no retained scalar measurements")
        data_loss = residual.square()[scalar_mask].mean()
    elif method == "pinn_ebm":
        update_running_std(running_std, flat_residual, std_momentum)
        scaled = flat_residual / running_std.detach()
        _, data_loss = estimator.mean_nll(scaled, detach_residual=False)
        estimator_loss = data_loss
    elif method in NAPINN_METHODS:
        if resolved_data_loss is None:
            raise ValueError(f"{method} requires a resolved reconstruction loss")
        update_running_std(running_std, flat_residual, std_momentum)
        scaled = flat_residual / running_std.detach()
        # The estimator sees scalar flattened u/v residuals. Its likelihood
        # fit uses detached residuals, while the PINN receives the gated data
        # gradient. A single optimizer performs one update of every parameter.
        _, estimator_loss = estimator.mean_nll(scaled.detach(), detach_residual=True)
        log_density = estimator(scaled.detach()).detach()
        weights, rejection_loss = gate(log_density)
        reconstruction_loss = (
            weights * pointwise_data_loss(flat_residual, resolved_data_loss)
        ).mean()
        data_loss = reconstruction_loss + rejection_loss
    else:
        raise ValueError(method)

    total = pde_weight * pde_loss + data_weight * data_loss
    if method in NAPINN_METHODS:
        total = total + estimator_weight * estimator_loss
    finite_or_raise("total loss", total, step)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            finite_or_raise(f"model gradient {name}", parameter.grad, step)
    for index, parameter in enumerate(experiment.extra_params()):
        if parameter.grad is not None:
            finite_or_raise(f"experiment gradient {index}", parameter.grad, step)
    if method in STAGED_METHODS:
        for name, parameter in estimator.named_parameters():
            if parameter.grad is not None:
                finite_or_raise(f"estimator gradient {name}", parameter.grad, step)
    optimizer.step()

    payload = {
        "loss_total": float(total.detach().cpu()),
        "loss_pde": float(pde_loss.detach().cpu()),
        "loss_data": float(data_loss.detach().cpu()),
    }
    if method in STAGED_METHODS:
        payload["loss_estimator_nll"] = float(estimator_loss.detach().cpu())
        payload["running_std"] = float(running_std.detach().cpu())
    if method in NAPINN_METHODS:
        payload["loss_reconstruction"] = float(reconstruction_loss.detach().cpu())
        payload["loss_rejection"] = float(rejection_loss.detach().cpu())
        payload["batch_retained_fraction"] = float(
            (weights.detach() >= 0.5).float().mean().cpu()
        )
        payload["batch_mean_gate_weight"] = float(weights.detach().mean().cpu())
    return payload


def compute_mad_scalar_screen(
    absolute_residual: np.ndarray,
    labels: np.ndarray,
    *,
    divisor: float,
    threshold_multiplier: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the audited Peng scalar rule and summarize optional labels."""
    if absolute_residual.ndim != 2 or absolute_residual.shape[1] != 2:
        raise ValueError("MAD residuals must have shape (measurements, 2)")
    if labels.shape != absolute_residual.shape:
        raise ValueError("MAD corruption labels and residuals must have equal shape")
    if divisor != 1.6777:
        raise ValueError("Faithful Peng MAD-PINN requires sigma_divisor=1.6777")
    if threshold_multiplier != 3.0:
        raise ValueError(
            "Canonical Peng MAD-PINN comparison requires threshold_multiplier=3.0"
        )
    median_absolute_residual = float(np.median(absolute_residual))
    sigma_hat = median_absolute_residual / divisor
    threshold = threshold_multiplier * sigma_hat
    retained = absolute_residual <= threshold
    if not retained.any():
        raise ValueError("MAD screening rejected every scalar measurement")
    metrics: dict[str, Any] = {
        "mad_median_absolute_residual": median_absolute_residual,
        "mad_sigma_hat": sigma_hat,
        "mad_sigma_divisor": divisor,
        "mad_threshold_multiplier": threshold_multiplier,
        "mad_threshold": threshold,
        "mad_retained_scalar_measurements": int(retained.sum()),
        "mad_total_scalar_measurements": int(retained.size),
        "mad_retained_fraction": float(retained.mean()),
        "mad_u_retained_fraction": float(retained[:, 0].mean()),
        "mad_v_retained_fraction": float(retained[:, 1].mean()),
        "mad_rows_both_retained": int(retained.all(axis=1).sum()),
        "mad_rows_one_retained": int(
            np.logical_xor(retained[:, 0], retained[:, 1]).sum()
        ),
        "mad_rows_none_retained": int((~retained.any(axis=1)).sum()),
    }
    if labels.any():
        metrics.update(
            {
                "mad_known_failed_rejection_rate": float((~retained)[labels].mean()),
                "mad_known_clean_rejection_rate": float((~retained)[~labels].mean()),
                "mad_known_failed_scalar_measurements": int(labels.sum()),
                "mad_known_clean_scalar_measurements": int((~labels).sum()),
            }
        )
    return retained, metrics


def load_mad_stage1_and_screen(
    *,
    checkpoint_path: Path,
    model,
    experiment,
    config: dict[str, Any],
    model_config: dict[str, Any],
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any], np.ndarray]:
    """Restore a completed LAD model and apply Peng et al.'s scalar screen."""
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MAD stage-1 checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=experiment.device,
        weights_only=False,
    )
    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("MAD stage-1 checkpoint lacks model/config state")
    stage1_config = checkpoint["config"]
    stage1_method = str(stage1_config.get("method", {}).get("kind", "")).lower()
    if stage1_method != "lad":
        raise ValueError(f"MAD stage 1 must be LAD-PINN, got {stage1_method!r}")
    stage1_schedule = stage1_config.get("effective_schedule", {})
    if bool(stage1_schedule.get("smoke_test", True)):
        raise ValueError("MAD stage 1 must be a completed non-smoke LAD run")
    stage1_steps = int(stage1_schedule.get("pinn_update_steps", -1))
    if stage1_steps != 30000 or int(checkpoint.get("pinn_step", -1)) != 30000:
        raise ValueError("MAD stage 1 must contain exactly 30,000 LAD-PINN updates")
    if int(stage1_config.get("seed", -1)) != seed:
        raise ValueError(
            f"MAD stage-1 seed {stage1_config.get('seed')} != requested seed {seed}"
        )
    stage1_data = Path(stage1_config["data"]["path"]).resolve()
    current_data = Path(config["data"]["path"]).resolve()
    if stage1_data != current_data:
        raise ValueError(
            f"MAD stage-1 data {stage1_data} != current data {current_data}"
        )
    if stage1_config.get("model") != model_config:
        raise ValueError("MAD stage-1 and stage-2 model configurations differ")

    model.load_state_dict(checkpoint["model"])
    if "experiment" in checkpoint:
        experiment.load_state_dict(checkpoint["experiment"])
    model.eval()
    with torch.no_grad():
        residual = experiment.y_data - model(experiment.X_data)[:, :2]
    absolute_residual = residual.detach().cpu().numpy().astype(np.float64)
    absolute_residual = np.abs(absolute_residual)
    mad_cfg = config["mad"]
    divisor = float(mad_cfg["sigma_divisor"])
    threshold_multiplier = float(mad_cfg["threshold_multiplier"])
    labels = experiment.data_corruption_labels.detach().cpu().numpy()
    retained_numpy, screening_metrics = compute_mad_scalar_screen(
        absolute_residual,
        labels,
        divisor=divisor,
        threshold_multiplier=threshold_multiplier,
    )
    retained_mask = torch.tensor(
        retained_numpy,
        dtype=torch.bool,
        device=experiment.device,
    )

    stage1_metrics_path = checkpoint_path.parent / "metrics.json"
    stage1_metrics = {}
    if stage1_metrics_path.is_file():
        stage1_metrics = json.loads(stage1_metrics_path.read_text(encoding="utf-8"))
        if bool(stage1_metrics.get("smoke_test", True)):
            raise ValueError("MAD stage-1 metrics identify a smoke run")
    stage1_provenance = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics_path": (
            str(stage1_metrics_path.resolve())
            if stage1_metrics_path.is_file()
            else None
        ),
        "metrics_sha256": (
            sha256_file(stage1_metrics_path) if stage1_metrics_path.is_file() else None
        ),
        "method": "lad",
        "seed": seed,
        "pinn_update_steps": stage1_steps,
        "effective_schedule": stage1_schedule,
        "training_wall_sec": stage1_metrics.get("training_wall_sec"),
        "end_to_end_wall_sec": stage1_metrics.get("end_to_end_wall_sec"),
    }
    return (
        retained_mask,
        screening_metrics,
        stage1_provenance,
        absolute_residual,
    )


def sample_mad_batch(
    *,
    experiment,
    n_f: int,
    generator: torch.Generator,
    retained_mask: torch.Tensor,
):
    """Sample coordinate rows while preserving independent u/v scalar masks."""
    eligible_rows = torch.nonzero(retained_mask.any(dim=1), as_tuple=False).reshape(-1)
    if eligible_rows.numel() == 0:
        raise ValueError("MAD screening left no eligible coordinate rows")
    sample_positions = torch.randint(
        0,
        eligible_rows.numel(),
        (min(experiment.n_data_batch, eligible_rows.numel()),),
        generator=generator,
        device=experiment.device,
    )
    data_indices = eligible_rows[sample_positions]
    return {
        "X_f": experiment._sample_collocation(n_f, generator),
        "X_d": experiment.X_data[data_indices],
        "y_d": experiment.y_data[data_indices],
        "data_scalar_mask": retained_mask[data_indices],
    }


def initialize_estimator(
    *,
    model,
    experiment,
    estimator,
    running_std,
    steps: int,
    momentum: float,
    generator: torch.Generator,
    n_f: int,
    log_every: int,
) -> float:
    start = time.perf_counter()
    model.eval()
    for step in range(steps):
        batch = experiment.sample_batch(n_f=n_f, generator=generator)
        with torch.no_grad():
            residual = experiment.measurement_residual(model, batch).reshape(-1, 1)
            update_running_std(running_std, residual, momentum)
            scaled = residual / running_std
        _, mean_nll = estimator.train_step(scaled)
        finite_or_raise("estimator initialization NLL", mean_nll, step)
        if log_every > 0 and ((step + 1) % log_every == 0 or step + 1 == steps):
            print(
                f"estimator_init {step + 1}/{steps} "
                f"nll={float(mean_nll.cpu()):.6e} "
                f"std={float(running_std.cpu()):.6e}",
                flush=True,
            )
    return time.perf_counter() - start


def evaluate_gate(
    model,
    experiment,
    estimator,
    gate,
    running_std,
    label_semantics: dict[str, str],
):
    if gate is None:
        return {}
    model.eval()
    estimator.eval()
    gate.eval()
    with torch.no_grad():
        residual = (experiment.y_data - model(experiment.X_data)[:, :2]).reshape(-1, 1)
        log_density = estimator(residual / running_std)
        weights, _ = gate(log_density)
    metrics = {
        "retained_fraction": float((weights >= 0.5).float().mean().cpu()),
        "mean_gate_weight": float(weights.mean().cpu()),
        "n_gated_scalar_measurements": int(weights.numel()),
    }
    labels = experiment.data_corruption_labels.reshape(-1)
    if bool(labels.any()) and bool((~labels).any()):
        flat_weights = weights.reshape(-1)
        positive = labels
        negative = ~labels
        rejected = flat_weights < 0.5
        positive_count = int(positive.sum().cpu())
        negative_count = int(negative.sum().cpu())
        detection_scores = (1.0 - flat_weights).detach().cpu().numpy()
        label_array = positive.detach().cpu().numpy()
        order = np.argsort(detection_scores, kind="mergesort")
        sorted_scores = detection_scores[order]
        ranks = np.empty(order.size, dtype=np.float64)
        start = 0
        while start < order.size:
            stop = start + 1
            while stop < order.size and sorted_scores[stop] == sorted_scores[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
            start = stop
        positive_rank_sum = ranks[label_array].sum()
        auroc = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
            positive_count * negative_count
        )
        positive_name = label_semantics["positive"]
        negative_name = label_semantics["negative"]
        metrics.update(
            {
                f"known_{positive_name}_scalar_measurements": positive_count,
                f"known_{negative_name}_scalar_measurements": negative_count,
                f"{positive_name}_rejection_rate": float(
                    rejected[positive].float().mean().cpu()
                ),
                f"{negative_name}_rejection_rate": float(
                    rejected[negative].float().mean().cpu()
                ),
                f"{positive_name}_mean_gate_weight": float(
                    flat_weights[positive].mean().cpu()
                ),
                f"{negative_name}_mean_gate_weight": float(
                    flat_weights[negative].mean().cpu()
                ),
                label_semantics["auroc"]: float(auroc),
            }
        )
    return metrics


def rank_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary AUROC with average ranks for tied scores."""
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(order.size, dtype=np.float64)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = ranks[labels].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def evaluate_estimator(
    model,
    experiment,
    estimator,
    running_std,
    label_semantics: dict[str, str],
):
    """Evaluate raw EBM surprise independently of any trainable gate."""
    if estimator is None:
        return {}
    model.eval()
    estimator.eval()
    with torch.no_grad():
        residual = (experiment.y_data - model(experiment.X_data)[:, :2]).reshape(-1, 1)
        scaled_residual = residual / running_std
        log_density = estimator(scaled_residual).reshape(-1)
        normalization_grid = estimator.make_grid()
        grid_min = normalization_grid.min()
        grid_max = normalization_grid.max()
        outside_grid = (scaled_residual < grid_min) | (scaled_residual > grid_max)
    labels = experiment.data_corruption_labels.reshape(-1)
    metrics = {
        "estimator_mean_negative_log_density": float((-log_density).mean().cpu()),
        "n_estimator_scored_scalar_measurements": int(log_density.numel()),
        "estimator_running_std": float(running_std.cpu()),
        "estimator_scaled_residual_max_abs": float(
            scaled_residual.abs().max().cpu()
        ),
        "estimator_normalization_grid_min": float(grid_min.cpu()),
        "estimator_normalization_grid_max": float(grid_max.cpu()),
        "estimator_scaled_residual_outside_normalization_grid_count": int(
            outside_grid.sum().cpu()
        ),
        "estimator_scaled_residual_outside_normalization_grid_fraction": float(
            outside_grid.float().mean().cpu()
        ),
    }
    if bool(labels.any()) and bool((~labels).any()):
        score_array = (-log_density).detach().cpu().numpy()
        label_array = labels.detach().cpu().numpy()
        positive_name = label_semantics["positive"]
        negative_name = label_semantics["negative"]
        metrics.update(
            {
                f"estimator_{label_semantics['auroc']}": rank_auroc(
                    label_array, score_array
                ),
                f"estimator_{positive_name}_mean_negative_log_density": float(
                    (-log_density[labels]).mean().cpu()
                ),
                f"estimator_{negative_name}_mean_negative_log_density": float(
                    (-log_density[~labels]).mean().cpu()
                ),
            }
        )
    return metrics


def save_checkpoint(
    path: Path,
    *,
    model,
    experiment,
    estimator,
    gate,
    running_std,
    config,
    step,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "experiment": experiment.state_dict(),
        "config": config,
        "pinn_step": step,
    }
    if estimator is not None:
        checkpoint["estimator"] = estimator.state_dict()
    if gate is not None:
        checkpoint["gate"] = gate.state_dict()
    if running_std is not None:
        checkpoint["running_std"] = running_std.detach().cpu()
    torch.save(checkpoint, path)


def main(args: argparse.Namespace) -> Path:
    common = load_yaml(args.common_config)
    method_override = load_yaml(args.exp_config)
    config = deep_merge(common, method_override)
    if args.variant_config is not None:
        config = deep_merge(config, load_yaml(args.variant_config))
        suffix = str(config.get("variant_tag_suffix", ""))
        if suffix and not str(config["tag"]).endswith(suffix):
            config["tag"] = f"{config['tag']}{suffix}"
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    # Run artifacts are canonicalized under outputs/.  The experiment YAML
    # still records historical roots for provenance, but a fresh execution
    # must not silently return to analysis/results/runs.
    config.setdefault("output", {})["root"] = str(args.output_root)

    method = str(config["method"]["kind"]).lower()
    if method not in ALL_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose {sorted(ALL_METHODS)}")
    resolved_data_loss = resolve_data_loss(config, method)
    config["resolved_data_loss"] = resolved_data_loss
    input_artifact_path = Path(config["data"]["path"]).resolve()
    if not input_artifact_path.is_file():
        raise FileNotFoundError(
            f"RealPDEBench input artifact not found: {input_artifact_path}"
        )
    input_artifact_sha256 = sha256_file(input_artifact_path)
    config["data"]["input_artifact_sha256"] = input_artifact_sha256
    smoke_test = args.smoke_steps is not None
    if smoke_test and args.smoke_steps <= 0:
        raise ValueError("--smoke-steps must be positive")

    requested_device = str(config["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested_device}")
    device = torch.device(requested_device)
    seed = int(config["seed"])
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    model_config = load_yaml(args.model_config)
    model_config["in_features"] = int(config["in_features"])
    model_config["out_features"] = int(config["out_features"])
    experiment = get_experiment(config["experiment_name"])(config, device)
    model = get_model(config["model_name"])(model_config).to(device)

    training = config["training"]
    mad_retained_mask = None
    mad_screening_metrics: dict[str, Any] = {}
    mad_stage1_provenance: dict[str, Any] = {}
    mad_absolute_residual = None
    if method == MAD_METHOD:
        stage1_value = (
            args.stage1_checkpoint
            if args.stage1_checkpoint is not None
            else config.get("mad", {}).get("stage1_checkpoint")
        )
        if not stage1_value:
            raise ValueError(
                "MAD-PINN requires --stage1-checkpoint pointing to a completed "
                "30,000-update LAD final.pt"
            )
        (
            mad_retained_mask,
            mad_screening_metrics,
            mad_stage1_provenance,
            mad_absolute_residual,
        ) = load_mad_stage1_and_screen(
            checkpoint_path=Path(stage1_value),
            model=model,
            experiment=experiment,
            config=config,
            model_config=model_config,
            seed=seed,
        )
        warmup_steps = 0
        estimator_init_steps = 0
        joint_steps = (
            args.smoke_steps if smoke_test else int(config["mad"]["stage2_steps"])
        )
        if not smoke_test and joint_steps != 30000:
            raise ValueError("MAD-PINN stage 2 must use exactly 30,000 updates")
    elif method in BASELINE_METHODS:
        warmup_steps = 0
        estimator_init_steps = 0
        joint_steps = args.smoke_steps if smoke_test else int(training["total_steps"])
    else:
        warmup_steps = args.smoke_steps if smoke_test else int(training["warmup_steps"])
        estimator_init_steps = (
            args.smoke_steps if smoke_test else int(training["estimator_init_steps"])
        )
        joint_steps = args.smoke_steps if smoke_test else int(training["joint_steps"])
    effective_pinn_steps = warmup_steps + joint_steps
    if not smoke_test and effective_pinn_steps != int(training["total_steps"]):
        raise ValueError(
            "Staged schedule must preserve total PINN update budget: "
            f"{warmup_steps}+{joint_steps}!={training['total_steps']}"
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_label = args.run_name or f"seed_{seed}_{timestamp}"
    if smoke_test:
        run_label = f"smoke_{run_label}"
    out_dir = (
        Path(config["output"]["root"])
        / config["output"]["benchmark"]
        / str(config["tag"])
        / run_label
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    effective_config = {
        **config,
        "model": model_config,
        "effective_schedule": {
            "warmup_steps": warmup_steps,
            "estimator_init_steps": estimator_init_steps,
            "joint_steps": joint_steps,
            "pinn_update_steps": effective_pinn_steps,
            "smoke_test": smoke_test,
        },
    }
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(effective_config, stream, sort_keys=True)
    run_metadata = {
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "seed": seed,
        "method": method,
        "hardware": hardware_metadata(device),
        "git": git_metadata(),
        "data_metadata": experiment.metadata,
        "input_artifact_path": str(input_artifact_path),
        "input_artifact_sha256": input_artifact_sha256,
        "resolved_data_loss": resolved_data_loss,
        "software": package_metadata(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "smoke_test": smoke_test,
        "evidence_status": (
            "smoke_test_not_paper_evidence"
            if smoke_test
            else "full_run_pending_aggregation"
        ),
    }
    if method == MAD_METHOD:
        run_metadata["mad_stage1"] = mad_stage1_provenance
        run_metadata["mad_screening"] = mad_screening_metrics
    write_json(out_dir / "run_metadata.json", run_metadata)
    if method == MAD_METHOD:
        np.savez_compressed(
            out_dir / "mad_screening.npz",
            absolute_lad_residual=mad_absolute_residual,
            retained_scalar_mask=mad_retained_mask.detach().cpu().numpy(),
            known_corruption_labels=(
                experiment.data_corruption_labels.detach().cpu().numpy()
            ),
            sigma_hat=np.asarray(mad_screening_metrics["mad_sigma_hat"]),
            threshold=np.asarray(mad_screening_metrics["mad_threshold"]),
            stage1_checkpoint_sha256=np.asarray(
                mad_stage1_provenance["checkpoint_sha256"]
            ),
        )

    estimator = None
    gate = None
    running_std = None
    if method in STAGED_METHODS:
        estimator = create_density_estimator(
            config["estimator"], input_dim=1, device=device
        )
        if estimator.__class__.__name__ != "EBM":
            raise ValueError("Canonical real-data staged methods require EBM")
        running_std = torch.ones((), device=device)
    if method in NAPINN_METHODS:
        gate_cfg = config["gate"]
        gate = TrainableLikelihoodGate(
            init_cutoff_sigma=float(gate_cfg["init_cutoff_sigma"]),
            init_steepness=float(gate_cfg["init_steepness"]),
            device=device,
            rejection_cost=float(gate_cfg["rejection_cost"]),
        )

    n_f = int(config["batch"]["n_f"])
    pinn_parameters = list(model.parameters()) + list(experiment.extra_params())
    optimizer = torch.optim.Adam(
        pinn_parameters,
        lr=float(training["model_lr"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(seed + 1000)
    init_generator = torch.Generator(device=device)
    init_generator.manual_seed(seed + 2000)
    log_every = int(config["output"].get("log_every", 100))
    checkpoint_every = int(config["output"].get("checkpoint_every", 5000))
    history_path = out_dir / "train_history.jsonl"

    def run_phase(
        phase_name: str,
        phase_steps: int,
        phase_method: str,
        offset: int,
        *,
        phase_pde_weight: float | None = None,
    ):
        phase_start = time.perf_counter()
        with history_path.open("a", encoding="utf-8") as history:
            for local_step in range(phase_steps):
                global_step = offset + local_step
                if phase_method == MAD_METHOD:
                    batch = sample_mad_batch(
                        experiment=experiment,
                        n_f=n_f,
                        generator=train_generator,
                        retained_mask=mad_retained_mask,
                    )
                else:
                    batch = experiment.sample_batch(n_f=n_f, generator=train_generator)
                values = train_step(
                    method=phase_method,
                    model=model,
                    experiment=experiment,
                    optimizer=optimizer,
                    batch=batch,
                    pde_weight=(
                        float(training["pde_weight"])
                        if phase_pde_weight is None
                        else phase_pde_weight
                    ),
                    data_weight=float(training["data_weight"]),
                    estimator=estimator,
                    gate=gate,
                    running_std=running_std,
                    std_momentum=float(config["estimator"]["std_momentum"]),
                    estimator_weight=float(training["estimator_weight"]),
                    resolved_data_loss=resolved_data_loss,
                    step=global_step,
                )
                record = {
                    "phase": phase_name,
                    "phase_step": local_step + 1,
                    "pinn_step": global_step + 1,
                    **values,
                }
                history.write(json.dumps(record, sort_keys=True) + "\n")
                if log_every > 0 and (
                    (local_step + 1) % log_every == 0 or local_step + 1 == phase_steps
                ):
                    print(
                        f"{phase_name} {local_step + 1}/{phase_steps} "
                        f"total={values['loss_total']:.6e} "
                        f"pde={values['loss_pde']:.6e} "
                        f"data={values['loss_data']:.6e}",
                        flush=True,
                    )
                if checkpoint_every > 0 and (global_step + 1) % checkpoint_every == 0:
                    save_checkpoint(
                        out_dir / f"step_{global_step + 1}.pt",
                        model=model,
                        experiment=experiment,
                        estimator=estimator,
                        gate=gate,
                        running_std=running_std,
                        config=effective_config,
                        step=global_step + 1,
                    )
        return time.perf_counter() - phase_start

    end_to_end_start = time.perf_counter()
    phase_times = {"warmup_sec": 0.0, "estimator_init_sec": 0.0, "joint_sec": 0.0}
    if method in BASELINE_METHODS or method == MAD_METHOD:
        phase_times["joint_sec"] = run_phase(method, joint_steps, method, 0)
    else:
        phase_times["warmup_sec"] = run_phase("mse_warmup", warmup_steps, "mse", 0)
        phase_times["estimator_init_sec"] = initialize_estimator(
            model=model,
            experiment=experiment,
            estimator=estimator,
            running_std=running_std,
            steps=estimator_init_steps,
            momentum=float(config["estimator"]["std_momentum"]),
            generator=init_generator,
            n_f=n_f,
            log_every=log_every,
        )
        # A fresh optimizer mirrors the staged naPINN schedule. The internal
        # EBM optimizer is used only above and is never called after this point.
        parameter_groups = [
            {
                "params": (list(model.parameters()) + list(experiment.extra_params())),
                "lr": float(training["model_lr"]),
            },
            {
                "params": list(estimator.parameters()),
                "lr": float(config["estimator"]["lr"]),
            },
        ]
        if gate is not None:
            parameter_groups.append(
                {
                    "params": list(gate.parameters()),
                    "lr": float(config["gate"]["lr"]),
                }
            )
        optimizer = torch.optim.Adam(
            parameter_groups,
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        phase_times["joint_sec"] = run_phase(
            "joint",
            joint_steps,
            method,
            warmup_steps,
            phase_pde_weight=float(
                training.get("joint_pde_weight", training["pde_weight"])
            ),
        )

    training_wall_sec = sum(phase_times.values())
    save_checkpoint(
        out_dir / "final.pt",
        model=model,
        experiment=experiment,
        estimator=estimator,
        gate=gate,
        running_std=running_std,
        config=effective_config,
        step=effective_pinn_steps,
    )

    evaluation_start = time.perf_counter()
    field_metrics = experiment.eval_on_grid(model)
    physics_metrics = experiment.evaluate_physics(model)
    label_semantics = corruption_label_semantics(experiment.metadata)
    gate_metrics = evaluate_gate(
        model,
        experiment,
        estimator,
        gate,
        running_std,
        label_semantics,
    )
    estimator_metrics = evaluate_estimator(
        model,
        experiment,
        estimator,
        running_std,
        label_semantics,
    )
    effective_reynolds = float(experiment.effective_reynolds().detach().cpu())
    reynolds_metrics = {
        "metadata_reynolds": float(experiment.reynolds),
        "effective_reynolds": effective_reynolds,
        "effective_reynolds_relative_error": float(
            abs(effective_reynolds - experiment.reynolds) / experiment.reynolds
        ),
        "learned_reynolds": bool(experiment.learn_reynolds),
    }
    corruption_metrics = corruption_metrics_from_metadata(experiment.metadata)
    evaluation_sec = time.perf_counter() - evaluation_start
    metrics = {
        "status": "complete",
        "benchmark": str(experiment.metadata["dataset"]),
        "dataset_family": str(
            experiment.metadata.get("dataset_family", "cylinder")
        ),
        "method": method,
        "tag": str(config["tag"]),
        "resolved_data_loss": resolved_data_loss,
        "reconstruction_loss_kind": resolved_data_loss["kind"],
        "reconstruction_loss_q": resolved_data_loss["q"],
        "seed": seed,
        "smoke_test": smoke_test,
        "evidence_status": (
            "smoke_test_not_paper_evidence"
            if smoke_test
            else "full_run_complete_unaggregated"
        ),
        "sim_id": experiment.metadata["sim_id"],
        "physics_interpretation": experiment.metadata.get(
            "physics_interpretation",
            "Nominal pressure-latent 2-D incompressible Navier--Stokes.",
        ),
        "geometry_exclusion": experiment.metadata.get(
            "geometry_exclusion", experiment.metadata.get("cylinder_mask")
        ),
        "reference_scale_provenance": experiment.metadata.get(
            "reference_scale_provenance"
        ),
        "source_sha256": experiment.metadata["source_sha256"],
        "input_artifact_path": str(input_artifact_path),
        "input_artifact_sha256": input_artifact_sha256,
        "software": package_metadata(),
        "sensor_seed": experiment.metadata["sensor_seed"],
        "n_spatial_sensors": experiment.metadata["n_spatial_sensors"],
        "n_frames": experiment.metadata["n_frames"],
        "pinn_update_steps": effective_pinn_steps,
        "estimator_init_steps": estimator_init_steps,
        **field_metrics,
        **physics_metrics,
        **gate_metrics,
        **estimator_metrics,
        **reynolds_metrics,
        **corruption_metrics,
        **phase_times,
        "training_wall_sec": training_wall_sec,
        "evaluation_wall_sec": evaluation_sec,
        "end_to_end_wall_sec": time.perf_counter() - end_to_end_start,
    }
    if method == MAD_METHOD:
        stage1_training_wall = mad_stage1_provenance.get("training_wall_sec")
        metrics.update(
            {
                **mad_screening_metrics,
                "mad_stage1_checkpoint_path": mad_stage1_provenance["checkpoint_path"],
                "mad_stage1_checkpoint_sha256": mad_stage1_provenance[
                    "checkpoint_sha256"
                ],
                "mad_stage1_pinn_update_steps": mad_stage1_provenance[
                    "pinn_update_steps"
                ],
                "mad_stage2_pinn_update_steps": effective_pinn_steps,
                "mad_total_pipeline_pinn_update_steps": (
                    mad_stage1_provenance["pinn_update_steps"] + effective_pinn_steps
                ),
                "compute_matching_status": (
                    "additional_stage2_work_not_compute_matched_to_"
                    "single_30000_update_methods"
                ),
            }
        )
        if stage1_training_wall is not None:
            metrics["mad_pipeline_training_wall_sec"] = (
                float(stage1_training_wall) + training_wall_sec
            )
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        metrics.update(
            {
                "gpu_peak_memory_allocated_bytes": peak_allocated,
                "gpu_peak_memory_reserved_bytes": peak_reserved,
                "gpu_peak_memory_mb": peak_allocated / 1024**2,
            }
        )
    for key, value in metrics.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise FloatingPointError(f"Final metric {key} is non-finite: {value}")
    write_json(out_dir / "metrics.json", metrics)
    run_metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_metadata["status"] = "complete"
    run_metadata["evidence_status"] = metrics["evidence_status"]
    write_json(out_dir / "run_metadata.json", run_metadata)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Artifacts: {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-config",
        type=Path,
        default=Path("configs/experiment/realpdebench_cylinder_common.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/mlp.yaml"),
    )
    parser.add_argument("--exp-config", type=Path, required=True)
    parser.add_argument(
        "--variant-config",
        "--experiment-config",
        dest="variant_config",
        type=Path,
        help=(
            "Optional dataset/condition experiment overlay merged after the "
            "method config. --experiment-config is the generic alias."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/realpde"),
        help=(
            "Root for checkpoints, metrics, resolved configs, histories, and "
            "run metadata. Fresh runs default to outputs/."
        ),
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        help=("Completed 30,000-update LAD final.pt required by MAD-PINN stage 2."),
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help=(
            "Run this many steps per active stage and label all outputs as "
            "smoke-test artifacts, never paper evidence."
        ),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
