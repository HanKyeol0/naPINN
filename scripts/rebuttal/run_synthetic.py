#!/usr/bin/env python3
"""Auditable synthetic-PDE rebuttal runner.

The runner freezes identical corrupted observations across methods by seeding
before experiment construction. It also keeps the two EBM optimization paths
explicit:

* ``pinn_ebm``: EBM negative log likelihood is the PINN data loss and its
  gradient jointly updates PINN, EBM, and the unknown PDE parameter.
* ``napinn``: the EBM is updated by its own NLL on detached residuals; the PINN
  is updated by gate-weighted MSE plus the rejection-cost regularizer.

Outputs are written to a unique method/noise/ratio/seed directory. Existing
completed outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from pinnlab.registry import get_experiment, get_model
from pinnlab.utils.data_loss import data_loss_q_gaussian
from pinnlab.utils.seed import seed_everything


METHODS = {
    "mse",
    "lad",
    "orpinn_q19",
    "orpinn_q29",
    "pinn_ebm",
    "napinn",
    "napinn_lad",
    "napinn_q29",
    "quantile",
    "threshold",
}
NAPINN_METHODS = {"napinn", "napinn_lad", "napinn_q29"}
STAGED_METHODS = {"pinn_ebm", *NAPINN_METHODS, "quantile", "threshold"}
OUTLIER_POINTS = {
    "allencahn2d": {0.05: 1120, 0.10: 2250, 0.15: 3375},
    "burgers2d": {0.05: 1690, 0.10: 3360, 0.15: 5030},
    "lambdaomega2d": {0.05: 2800, 0.10: 5700, 0.15: 8450},
}
TRUE_PARAMETER = {
    "allencahn2d": ("eps", 0.3),
    "burgers2d": ("nu", 0.01),
    "lambdaomega2d": ("beta", 1.0),
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_metadata() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_experiment(
    config: dict[str, Any],
    *,
    experiment_name: str,
    method: str,
    device: str,
    outlier_ratio: float,
    noise_kind: str,
    ema_momentum: float,
    rejection_cost: float,
    estimator_init_steps: int,
) -> dict[str, Any]:
    config["device"] = device
    ratio_key = min(
        OUTLIER_POINTS[experiment_name],
        key=lambda key: abs(key - outlier_ratio),
    )
    if not math.isclose(ratio_key, outlier_ratio, abs_tol=1.0e-9):
        raise ValueError(f"Unsupported outlier ratio: {outlier_ratio}")
    config["noise"]["extra_noise"]["n_points"] = OUTLIER_POINTS[
        experiment_name
    ][ratio_key]
    config["noise"]["kind"] = noise_kind
    if noise_kind == "G":
        config["noise"]["pars"] = [0.0, 4.0]
    elif noise_kind == "Laplace":
        config["noise"]["pars"] = [0.0, 4.0]
    elif noise_kind == "StudentT":
        config["noise"]["pars"] = [3.0, 0.0, 1.0]
    elif noise_kind != "4G":
        raise ValueError(f"Unsupported noise kind: {noise_kind}")

    config["ebm"]["momentum"] = ema_momentum
    config["ebm"]["init_train_epochs"] = estimator_init_steps
    config["data_loss_balancer"]["rejection_cost"] = rejection_cost
    config["phase"]["enabled"] = method in STAGED_METHODS

    if method in NAPINN_METHODS:
        config["ebm"]["enabled"] = True
        if method == "napinn_lad":
            config["data_loss"]["kind"] = "L1"
        elif method == "napinn_q29":
            config["data_loss"]["kind"] = "q_gaussian"
            config["data_loss"]["q"] = 2.9
        else:
            config["data_loss"]["kind"] = "mse"
        config["data_loss_balancer"]["use_loss_balancer"] = True
        config["data_loss_balancer"]["kind"] = "gated_trainable"
    elif method == "pinn_ebm":
        config["ebm"]["enabled"] = True
        config["data_loss"]["kind"] = "mse"
        config["data_loss_balancer"]["use_loss_balancer"] = False
    elif method == "threshold":
        config["ebm"]["enabled"] = False
        config["data_loss"]["kind"] = "mse"
        config["data_loss_balancer"]["use_loss_balancer"] = True
        config["data_loss_balancer"]["kind"] = "threshold"
    elif method == "quantile":
        config["ebm"]["enabled"] = False
        config["data_loss"]["kind"] = "mse"
        config["data_loss_balancer"]["use_loss_balancer"] = True
        config["data_loss_balancer"]["kind"] = "quantile"
    else:
        config["ebm"]["enabled"] = False
        config["data_loss_balancer"]["use_loss_balancer"] = False
        if method == "lad":
            config["data_loss"]["kind"] = "L1"
        elif method.startswith("orpinn"):
            config["data_loss"]["kind"] = "q_gaussian"
            config["data_loss"]["q"] = (
                1.9 if method == "orpinn_q19" else 2.9
            )
        else:
            config["data_loss"]["kind"] = "mse"
    return config


def residual_from_batch(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["y_d"] - model(batch["X_d"])


def update_running_std(
    running_std: torch.Tensor,
    residual: torch.Tensor,
    momentum: float,
) -> None:
    with torch.no_grad():
        current = residual.detach().std(unbiased=True)
        current = torch.clamp(
            current,
            min=1.0e-6,
            max=torch.clamp(running_std * 10.0, min=1.0e-5),
        )
        running_std.mul_(1.0 - momentum).add_(momentum * current)


def baseline_data_loss(method: str, residual: torch.Tensor) -> torch.Tensor:
    if method == "mse":
        return residual.square().mean()
    if method == "lad":
        return residual.abs().mean()
    if method == "orpinn_q19":
        return data_loss_q_gaussian(residual, q=1.9).mean()
    if method == "orpinn_q29":
        return data_loss_q_gaussian(residual, q=2.9).mean()
    raise ValueError(method)


def make_optimizer(
    model,
    experiment,
    *,
    method: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    parameters = list(model.parameters()) + list(experiment.extra_params())
    if method == "pinn_ebm":
        parameters += list(experiment.ebm.parameters())
    return torch.optim.Adam(parameters, lr=learning_rate)


def train_range(
    *,
    label: str,
    steps: int,
    method: str,
    phase: int,
    model,
    experiment,
    optimizer,
    n_f: int,
    pde_weight: float,
    data_weight: float,
    start_step: int,
) -> None:
    for local_step in range(steps):
        step = start_step + local_step + 1
        model.train()
        batch = experiment.sample_batch(n_f=n_f)
        pde_loss = experiment.pde_residual_loss(model, batch).mean()
        residual = residual_from_batch(model, batch)

        if phase == 1:
            data_loss = residual.square().mean()
        elif method in {"mse", "lad", "orpinn_q19", "orpinn_q29"}:
            data_loss = baseline_data_loss(method, residual)
        elif method == "pinn_ebm":
            flat = residual.reshape(-1, 1)
            update_running_std(
                experiment.running_std,
                flat,
                experiment.momentum,
            )
            scaled = flat / experiment.running_std.detach()
            _, data_loss = experiment.ebm.mean_nll(
                scaled,
                detach_residual=False,
            )
        elif method in {*NAPINN_METHODS, "quantile", "threshold"}:
            data_loss = experiment.data_loss(model, batch, phase=2).mean()
        else:
            raise ValueError(method)

        total = pde_weight * pde_loss + data_weight * data_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(
                f"Non-finite total loss in {label} at step {step}: {total}"
            )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        if step == 1 or step % 1000 == 0 or local_step + 1 == steps:
            payload = {
                "phase": label,
                "step": step,
                "loss_total": float(total.detach().cpu()),
                "loss_pde": float(pde_loss.detach().cpu()),
                "loss_data": float(data_loss.detach().cpu()),
            }
            parameter_name, _ = TRUE_PARAMETER[experiment_name_for(experiment)]
            parameter = getattr(experiment, parameter_name)
            if isinstance(parameter, torch.Tensor):
                payload[f"pde_{parameter_name}"] = float(
                    parameter.detach().cpu()
                )
            print(json.dumps(payload, sort_keys=True), flush=True)


def experiment_name_for(experiment) -> str:
    class_name = experiment.__class__.__name__
    mapping = {
        "AllenCahn2D": "allencahn2d",
        "Burgers2D": "burgers2d",
        "LambdaOmega2D": "lambdaomega2d",
    }
    return mapping[class_name]


def gate_metrics(model, experiment) -> dict[str, Any]:
    if experiment.gate_module is None or experiment.ebm is None:
        return {}
    model.eval()
    experiment.ebm.eval()
    with torch.no_grad():
        residual = (experiment.y_data - model(experiment.X_data)).reshape(-1, 1)
        scaled = residual / experiment.running_std
        log_density = experiment.ebm(scaled)
        weights, _ = experiment.gate_module(log_density)
    labels = np.zeros(experiment.y_data.shape, dtype=bool)
    indices = np.asarray(getattr(experiment, "outlier_indices", []), dtype=int)
    if indices.size:
        labels[indices, :] = True
    flat_labels = labels.reshape(-1)
    flat_weights = weights.detach().cpu().numpy().reshape(-1)
    metrics: dict[str, Any] = {
        "mean_gate_weight": float(flat_weights.mean()),
        "rejected_fraction_at_0_5": float((flat_weights < 0.5).mean()),
    }
    if flat_labels.any() and (~flat_labels).any():
        metrics.update(
            {
                "outlier_detection_auroc": float(
                    roc_auc_score(flat_labels, 1.0 - flat_weights)
                ),
                "known_outlier_rejection_rate": float(
                    (flat_weights[flat_labels] < 0.5).mean()
                ),
                "known_clean_rejection_rate": float(
                    (flat_weights[~flat_labels] < 0.5).mean()
                ),
            }
        )
    return metrics


def estimator_score_metrics(model, experiment) -> dict[str, Any]:
    """Evaluate raw learned-density scores for both gated and no-gate EBM."""
    if experiment.ebm is None:
        return {}
    model.eval()
    experiment.ebm.eval()
    with torch.no_grad():
        residual = (experiment.y_data - model(experiment.X_data)).reshape(-1, 1)
        log_density = experiment.ebm(
            residual / experiment.running_std
        ).reshape(-1)
    labels = np.zeros(experiment.y_data.shape, dtype=bool)
    indices = np.asarray(getattr(experiment, "outlier_indices", []), dtype=int)
    if indices.size:
        labels[indices, :] = True
    flat_labels = labels.reshape(-1)
    anomaly_score = (-log_density).detach().cpu().numpy()
    result = {
        "mean_negative_log_unnormalized_density": float(anomaly_score.mean()),
    }
    if flat_labels.any() and (~flat_labels).any():
        result.update(
            {
                "outlier_detection_auroc": float(
                    roc_auc_score(flat_labels, anomaly_score)
                ),
                "known_outlier_mean_score": float(
                    anomaly_score[flat_labels].mean()
                ),
                "known_clean_mean_score": float(
                    anomaly_score[~flat_labels].mean()
                ),
            }
        )
    return result


def screening_metrics(model, experiment) -> dict[str, Any]:
    """Evaluate the final explicit selector, including fixed ablations."""
    if not bool(getattr(experiment, "use_data_loss_balancer", False)):
        return {}
    if not hasattr(experiment, "_get_weights"):
        return {}
    model.eval()
    for attribute in ("gate_module", "threshold_gate", "quantile_gate"):
        module = getattr(experiment, attribute, None)
        if module is not None:
            module.eval()
    with torch.no_grad():
        residual = (
            experiment.y_data - model(experiment.X_data)
        ).reshape(-1, 1)
        scaled = residual / experiment.running_std
        weights, _ = experiment._get_weights(scaled)
    labels = np.zeros(experiment.y_data.shape, dtype=bool)
    indices = np.asarray(getattr(experiment, "outlier_indices", []), dtype=int)
    if indices.size:
        labels[indices, :] = True
    flat_labels = labels.reshape(-1)
    flat_weights = weights.detach().cpu().numpy().reshape(-1)
    result: dict[str, Any] = {
        "kind": str(experiment.data_loss_balancer_kind),
        "mean_weight": float(flat_weights.mean()),
        "rejected_fraction_at_0_5": float((flat_weights < 0.5).mean()),
    }
    if flat_labels.any() and (~flat_labels).any():
        result.update(
            {
                "outlier_detection_auroc": float(
                    roc_auc_score(flat_labels, 1.0 - flat_weights)
                ),
                "known_outlier_rejection_rate": float(
                    (flat_weights[flat_labels] < 0.5).mean()
                ),
                "known_clean_rejection_rate": float(
                    (flat_weights[~flat_labels] < 0.5).mean()
                ),
            }
        )
    return result


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_bytes": props.total_memory,
            }
        )
    return result


def main(args: argparse.Namespace) -> None:
    if args.method not in METHODS:
        raise ValueError(args.method)
    if not torch.cuda.is_available():
        raise RuntimeError("Rebuttal experiments require CUDA.")
    if not args.device.startswith("cuda:"):
        raise ValueError("--device must be an explicit CUDA device.")

    common = load_yaml(args.common_config)
    model_config = load_yaml(args.model_config)
    experiment_config = configure_experiment(
        load_yaml(args.experiment_config),
        experiment_name=args.experiment_name,
        method=args.method,
        device=args.device,
        outlier_ratio=args.outlier_ratio,
        noise_kind=args.noise_kind,
        ema_momentum=args.ema_momentum,
        rejection_cost=args.rejection_cost,
        estimator_init_steps=args.estimator_init_steps,
    )
    common["seed"] = args.seed
    common["device"] = args.device
    model_config["in_features"] = experiment_config["in_features"]
    model_config["out_features"] = experiment_config["out_features"]

    ratio_label = f"{int(round(100 * args.outlier_ratio)):02d}pct"
    configured_pde_weight = float(common["train"]["loss_weights"]["res"])
    configured_data_weight = float(common["train"]["loss_weights"]["data"])
    pde_weight = (
        configured_pde_weight
        if args.pde_weight is None
        else float(args.pde_weight)
    )
    data_weight = (
        configured_data_weight
        if args.data_weight is None
        else float(args.data_weight)
    )
    sensitivity_label = (
        f"ema_{args.ema_momentum:g}_reject_{args.rejection_cost:g}"
        f"_pde_{pde_weight:g}_data_{data_weight:g}"
    )
    output_dir = (
        args.output_root
        / args.experiment_name
        / args.noise_kind
        / ratio_label
        / args.method
        / sensitivity_label
        / f"seed_{args.seed}"
    )
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"Completed output already exists: {metrics_path}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite incomplete non-empty run: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    experiment = get_experiment(args.experiment_name)(
        experiment_config,
        device,
    )
    # EBM construction consumes RNG only for methods that instantiate it.
    # Reseeding here keeps the PINN initialization and subsequent batch draws
    # paired across methods while preserving the already-created observations.
    seed_everything(args.seed)
    model = get_model(args.model_name)(model_config).to(device)
    n_f = int(experiment_config["batch"]["n_f"])
    learning_rate = float(common["train"]["optimizer"]["lr"])
    resolved_config = {
        "common": common,
        "model": model_config,
        "experiment": experiment_config,
        "run": {
            "method": args.method,
            "seed": args.seed,
            "device": args.device,
            "noise_kind": args.noise_kind,
            "outlier_ratio": args.outlier_ratio,
            "ema_momentum": args.ema_momentum,
            "rejection_cost": args.rejection_cost,
            "pde_weight": pde_weight,
            "data_weight": data_weight,
            "warmup_steps": args.warmup_steps,
            "joint_steps": args.joint_steps,
            "baseline_steps": args.baseline_steps,
            "estimator_init_steps": args.estimator_init_steps,
        },
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved_config, stream, sort_keys=False)

    phase_times: dict[str, float] = {}
    sync(device)
    training_start = time.perf_counter()

    if args.method in STAGED_METHODS:
        optimizer = make_optimizer(
            model,
            experiment,
            method="napinn" if args.method in NAPINN_METHODS else "threshold",
            learning_rate=learning_rate,
        )
        sync(device)
        phase_start = time.perf_counter()
        train_range(
            label="warmup",
            steps=args.warmup_steps,
            method=args.method,
            phase=1,
            model=model,
            experiment=experiment,
            optimizer=optimizer,
            n_f=n_f,
            pde_weight=pde_weight,
            data_weight=data_weight,
            start_step=0,
        )
        sync(device)
        phase_times["warmup_seconds"] = time.perf_counter() - phase_start

        sync(device)
        phase_start = time.perf_counter()
        if args.method in {*NAPINN_METHODS, "pinn_ebm"}:
            experiment.initialize_EBM(model)
        sync(device)
        phase_times["estimator_initialization_seconds"] = (
            time.perf_counter() - phase_start
        )

        optimizer = make_optimizer(
            model,
            experiment,
            method=args.method,
            learning_rate=learning_rate,
        )
        sync(device)
        phase_start = time.perf_counter()
        train_range(
            label="joint",
            steps=args.joint_steps,
            method=args.method,
            phase=2,
            model=model,
            experiment=experiment,
            optimizer=optimizer,
            n_f=n_f,
            pde_weight=pde_weight,
            data_weight=data_weight,
            start_step=args.warmup_steps,
        )
        sync(device)
        phase_times["joint_seconds"] = time.perf_counter() - phase_start
        pinn_updates = args.warmup_steps + args.joint_steps
    else:
        optimizer = make_optimizer(
            model,
            experiment,
            method=args.method,
            learning_rate=learning_rate,
        )
        sync(device)
        phase_start = time.perf_counter()
        train_range(
            label="ordinary",
            steps=args.baseline_steps,
            method=args.method,
            phase=0,
            model=model,
            experiment=experiment,
            optimizer=optimizer,
            n_f=n_f,
            pde_weight=pde_weight,
            data_weight=data_weight,
            start_step=0,
        )
        sync(device)
        phase_times["ordinary_training_seconds"] = (
            time.perf_counter() - phase_start
        )
        pinn_updates = args.baseline_steps

    sync(device)
    training_seconds = time.perf_counter() - training_start
    checkpoint = {
        "model": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "experiment": experiment.state_dict(),
        "resolved_config": resolved_config,
    }
    torch.save(checkpoint, output_dir / "final.pt")

    evaluation_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        evaluation = experiment.eval_on_grid(model, common["eval"]["grid"])
    parameter_name, true_value = TRUE_PARAMETER[args.experiment_name]
    learned_parameter = getattr(experiment, parameter_name)
    learned_value = float(
        learned_parameter.detach().cpu()
        if isinstance(learned_parameter, torch.Tensor)
        else learned_parameter
    )
    result = {
        "status": "complete",
        "experiment_name": args.experiment_name,
        "method": args.method,
        "seed": args.seed,
        "noise_kind": args.noise_kind,
        "outlier_ratio": args.outlier_ratio,
        "outlier_points": experiment_config["noise"]["extra_noise"]["n_points"],
        "pde_loss_weight": pde_weight,
        "data_loss_weight": data_weight,
        "field_rMAE": float(evaluation["rMAE"]),
        "field_rMSE": float(evaluation["rMSE"]),
        "pde_parameter_name": parameter_name,
        "pde_parameter_true": true_value,
        "pde_parameter_learned": learned_value,
        "pde_parameter_absolute_error": abs(learned_value - true_value),
        "pinn_update_steps": pinn_updates,
        "estimator_only_steps": (
            args.estimator_init_steps
            if args.method in {*NAPINN_METHODS, "pinn_ebm"}
            else 0
        ),
        "training_seconds": training_seconds,
        "evaluation_seconds": time.perf_counter() - evaluation_start,
        "phase_times": phase_times,
        "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(
            device
        ),
        "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(
            device
        ),
        "gate": gate_metrics(model, experiment),
        "screening": screening_metrics(model, experiment),
        "estimator_scores": estimator_score_metrics(model, experiment),
        "hardware": hardware_metadata(device),
        "git": git_metadata(),
    }
    write_json(metrics_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        required=True,
        choices=sorted(OUTLIER_POINTS),
    )
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--noise-kind", default="4G")
    parser.add_argument("--outlier-ratio", type=float, default=0.15)
    parser.add_argument("--ema-momentum", type=float, default=0.05)
    parser.add_argument("--rejection-cost", type=float, default=0.5)
    parser.add_argument("--pde-weight", type=float)
    parser.add_argument("--data-weight", type=float)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--joint-steps", type=int, default=25000)
    parser.add_argument("--baseline-steps", type=int, default=30000)
    parser.add_argument("--estimator-init-steps", type=int, default=5000)
    parser.add_argument(
        "--common-config",
        type=Path,
        default=Path("configs/rebuttal/common.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/rebuttal/model_mlp.yaml"),
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--model-name", default="mlp")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/results/runs/rebuttal_synthetic"),
    )
    main(parser.parse_args())
