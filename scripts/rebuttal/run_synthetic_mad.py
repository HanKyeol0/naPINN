#!/usr/bin/env python3
"""Faithful two-stage MAD-PINN on a completed synthetic LAD-PINN run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pinnlab.registry import get_experiment, get_model
from pinnlab.utils.seed import seed_everything
from scripts.rebuttal.run_synthetic import (
    TRUE_PARAMETER,
    gate_metrics,
    git_metadata,
    hardware_metadata,
    sync,
    write_json,
)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main(args):
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("MAD-PINN rebuttal runs require an explicit CUDA device")
    smoke_test = args.stage2_steps != 30000
    if smoke_test and not args.allow_smoke:
        raise ValueError(
            "Noncanonical MAD stage-two length requires --allow-smoke"
        )
    stage1_dir = args.lad_checkpoint.parent
    stage1_metrics_path = stage1_dir / "metrics.json"
    stage1_config_path = stage1_dir / "config.yaml"
    if not args.lad_checkpoint.is_file():
        raise FileNotFoundError(args.lad_checkpoint)
    if not stage1_metrics_path.is_file() or not stage1_config_path.is_file():
        raise FileNotFoundError("LAD checkpoint lacks config.yaml or metrics.json")
    stage1_metrics = json.loads(stage1_metrics_path.read_text(encoding="utf-8"))
    resolved = load_yaml(stage1_config_path)
    run = resolved["run"]
    if run["method"] != "lad":
        raise ValueError("MAD stage 1 must be LAD-PINN")
    if int(run["seed"]) != args.seed:
        raise ValueError("MAD and LAD seeds differ")
    if int(stage1_metrics["pinn_update_steps"]) != 30000:
        raise ValueError("MAD stage 1 must contain 30,000 LAD updates")
    if stage1_metrics["experiment_name"] != args.experiment_name:
        raise ValueError("MAD and LAD benchmarks differ")

    output_dir = (
        args.output_root
        / args.experiment_name
        / str(run["noise_kind"])
        / f"{int(round(100 * float(run['outlier_ratio']))):02d}pct"
        / "mad_pinn"
        / "ema_0.05_reject_0.5_pde_1_data_1"
        / f"seed_{args.seed}"
    )
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"Completed output already exists: {metrics_path}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(args.seed)
    experiment_config = resolved["experiment"]
    experiment_config["device"] = args.device
    experiment = get_experiment(args.experiment_name)(
        experiment_config, device
    )
    seed_everything(args.seed)
    model_config = resolved["model"]
    model = get_model(args.model_name)(model_config).to(device)
    checkpoint = torch.load(
        args.lad_checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    experiment.load_state_dict(checkpoint["experiment"])

    model.eval()
    with torch.no_grad():
        residual = experiment.y_data - model(experiment.X_data)
    absolute_residual = residual.abs()
    median_absolute_residual = float(
        torch.median(absolute_residual).detach().cpu()
    )
    sigma_hat = median_absolute_residual / args.sigma_divisor
    threshold = args.threshold_multiplier * sigma_hat
    retained_mask = absolute_residual <= threshold
    eligible_rows = torch.nonzero(
        retained_mask.any(dim=1), as_tuple=False
    ).reshape(-1)
    if eligible_rows.numel() == 0:
        raise ValueError("MAD screen rejected every coordinate row")

    labels = np.zeros(experiment.y_data.shape, dtype=bool)
    outlier_indices = np.asarray(
        getattr(experiment, "outlier_indices", []), dtype=int
    )
    if outlier_indices.size:
        labels[outlier_indices, :] = True
    retained_numpy = retained_mask.detach().cpu().numpy()
    screening = {
        "mad_median_absolute_residual": median_absolute_residual,
        "mad_sigma_divisor": args.sigma_divisor,
        "mad_sigma_hat": sigma_hat,
        "mad_threshold_multiplier": args.threshold_multiplier,
        "mad_threshold": threshold,
        "mad_retained_fraction": float(retained_numpy.mean()),
        "mad_retained_scalar_measurements": int(retained_numpy.sum()),
        "mad_total_scalar_measurements": int(retained_numpy.size),
        "mad_known_outlier_rejection_rate": float(
            (~retained_numpy)[labels].mean()
        ),
        "mad_known_clean_rejection_rate": float(
            (~retained_numpy)[~labels].mean()
        ),
    }
    np.savez_compressed(
        output_dir / "mad_screening.npz",
        absolute_lad_residual=absolute_residual.detach().cpu().numpy(),
        retained_scalar_mask=retained_numpy,
        known_outlier_labels=labels,
        sigma_hat=np.asarray(sigma_hat),
        threshold=np.asarray(threshold),
    )

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(experiment.extra_params()),
        lr=float(resolved["common"]["train"]["optimizer"]["lr"]),
    )
    n_f = int(experiment_config["batch"]["n_f"])
    n_data_batch = int(experiment.n_data_batch)
    sync(device)
    start = time.perf_counter()
    for step in range(args.stage2_steps):
        model.train()
        batch = experiment.sample_batch(n_f=n_f)
        sample_positions = torch.randint(
            0,
            eligible_rows.numel(),
            (min(n_data_batch, eligible_rows.numel()),),
            device=device,
        )
        data_indices = eligible_rows[sample_positions]
        batch["X_d"] = experiment.X_data[data_indices]
        batch["y_d"] = experiment.y_data[data_indices]
        scalar_mask = retained_mask[data_indices]
        pde_loss = experiment.pde_residual_loss(model, batch).mean()
        stage2_residual = batch["y_d"] - model(batch["X_d"])
        data_loss = stage2_residual.square()[scalar_mask].mean()
        total = pde_loss + data_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 1000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "mad_mse",
                        "step": step + 1,
                        "loss_total": float(total.detach().cpu()),
                        "loss_pde": float(pde_loss.detach().cpu()),
                        "loss_data": float(data_loss.detach().cpu()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    sync(device)
    stage2_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        field = experiment.eval_on_grid(
            model, resolved["common"]["eval"]["grid"]
        )
    parameter_name, true_value = TRUE_PARAMETER[args.experiment_name]
    parameter = getattr(experiment, parameter_name)
    learned = float(parameter.detach().cpu())
    final_checkpoint = {
        "model": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "experiment": experiment.state_dict(),
        "stage1_checkpoint": str(args.lad_checkpoint.resolve()),
        "screening": screening,
    }
    torch.save(final_checkpoint, output_dir / "final.pt")
    stage1_training = float(stage1_metrics["training_seconds"])
    metrics = {
        "status": "complete",
        "evidence_status": (
            "smoke_test_not_rebuttal_evidence"
            if smoke_test
            else "full_run_complete_unaggregated"
        ),
        "experiment_name": args.experiment_name,
        "method": "mad_pinn",
        "seed": args.seed,
        "noise_kind": run["noise_kind"],
        "outlier_ratio": float(run["outlier_ratio"]),
        "outlier_points": int(stage1_metrics["outlier_points"]),
        "pde_loss_weight": 1.0,
        "data_loss_weight": 1.0,
        "field_rMAE": float(field["rMAE"]),
        "field_rMSE": float(field["rMSE"]),
        "pde_parameter_name": parameter_name,
        "pde_parameter_true": true_value,
        "pde_parameter_learned": learned,
        "pde_parameter_absolute_error": abs(learned - true_value),
        "pinn_update_steps": 30000 + args.stage2_steps,
        "mad_stage1_pinn_updates": 30000,
        "mad_stage2_pinn_updates": args.stage2_steps,
        "estimator_only_steps": 0,
        "training_seconds": stage1_training + stage2_seconds,
        "mad_stage2_training_seconds": stage2_seconds,
        "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(
            device
        ),
        "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(
            device
        ),
        "screening": screening,
        "gate": gate_metrics(model, experiment),
        "hardware": hardware_metadata(device),
        "git": git_metadata(),
        "stage1_metrics_path": str(stage1_metrics_path.resolve()),
        "stage1_checkpoint_path": str(args.lad_checkpoint.resolve()),
        "compute_matching_status": (
            f"not_compute_matched; 30000 LAD plus {args.stage2_steps} "
            "retained-MSE updates"
        ),
    }
    write_json(metrics_path, metrics)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "stage1_resolved_config": resolved,
                "mad": {
                    "sigma_divisor": args.sigma_divisor,
                    "threshold_multiplier": args.threshold_multiplier,
                    "stage2_steps": args.stage2_steps,
                },
            },
            stream,
            sort_keys=False,
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        required=True,
        choices=sorted(TRUE_PARAMETER),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lad-checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="mlp")
    parser.add_argument("--sigma-divisor", type=float, default=1.6777)
    parser.add_argument("--threshold-multiplier", type=float, default=3.0)
    parser.add_argument("--stage2-steps", type=int, default=30000)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/synthetic"),
    )
    main(parser.parse_args())
