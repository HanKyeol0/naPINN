#!/usr/bin/env python3
"""One-step forward/backward smoke for every dataset/method configuration.

This is a component-health check only.  It does not run evaluation, does not
produce a training checkpoint, and is explicitly marked non-evidentiary.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinnlab.registry import get_experiment, get_model
from pinnlab.utils.density import create_density_estimator
from pinnlab.utils.ebm import TrainableLikelihoodGate
from pinnlab.utils.seed import seed_everything
from scripts.rebuttal.run_realpdebench import (
    NAPINN_METHODS,
    STAGED_METHODS,
    deep_merge,
    load_yaml,
    resolve_data_loss,
    train_step,
)
from scripts.rebuttal.run_realpdebench_multidataset_queue import (
    MANIFEST,
    build_jobs,
    load_protocol,
)


def smoke(device_name: str) -> dict:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(20260726)
    protocol = load_protocol(MANIFEST)
    jobs = [
        job
        for job in build_jobs(protocol)
        if job.seed == 40 and job.ratio_percent == 10
    ]
    if len(jobs) != 24:
        raise AssertionError("Expected 3 datasets x 8 methods for smoke")
    common = load_yaml(Path(protocol["runner"]["common_config"]))
    model_base = load_yaml(Path(protocol["runner"]["model_config"]))
    records = []
    for job in jobs:
        config = deep_merge(common, load_yaml(Path(job.method_config)))
        config = deep_merge(config, load_yaml(Path(job.data_overlay)))
        config["seed"] = 40
        config["device"] = device_name
        config["batch"]["n_data"] = 8
        method = str(config["method"]["kind"]).lower()
        resolved = resolve_data_loss(config, method)
        experiment = get_experiment(config["experiment_name"])(config, device)
        model_config = {
            **model_base,
            "in_features": int(config["in_features"]),
            "out_features": int(config["out_features"]),
        }
        model = get_model(config["model_name"])(model_config).to(device)
        estimator = None
        gate = None
        running_std = None
        if method in STAGED_METHODS:
            estimator = create_density_estimator(
                config["estimator"], input_dim=1, device=device
            )
            running_std = torch.ones((), device=device)
        if method in NAPINN_METHODS:
            gate = TrainableLikelihoodGate(
                init_cutoff_sigma=float(config["gate"]["init_cutoff_sigma"]),
                init_steepness=float(config["gate"]["init_steepness"]),
                device=device,
                rejection_cost=float(config["gate"]["rejection_cost"]),
            )
        parameters = list(model.parameters()) + list(experiment.extra_params())
        if estimator is not None:
            parameters += list(estimator.parameters())
        if gate is not None:
            parameters += list(gate.parameters())
        optimizer = torch.optim.Adam(parameters, lr=1.0e-3)
        generator = torch.Generator(device=device)
        generator.manual_seed(4000)
        batch = experiment.sample_batch(n_f=4, generator=generator)
        values = train_step(
            method=method,
            model=model,
            experiment=experiment,
            optimizer=optimizer,
            batch=batch,
            pde_weight=float(
                config["training"].get(
                    "joint_pde_weight", config["training"]["pde_weight"]
                )
            ),
            data_weight=float(config["training"]["data_weight"]),
            estimator=estimator,
            gate=gate,
            running_std=running_std,
            std_momentum=float(config["estimator"]["std_momentum"]),
            estimator_weight=float(config["training"]["estimator_weight"]),
            resolved_data_loss=resolved,
            step=0,
        )
        if not all(np.isfinite(float(value)) for value in values.values()):
            raise FloatingPointError(f"Non-finite smoke values for {job.key}")
        records.append(
            {
                "dataset": job.dataset,
                "method": job.method,
                "runner_method": method,
                "input_sha256": job.input_sha256,
                "losses": values,
            }
        )
        del experiment, model, estimator, gate, optimizer, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "status": "complete",
        "evidence_status": "component_smoke_not_performance_evidence",
        "device": device_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "record_count": len(records),
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = smoke(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "device": result["device"],
                "record_count": result["record_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
