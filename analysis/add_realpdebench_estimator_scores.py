"""Backfill raw EBM surprise metrics for completed staged PIV checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from pinnlab.registry import get_experiment, get_model
from pinnlab.utils.density import create_density_estimator
from scripts.rebuttal.run_realpdebench import evaluate_estimator, sha256_file


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main(args):
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Use an explicit available CUDA device")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model_template = load_yaml(args.model_config)
    updated = 0
    skipped = 0
    for metrics_path in sorted(args.root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("smoke_test", True):
            continue
        if metrics.get("method") not in {"pinn_ebm", "napinn"}:
            continue
        if not metrics.get("corruption_kind"):
            continue
        if metrics.get("estimator_failure_detection_auroc") is not None:
            skipped += 1
            continue
        checkpoint_path = metrics_path.parent / "final.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = checkpoint["config"]
        config["device"] = args.device
        experiment = get_experiment(config["experiment_name"])(
            config, device
        )
        model_config = dict(model_template)
        model_config["in_features"] = int(config["in_features"])
        model_config["out_features"] = int(config["out_features"])
        model = get_model(config["model_name"])(model_config).to(device)
        model.load_state_dict(checkpoint["model"])
        experiment.load_state_dict(checkpoint["experiment"])
        estimator = create_density_estimator(
            config["estimator"], input_dim=1, device=device
        )
        estimator.load_state_dict(checkpoint["estimator"])
        running_std = checkpoint["running_std"].to(device)
        scores = evaluate_estimator(
            model, experiment, estimator, running_std
        )
        if "estimator_failure_detection_auroc" not in scores:
            raise ValueError(
                f"Missing corruption labels while scoring {checkpoint_path}"
            )
        metrics.update(scores)
        metrics["estimator_score_backfill"] = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "device": args.device,
        }
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        updated += 1
        print(
            f"updated {metrics_path}: "
            f"AUROC={scores['estimator_failure_detection_auroc']:.6f}"
        )
    print(f"updated={updated} skipped_existing={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde/cylinder_real_piv"
        ),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/mlp.yaml"),
    )
    main(parser.parse_args())
