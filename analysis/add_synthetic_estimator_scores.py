"""Backfill raw EBM density-score diagnostics for completed synthetic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from pinnlab.registry import get_experiment, get_model
from pinnlab.utils.seed import seed_everything
from scripts.rebuttal.run_synthetic import estimator_score_metrics


def main(args):
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("Use an explicit CUDA device for score evaluation")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    updated = 0
    for metrics_path in sorted(args.input_root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("method") not in {
            "pinn_ebm",
            "napinn",
            "napinn_lad",
            "napinn_q29",
        }:
            continue
        if "estimator_scores" in metrics:
            continue
        run_dir = metrics_path.parent
        config = yaml.safe_load(
            (run_dir / "config.yaml").read_text(encoding="utf-8")
        )
        seed = int(config["run"]["seed"])
        seed_everything(seed)
        experiment = get_experiment(metrics["experiment_name"])(
            config["experiment"], device
        )
        seed_everything(seed)
        model = get_model(args.model_name)(config["model"]).to(device)
        checkpoint = torch.load(
            run_dir / "final.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        experiment.load_state_dict(checkpoint["experiment"])
        metrics["estimator_scores"] = estimator_score_metrics(
            model, experiment
        )
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        updated += 1
        print(f"Updated: {metrics_path}")
    print(f"Backfilled {updated} completed runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("analysis/results/runs/rebuttal_synthetic"),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--model-name", default="mlp")
    main(parser.parse_args())
