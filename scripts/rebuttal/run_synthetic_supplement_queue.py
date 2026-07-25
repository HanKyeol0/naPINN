#!/usr/bin/env python3
"""Run a shard of the frozen sensitivity/noise/ablation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CONFIGS = {
    "allencahn2d": Path("configs/rebuttal/allencahn2d.yaml"),
    "burgers2d": Path("configs/rebuttal/burgers2d.yaml"),
    "lambdaomega2d": Path("configs/rebuttal/lambdaomega2d.yaml"),
}


def jobs():
    result = []
    # Reviewer-requested EMA coefficient sensitivity. momentum is the new
    # batch-statistic weight; old-state decay is 1-momentum.
    for seed in (40, 41, 42):
        for momentum in (0.01, 0.05, 0.10):
            result.append(
                {
                    "track": "ema",
                    "experiment": "allencahn2d",
                    "method": "napinn",
                    "seed": seed,
                    "noise": "4G",
                    "ratio": 0.15,
                    "momentum": momentum,
                    "rejection": 0.5,
                }
            )
    for seed in (40, 41, 42):
        for rejection in (0.10, 0.30, 0.50, 0.70, 1.00):
            result.append(
                {
                    "track": "rejection",
                    "experiment": "allencahn2d",
                    "method": "napinn",
                    "seed": seed,
                    "noise": "4G",
                    "ratio": 0.15,
                    "momentum": 0.05,
                    "rejection": rejection,
                }
            )
    # Submitted appendix scope plus Student-t: lambda-omega, 15% outliers.
    for noise in ("G", "Laplace", "StudentT"):
        for seed in (40, 41, 42):
            for method in (
                "mse",
                "lad",
                "orpinn_q29",
                "pinn_ebm",
                "napinn",
                "napinn_lad",
                "napinn_q29",
            ):
                result.append(
                    {
                        "track": "noise",
                        "experiment": "lambdaomega2d",
                        "method": method,
                        "seed": seed,
                        "noise": noise,
                        "ratio": 0.15,
                        "momentum": 0.05,
                        "rejection": 0.5,
                    }
                )
    # The core queue already covers the five ordinary/base methods for 4G.
    # Add only the two naPINN backbone-loss variants that are not in core.
    for seed in (40, 41, 42):
        for method in ("napinn_lad", "napinn_q29"):
            result.append(
                {
                    "track": "noise",
                    "experiment": "lambdaomega2d",
                    "method": method,
                    "seed": seed,
                    "noise": "4G",
                    "ratio": 0.15,
                    "momentum": 0.05,
                    "rejection": 0.5,
                }
            )
    # Estimator-free selector ablations at the most severe setting.
    # ``quantile`` has a fixed quantile rule; ``threshold`` learns its
    # threshold and steepness and must not be described as fixed.
    for experiment in CONFIGS:
        for seed in (40, 41, 42):
            for method in ("quantile", "threshold"):
                result.append(
                    {
                        "track": "ablation",
                        "experiment": experiment,
                        "method": method,
                        "seed": seed,
                        "noise": "4G",
                        "ratio": 0.15,
                        "momentum": 0.05,
                        "rejection": 0.5,
                    }
                )
    return result


def main(args):
    all_jobs = jobs()
    shard = [
        job
        for index, job in enumerate(all_jobs)
        if index % args.num_shards == args.shard_index
    ]
    status_path = (
        args.status_root
        / f"supplement_cuda_{args.gpu}_shard_{args.shard_index}.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_jobs": len(shard),
        "completed": [],
        "failed": [],
        "started_at_unix": time.time(),
    }
    for job in shard:
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_synthetic",
            "--experiment-name",
            job["experiment"],
            "--experiment-config",
            str(CONFIGS[job["experiment"]]),
            "--method",
            job["method"],
            "--seed",
            str(job["seed"]),
            "--device",
            f"cuda:{args.gpu}",
            "--noise-kind",
            job["noise"],
            "--outlier-ratio",
            str(job["ratio"]),
            "--ema-momentum",
            str(job["momentum"]),
            "--rejection-cost",
            str(job["rejection"]),
            "--output-root",
            str(args.output_root),
        ]
        print(
            json.dumps({"event": "job_start", **job}, sort_keys=True),
            flush=True,
        )
        started = time.time()
        result = subprocess.run(command, check=False)
        record = {
            **job,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
        }
        status[
            "completed" if result.returncode == 0 else "failed"
        ].append(record)
        status["updated_at_unix"] = time.time()
        status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if result.returncode != 0 and args.fail_fast:
            raise SystemExit(result.returncode)
    status["finished_at_unix"] = time.time()
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/results/runs/rebuttal_synthetic"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("analysis/results/runs/rebuttal_synthetic_queue"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
