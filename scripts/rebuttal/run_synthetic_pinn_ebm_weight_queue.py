#!/usr/bin/env python3
"""Run the full direct PINN-EBM PDE-weight sensitivity.

The equal-weight held-out results were inspected before the planned seed-39
calibration completed. To avoid a post-hoc favorable choice, this queue keeps
the seed-39 calibration context but evaluates every additional weight on all
held-out seeds as well. Existing weight-1 held-out artifacts are reused.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


CONFIGS = {
    "allencahn2d": Path("configs/rebuttal/allencahn2d.yaml"),
    "burgers2d": Path("configs/rebuttal/burgers2d.yaml"),
    "lambdaomega2d": Path("configs/rebuttal/lambdaomega2d.yaml"),
}


def jobs(phase: str = "all"):
    calibration = list(product([39], [1.0, 10.0, 50.0], CONFIGS))
    held_out = list(product([40, 41, 42], [10.0, 50.0], CONFIGS))
    if phase == "calibration":
        return calibration
    if phase == "heldout":
        return held_out
    if phase == "all":
        return calibration + held_out
    raise ValueError(f"Unknown phase: {phase}")


def main(args):
    all_jobs = jobs(args.phase)
    shard = [
        job
        for index, job in enumerate(all_jobs)
        if index % args.num_shards == args.shard_index
    ]
    status_path = (
        args.status_root
        / (
            f"pinn_ebm_weight_cuda_{args.gpu}"
            f"_shard_{args.shard_index}.json"
        )
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "campaign": "synthetic_pinn_ebm_pde_weight",
        "phase": args.phase,
        "expected_phase_jobs": len(all_jobs),
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_jobs": len(shard),
        "completed": [],
        "failed": [],
        "started_at_unix": time.time(),
    }
    for seed, weight, experiment in shard:
        descriptor = {
            "experiment": experiment,
            "method": "pinn_ebm",
            "ratio": 0.15,
            "seed": seed,
            "pde_weight": weight,
        }
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_synthetic",
            "--experiment-name",
            experiment,
            "--method",
            "pinn_ebm",
            "--seed",
            str(seed),
            "--device",
            f"cuda:{args.gpu}",
            "--noise-kind",
            "4G",
            "--outlier-ratio",
            "0.15",
            "--pde-weight",
            str(weight),
            "--experiment-config",
            str(CONFIGS[experiment]),
            "--output-root",
            str(args.output_root),
        ]
        print(
            json.dumps({"event": "job_start", **descriptor}, sort_keys=True),
            flush=True,
        )
        started = time.time()
        result = subprocess.run(command, check=False)
        record = {
            **descriptor,
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
    parser.add_argument("--num-shards", type=int, default=7)
    parser.add_argument(
        "--phase",
        choices=("calibration", "heldout", "all"),
        default="all",
        help=(
            "Run seed-39 calibration or seeds-40--42 held-out cells "
            "separately so each phase can use an independent immutable "
            "output root and strict aggregate."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/synthetic"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/rebuttal_synthetic"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
