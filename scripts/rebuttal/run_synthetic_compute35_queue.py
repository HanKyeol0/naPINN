#!/usr/bin/env python3
"""Run conservative 35,000-update synthetic baseline comparisons.

PINN-EBM and naPINN use 30,000 PINN updates plus 5,000 estimator-only
updates.  This queue gives the non-estimator baselines 35,000 full PINN
updates, which is a conservative update-count match rather than a
wall-clock match.  It is kept in a separate output root so canonical 30k
artifacts cannot be overwritten.
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


def main(args):
    all_jobs = list(
        product(
            [40, 41, 42],
            ["mse", "lad", "orpinn_q29"],
            ["allencahn2d", "burgers2d", "lambdaomega2d"],
        )
    )
    shard = [
        job
        for index, job in enumerate(all_jobs)
        if index % args.num_shards == args.shard_index
    ]
    status_path = (
        args.status_root
        / f"compute35_cuda_{args.gpu}_shard_{args.shard_index}.json"
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
    for seed, method, experiment in shard:
        descriptor = {
            "experiment": experiment,
            "method": method,
            "ratio": 0.15,
            "seed": seed,
            "baseline_steps": 35000,
        }
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_synthetic",
            "--experiment-name",
            experiment,
            "--method",
            method,
            "--seed",
            str(seed),
            "--device",
            f"cuda:{args.gpu}",
            "--noise-kind",
            "4G",
            "--outlier-ratio",
            "0.15",
            "--baseline-steps",
            "35000",
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
        target = "completed" if result.returncode == 0 else "failed"
        status[target].append(record)
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
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/synthetic_compute35"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/rebuttal_synthetic_compute35"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
