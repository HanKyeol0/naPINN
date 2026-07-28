#!/usr/bin/env python3
"""Run a deterministic shard of the frozen synthetic rebuttal matrix."""

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


def core_jobs():
    # Severe corruption first so the closest-prior, parameter, and stronger
    # baseline questions receive evidence before the lower-ratio extension.
    ratios = [0.15, 0.05, 0.10]
    methods = [
        "napinn",
        "pinn_ebm",
        "lad",
        "orpinn_q29",
        "mse",
        "orpinn_q19",
    ]
    seeds = [40, 41, 42]
    experiments = ["allencahn2d", "burgers2d", "lambdaomega2d"]
    return list(product(ratios, seeds, methods, experiments))


def main(args):
    jobs = core_jobs()
    shard = [
        job
        for index, job in enumerate(jobs)
        if index % args.num_shards == args.shard_index
    ]
    status_path = (
        args.status_root / f"core_cuda_{args.gpu}_shard_{args.shard_index}.json"
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
    for ratio, seed, method, experiment in shard:
        descriptor = {
            "experiment": experiment,
            "method": method,
            "ratio": ratio,
            "seed": seed,
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
            str(ratio),
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
        if result.returncode == 0:
            status["completed"].append(record)
        else:
            status["failed"].append(record)
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
        default=Path("outputs/rebuttal/synthetic"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/rebuttal_synthetic"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
