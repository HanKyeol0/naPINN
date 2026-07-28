#!/usr/bin/env python3
"""Run the predeclared synthetic 4G background-noise-only diagnostic.

The training data retain the configured four-Gaussian background noise, while
the number of additional gross-corrupted rows is fixed to zero.  This queue
is intentionally separate from the original 5/10/15% evidence matrix.
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


def jobs() -> list[tuple[int, str, str]]:
    seeds = (40, 41, 42)
    methods = ("pinn_ebm", "napinn")
    experiments = ("allencahn2d", "burgers2d", "lambdaomega2d")
    return list(product(seeds, methods, experiments))


def main(args: argparse.Namespace) -> None:
    shard = [
        job
        for index, job in enumerate(jobs())
        if index % args.num_shards == args.shard_index
    ]
    status_path = (
        args.status_root
        / f"background_only_cuda_{args.gpu}_shard_{args.shard_index}.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "campaign": "synthetic_4g_background_only_no_gross_rows",
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
            "outlier_ratio": 0.0,
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
            "0.0",
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
        destination = (
            status["completed"] if result.returncode == 0 else status["failed"]
        )
        assert isinstance(destination, list)
        destination.append(record)
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
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/synthetic_background_only"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path(
            "outputs/status/rebuttal_synthetic_background_only"
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
