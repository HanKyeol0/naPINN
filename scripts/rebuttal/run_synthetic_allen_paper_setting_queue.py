#!/usr/bin/env python3
"""Run the submitted-paper Allen--Cahn rejection cost of 1.0."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


CONFIG = Path("configs/rebuttal/allencahn2d.yaml")


def main(args):
    all_jobs = list(product([0.15, 0.05, 0.10], [40, 41, 42]))
    shard = [
        job
        for index, job in enumerate(all_jobs)
        if index % args.num_shards == args.shard_index
    ]
    status_path = (
        args.status_root
        / (
            f"allen_paper_reject1_cuda_{args.gpu}"
            f"_shard_{args.shard_index}.json"
        )
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
    for ratio, seed in shard:
        descriptor = {
            "experiment": "allencahn2d",
            "method": "napinn",
            "ratio": ratio,
            "seed": seed,
            "rejection_cost": 1.0,
        }
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_synthetic",
            "--experiment-name",
            "allencahn2d",
            "--method",
            "napinn",
            "--seed",
            str(seed),
            "--device",
            f"cuda:{args.gpu}",
            "--noise-kind",
            "4G",
            "--outlier-ratio",
            str(ratio),
            "--rejection-cost",
            "1.0",
            "--experiment-config",
            str(CONFIG),
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
        default=Path("outputs/rebuttal/synthetic"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/rebuttal_synthetic"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
