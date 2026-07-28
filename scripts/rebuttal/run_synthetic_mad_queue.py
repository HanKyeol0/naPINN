#!/usr/bin/env python3
"""Run severe-corruption MAD-PINN after canonical LAD checkpoints exist."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


def find_lad_checkpoint(root: Path, experiment: str, seed: int) -> Path:
    run_dir = (
        root
        / experiment
        / "4G"
        / "15pct"
        / "lad"
        / "ema_0.05_reject_0.5_pde_1_data_1"
        / f"seed_{seed}"
    )
    checkpoint = run_dir / "final.pt"
    metrics_path = run_dir / "metrics.json"
    if not checkpoint.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing completed LAD artifact for {experiment=} {seed=}: "
            f"{run_dir}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics.get("status") != "complete"
        or metrics.get("method") != "lad"
        or int(metrics.get("pinn_update_steps", -1)) != 30000
    ):
        raise ValueError(f"Invalid LAD stage-one metrics: {metrics_path}")
    return checkpoint


def main(args):
    all_jobs = list(
        product(
            [40, 41, 42],
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
        / f"mad_cuda_{args.gpu}_shard_{args.shard_index}.json"
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
    for seed, experiment in shard:
        descriptor = {"experiment": experiment, "seed": seed}
        try:
            checkpoint = find_lad_checkpoint(
                args.stage1_root, experiment, seed
            )
        except Exception as error:
            status["failed"].append({**descriptor, "error": str(error)})
            continue
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_synthetic_mad",
            "--experiment-name",
            experiment,
            "--seed",
            str(seed),
            "--device",
            f"cuda:{args.gpu}",
            "--lad-checkpoint",
            str(checkpoint),
            "--stage2-steps",
            "30000",
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
            "stage1_checkpoint": str(checkpoint),
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
        "--stage1-root",
        type=Path,
        default=Path("outputs/rebuttal/synthetic"),
        help=(
            "Read-only root containing the canonical completed LAD runs. "
            "This is deliberately separate from --output-root so MAD stage "
            "2 cannot contaminate the exact core aggregation matrix."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/synthetic_mad"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/rebuttal_synthetic_mad"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
