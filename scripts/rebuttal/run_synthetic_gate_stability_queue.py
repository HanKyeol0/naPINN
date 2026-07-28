#!/usr/bin/env python3
"""Gate-initialization sensitivity and staged-training stability.

Two reviewer requests are covered here.

* Gate initializer. The submitted YAML exposes ``init_threshold`` and
  ``init_steepness``, but those keys are consumed by LearnableThresholdGate.
  The paper's main gated_trainable path never received them, so the trainable
  likelihood gate always started from the constructor values (cutoff 2.0,
  steepness 30.0). This queue sweeps the values that the gate actually uses,
  through the newly added ``gate_init_*`` keys, one factor at a time around
  that historical operating point.

* Staged-training stability. The submitted staged-training ablation has no
  run artifact in this checkout. Here the no-warm-up arm keeps the same 30,000
  PINN updates and moves all of them into the joint stage, so the comparison
  isolates the warm-up schedule rather than the update budget.

Everything runs on a single GPU that is not shared with the isolated
wall-clock campaign.
"""

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
SEEDS = [40, 41, 42]
# Historical operating point, included so the sweep is self-contained rather
# than compared across output roots.
GATE_CELLS = [
    {"label": "baseline_cut2_steep30", "cutoff": None, "steepness": None},
    {"label": "cut1", "cutoff": 1.0, "steepness": None},
    {"label": "cut4", "cutoff": 4.0, "steepness": None},
    {"label": "steep10", "cutoff": None, "steepness": 10.0},
    {"label": "steep60", "cutoff": None, "steepness": 60.0},
]


def build_jobs():
    jobs = []
    for cell in GATE_CELLS:
        for seed in SEEDS:
            jobs.append(
                {
                    "track": "gate_init",
                    "experiment": "allencahn2d",
                    "method": "napinn",
                    "seed": seed,
                    "cell": cell["label"],
                    "gate_init_cutoff_sigma": cell["cutoff"],
                    "gate_init_steepness": cell["steepness"],
                    "warmup_steps": 5000,
                    "joint_steps": 25000,
                }
            )
    for experiment in CONFIGS:
        for seed in SEEDS:
            jobs.append(
                {
                    "track": "staged_training",
                    "experiment": experiment,
                    "method": "napinn",
                    "seed": seed,
                    "cell": "no_warmup_30k_joint",
                    "gate_init_cutoff_sigma": None,
                    "gate_init_steepness": None,
                    "warmup_steps": 0,
                    "joint_steps": 30000,
                }
            )
    return jobs


def main(args: argparse.Namespace) -> None:
    jobs = build_jobs()
    args.status_root.mkdir(parents=True, exist_ok=True)
    status_path = args.status_root / f"gate_stability_cuda_{args.gpu}.json"
    status: dict = {
        "gpu": args.gpu,
        "total_jobs": len(jobs),
        "completed": [],
        "failed": [],
        "started_at_unix": time.time(),
    }

    def save() -> None:
        status["updated_at_unix"] = time.time()
        status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    save()
    for job in jobs:
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_synthetic",
            "--experiment-name",
            job["experiment"],
            "--method",
            job["method"],
            "--seed",
            str(job["seed"]),
            "--device",
            f"cuda:{args.gpu}",
            "--noise-kind",
            "4G",
            "--outlier-ratio",
            "0.15",
            "--warmup-steps",
            str(job["warmup_steps"]),
            "--joint-steps",
            str(job["joint_steps"]),
            "--experiment-config",
            str(CONFIGS[job["experiment"]]),
            "--output-root",
            str(args.output_root / job["track"]),
        ]
        if job["gate_init_cutoff_sigma"] is not None:
            command += [
                "--gate-init-cutoff-sigma",
                str(job["gate_init_cutoff_sigma"]),
            ]
        if job["gate_init_steepness"] is not None:
            command += [
                "--gate-init-steepness",
                str(job["gate_init_steepness"]),
            ]
        print(json.dumps({"event": "job_start", **job}, sort_keys=True), flush=True)
        started = time.time()
        result = subprocess.run(command, check=False)
        record = {
            **job,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
        }
        status["completed" if result.returncode == 0 else "failed"].append(record)
        save()
        if result.returncode != 0 and args.fail_fast:
            raise SystemExit(result.returncode)
    status["finished_at_unix"] = time.time()
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/gate_stability_20260728"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/gate_stability_20260728"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
