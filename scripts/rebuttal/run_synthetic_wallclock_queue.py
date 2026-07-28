#!/usr/bin/env python3
"""Hardware-isolated equal-wall-clock synthetic comparison.

Every previous compute comparison in this rebuttal was either an update-count
match (the 35,000-update queue) or a wall time observed on a shared server.
Neither answers the reviewers' equal-wall-clock question. This queue runs the
whole comparison serially on one exclusively reserved GPU:

* Stage 1 measures the end-to-end time of naPINN and direct PINN-EBM under
  their normal schedule (5,000 warm-up + 5,000 estimator-only + 25,000 joint).
* Stage 2 gives every ordinary baseline the *same elapsed seconds* as the
  stage-1 naPINN reference for that PDE, letting each baseline take as many
  PINN updates as it can fit.

The reference budget is the per-PDE mean naPINN end-to-end training time over
the reporting seeds, computed from stage 1 and frozen into
``wallclock_budget.json`` before stage 2 starts. It is therefore not selected
after seeing any baseline result.

The isolation is the whole point: run this with a single ``--gpu`` and do not
schedule any other job on that device while it runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


CONFIGS = {
    "allencahn2d": Path("configs/rebuttal/allencahn2d.yaml"),
    "burgers2d": Path("configs/rebuttal/burgers2d.yaml"),
    "lambdaomega2d": Path("configs/rebuttal/lambdaomega2d.yaml"),
}
EXPERIMENTS = ["allencahn2d", "burgers2d", "lambdaomega2d"]
SEEDS = [40, 41, 42]
REFERENCE_METHODS = ["napinn", "pinn_ebm"]
BUDGETED_METHODS = ["mse", "lad", "orpinn_q29"]
BUDGET_REFERENCE_METHOD = "napinn"
OUTLIER_RATIO = "0.15"
NOISE_KIND = "4G"
# Safety cap only. The time budget must be the binding constraint; the runner
# raises if a baseline hits this cap before its budget expires.
BASELINE_STEP_CAP = 2_000_000


def run_directory(root: Path, experiment: str, method: str, seed: int) -> Path:
    return (
        root
        / experiment
        / NOISE_KIND
        / f"{int(round(100 * float(OUTLIER_RATIO))):02d}pct"
        / method
        / "ema_0.05_reject_0.5_pde_1_data_1"
        / f"seed_{seed}"
    )


def launch(
    *,
    experiment: str,
    method: str,
    seed: int,
    gpu: int,
    output_root: Path,
    time_budget_seconds: float | None,
) -> tuple[int, float]:
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
        f"cuda:{gpu}",
        "--noise-kind",
        NOISE_KIND,
        "--outlier-ratio",
        OUTLIER_RATIO,
        "--experiment-config",
        str(CONFIGS[experiment]),
        "--output-root",
        str(output_root),
    ]
    if time_budget_seconds is not None:
        command += [
            "--time-budget-seconds",
            f"{time_budget_seconds:.6f}",
            "--baseline-steps",
            str(BASELINE_STEP_CAP),
        ]
    started = time.time()
    result = subprocess.run(command, check=False)
    return result.returncode, time.time() - started


def read_training_seconds(metrics_path: Path) -> float:
    return float(json.loads(metrics_path.read_text())["training_seconds"])


def main(args: argparse.Namespace) -> None:
    args.status_root.mkdir(parents=True, exist_ok=True)
    status_path = args.status_root / f"wallclock_cuda_{args.gpu}.json"
    status: dict = {
        "gpu": args.gpu,
        "isolation": "single exclusively reserved GPU, serial execution",
        "budget_reference_method": BUDGET_REFERENCE_METHOD,
        "stage1_reference": [],
        "stage2_budgeted": [],
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

    # ---- Stage 1: reference schedule, measured on the isolated GPU ----
    for experiment in EXPERIMENTS:
        for method in REFERENCE_METHODS:
            for seed in SEEDS:
                descriptor = {
                    "stage": 1,
                    "experiment": experiment,
                    "method": method,
                    "seed": seed,
                }
                print(
                    json.dumps({"event": "job_start", **descriptor}, sort_keys=True),
                    flush=True,
                )
                code, elapsed = launch(
                    experiment=experiment,
                    method=method,
                    seed=seed,
                    gpu=args.gpu,
                    output_root=args.output_root,
                    time_budget_seconds=None,
                )
                record = {**descriptor, "returncode": code, "queue_seconds": elapsed}
                if code == 0:
                    status["stage1_reference"].append(record)
                else:
                    status["failed"].append(record)
                    if args.fail_fast:
                        save()
                        raise SystemExit(code)
                save()

    # ---- Freeze the budget from stage 1 before any baseline runs ----
    budget: dict[str, float] = {}
    for experiment in EXPERIMENTS:
        samples = []
        for seed in SEEDS:
            metrics = (
                run_directory(
                    args.output_root, experiment, BUDGET_REFERENCE_METHOD, seed
                )
                / "metrics.json"
            )
            if not metrics.is_file():
                raise SystemExit(
                    f"Missing stage-1 reference metrics for the budget: {metrics}"
                )
            samples.append(read_training_seconds(metrics))
        budget[experiment] = statistics.fmean(samples)
    budget_path = args.output_root / "wallclock_budget.json"
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    budget_path.write_text(
        json.dumps(
            {
                "reference_method": BUDGET_REFERENCE_METHOD,
                "reference_seeds": SEEDS,
                "definition": (
                    "per-PDE mean end-to-end naPINN training seconds "
                    "(warm-up + estimator initialization + joint), measured on "
                    "the same exclusively reserved GPU"
                ),
                "budget_seconds": budget,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    status["frozen_budget_seconds"] = budget
    save()

    # ---- Stage 2: ordinary baselines under the frozen equal-time budget ----
    for experiment in EXPERIMENTS:
        for method in BUDGETED_METHODS:
            for seed in SEEDS:
                descriptor = {
                    "stage": 2,
                    "experiment": experiment,
                    "method": method,
                    "seed": seed,
                    "time_budget_seconds": budget[experiment],
                }
                print(
                    json.dumps({"event": "job_start", **descriptor}, sort_keys=True),
                    flush=True,
                )
                code, elapsed = launch(
                    experiment=experiment,
                    method=method,
                    seed=seed,
                    gpu=args.gpu,
                    output_root=args.output_root,
                    time_budget_seconds=budget[experiment],
                )
                record = {**descriptor, "returncode": code, "queue_seconds": elapsed}
                if code == 0:
                    status["stage2_budgeted"].append(record)
                else:
                    status["failed"].append(record)
                    if args.fail_fast:
                        save()
                        raise SystemExit(code)
                save()

    status["finished_at_unix"] = time.time()
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/wallclock_isolated_20260728"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/wallclock_isolated_20260728"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
