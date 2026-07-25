#!/usr/bin/env python3
"""Run inverse-Re recovery on the severe injected real-PIV condition."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


METHOD_CONFIGS = {
    "mse": Path("configs/experiment/realpdebench_cylinder_mse.yaml"),
    "lad": Path("configs/experiment/realpdebench_cylinder_lad.yaml"),
    "pinn_ebm": Path(
        "configs/experiment/realpdebench_cylinder_pinn_ebm.yaml"
    ),
    "napinn": Path(
        "configs/experiment/realpdebench_cylinder_napinn_rejection_001.yaml"
    ),
}
VARIANT_CONFIG = Path(
    "configs/experiment/"
    "realpdebench_cylinder_inverse_re_sensor_drift30.yaml"
)


def main(args):
    method_configs = dict(METHOD_CONFIGS)
    method_configs["napinn"] = args.napinn_config
    all_jobs = list(product([40, 41, 42], args.methods))
    shard = [
        job
        for index, job in enumerate(all_jobs)
        if index % args.num_shards == args.shard_index
    ]
    status_path = (
        args.status_root
        / f"inverse_re_cuda_{args.gpu}_shard_{args.shard_index}.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "methods": args.methods,
        "napinn_config": str(args.napinn_config),
        "total_jobs": len(shard),
        "completed": [],
        "failed": [],
        "started_at_unix": time.time(),
    }
    for seed, method in shard:
        descriptor = {"seed": seed, "method": method}
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_realpdebench",
            "--exp-config",
            str(method_configs[method]),
            "--variant-config",
            str(VARIANT_CONFIG),
            "--seed",
            str(seed),
            "--device",
            f"cuda:{args.gpu}",
            "--run-name",
            f"heldout_seed_{seed}",
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
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHOD_CONFIGS),
        required=True,
    )
    parser.add_argument(
        "--napinn-config",
        type=Path,
        default=Path(
            "configs/experiment/"
            "realpdebench_cylinder_napinn_rejection_001.yaml"
        ),
        help="Calibration-selected naPINN experiment YAML.",
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("analysis/results/runs/rebuttal_realpde_queue"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
