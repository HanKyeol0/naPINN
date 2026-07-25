#!/usr/bin/env python3
"""Run a deterministic shard of the injected-real-PIV rebuttal matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import yaml


METHOD_CONFIGS = {
    "mse": Path("configs/experiment/realpdebench_cylinder_mse.yaml"),
    "lad": Path("configs/experiment/realpdebench_cylinder_lad.yaml"),
    "orpinn_q29": Path(
        "configs/experiment/realpdebench_cylinder_orpinn_q29.yaml"
    ),
    "pinn_ebm": Path(
        "configs/experiment/realpdebench_cylinder_pinn_ebm.yaml"
    ),
    "napinn": Path(
        "configs/experiment/realpdebench_cylinder_napinn_rejection_001.yaml"
    ),
}
VARIANTS = {
    "sensor_drift30": Path(
        "configs/experiment/realpdebench_cylinder_sensor_drift30.yaml"
    ),
    "sensor_drift20": Path(
        "configs/experiment/realpdebench_cylinder_sensor_drift20.yaml"
    ),
    "sensor_drift10": Path(
        "configs/experiment/realpdebench_cylinder_sensor_drift10.yaml"
    ),
    "ar1": Path("configs/experiment/realpdebench_cylinder_ar1.yaml"),
    "spatial_burst": Path(
        "configs/experiment/realpdebench_cylinder_spatial_burst.yaml"
    ),
}


def jobs():
    # Severe persistent failures are evaluated first; correlated families
    # follow. All cells are retained regardless of which method wins.
    variants = [
        "sensor_drift30",
        "sensor_drift20",
        "sensor_drift10",
        "ar1",
        "spatial_burst",
    ]
    methods = ["napinn", "lad", "orpinn_q29", "mse", "pinn_ebm"]
    seeds = [40, 41, 42]
    return list(product(variants, seeds, methods))


def config_tag(config_path: Path) -> str:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tag = payload.get("tag") if isinstance(payload, dict) else None
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"Missing non-empty tag in {config_path}")
    return tag


def completed(
    root: Path,
    variant: str,
    method: str,
    seed: int,
    *,
    method_configs: dict[str, Path],
) -> bool:
    expected_tag = f"{config_tag(method_configs[method])}_{variant}"
    tag_root = root / "cylinder_real_piv" / expected_tag
    if not tag_root.is_dir():
        return False
    for metrics_path in tag_root.rglob("metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            int(payload.get("seed", -1)) == seed
            and not bool(payload.get("smoke_test", True))
            and payload.get("evidence_status")
            == "full_run_complete_unaggregated"
        ):
            return True
    return False


def main(args):
    method_configs = dict(METHOD_CONFIGS)
    method_configs["napinn"] = args.napinn_config
    all_jobs = [
        job for job in jobs() if job[2] in set(args.methods)
    ]
    shard = [
        job
        for index, job in enumerate(all_jobs)
        if index % args.num_shards == args.shard_index
    ]
    status_path = (
        args.status_root
        / (
            f"injected_piv_{args.status_label}_cuda_{args.gpu}"
            f"_shard_{args.shard_index}.json"
        )
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "napinn_config": str(args.napinn_config),
        "total_jobs": len(shard),
        "completed": [],
        "skipped_existing": [],
        "failed": [],
        "started_at_unix": time.time(),
    }
    for variant, seed, method in shard:
        descriptor = {"variant": variant, "seed": seed, "method": method}
        if completed(
            args.output_root,
            variant,
            method,
            seed,
            method_configs=method_configs,
        ):
            status["skipped_existing"].append(descriptor)
            continue
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_realpdebench",
            "--exp-config",
            str(method_configs[method]),
            "--variant-config",
            str(VARIANTS[variant]),
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
        default=sorted(METHOD_CONFIGS),
    )
    parser.add_argument(
        "--napinn-config",
        type=Path,
        default=Path(
            "configs/experiment/"
            "realpdebench_cylinder_napinn_rejection_001.yaml"
        ),
        help=(
            "Calibration-selected naPINN config. The output tag is read "
            "from this YAML so held-out completion checks cannot silently "
            "reuse a different rejection cost."
        ),
    )
    parser.add_argument("--status-label", default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/results/runs/rebuttal_realpde"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("analysis/results/runs/rebuttal_realpde_queue"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
