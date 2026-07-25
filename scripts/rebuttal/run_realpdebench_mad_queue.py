#!/usr/bin/env python3
"""Run MAD-PINN stage 2 after the injected-PIV LAD runs are complete.

The MAD implementation requires the seed- and corruption-matched 30,000-step
LAD checkpoint.  Keeping this as a second queue makes that dependency explicit
and prevents a shard from starting MAD before another shard has finished LAD.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


MAD_CONFIG = Path(
    "configs/experiment/realpdebench_cylinder_mad_pinn.yaml"
)
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


def find_lad_checkpoint(root: Path, variant: str, seed: int) -> Path:
    tag_root = root / "cylinder_real_piv" / f"lad_{variant}"
    matches: list[Path] = []
    for metrics_path in tag_root.rglob("metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            int(payload.get("seed", -1)) == seed
            and not bool(payload.get("smoke_test", True))
            and payload.get("evidence_status")
            == "full_run_complete_unaggregated"
            and payload.get("method") == "lad"
            and int(payload.get("pinn_update_steps", -1)) == 30000
        ):
            checkpoint = metrics_path.parent / "final.pt"
            if checkpoint.is_file():
                matches.append(checkpoint)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one completed LAD checkpoint for {variant=} {seed=}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def completed(root: Path, variant: str, seed: int) -> bool:
    tag_root = root / "cylinder_real_piv" / f"mad_pinn_{variant}"
    for metrics_path in tag_root.rglob("metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            int(payload.get("seed", -1)) == seed
            and not bool(payload.get("smoke_test", True))
            and payload.get("evidence_status")
            == "full_run_complete_unaggregated"
            and payload.get("method") == "mad_pinn"
        ):
            return True
    return False


def main(args):
    all_jobs = list(
        product(
            args.variants,
            [40, 41, 42],
        )
    )
    shard = [
        job
        for index, job in enumerate(all_jobs)
        if index % args.num_shards == args.shard_index
    ]
    status_path = (
        args.status_root
        / (
            f"injected_piv_mad_{args.status_label}_cuda_{args.gpu}"
            f"_shard_{args.shard_index}.json"
        )
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "variants": args.variants,
        "total_jobs": len(shard),
        "completed": [],
        "skipped_existing": [],
        "failed": [],
        "started_at_unix": time.time(),
    }
    for variant, seed in shard:
        descriptor = {"variant": variant, "seed": seed, "method": "mad_pinn"}
        if completed(args.output_root, variant, seed):
            status["skipped_existing"].append(descriptor)
            continue
        try:
            checkpoint = find_lad_checkpoint(args.output_root, variant, seed)
        except Exception as error:
            status["failed"].append({**descriptor, "error": str(error)})
            status["updated_at_unix"] = time.time()
            status_path.write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if args.fail_fast:
                raise
            continue
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_realpdebench",
            "--exp-config",
            str(MAD_CONFIG),
            "--variant-config",
            str(VARIANTS[variant]),
            "--stage1-checkpoint",
            str(checkpoint),
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
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=list(VARIANTS),
        help=(
            "Corruption variants whose completed, seed-matched LAD "
            "checkpoints should be consumed."
        ),
    )
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
    parser.add_argument("--status-label", default="all")
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
