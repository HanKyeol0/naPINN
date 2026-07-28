#!/usr/bin/env python3
"""Run one deterministic shard of the exact-center legacy-4G PIV matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


METHOD_CONFIGS = {
    "mse": Path("configs/experiment/realpdebench_cylinder_mse.yaml"),
    "lad": Path("configs/experiment/realpdebench_cylinder_lad.yaml"),
    "orpinn_q29": Path(
        "configs/experiment/realpdebench_cylinder_orpinn_q29.yaml"
    ),
    "pinn_ebm_equal": Path(
        "configs/experiment/realpdebench_cylinder_pinn_ebm_equal_weight.yaml"
    ),
    "pinn_ebm_prior": Path(
        "configs/experiment/realpdebench_cylinder_pinn_ebm.yaml"
    ),
    "napinn_mse": Path(
        "configs/experiment/realpdebench_cylinder_napinn.yaml"
    ),
    "napinn_lad": Path(
        "configs/experiment/realpdebench_cylinder_napinn_lad.yaml"
    ),
    "napinn_q29": Path(
        "configs/experiment/realpdebench_cylinder_napinn_q29.yaml"
    ),
}
STAGED_METHODS = {
    "pinn_ebm_equal",
    "pinn_ebm_prior",
    "napinn_mse",
    "napinn_lad",
    "napinn_q29",
}
VARIANT_CONFIGS = {
    (seed, ratio): Path(
        "configs/experiment/"
        f"realpdebench_cylinder_legacy4g_seed{seed}_"
        f"r{int(round(100 * ratio))}_b1_o1.yaml"
    )
    for seed in (40, 41, 42)
    for ratio in (0.10, 0.15)
}


def config_tag(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    tag = payload.get("tag") if isinstance(payload, dict) else None
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"Missing non-empty tag in {path}")
    return tag


def variant_suffix(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    suffix = (
        payload.get("variant_tag_suffix") if isinstance(payload, dict) else None
    )
    if not isinstance(suffix, str) or not suffix:
        raise ValueError(f"Missing variant_tag_suffix in {path}")
    return suffix


def jobs() -> list[tuple[int, float, str]]:
    return [
        (seed, ratio, method)
        for seed in (40, 41, 42)
        for ratio in (0.10, 0.15)
        for method in METHOD_CONFIGS
    ]


def assigned_jobs(
    *, shard_index: int, num_shards: int
) -> list[tuple[int, float, str]]:
    """Greedily balance staged and ordinary methods across deterministic shards."""
    assignments: list[list[tuple[int, float, str]]] = [
        [] for _ in range(num_shards)
    ]
    loads = [0.0 for _ in range(num_shards)]
    weighted = sorted(
        jobs(),
        key=lambda job: (
            0 if job[2] in STAGED_METHODS else 1,
            job[0],
            job[1],
            job[2],
        ),
    )
    for job in weighted:
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        assignments[target].append(job)
        loads[target] += 1.5 if job[2] in STAGED_METHODS else 1.0
    return assignments[shard_index]


def completed(
    output_root: Path, *, seed: int, ratio: float, method: str
) -> bool:
    variant_config = VARIANT_CONFIGS[(seed, ratio)]
    tag = f"{config_tag(METHOD_CONFIGS[method])}{variant_suffix(variant_config)}"
    tag_root = output_root / "cylinder_real_piv_legacy4g" / tag
    if not tag_root.is_dir():
        return False
    for metrics_path in tag_root.rglob("metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            int(payload.get("seed", -1)) == seed
            and not bool(payload.get("smoke_test", True))
            and payload.get("evidence_status")
            == "full_run_complete_unaggregated"
            and payload.get("status") == "complete"
        ):
            return True
    return False


def main(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    selected = assigned_jobs(
        shard_index=args.shard_index, num_shards=args.num_shards
    )
    status_path = (
        args.status_root
        / f"legacy4g_exact_cuda_{args.gpu}_shard_{args.shard_index}.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "protocol": "legacy4g_exact_center_b1_o1",
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_jobs": len(selected),
        "completed": [],
        "skipped_existing": [],
        "failed": [],
        "started_at_unix": time.time(),
    }
    for seed, ratio, method in selected:
        descriptor = {
            "seed": seed,
            "ratio": ratio,
            "method": method,
        }
        if completed(
            args.output_root, seed=seed, ratio=ratio, method=method
        ):
            status["skipped_existing"].append(descriptor)
            continue
        ratio_token = int(round(100 * ratio))
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_realpdebench",
            "--exp-config",
            str(METHOD_CONFIGS[method]),
            "--variant-config",
            str(VARIANT_CONFIGS[(seed, ratio)]),
            "--seed",
            str(seed),
            "--device",
            f"cuda:{args.gpu}",
            "--output-root",
            str(args.output_root),
            "--run-name",
            f"report_seed_{seed}_r{ratio_token}_b1_o1",
        ]
        print(
            json.dumps(
                {"event": "job_start", "command": command, **descriptor},
                sort_keys=True,
            ),
            flush=True,
        )
        started = time.time()
        result = subprocess.run(command, check=False)
        record = {
            **descriptor,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
        }
        status[
            "completed" if result.returncode == 0 else "failed"
        ].append(record)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/realpde_legacy_4g"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/rebuttal_realpde_legacy_4g"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
