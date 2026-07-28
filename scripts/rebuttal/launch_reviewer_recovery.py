#!/usr/bin/env python3
"""Launch one frozen destination-server reviewer-recovery campaign.

This launcher is intentionally thin: it reads exact roots from
``reviewer_recovery_manifest.yaml``, creates one worker per deterministic
shard, assigns the requested GPUs round-robin, captures a separate log for
each worker, and records every return code.  It does not aggregate results
and never interprets partial metrics as evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = Path("configs/rebuttal/reviewer_recovery_manifest.yaml")
CAMPAIGNS = {
    "synthetic_supplement": {
        "script": "scripts/rebuttal/run_synthetic_supplement_queue.py",
    },
    "synthetic_pinn_ebm_weight_calibration": {
        "script": "scripts/rebuttal/run_synthetic_pinn_ebm_weight_queue.py",
        "extra": ["--phase", "calibration"],
    },
    "synthetic_pinn_ebm_weight_heldout": {
        "script": "scripts/rebuttal/run_synthetic_pinn_ebm_weight_queue.py",
        "extra": ["--phase", "heldout"],
    },
    "synthetic_compute_35k": {
        "script": "scripts/rebuttal/run_synthetic_compute35_queue.py",
    },
    "synthetic_mad": {
        "script": "scripts/rebuttal/run_synthetic_mad_queue.py",
        "stage1": True,
    },
    "structured_piv_fixed_re": {
        "script": "scripts/rebuttal/run_realpdebench_queue.py",
    },
    "structured_piv_inverse_re": {
        "script": "scripts/rebuttal/run_realpdebench_inverse_queue.py",
        "extra": [
            "--methods",
            "mse",
            "lad",
            "pinn_ebm",
            "napinn",
        ],
    },
    "structured_piv_mad": {
        "script": "scripts/rebuttal/run_realpdebench_mad_queue.py",
    },
}


def load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    queued = payload.get("queued_after_primary_gpu_release")
    if not isinstance(queued, dict):
        raise ValueError(f"Missing queued campaign mapping in {path}")
    return payload


def build_worker_commands(
    *,
    manifest: dict[str, Any],
    campaign: str,
    gpus: list[int],
    num_shards: int,
    python_executable: str = sys.executable,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if campaign not in CAMPAIGNS:
        raise ValueError(f"Unknown campaign {campaign!r}")
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("--gpus must contain unique non-negative indices")
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    record = manifest["queued_after_primary_gpu_release"][campaign]
    if not isinstance(record, dict):
        raise ValueError(f"Invalid manifest record for {campaign}")
    if int(record["expected_runs"]) <= 0:
        raise ValueError(f"Invalid expected run count for {campaign}")
    output_root = Path(str(record["output_root"]))
    status_root = Path(str(record["status_root"]))
    if output_root.resolve() == status_root.resolve():
        raise ValueError("Output and status roots must be separate")

    spec = CAMPAIGNS[campaign]
    commands = []
    for shard_index in range(num_shards):
        gpu = gpus[shard_index % len(gpus)]
        command = [
            python_executable,
            str(spec["script"]),
            "--gpu",
            str(gpu),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            str(num_shards),
            "--output-root",
            str(output_root),
            "--status-root",
            str(status_root),
            "--fail-fast",
        ]
        if spec.get("stage1"):
            stage1_root = record.get("stage1_root")
            if not stage1_root:
                raise ValueError(f"{campaign} lacks stage1_root")
            command.extend(["--stage1-root", str(stage1_root)])
        if record.get("napinn_config"):
            command.extend(
                ["--napinn-config", str(record["napinn_config"])]
            )
        command.extend(spec.get("extra", []))
        commands.append(
            {
                "shard_index": shard_index,
                "gpu": gpu,
                "command": command,
            }
        )
    return record, commands


def validate_dependencies(campaign: str, record: dict[str, Any]) -> dict[str, Any]:
    """Validate checkpoint dependencies immediately before worker launch."""
    if campaign == "synthetic_mad":
        from scripts.rebuttal.run_synthetic_mad_queue import (
            find_lad_checkpoint,
        )

        root = Path(str(record["stage1_root"]))
        checkpoints = []
        for experiment in (
            "allencahn2d",
            "burgers2d",
            "lambdaomega2d",
        ):
            for seed in (40, 41, 42):
                checkpoints.append(
                    str(find_lad_checkpoint(root, experiment, seed).resolve())
                )
        return {
            "dependency": "synthetic_seed_matched_lad",
            "validated_checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
        }
    if campaign == "structured_piv_mad":
        from scripts.rebuttal.run_realpdebench_mad_queue import (
            VARIANTS,
            find_lad_checkpoint,
        )

        root = Path(str(record["output_root"]))
        checkpoints = []
        for variant in VARIANTS:
            for seed in (40, 41, 42):
                checkpoints.append(
                    str(find_lad_checkpoint(root, variant, seed).resolve())
                )
        return {
            "dependency": "structured_piv_seed_matched_lad",
            "validated_checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
        }
    return {
        "dependency": None,
        "validated_checkpoint_count": 0,
        "checkpoints": [],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    record, commands = build_worker_commands(
        manifest=manifest,
        campaign=args.campaign,
        gpus=args.gpus,
        num_shards=args.num_shards,
    )
    payload: dict[str, Any] = {
        "campaign": args.campaign,
        "manifest": str(args.manifest.resolve()),
        "manifest_record": record,
        "gpus": args.gpus,
        "num_shards": args.num_shards,
        "workers": commands,
        "evidence_status": "launcher_metadata_not_performance_evidence",
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    payload["dependency_preflight"] = validate_dependencies(
        args.campaign, record
    )
    available = torch.cuda.device_count()
    invalid = [gpu for gpu in args.gpus if gpu >= available]
    if invalid:
        raise ValueError(
            f"Requested unavailable GPUs {invalid}; torch sees {available}"
        )

    status_root = Path(str(record["status_root"]))
    launcher_status = (
        status_root / f"launcher_{args.campaign}_{args.run_id}.json"
    )
    if launcher_status.exists():
        raise FileExistsError(
            f"Refusing to overwrite launcher status: {launcher_status}"
        )
    status_root.mkdir(parents=True, exist_ok=True)
    payload["run_id"] = args.run_id
    payload["started_at_unix"] = time.time()
    payload["status"] = "running"
    write_json_atomic(launcher_status, payload)

    processes: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
    for worker in commands:
        log_path = status_root / (
            f"launcher_{args.run_id}_gpu{worker['gpu']}"
            f"_shard{worker['shard_index']}.log"
        )
        log_stream = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            worker["command"],
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        worker["pid"] = process.pid
        worker["log_path"] = str(log_path)
        processes.append((worker, process, log_stream))
        write_json_atomic(launcher_status, payload)

    failed = False
    for worker, process, log_stream in processes:
        returncode = process.wait()
        log_stream.close()
        worker["returncode"] = returncode
        worker["finished_at_unix"] = time.time()
        failed = failed or returncode != 0
        write_json_atomic(launcher_status, payload)
    payload["finished_at_unix"] = time.time()
    payload["status"] = "failed" if failed else "workers_complete"
    payload["evidence_status"] = (
        "worker_failure_not_aggregated"
        if failed
        else "workers_complete_pending_strict_aggregation"
    )
    write_json_atomic(launcher_status, payload)
    if failed:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument(
        "--run-id",
        default=time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
