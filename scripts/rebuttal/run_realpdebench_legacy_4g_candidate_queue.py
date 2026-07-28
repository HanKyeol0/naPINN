#!/usr/bin/env python3
"""Run one deterministic shard of frozen legacy-4G candidate confirmations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from scripts.rebuttal.run_realpdebench_legacy_4g_queue import (
    METHOD_CONFIGS,
    STAGED_METHODS,
    config_tag,
)


def jobs(manifest: dict) -> list[dict]:
    output = []
    for record in manifest["artifacts_and_configs"]:
        for method in METHOD_CONFIGS:
            output.append({**record, "method": method})
    expected = int(manifest["expected_full_run_count"])
    if len(output) != expected:
        raise ValueError(f"Manifest expected {expected} jobs, resolved {len(output)}")
    return output


def assigned_jobs(
    manifest: dict, *, shard_index: int, num_shards: int
) -> list[dict]:
    assignments: list[list[dict]] = [[] for _ in range(num_shards)]
    loads = [0.0 for _ in range(num_shards)]
    weighted = sorted(
        jobs(manifest),
        key=lambda job: (
            0 if job["method"] in STAGED_METHODS else 1,
            job["rank"],
            job["seed"],
            job["method"],
        ),
    )
    for job in weighted:
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        assignments[target].append(job)
        loads[target] += 1.5 if job["method"] in STAGED_METHODS else 1.0
    return assignments[shard_index]


def variant_suffix(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    suffix = payload.get("variant_tag_suffix")
    if not isinstance(suffix, str) or not suffix:
        raise ValueError(f"Missing variant_tag_suffix in {path}")
    return suffix


def completed(output_root: Path, job: dict) -> bool:
    variant = Path(job["variant_config_path"])
    tag = f"{config_tag(METHOD_CONFIGS[job['method']])}{variant_suffix(variant)}"
    tag_root = output_root / "cylinder_real_piv_legacy4g_candidates" / tag
    if not tag_root.is_dir():
        return False
    for metrics_path in tag_root.rglob("metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            int(payload.get("seed", -1)) == int(job["seed"])
            and not bool(payload.get("smoke_test", True))
            and payload.get("status") == "complete"
            and payload.get("evidence_status")
            == "full_run_complete_unaggregated"
        ):
            return True
    return False


def main(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    selected = assigned_jobs(
        manifest, shard_index=args.shard_index, num_shards=args.num_shards
    )
    status_path = (
        args.status_root
        / f"legacy4g_candidates_cuda_{args.gpu}_shard_{args.shard_index}.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "protocol": "legacy4g_reporting_candidate_confirmation",
        "selection_id": manifest["selection_id"],
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "concurrency": args.concurrency,
        "total_jobs": len(selected),
        "completed": [],
        "skipped_existing": [],
        "failed": [],
        "started_at_unix": time.time(),
    }
    pending: list[dict] = []
    for job in selected:
        descriptor = {
            key: job[key]
            for key in (
                "candidate_id",
                "rank",
                "seed",
                "ratio",
                "background_scale_multiplier",
                "gross_scale_multiplier",
                "selected_napinn_variant",
                "method",
            )
        }
        if completed(args.output_root, job):
            status["skipped_existing"].append(descriptor)
            continue
        pending.append(job)

    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    job_logs = args.status_root / "job_logs"
    job_logs.mkdir(parents=True, exist_ok=True)

    def execute(job: dict) -> dict:
        descriptor = {
            key: job[key]
            for key in (
                "candidate_id",
                "rank",
                "seed",
                "ratio",
                "background_scale_multiplier",
                "gross_scale_multiplier",
                "selected_napinn_variant",
                "method",
            )
        }
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_realpdebench",
            "--exp-config",
            str(METHOD_CONFIGS[job["method"]]),
            "--variant-config",
            str(job["variant_config_path"]),
            "--seed",
            str(job["seed"]),
            "--device",
            f"cuda:{args.gpu}",
            "--output-root",
            str(args.output_root),
            "--run-name",
            f"confirmation_{job['candidate_id']}_seed{job['seed']}",
        ]
        log_path = job_logs / (
            f"{job['candidate_id']}_seed{job['seed']}_{job['method']}.log"
        )
        if log_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite candidate job log: {log_path}"
            )
        print(
            json.dumps(
                {
                    "event": "job_start",
                    "command": command,
                    "log_path": str(log_path),
                    **descriptor,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        started = time.time()
        with log_path.open("x", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                check=False,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        return {
            **descriptor,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
            "log_path": str(log_path),
        }

    stop_launching = False
    pending_iterator = iter(pending)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        active: dict[concurrent.futures.Future[dict], dict] = {}

        def fill_active() -> None:
            while not stop_launching and len(active) < args.concurrency:
                try:
                    job = next(pending_iterator)
                except StopIteration:
                    return
                active[executor.submit(execute, job)] = job

        fill_active()
        while active:
            done, _ = concurrent.futures.wait(
                active,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                job = active.pop(future)
                descriptor = {
                    key: job[key]
                    for key in (
                        "candidate_id",
                        "rank",
                        "seed",
                        "ratio",
                        "background_scale_multiplier",
                        "gross_scale_multiplier",
                        "selected_napinn_variant",
                        "method",
                    )
                }
                try:
                    record = future.result()
                except BaseException as error:
                    record = {
                        **descriptor,
                        "returncode": -1,
                        "elapsed_seconds": 0.0,
                        "exception": repr(error),
                    }
                status[
                    "completed"
                    if record["returncode"] == 0
                    else "failed"
                ].append(record)
                if record["returncode"] != 0 and args.fail_fast:
                    stop_launching = True
                status["updated_at_unix"] = time.time()
                status_path.write_text(
                    json.dumps(status, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            fill_active()

    status["finished_at_unix"] = time.time()
    status["stopped_launching_after_failure"] = stop_launching
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if status["failed"] and args.fail_fast:
        raise SystemExit(
            next(
                (
                    int(record["returncode"])
                    for record in status["failed"]
                    if int(record["returncode"]) > 0
                ),
                1,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help=(
            "Concurrent child runs on this GPU. The campaign default of 5 "
            "was validated on the destination server's RTX A6000 GPUs."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "analysis/results/runs/"
            "rebuttal_realpde_legacy_4g_candidates/"
            "candidate_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/realpde_legacy_4g_candidates"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/rebuttal_realpde_legacy_4g_candidates"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
