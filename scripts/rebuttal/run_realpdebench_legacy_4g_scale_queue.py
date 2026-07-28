#!/usr/bin/env python3
"""Run one deterministic shard of the frozen seed-39 legacy-4G scale grid."""

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


SEED = 39
RATIOS = (0.10, 0.15)
MULTIPLIERS = (0.5, 1.0, 2.0)


def number_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def variant_config(ratio: float, base: float, gross: float) -> Path:
    return Path(
        "configs/experiment/"
        f"realpdebench_cylinder_legacy4g_seed{SEED}_"
        f"r{int(round(100 * ratio))}_"
        f"b{number_token(base)}_o{number_token(gross)}.yaml"
    )


def variant_suffix(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    suffix = (
        payload.get("variant_tag_suffix") if isinstance(payload, dict) else None
    )
    if not isinstance(suffix, str) or not suffix:
        raise ValueError(f"Missing variant_tag_suffix in {path}")
    return suffix


def jobs() -> list[tuple[float, float, float, str]]:
    return [
        (ratio, base, gross, method)
        for ratio in RATIOS
        for base in MULTIPLIERS
        for gross in MULTIPLIERS
        for method in METHOD_CONFIGS
    ]


def assigned_jobs(
    *, shard_index: int, num_shards: int
) -> list[tuple[float, float, float, str]]:
    assignments: list[list[tuple[float, float, float, str]]] = [
        [] for _ in range(num_shards)
    ]
    loads = [0.0 for _ in range(num_shards)]
    weighted = sorted(
        jobs(),
        key=lambda job: (
            0 if job[3] in STAGED_METHODS else 1,
            job[0],
            job[1],
            job[2],
            job[3],
        ),
    )
    for job in weighted:
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        assignments[target].append(job)
        loads[target] += 1.5 if job[3] in STAGED_METHODS else 1.0
    return assignments[shard_index]


def completed(
    output_root: Path,
    *,
    ratio: float,
    base: float,
    gross: float,
    method: str,
) -> bool:
    variant = variant_config(ratio, base, gross)
    tag = f"{config_tag(METHOD_CONFIGS[method])}{variant_suffix(variant)}"
    tag_root = output_root / "cylinder_real_piv_legacy4g_scale" / tag
    if not tag_root.is_dir():
        return False
    for metrics_path in tag_root.rglob("metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            int(payload.get("seed", -1)) == SEED
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
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    selected = assigned_jobs(
        shard_index=args.shard_index, num_shards=args.num_shards
    )
    status_path = (
        args.status_root
        / f"legacy4g_scale_cuda_{args.gpu}_shard_{args.shard_index}.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "protocol": "legacy4g_seed39_full_3x3_scale_grid",
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
    pending: list[tuple[float, float, float, str]] = []
    for ratio, base, gross, method in selected:
        descriptor = {
            "seed": SEED,
            "ratio": ratio,
            "base_multiplier": base,
            "gross_multiplier": gross,
            "method": method,
        }
        if completed(
            args.output_root,
            ratio=ratio,
            base=base,
            gross=gross,
            method=method,
        ):
            status["skipped_existing"].append(descriptor)
            continue
        pending.append((ratio, base, gross, method))

    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    job_logs = args.status_root / "job_logs"
    job_logs.mkdir(parents=True, exist_ok=True)

    def execute(job: tuple[float, float, float, str]) -> dict:
        ratio, base, gross, method = job
        descriptor = {
            "seed": SEED,
            "ratio": ratio,
            "base_multiplier": base,
            "gross_multiplier": gross,
            "method": method,
        }
        variant = variant_config(ratio, base, gross)
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_realpdebench",
            "--exp-config",
            str(METHOD_CONFIGS[method]),
            "--variant-config",
            str(variant),
            "--seed",
            str(SEED),
            "--device",
            f"cuda:{args.gpu}",
            "--output-root",
            str(args.output_root),
            "--run-name",
            (
                f"development_seed_{SEED}_r{int(round(100 * ratio))}_"
                f"b{number_token(base)}_o{number_token(gross)}"
            ),
        ]
        log_path = job_logs / (
            f"seed{SEED}_r{int(round(100 * ratio))}_"
            f"b{number_token(base)}_o{number_token(gross)}_{method}.log"
        )
        if log_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite scale job log: {log_path}"
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
        active: dict[
            concurrent.futures.Future[dict],
            tuple[float, float, float, str],
        ] = {}

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
                ratio, base, gross, method = job
                descriptor = {
                    "seed": SEED,
                    "ratio": ratio,
                    "base_multiplier": base,
                    "gross_multiplier": gross,
                    "method": method,
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


if __name__ == "__main__":
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
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/realpde_legacy_4g_scale"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/rebuttal_realpde_legacy_4g_scale"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    main(parser.parse_args())
