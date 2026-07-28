#!/usr/bin/env python3
"""Launch the frozen nine-job natural-PIV naPINN robust-loss matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


EXPECTED_INPUT_SHA256 = (
    "e2513cccae4bdd75da1bf33a85bf23bb66a2dc8350c3e951a5c3e95ebb94b1bd"
)
SEEDS = (40, 41, 42)
METHODS: dict[str, dict[str, Any]] = {
    "napinn": {
        "config": Path("configs/experiment/realpdebench_cylinder_napinn.yaml"),
        "tag": "napinn",
        "loss": "mse",
        "q": None,
    },
    "napinn_lad": {
        "config": Path(
            "configs/experiment/realpdebench_cylinder_napinn_lad.yaml"
        ),
        "tag": "napinn_l1",
        "loss": "l1",
        "q": None,
    },
    "napinn_q29": {
        "config": Path(
            "configs/experiment/realpdebench_cylinder_napinn_q29.yaml"
        ),
        "tag": "napinn_q29",
        "loss": "q_gaussian",
        "q": 2.9,
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol(args: argparse.Namespace) -> Path:
    common = load_yaml(args.common_config)
    data_path = Path(common["data"]["path"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Natural-PIV input is missing: {data_path}")
    observed_sha256 = sha256_file(data_path)
    if observed_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "Natural-PIV input checksum mismatch: "
            f"expected {EXPECTED_INPUT_SHA256}, observed {observed_sha256}"
        )

    expected_schedule = {
        "total_steps": 30000,
        "warmup_steps": 5000,
        "estimator_init_steps": 5000,
        "joint_steps": 25000,
    }
    training = common.get("training", {})
    observed_schedule = {
        key: int(training.get(key, -1)) for key in expected_schedule
    }
    if observed_schedule != expected_schedule:
        raise ValueError(
            f"Natural-PIV schedule changed: {observed_schedule}"
        )
    if common.get("pde", {}).get("learn_reynolds") is not False:
        raise ValueError("Natural-PIV matrix requires fixed Reynolds number")
    if common.get("batch") != {"n_f": 8192, "n_data": 2048}:
        raise ValueError(
            f"Natural-PIV batch protocol changed: {common.get('batch')}"
        )

    seen_tags: set[str] = set()
    for method, expected in METHODS.items():
        config = load_yaml(expected["config"])
        if config.get("tag") != expected["tag"]:
            raise ValueError(
                f"{method} tag changed: {config.get('tag')!r}"
            )
        if config.get("method", {}).get("kind") != method:
            raise ValueError(f"{method} config selects a different method")
        raw_kind = str(config.get("data_loss", {}).get("kind", "")).lower()
        normalized_kind = "l1" if raw_kind in {"l1", "lad"} else raw_kind
        if normalized_kind != expected["loss"]:
            raise ValueError(
                f"{method} reconstruction loss changed: {raw_kind!r}"
            )
        raw_q = config.get("data_loss", {}).get("q")
        expected_q = expected["q"]
        if expected_q is None and raw_q is not None:
            raise ValueError(f"{method} unexpectedly sets q={raw_q}")
        if expected_q is not None and float(raw_q) != float(expected_q):
            raise ValueError(f"{method} requires q={expected_q}, got {raw_q}")
        if expected["tag"] in seen_tags:
            raise ValueError(f"Duplicate natural-PIV tag: {expected['tag']}")
        seen_tags.add(expected["tag"])
    return data_path


def all_jobs() -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "seed": seed,
            "tag": spec["tag"],
            "loss": spec["loss"],
            "q": spec["q"],
            "config": str(spec["config"]),
        }
        for seed in SEEDS
        for method, spec in METHODS.items()
    ]


def assigned_jobs(
    *, shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    return [
        job
        for index, job in enumerate(all_jobs())
        if index % num_shards == shard_index
    ]


def run_name(job: dict[str, Any]) -> str:
    return f"natural_piv_{job['method']}_seed_{job['seed']}"


def run_dir(root: Path, job: dict[str, Any]) -> Path:
    return (
        root
        / "runs"
        / "cylinder_real_piv"
        / str(job["tag"])
        / run_name(job)
    )


def completed(root: Path, job: dict[str, Any]) -> bool:
    metrics_path = run_dir(root, job) / "metrics.json"
    if not metrics_path.is_file():
        return False
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return (
        metrics.get("status") == "complete"
        and metrics.get("evidence_status")
        == "full_run_complete_unaggregated"
        and not bool(metrics.get("smoke_test", True))
        and metrics.get("method") == job["method"]
        and metrics.get("tag") == job["tag"]
        and int(metrics.get("seed", -1)) == int(job["seed"])
        and metrics.get("input_artifact_sha256") == EXPECTED_INPUT_SHA256
        and int(metrics.get("pinn_update_steps", -1)) == 30000
        and int(metrics.get("estimator_init_steps", -1)) == 5000
    )


def execute_job(
    *,
    args: argparse.Namespace,
    job: dict[str, Any],
    data_path: Path,
) -> dict[str, Any]:
    descriptor = {
        key: job[key] for key in ("method", "tag", "loss", "q", "seed")
    }
    expected_dir = run_dir(args.root, job)
    if expected_dir.exists():
        raise FileExistsError(
            "Refusing to reuse an incomplete natural-PIV run directory: "
            f"{expected_dir}"
        )

    log_path = (
        args.root
        / "logs"
        / f"{job['method']}_seed_{job['seed']}_cuda_{args.gpu}.log"
    )
    if log_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite natural-PIV job log: {log_path}"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = (
        args.root
        / "status"
        / "jobs"
        / f"{job['method']}_seed_{job['seed']}.json"
    )
    command = [
        sys.executable,
        "-m",
        "scripts.rebuttal.run_realpdebench",
        "--common-config",
        str(args.common_config),
        "--model-config",
        str(args.model_config),
        "--exp-config",
        str(job["config"]),
        "--seed",
        str(job["seed"]),
        "--device",
        f"cuda:{args.gpu}",
        "--run-name",
        run_name(job),
        "--output-root",
        str(args.root / "runs"),
    ]
    started = time.time()
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        write_json(
            status_path,
            {
                "status": "running",
                "pid": process.pid,
                "gpu": args.gpu,
                "command": command,
                "log_path": str(log_path),
                "expected_run_dir": str(expected_dir),
                "input_artifact_path": str(data_path.resolve()),
                "input_artifact_sha256": EXPECTED_INPUT_SHA256,
                "started_at_unix": started,
                **descriptor,
            },
        )
        returncode = process.wait()

    record = {
        "status": "complete" if returncode == 0 else "failed",
        "pid": process.pid,
        "returncode": returncode,
        "gpu": args.gpu,
        "command": command,
        "log_path": str(log_path),
        "expected_run_dir": str(expected_dir),
        "input_artifact_path": str(data_path.resolve()),
        "input_artifact_sha256": EXPECTED_INPUT_SHA256,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        **descriptor,
    }
    if returncode == 0 and not completed(args.root, job):
        record["status"] = "failed"
        record["returncode"] = -2
        record["validation_error"] = (
            "Child returned zero but no strict completed metrics were found"
        )
    write_json(status_path, record)
    return record


def main(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    data_path = validate_protocol(args)
    selected = assigned_jobs(
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    status_path = (
        args.root
        / "status"
        / f"natural_piv_cuda_{args.gpu}_shard_{args.shard_index}.json"
    )
    status: dict[str, Any] = {
        "protocol": "natural_piv_napinn_robust_loss",
        "input_artifact_path": str(data_path.resolve()),
        "input_artifact_sha256": EXPECTED_INPUT_SHA256,
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "concurrency": args.concurrency,
        "total_jobs": len(selected),
        "assigned_jobs": selected,
        "completed": [],
        "skipped_existing": [],
        "failed": [],
        "started_at_unix": time.time(),
    }
    pending = []
    for job in selected:
        if completed(args.root, job):
            status["skipped_existing"].append(job)
        else:
            pending.append(job)
    write_json(status_path, status)

    stop_launching = False
    pending_iterator = iter(pending)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        active: dict[
            concurrent.futures.Future[dict[str, Any]], dict[str, Any]
        ] = {}

        def fill_active() -> None:
            while not stop_launching and len(active) < args.concurrency:
                try:
                    job = next(pending_iterator)
                except StopIteration:
                    return
                future = executor.submit(
                    execute_job,
                    args=args,
                    job=job,
                    data_path=data_path,
                )
                active[future] = job

        fill_active()
        while active:
            done, _ = concurrent.futures.wait(
                active,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                job = active.pop(future)
                try:
                    record = future.result()
                except BaseException as error:
                    record = {
                        **job,
                        "status": "failed",
                        "returncode": -1,
                        "exception": repr(error),
                    }
                status[
                    "completed"
                    if record.get("status") == "complete"
                    else "failed"
                ].append(record)
                if record.get("status") != "complete" and args.fail_fast:
                    stop_launching = True
                status["updated_at_unix"] = time.time()
                write_json(status_path, status)
            fill_active()

    status["finished_at_unix"] = time.time()
    status["stopped_launching_after_failure"] = stop_launching
    write_json(status_path, status)
    if status["failed"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Concurrent natural-PIV child runs assigned to this GPU.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/rebuttal/natural_piv"),
    )
    parser.add_argument(
        "--common-config",
        type=Path,
        default=Path(
            "configs/experiment/realpdebench_cylinder_common.yaml"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/mlp.yaml"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
