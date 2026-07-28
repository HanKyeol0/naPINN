#!/usr/bin/env python3
"""Run/resume the frozen 144-cell non-Cylinder RealPDEBench matrix.

Eight deterministic shards are balanced by staged-method cost.  ``--launch-all``
starts two shard workers on each of GPUs 0--3.  A worker skips a run only after
strictly validating complete metrics, the final checkpoint, the exact input
hash/path, update budget, and resolved config.  A pre-existing partial or
mismatched run directory is reported as blocked and is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


MANIFEST = Path("configs/rebuttal/realpdebench_multidataset/manifest.yaml")
STAGED = {
    "pinn_ebm_w1",
    "pinn_ebm_w50",
    "napinn_mse",
    "napinn_l1",
    "napinn_q29",
}


@dataclass(frozen=True)
class Job:
    dataset: str
    seed: int
    ratio_percent: int
    method: str
    method_config: str
    data_overlay: str
    input_artifact: str
    input_sha256: str
    benchmark: str
    tag: str

    @property
    def key(self) -> str:
        return (
            f"{self.dataset}|seed{self.seed}|r{self.ratio_percent}|"
            f"{self.method}"
        )

    @property
    def run_name(self) -> str:
        return (
            f"full_{self.dataset}_seed{self.seed}_r{self.ratio_percent}_"
            f"{self.method}"
        )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_protocol(path: Path = MANIFEST) -> dict[str, Any]:
    protocol = load_yaml(path)
    if int(protocol["matrix"]["expected_runs"]) != 144:
        raise ValueError("Frozen manifest must declare exactly 144 runs")
    for record in protocol["input_manifests"].values():
        source = Path(record["path"])
        if sha256_file(source) != str(record["sha256"]):
            raise ValueError(f"Input manifest hash mismatch: {source}")
    return protocol


def build_jobs(protocol: dict[str, Any]) -> list[Job]:
    corruption_path = Path(
        protocol["input_manifests"]["corruption"]["path"]
    )
    corruption = json.loads(corruption_path.read_text(encoding="utf-8"))
    artifacts: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in corruption["artifacts"]:
        metadata = record["corruption"]
        key = (
            str(record["dataset_family"]),
            int(metadata["corruption_seed"]),
            int(round(100 * float(metadata["gross_row_ratio_requested"]))),
        )
        if key in artifacts:
            raise ValueError(f"Duplicate corruption artifact key {key}")
        artifacts[key] = record

    jobs: list[Job] = []
    config_root = Path("configs/rebuttal/realpdebench_multidataset")
    for dataset in protocol["protocol"]["datasets"]:
        for seed in protocol["protocol"]["corruption_seeds"]:
            for ratio in protocol["protocol"]["gross_row_ratios"]:
                ratio_percent = int(round(100 * float(ratio)))
                artifact = artifacts[(dataset, int(seed), ratio_percent)]
                overlay = (
                    config_root
                    / "data"
                    / f"{dataset}_seed{seed}_r{ratio_percent}.yaml"
                )
                overlay_payload = load_yaml(overlay)
                artifact_path = Path(artifact["artifact_path"]).resolve()
                configured_path = Path(
                    overlay_payload["data"]["path"]
                ).resolve()
                if configured_path != artifact_path:
                    raise ValueError(
                        f"Overlay/artifact mismatch: {overlay} {artifact_path}"
                    )
                if sha256_file(artifact_path) != artifact["artifact_sha256"]:
                    raise ValueError(f"Artifact hash mismatch: {artifact_path}")
                benchmark = str(overlay_payload["output"]["benchmark"])
                for method, method_record in protocol["methods"].items():
                    method_config = Path(method_record["config"])
                    tag = str(load_yaml(method_config)["tag"])
                    jobs.append(
                        Job(
                            dataset=dataset,
                            seed=int(seed),
                            ratio_percent=ratio_percent,
                            method=method,
                            method_config=str(method_config),
                            data_overlay=str(overlay),
                            input_artifact=str(artifact_path),
                            input_sha256=str(artifact["artifact_sha256"]),
                            benchmark=benchmark,
                            tag=tag,
                        )
                    )
    if len(jobs) != 144 or len({job.key for job in jobs}) != 144:
        raise AssertionError("Job expansion is not exactly 144 unique cells")
    return jobs


def shard_assignments(jobs: list[Job], num_shards: int) -> list[list[Job]]:
    if num_shards != 8:
        raise ValueError("Frozen launch protocol requires --num-shards 8")
    assignments: list[list[Job]] = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    ordered = sorted(
        jobs,
        key=lambda job: (
            0 if job.method in STAGED else 1,
            job.dataset,
            job.seed,
            job.ratio_percent,
            job.method,
        ),
    )
    for job in ordered:
        target = min(range(num_shards), key=lambda i: (loads[i], i))
        assignments[target].append(job)
        loads[target] += 1.5 if job.method in STAGED else 1.0
    flattened = [job.key for shard in assignments for job in shard]
    if len(flattened) != 144 or len(set(flattened)) != 144:
        raise AssertionError("Shard assignment loses or duplicates jobs")
    if max(map(len, assignments)) - min(map(len, assignments)) > 1:
        raise AssertionError("Shard job counts are unexpectedly imbalanced")
    return assignments


def expected_config(
    protocol: dict[str, Any], job: Job, output_root: Path
) -> dict[str, Any]:
    config = deep_merge(
        load_yaml(Path(protocol["runner"]["common_config"])),
        load_yaml(Path(job.method_config)),
    )
    config = deep_merge(config, load_yaml(Path(job.data_overlay)))
    config["seed"] = job.seed
    config.setdefault("output", {})["root"] = str(output_root)
    return config


def run_directory(output_root: Path, job: Job) -> Path:
    return output_root / job.benchmark / job.tag / job.run_name


def _same_subset(
    actual: dict[str, Any], expected: dict[str, Any], keys: tuple[str, ...]
) -> bool:
    return all(actual.get(key) == expected.get(key) for key in keys)


def validate_completed(
    protocol: dict[str, Any], job: Job, output_root: Path
) -> tuple[bool, str]:
    run_dir = run_directory(output_root, job)
    if not run_dir.exists():
        return False, "absent"
    required = [
        run_dir / "metrics.json",
        run_dir / "config.yaml",
        run_dir / "run_metadata.json",
        run_dir / "final.pt",
    ]
    if not all(path.is_file() for path in required):
        return False, "partial_existing_directory"
    try:
        metrics = json.loads(required[0].read_text(encoding="utf-8"))
        config = load_yaml(required[1])
        run_metadata = json.loads(required[2].read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return False, "unreadable_existing_directory"
    expected = expected_config(protocol, job, output_root)
    staged = job.method in STAGED
    checks = {
        "metrics_status": metrics.get("status") == "complete",
        "metrics_evidence": (
            metrics.get("evidence_status")
            == "full_run_complete_unaggregated"
        ),
        "not_smoke": not bool(metrics.get("smoke_test", True)),
        "seed": int(metrics.get("seed", -1)) == job.seed,
        "method": metrics.get("tag") == job.tag,
        "input_hash": metrics.get("input_artifact_sha256")
        == job.input_sha256,
        "input_path": Path(metrics.get("input_artifact_path", "")).resolve()
        == Path(job.input_artifact),
        "pinn_budget": int(metrics.get("pinn_update_steps", -1)) == 30000,
        "estimator_budget": int(metrics.get("estimator_init_steps", -1))
        == (5000 if staged else 0),
        "run_metadata_status": run_metadata.get("status") == "complete",
        "run_metadata_hash": run_metadata.get("input_artifact_sha256")
        == job.input_sha256,
        "config_core": _same_subset(
            config,
            expected,
            (
                "experiment_name",
                "model_name",
                "in_features",
                "out_features",
                "seed",
                "tag",
                "method",
                "data_loss",
                "batch",
                "training",
                "estimator",
                "gate",
                "eval",
            ),
        ),
        "config_data_path": Path(config["data"]["path"]).resolve()
        == Path(job.input_artifact),
        "config_input_hash": config["data"].get("input_artifact_sha256")
        == job.input_sha256,
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        return False, "mismatched_existing_directory:" + ",".join(failed)
    return True, "strict_complete"


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_worker(args: argparse.Namespace) -> None:
    if args.gpu is None or args.shard_index is None:
        raise ValueError("Worker requires --gpu and --shard-index")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    protocol = load_protocol(args.manifest)
    assignments = shard_assignments(build_jobs(protocol), args.num_shards)
    selected = assignments[args.shard_index]
    status_path = (
        args.status_root
        / f"worker_gpu{args.gpu}_shard{args.shard_index}.json"
    )
    status: dict[str, Any] = {
        "campaign": "realpdebench_multidataset_frozen_144",
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_job_count": len(selected),
        "assigned_job_keys": [job.key for job in selected],
        "completed": [],
        "skipped_strict_complete": [],
        "blocked_existing": [],
        "failed": [],
        "started_at_unix": time.time(),
    }
    for job in selected:
        descriptor = asdict(job)
        descriptor["key"] = job.key
        complete, reason = validate_completed(
            protocol, job, args.output_root
        )
        if complete:
            status["skipped_strict_complete"].append(
                {**descriptor, "reason": reason}
            )
            write_status(status_path, status)
            continue
        if reason != "absent":
            status["blocked_existing"].append(
                {**descriptor, "reason": reason}
            )
            write_status(status_path, status)
            if args.fail_fast:
                raise SystemExit(2)
            continue
        command = [
            sys.executable,
            "-m",
            "scripts.rebuttal.run_realpdebench",
            "--common-config",
            str(protocol["runner"]["common_config"]),
            "--model-config",
            str(protocol["runner"]["model_config"]),
            "--exp-config",
            job.method_config,
            "--experiment-config",
            job.data_overlay,
            "--seed",
            str(job.seed),
            "--device",
            f"cuda:{args.gpu}",
            "--output-root",
            str(args.output_root),
            "--run-name",
            job.run_name,
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
        if result.returncode == 0:
            complete, reason = validate_completed(
                protocol, job, args.output_root
            )
            if complete:
                status["completed"].append(record)
            else:
                record["post_run_validation"] = reason
                status["failed"].append(record)
        else:
            status["failed"].append(record)
        status["updated_at_unix"] = time.time()
        write_status(status_path, status)
        if status["failed"] and args.fail_fast:
            raise SystemExit(result.returncode or 3)
    status["finished_at_unix"] = time.time()
    write_status(status_path, status)
    if status["blocked_existing"] or status["failed"]:
        raise SystemExit(2)


def launch_all(args: argparse.Namespace) -> None:
    if args.num_shards != 8:
        raise ValueError("--launch-all requires --num-shards 8")
    protocol = load_protocol(args.manifest)
    assignments = shard_assignments(build_jobs(protocol), args.num_shards)
    if any(len(shard) != 18 for shard in assignments):
        raise AssertionError("Each of eight frozen shards must contain 18 jobs")
    args.status_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, int, subprocess.Popen[Any], Any]] = []
    for shard_index in range(8):
        gpu = shard_index // 2
        log_path = (
            args.status_root
            / f"worker_gpu{gpu}_shard{shard_index}.log"
        )
        log_stream = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--manifest",
            str(args.manifest),
            "--gpu",
            str(gpu),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            "8",
            "--output-root",
            str(args.output_root),
            "--status-root",
            str(args.status_root),
        ]
        if args.fail_fast:
            command.append("--fail-fast")
        process = subprocess.Popen(
            command,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_index, gpu, process, log_stream))
    failures = []
    for shard_index, gpu, process, log_stream in processes:
        returncode = process.wait()
        log_stream.close()
        if returncode != 0:
            failures.append(
                {"shard_index": shard_index, "gpu": gpu, "returncode": returncode}
            )
    summary = {
        "campaign": "realpdebench_multidataset_frozen_144",
        "num_shards": 8,
        "workers_per_gpu": 2,
        "gpu_ids": [0, 1, 2, 3],
        "failures": failures,
        "finished_at_unix": time.time(),
    }
    write_status(args.status_root / "launch_summary.json", summary)
    if failures:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--launch-all", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/rebuttal/realpdebench_multidataset/runs"
        ),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("outputs/status/realpdebench_multidataset"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.launch_all:
        launch_all(arguments)
    else:
        run_worker(arguments)
