#!/usr/bin/env python3
"""Read-only progress and EBM tail-retry monitor for official PINN-EBM runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

MODEL_ORDER = [0, 3, 2]
UPDATES_PER_MODEL = 100000
RUNS_PER_JOB = 5
TOTAL_UPDATES = len(MODEL_ORDER) * UPDATES_PER_MODEL * RUNS_PER_JOB
RUN_PATTERN = re.compile(r"^run #(\d+)\s*$", re.MULTILINE)
PROGRESS_PATTERN = re.compile(r"j(\d+) itges(\d+) epd(\d+)")
MODEL_DONE_PATTERN = re.compile(r"tm([032]):\s*([0-9.]+)")


def parse_timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def split_run_segments(text: str) -> list[tuple[int, str]]:
    matches = list(RUN_PATTERN.finditer(text))
    segments = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segments.append((int(match.group(1)), text[match.end() : end]))
    return segments


def parse_segment(run_index: int, segment: str) -> dict[str, Any]:
    completed_models = [
        {"model_index": int(model), "seconds": float(seconds)}
        for model, seconds in MODEL_DONE_PATTERN.findall(segment)
    ]
    progress_matches = list(PROGRESS_PATTERN.finditer(segment))
    latest = progress_matches[-1] if progress_matches else None
    completed_count = len(completed_models)
    current_model = MODEL_ORDER[completed_count] if completed_count < 3 else None
    current_updates = int(latest.group(2)) if latest and current_model is not None else 0
    ebm_initializations = segment.count("init_ebm!")
    escalations = segment.count("updated!")

    if current_model != 3:
        tail_status = "not_currently_in_pinn_ebm"
    elif current_updates < 10000 and ebm_initializations == 0:
        tail_status = "not_reached"
    elif current_updates <= 10000:
        tail_status = "initializing_or_retrying"
    else:
        tail_status = "passed_and_joint_training_resumed"

    return {
        "run_index": run_index,
        "completed_models": completed_models,
        "current_model_index": current_model,
        "current_model_updates": current_updates,
        "latest_outer_epoch": int(latest.group(1)) if latest else None,
        "ebm_initialization_attempts": ebm_initializations,
        "ebm_tail_retries": max(0, ebm_initializations - 1),
        "ebm_retry_escalations": escalations,
        "ebm_tail_status": tail_status,
        "completed_update_equivalents": (
            completed_count * UPDATES_PER_MODEL + current_updates
        ),
    }


def parse_run(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    log_path = run_dir / "stdout_stderr.log"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    text = (
        log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        if log_path.exists()
        else ""
    )
    segments = [
        parse_segment(run_index, segment)
        for run_index, segment in split_run_segments(text)
    ]
    completed_updates = sum(
        segment["completed_update_equivalents"] for segment in segments
    )
    started = parse_timestamp(metadata["started_at_utc"])
    now = dt.datetime.now(dt.timezone.utc)
    elapsed = max((now - started).total_seconds(), 1e-9)
    rate = completed_updates / elapsed
    remaining = max(TOTAL_UPDATES - completed_updates, 0)
    eta_seconds = remaining / rate if rate > 0 else None
    return {
        "variant_id": metadata["variant_id"],
        "run_dir": str(run_dir.resolve()),
        "metadata_status": metadata["status"],
        "log_size_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "elapsed_seconds": elapsed,
        "completed_update_equivalents": completed_updates,
        "total_update_equivalents": TOTAL_UPDATES,
        "progress_fraction": completed_updates / TOTAL_UPDATES,
        "observed_update_rate_per_second": rate,
        "naive_remaining_seconds": eta_seconds,
        "naive_eta_utc": (
            (now + dt.timedelta(seconds=eta_seconds)).isoformat()
            if eta_seconds is not None
            else None
        ),
        "segments": segments,
        "tail_retry_summary": {
            "initialization_attempts": sum(
                segment["ebm_initialization_attempts"] for segment in segments
            ),
            "retries": sum(segment["ebm_tail_retries"] for segment in segments),
            "escalations": sum(
                segment["ebm_retry_escalations"] for segment in segments
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse existing official PINN-EBM stdout without modifying or "
            "interrupting training."
        )
    )
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--write",
        type=Path,
        help="Optionally write the same snapshot JSON below outputs/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "monitor_semantics": (
            "Progress is inferred from upstream carriage-return logs. ETA is "
            "a naive extrapolation and includes no correction for EBM-only "
            "initialization, retries, evaluation, or plotting."
        ),
        "runs": [parse_run(path.resolve()) for path in args.run_dirs],
    }
    if args.write:
        output = args.write.resolve()
        try:
            output.relative_to(Path.cwd().resolve() / "outputs")
        except ValueError as error:
            raise ValueError("--write must be below the repository outputs/") from error
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
