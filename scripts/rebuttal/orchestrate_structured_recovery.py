#!/usr/bin/env python3
"""Finish the frozen structured-PIV and official PINN-EBM recoveries.

This script only advances a campaign after the recovery monitor observes the
exact number of complete, non-smoke metrics declared in the frozen manifest.
It does not inspect performance values or select favorable cells.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.monitor_reviewer_recovery import (
    MANIFEST,
    build_summary,
)


STRUCTURED_TRACKS = (
    ("structured_piv_fixed_re", 75),
    ("structured_piv_inverse_re", 12),
    ("structured_piv_mad", 15),
)


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, "time_unix": time.time(), **payload},
            sort_keys=True,
        ),
        flush=True,
    )


def summary_for(manifest: Path) -> dict[str, Any]:
    return build_summary(manifest)


def complete_count(summary: dict[str, Any], campaign: str) -> int:
    record = summary["campaigns"][campaign]
    return int(record.get("discovered_complete_non_smoke_metrics", 0))


def wait_for_complete_count(
    manifest: Path,
    campaign: str,
    expected: int,
    poll_seconds: float,
) -> dict[str, Any]:
    last_count = -1
    while True:
        summary = summary_for(manifest)
        count = complete_count(summary, campaign)
        if count != last_count:
            emit(
                "campaign_progress",
                campaign=campaign,
                complete_non_smoke=count,
                expected=expected,
            )
            last_count = count
        if count > expected:
            raise RuntimeError(
                f"{campaign} has {count} complete metrics; expected exactly "
                f"{expected}"
            )
        if count == expected:
            return summary
        time.sleep(poll_seconds)


def run_checked(command: list[str], event: str) -> None:
    emit(event, command=command)
    subprocess.run(command, cwd=ROOT, check=True)


def launch_campaign(
    campaign: str,
    *,
    gpus: list[int],
    num_shards: int,
    run_id: str,
) -> None:
    run_checked(
        [
            sys.executable,
            "scripts/rebuttal/launch_reviewer_recovery.py",
            "--campaign",
            campaign,
            "--gpus",
            *[str(gpu) for gpu in gpus],
            "--num-shards",
            str(num_shards),
            "--run-id",
            run_id,
        ],
        "campaign_launch",
    )


def finish_structured(args: argparse.Namespace) -> None:
    wait_for_complete_count(
        args.manifest,
        "structured_piv_fixed_re",
        75,
        args.poll_seconds,
    )

    summary = summary_for(args.manifest)
    inverse_count = complete_count(summary, "structured_piv_inverse_re")
    if inverse_count < 12:
        launch_campaign(
            "structured_piv_inverse_re",
            gpus=args.gpus,
            num_shards=args.num_shards,
            run_id=f"{args.run_id}_inverse",
        )
    wait_for_complete_count(
        args.manifest,
        "structured_piv_inverse_re",
        12,
        args.poll_seconds,
    )

    summary = summary_for(args.manifest)
    mad_count = complete_count(summary, "structured_piv_mad")
    if mad_count < 15:
        launch_campaign(
            "structured_piv_mad",
            gpus=args.gpus,
            num_shards=args.num_shards,
            run_id=f"{args.run_id}_mad",
        )
    wait_for_complete_count(
        args.manifest,
        "structured_piv_mad",
        15,
        args.poll_seconds,
    )

    summary = summary_for(args.manifest)
    combined = summary["campaigns"][
        "structured_piv_combined_strict_aggregate"
    ]
    if not bool(combined.get("evidence_ready", False)):
        run_checked(
            [
                sys.executable,
                "scripts/rebuttal/finalize_reviewer_recovery.py",
                "--campaign",
                "structured_piv_combined_strict_aggregate",
            ],
            "strict_aggregate_launch",
        )
    emit("structured_recovery_complete")


def finish_official_ab(args: argparse.Namespace) -> None:
    announced: dict[str, bool] = {}
    while True:
        summary = summary_for(args.manifest)
        record = summary["campaigns"]["pinn_ebm_upstream_a_b"]
        if bool(record.get("evidence_ready", False)):
            emit("official_ab_already_aggregated")
            return
        variants = record.get("variants", {})
        ready = {
            name: bool(value.get("official_metrics_exists", False))
            for name, value in variants.items()
        }
        if ready != announced:
            emit("official_ab_progress", official_metrics_exists=ready)
            announced = ready
        if variants and all(ready.values()):
            break
        time.sleep(args.poll_seconds)

    run_checked(
        [
            sys.executable,
            "scripts/rebuttal/finalize_reviewer_recovery.py",
            "--campaign",
            "pinn_ebm_upstream_a_b",
        ],
        "official_ab_strict_aggregate_launch",
    )
    emit("official_ab_recovery_complete")


def main(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise ValueError("--poll-seconds must be in (0, 60]")
    emit(
        "orchestrator_start",
        manifest=str(args.manifest),
        gpus=args.gpus,
        num_shards=args.num_shards,
        run_id=args.run_id,
    )
    finish_structured(args)
    finish_official_ab(args)
    emit("all_recoveries_complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--gpus", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--run-id", default="auto_20260727")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
