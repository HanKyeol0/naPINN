#!/usr/bin/env python3
"""Strictly aggregate one complete reviewer-recovery campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.monitor_reviewer_recovery import (
    MANIFEST,
    build_summary,
    load_yaml,
    write_json_atomic,
)


SYNTHETIC_CAMPAIGNS = {
    "synthetic_core": "active",
    "synthetic_supplement": "queued_after_primary_gpu_release",
    "synthetic_pinn_ebm_weight_calibration": (
        "queued_after_primary_gpu_release"
    ),
    "synthetic_pinn_ebm_weight_heldout": (
        "queued_after_primary_gpu_release"
    ),
    "synthetic_compute_35k": "queued_after_primary_gpu_release",
    "synthetic_mad": "queued_after_primary_gpu_release",
}
OTHER_CAMPAIGNS = {
    "realpdebench_multidataset": "active",
    "pinn_ebm_upstream_a_b": "active",
    "structured_piv_combined_strict_aggregate": (
        "queued_after_primary_gpu_release"
    ),
}
CAMPAIGNS = {**SYNTHETIC_CAMPAIGNS, **OTHER_CAMPAIGNS}


def campaign_record(
    manifest: dict[str, Any], campaign: str
) -> dict[str, Any]:
    if campaign not in CAMPAIGNS:
        raise ValueError(f"Unknown campaign: {campaign}")
    section = CAMPAIGNS[campaign]
    record = manifest[section][campaign]
    if not isinstance(record, dict):
        raise ValueError(f"Invalid manifest record for {campaign}")
    return record


def build_command(
    manifest: dict[str, Any],
    campaign: str,
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    record = campaign_record(manifest, campaign)
    aggregate = record.get("strict_aggregate")
    if not isinstance(aggregate, dict) or not aggregate.get("json"):
        raise ValueError(f"{campaign} lacks a strict aggregate destination")
    if campaign in SYNTHETIC_CAMPAIGNS:
        return [
            python_executable,
            "analysis/aggregate_rebuttal_synthetic.py",
            "--input-root",
            str(record["output_root"]),
            "--output-json",
            str(aggregate["json"]),
            "--output-csv",
            str(aggregate["csv"]),
            "--required-seeds",
            *[str(seed) for seed in record["required_seeds"]],
            "--strict",
            "--expected-runs",
            str(record["expected_runs"]),
            "--expected-groups",
            str(record["expected_groups"]),
            "--expected-pinn-steps",
            str(aggregate["expected_pinn_steps"]),
        ]
    if campaign == "realpdebench_multidataset":
        return [
            python_executable,
            "analysis/aggregate_realpdebench_multidataset.py",
            "--manifest",
            "configs/rebuttal/realpdebench_multidataset/manifest.yaml",
            "--run-root",
            str(record["output_root"]),
            "--output",
            str(aggregate["json"]),
        ]
    if campaign == "pinn_ebm_upstream_a_b":
        return [
            python_executable,
            "analysis/aggregate_pinn_ebm_upstream.py",
            "--run-dir-a",
            str(record["variants"]["A_upstream_active_0b74f6f"]),
            "--run-dir-b",
            str(record["variants"]["B_paper_spec_8x20"]),
            "--output",
            str(aggregate["json"]),
        ]
    if campaign == "structured_piv_combined_strict_aggregate":
        return [
            python_executable,
            "analysis/aggregate_injected_realpdebench.py",
            "--root",
            str(record["input_root"]),
            "--output",
            str(aggregate["json"]),
        ]
    raise AssertionError(campaign)


def main(args: argparse.Namespace) -> None:
    manifest = load_yaml(args.manifest)
    command = build_command(manifest, args.campaign)
    if args.dry_run:
        print(json.dumps({"campaign": args.campaign, "command": command}))
        return
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    summary = build_summary(args.manifest)
    status = summary["campaigns"][args.campaign]
    if not status.get("evidence_ready", False):
        raise RuntimeError(
            f"Strict aggregator returned success but recovery monitor did "
            f"not accept {args.campaign}: "
            f"{status.get('aggregate_validation_errors')}"
        )
    write_json_atomic(args.status_output, summary)
    print(
        json.dumps(
            {
                "campaign": args.campaign,
                "evidence_ready": True,
                "aggregate_json_sha256": status.get("json_sha256"),
                "status_output": str(args.status_output),
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS), required=True)
    parser.add_argument(
        "--status-output",
        type=Path,
        default=Path("outputs/status/reviewer_recovery_summary.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
