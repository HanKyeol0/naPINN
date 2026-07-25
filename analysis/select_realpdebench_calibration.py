"""Select the PIV naPINN rejection cost using the recorded seed-39 rule.

This script refuses to select until all five predeclared calibration cells
exist as complete, non-smoke artifacts. It reads only seed 39 and writes the
full candidate table, so the choice is reproducible and adverse calibration
outcomes remain visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANDIDATES = {
    0.005: (
        "napinn_scaled_rejection_sensor_drift30",
        Path(
            "configs/experiment/"
            "realpdebench_cylinder_napinn_scaled_rejection.yaml"
        ),
    ),
    0.01: (
        "napinn_rejection_001_sensor_drift30",
        Path(
            "configs/experiment/"
            "realpdebench_cylinder_napinn_rejection_001.yaml"
        ),
    ),
    0.05: (
        "napinn_rejection_005_sensor_drift30",
        Path(
            "configs/experiment/"
            "realpdebench_cylinder_napinn_rejection_005.yaml"
        ),
    ),
    0.10: (
        "napinn_rejection_01_sensor_drift30",
        Path(
            "configs/experiment/"
            "realpdebench_cylinder_napinn_rejection_01.yaml"
        ),
    ),
    0.50: (
        "napinn_sensor_drift30",
        Path("configs/experiment/realpdebench_cylinder_napinn.yaml"),
    ),
}


def load_candidate(root: Path, cost: float, tag: str) -> dict:
    matches = []
    for path in (root / tag).rglob("metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            int(payload.get("seed", -1)) == 39
            and not bool(payload.get("smoke_test", True))
            and payload.get("evidence_status")
            == "full_run_complete_unaggregated"
        ):
            matches.append((path, payload))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one full seed-39 artifact for cost={cost:g}, "
            f"tag={tag!r}; found {len(matches)}: "
            f"{[str(path) for path, _ in matches]}"
        )
    path, payload = matches[0]
    required = (
        "rMAE",
        "rMSE",
        "failure_detection_auroc",
        "failed_rejection_rate",
        "clean_rejection_rate",
        "retained_fraction",
    )
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise ValueError(f"Missing calibration metrics {missing}: {path}")
    failed = float(payload["failed_rejection_rate"])
    clean = float(payload["clean_rejection_rate"])
    return {
        "cost": cost,
        "tag": tag,
        "metrics_path": str(path.resolve()),
        "rMAE": float(payload["rMAE"]),
        "rMSE": float(payload["rMSE"]),
        "failure_detection_auroc": float(
            payload["failure_detection_auroc"]
        ),
        "failed_rejection_rate": failed,
        "clean_rejection_rate": clean,
        "retained_fraction": float(payload["retained_fraction"]),
        "eligible": failed >= 0.90 and clean <= 0.40,
        "balanced_rejection_error": 1.0 - failed + clean,
    }


def main(args: argparse.Namespace) -> None:
    rows = []
    config_by_cost = {}
    for cost, (tag, config_path) in CANDIDATES.items():
        rows.append(load_candidate(args.root, cost, tag))
        config_by_cost[cost] = config_path
    eligible = [row for row in rows if row["eligible"]]
    if eligible:
        selected = min(eligible, key=lambda row: (row["rMAE"], row["cost"]))
        branch = "eligible_then_lowest_seed39_rMAE"
    else:
        selected = min(
            rows,
            key=lambda row: (
                row["balanced_rejection_error"],
                row["rMAE"],
                row["cost"],
            ),
        )
        branch = "fallback_balanced_rejection_error_then_rMAE"
    selected_config = config_by_cost[float(selected["cost"])]
    payload = {
        "calibration_seed": 39,
        "condition": "30pct_persistent_sensor_bias_plus_linear_drift",
        "chronology_limitation": (
            "The 10% cost-0.01 held-out cells were completed before the "
            "0.05/0.10 grid refinement; do not describe this as fully "
            "held-out-blind tuning. Selection itself reads seed 39 only."
        ),
        "selection_rule": (
            "Among failed_rejection_rate>=0.90 and "
            "clean_rejection_rate<=0.40, minimize seed-39 rMAE. If none "
            "is eligible, minimize 1-failed_rejection_rate+"
            "clean_rejection_rate, then rMAE."
        ),
        "selection_branch": branch,
        "candidates": rows,
        "selected_cost": selected["cost"],
        "selected_tag": selected["tag"],
        "selected_config": str(selected_config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde/cylinder_real_piv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde/"
            "piv_rejection_calibration_selection.json"
        ),
    )
    main(parser.parse_args())
