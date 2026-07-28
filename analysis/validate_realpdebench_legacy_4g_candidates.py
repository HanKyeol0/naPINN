"""Validate candidate-confirmation aggregation against its frozen manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_GROUPS = {
    ("mse", None),
    ("lad", None),
    ("orpinn_q29", None),
    ("pinn_ebm", 1.0),
    ("pinn_ebm", 50.0),
    ("napinn", None),
    ("napinn_lad", None),
    ("napinn_q29", None),
}
SELECTED_METHOD = {
    "napinn_mse": ("napinn", None),
    "napinn_l1": ("napinn_lad", None),
    "napinn_q29": ("napinn_q29", None),
}
NON_NAPINN = {
    ("mse", None),
    ("lad", None),
    ("orpinn_q29", None),
    ("pinn_ebm", 1.0),
    ("pinn_ebm", 50.0),
}


def scale_key(value: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(value["ratio"]),
        float(value["background_scale_multiplier"]),
        float(value["gross_scale_multiplier"]),
    )


def method_key(group: dict[str, Any]) -> tuple[str, float | None]:
    weight = group.get("direct_pinn_ebm_joint_pde_weight")
    return (
        str(group["method"]),
        None if weight is None else float(weight),
    )


def main(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    aggregate = json.loads(args.aggregation.read_text(encoding="utf-8"))
    expected_runs = int(manifest["expected_full_run_count"])
    expected_candidates = int(manifest["candidate_count"])
    if aggregate.get("n_eligible_runs") != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} runs, found "
            f"{aggregate.get('n_eligible_runs')}"
        )
    if aggregate.get("n_groups") != expected_candidates * 8:
        raise ValueError("Candidate aggregate has an incorrect group count")
    if len(aggregate.get("paired_input_blocks", [])) != expected_candidates * 3:
        raise ValueError("Candidate aggregate has an incorrect paired-block count")

    candidates = {
        scale_key(candidate): candidate for candidate in manifest["candidates"]
    }
    if len(candidates) != expected_candidates:
        raise ValueError("Candidate manifest contains duplicate scale cells")
    expected_hashes = {
        (
            float(record["ratio"]),
            float(record["background_scale_multiplier"]),
            float(record["gross_scale_multiplier"]),
            int(record["seed"]),
        ): record["artifact_sha256"]
        for record in manifest["artifacts_and_configs"]
    }
    grouped: dict[tuple[float, float, float], dict[tuple, dict]] = {
        key: {} for key in candidates
    }
    for group in aggregate["groups"]:
        key = scale_key(group)
        if key not in grouped:
            raise ValueError(f"Unexpected candidate scale group: {key}")
        method = method_key(group)
        if method in grouped[key]:
            raise ValueError(f"Duplicate candidate method group: {key}, {method}")
        if not group.get("complete_for_reporting_seeds"):
            raise ValueError(f"Incomplete reporting seeds: {group['group_id']}")
        if group.get("seeds") != [40, 41, 42]:
            raise ValueError(f"Incorrect reporting seeds: {group['group_id']}")
        for seed_text, artifact in group["input_artifacts_by_seed"].items():
            expected = expected_hashes[(*key, int(seed_text))]
            if artifact["sha256"] != expected:
                raise ValueError(
                    f"Manifest/artifact mismatch: {group['group_id']}, "
                    f"seed={seed_text}"
                )
        grouped[key][method] = group

    conclusions = []
    for key, methods in sorted(grouped.items()):
        if set(methods) != EXPECTED_GROUPS:
            raise ValueError(
                f"Candidate method mismatch in {key}: "
                f"missing={EXPECTED_GROUPS-set(methods)}, "
                f"unexpected={set(methods)-EXPECTED_GROUPS}"
            )
        candidate = candidates[key]
        selected_key = SELECTED_METHOD[candidate["selected_napinn_variant"]]
        selected = methods[selected_key]
        selected_rmae = selected["metrics"]["rMAE"]["mean"]
        selected_rmse = selected["metrics"]["rMSE"]["mean"]
        best_baseline_rmae = min(
            methods[method]["metrics"]["rMAE"]["mean"]
            for method in NON_NAPINN
        )
        best_baseline_rmse = min(
            methods[method]["metrics"]["rMSE"]["mean"]
            for method in NON_NAPINN
        )
        confirmed = (
            selected_rmae < best_baseline_rmae
            and selected_rmse < best_baseline_rmse
        )
        conclusions.append(
            {
                **candidate,
                "selected_reporting_rMAE_mean": selected_rmae,
                "selected_reporting_rMSE_mean": selected_rmse,
                "best_non_napinn_rMAE_mean": best_baseline_rmae,
                "best_non_napinn_rMSE_mean": best_baseline_rmse,
                "confirmed_win_by_frozen_definition": confirmed,
                "rMAE_relative_gain_over_best_non_napinn": (
                    (best_baseline_rmae - selected_rmae)
                    / best_baseline_rmae
                ),
                "rMSE_relative_gain_over_best_non_napinn": (
                    (best_baseline_rmse - selected_rmse)
                    / best_baseline_rmse
                ),
            }
        )
    for conclusion in conclusions:
        for key, value in conclusion.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite conclusion {key}: {value}")
    payload = {
        "schema_version": 1,
        "selection_id": manifest["selection_id"],
        "response_status": "RESPONSE HOLD",
        "expected_full_run_count": expected_runs,
        "verified_full_run_count": aggregate["n_eligible_runs"],
        "candidate_count": expected_candidates,
        "all_candidate_groups_complete": True,
        "all_paired_input_blocks_verified": all(
            block.get("paired_input_verified")
            for block in aggregate["paired_input_blocks"]
        ),
        "candidate_conclusions": conclusions,
        "positive_mixed_adverse_results_retained_in_aggregation": True,
        "aggregation_path": str(args.aggregation.resolve()),
        "manifest_path": str(args.manifest.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Validated {expected_runs} full runs across "
        f"{expected_candidates} candidate(s)."
    )
    print(f"Output: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--aggregation",
        type=Path,
        default=Path(
            "analysis/results/runs/"
            "rebuttal_realpde_legacy_4g_candidates/"
            "aggregation.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/runs/"
            "rebuttal_realpde_legacy_4g_candidates/"
            "candidate_validation.json"
        ),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
