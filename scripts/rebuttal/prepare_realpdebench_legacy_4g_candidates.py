#!/usr/bin/env python3
"""Materialize paired reporting artifacts/configs for frozen scale candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.rebuttal.inject_realpdebench_legacy_4g import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    generate,
    sha256_file,
)


REPORTING_SEEDS = (40, 41, 42)


def number_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def artifact_path(
    *,
    output_dir: Path,
    parent_stem: str,
    seed: int,
    ratio: float,
    base: float,
    gross: float,
) -> Path:
    return output_dir / (
        f"{parent_stem}_legacy4g_seed{seed}_r{int(round(100 * ratio))}_"
        f"b{number_token(base)}_o{number_token(gross)}.npz"
    )


def write_yaml_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=True)


def prepare(args: argparse.Namespace) -> Path:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    plan_hash = sha256_file(args.plan)
    if selection.get("plan_sha256") != plan_hash:
        raise ValueError("Selection does not match the frozen plan SHA-256")
    selected = selection.get("selected_candidates")
    if not isinstance(selected, list) or len(selected) > 3:
        raise ValueError("Selection must contain zero to three candidates")
    selection_id = str(selection.get("selection_id", ""))
    if len(selection_id) != 64:
        raise ValueError("Selection lacks a valid selection_id")

    parent = args.input.resolve()
    output_dir = args.artifact_output_dir.resolve()
    config_dir = args.config_output_dir.resolve()
    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    config_targets = []
    candidate_specs = []
    for rank, candidate in enumerate(selected, start=1):
        ratio = float(candidate["ratio"])
        base = float(candidate["background_scale_multiplier"])
        gross = float(candidate["gross_scale_multiplier"])
        if base == 1.0 and gross == 1.0:
            raise ValueError("Exact-center cell cannot be a selected candidate")
        variant = str(candidate["variant"])
        candidate_id = (
            f"rank{rank}_r{int(round(100 * ratio))}_"
            f"b{number_token(base)}_o{number_token(gross)}_{variant}"
        )
        candidate_specs.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "ratio": ratio,
                "background_scale_multiplier": base,
                "gross_scale_multiplier": gross,
                "selected_napinn_variant": variant,
                "development_ranking_score": float(
                    candidate["ranking_score"]
                ),
            }
        )
        for seed in REPORTING_SEEDS:
            config_targets.append(
                config_dir / f"{candidate_id}_seed{seed}.yaml"
            )
    collisions = [str(path) for path in config_targets if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite candidate config(s): "
            + ", ".join(collisions)
        )

    records = []
    for candidate in candidate_specs:
        ratio = candidate["ratio"]
        base = candidate["background_scale_multiplier"]
        gross = candidate["gross_scale_multiplier"]
        for seed in REPORTING_SEEDS:
            expected_artifact = artifact_path(
                output_dir=output_dir,
                parent_stem=parent.stem,
                seed=seed,
                ratio=ratio,
                base=base,
                gross=gross,
            )
            if expected_artifact.exists():
                sidecar = expected_artifact.with_suffix(".manifest.json")
                if not sidecar.is_file():
                    raise FileExistsError(
                        f"Existing candidate artifact lacks sidecar: "
                        f"{expected_artifact}"
                    )
                artifact_manifest = json.loads(
                    sidecar.read_text(encoding="utf-8")
                )
                if (
                    artifact_manifest.get("artifact_sha256")
                    != sha256_file(expected_artifact)
                ):
                    raise ValueError(
                        f"Existing candidate artifact hash mismatch: "
                        f"{expected_artifact}"
                    )
            else:
                generated = generate(
                    input_path=parent,
                    output_dir=output_dir,
                    seed=seed,
                    ratios=[ratio],
                    base_multipliers=[base],
                    gross_multipliers=[gross],
                )
                if len(generated) != 1 or generated[0][0] != expected_artifact:
                    raise AssertionError(
                        f"Unexpected generated target: {generated}"
                    )
            artifact_hash = sha256_file(expected_artifact)
            config_path = (
                config_dir
                / f"{candidate['candidate_id']}_seed{seed}.yaml"
            )
            suffix = (
                f"_candidate_{candidate['candidate_id']}_seed{seed}"
            )
            config = {
                "variant_tag_suffix": suffix,
                "selection_id": selection_id,
                "candidate_id": candidate["candidate_id"],
                "data": {"path": str(expected_artifact)},
                "output": {
                    "root": str(args.run_output_root),
                    "benchmark": "cylinder_real_piv_legacy4g_candidates",
                },
            }
            write_yaml_exclusive(config_path, config)
            records.append(
                {
                    **candidate,
                    "seed": seed,
                    "artifact_path": str(expected_artifact),
                    "artifact_sha256": artifact_hash,
                    "variant_config_path": str(config_path),
                    "variant_tag_suffix": suffix,
                }
            )

    manifest = {
        "schema_version": 1,
        "selection_path": str(args.selection.resolve()),
        "selection_sha256": sha256_file(args.selection),
        "selection_id": selection_id,
        "frozen_plan_path": str(args.plan.resolve()),
        "frozen_plan_sha256": plan_hash,
        "reporting_seeds": list(REPORTING_SEEDS),
        "candidate_count": len(candidate_specs),
        "conditions_per_candidate_seed": 8,
        "expected_full_run_count": len(candidate_specs) * 3 * 8,
        "candidates": candidate_specs,
        "artifacts_and_configs": records,
        "response_status": "RESPONSE HOLD",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        f"Prepared {len(records)} paired candidate artifact/config records "
        f"for {manifest['expected_full_run_count']} full runs."
    )
    print(f"Manifest: {manifest_path}")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde_legacy_4g_scale/"
            "seed39_scale_selection.json"
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("rebuttal/legacy4g_scale_selection_plan.yaml"),
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--artifact-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--config-output-dir",
        type=Path,
        default=Path(
            "analysis/results/runs/"
            "rebuttal_realpde_legacy_4g_candidates/configs"
        ),
    )
    parser.add_argument(
        "--run-output-root",
        type=Path,
        default=Path(
            "analysis/results/runs/"
            "rebuttal_realpde_legacy_4g_candidates"
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
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
