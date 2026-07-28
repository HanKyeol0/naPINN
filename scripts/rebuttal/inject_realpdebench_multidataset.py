#!/usr/bin/env python3
"""Inject the frozen paired/nested Legacy-4G corruption into real PIV NPZs.

For each dataset and corruption seed, one four-Gaussian background tensor,
one gross-row permutation, and one positive gross-factor tensor are shared by
the 10% and 15% artifacts.  The 10% gross rows are therefore an exact prefix
of (and subset of) the 15% rows.  Every method consumes the same immutable
artifact; this script has no method-dependent branch.

The corruption matches the submitted implementation semantics: four equally
weighted Gaussian components ``(-9,2), (-0.3,4), (2.7,0.6), (8.5,1)`` scaled
by ``0.1 * mean(abs(y/U_ref))``, followed by positive additive (not
replacement) gross offsets with factors sampled uniformly from ``[3,10]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("outputs/rebuttal/realpdebench_multidataset")
COMPONENTS = np.asarray(
    [(-9.0, 2.0), (-0.3, 4.0), (2.7, 0.6), (8.5, 1.0)],
    dtype=np.float64,
)
RATIOS = (0.10, 0.15)
SEEDS = (40, 41, 42)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _load_parent(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {
        "x_coord",
        "y_coord",
        "t_coord",
        "u_observed",
        "v_observed",
        "fluid_mask",
        "train_sensor_flat_indices",
        "heldout_flat_indices",
        "metadata_json",
    }
    with np.load(path, allow_pickle=False) as raw:
        missing = required.difference(raw.files)
        if missing:
            raise ValueError(f"{path} lacks arrays {sorted(missing)}")
        arrays = {name: raw[name].copy() for name in raw.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if "corruption" in metadata:
        raise ValueError(f"{path} is already corrupted")
    return arrays, metadata


def _validate_parent(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> dict[str, Any]:
    u = arrays["u_observed"]
    v = arrays["v_observed"]
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError("u/v must share (frame,height,width)")
    n_frames = u.shape[0]
    grid_size = int(np.prod(u.shape[1:]))
    train = arrays["train_sensor_flat_indices"].astype(np.int64, copy=False)
    heldout = arrays["heldout_flat_indices"].astype(np.int64, copy=False)
    valid = np.flatnonzero(arrays["fluid_mask"].reshape(-1))
    if np.intersect1d(train, heldout).size:
        raise ValueError("Training and held-out points overlap")
    if not np.array_equal(
        np.sort(np.concatenate((train, heldout))), valid
    ):
        raise ValueError("Training/held-out points do not partition fluid_mask")
    if np.any(train < 0) or np.any(train >= grid_size):
        raise ValueError("Training index outside grid")
    velocity_scale = float(
        metadata["characteristic_velocity_source_units"]
    )
    if not np.isfinite(velocity_scale) or velocity_scale <= 0.0:
        raise ValueError("Invalid characteristic velocity scale")
    reference = np.stack(
        (
            u.reshape(n_frames, -1)[:, train],
            v.reshape(n_frames, -1)[:, train],
        ),
        axis=-1,
    ).astype(np.float64)
    reference /= velocity_scale
    if not np.isfinite(reference).all():
        raise ValueError("Training reference is non-finite")
    return {
        "n_frames": n_frames,
        "n_sensors": int(train.size),
        "n_rows": int(n_frames * train.size),
        "train": train,
        "heldout": heldout,
        "non_training": np.setdiff1d(
            np.arange(grid_size, dtype=np.int64), train
        ),
        "velocity_scale": velocity_scale,
        "reference": reference.reshape(-1, 2),
        "reference_mean_abs": float(np.mean(np.abs(reference))),
        "reference_sample_std": float(np.std(reference, ddof=1)),
    }


def _realization(seed: int, n_rows: int, parent_hash: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    choices = rng.integers(
        0, COMPONENTS.shape[0], size=(n_rows, 2), dtype=np.uint8
    )
    standard_normal = rng.standard_normal((n_rows, 2))
    raw_background = (
        COMPONENTS[choices, 0] + COMPONENTS[choices, 1] * standard_normal
    )
    permutation = rng.permutation(n_rows).astype(np.int64)
    gross_factors = rng.uniform(3.0, 10.0, size=(n_rows, 2))
    checksums = {
        "component_choices_sha256": sha256_array(choices),
        "standard_normal_sha256": sha256_array(standard_normal),
        "raw_four_gaussian_sha256": sha256_array(raw_background),
        "gross_row_permutation_sha256": sha256_array(permutation),
        "positive_gross_factors_sha256": sha256_array(gross_factors),
    }
    pairing_id = hashlib.sha256(
        json.dumps(
            {
                "parent_sha256": parent_hash,
                "seed": seed,
                **checksums,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "choices": choices,
        "standard_normal": standard_normal,
        "raw_background": raw_background,
        "permutation": permutation,
        "gross_factors": gross_factors,
        "checksums": checksums,
        "pairing_id": pairing_id,
    }


def _write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def inject_parent(
    parent: Path,
    output_dir: Path,
    seed: int,
    ratios: tuple[float, ...] = RATIOS,
) -> list[dict[str, Any]]:
    parent = parent.resolve()
    arrays, metadata = _load_parent(parent)
    info = _validate_parent(arrays, metadata)
    parent_hash = sha256_file(parent)
    realization = _realization(seed, info["n_rows"], parent_hash)
    counts = {ratio: int(round(ratio * info["n_rows"])) for ratio in ratios}
    if any(count <= 0 or count >= info["n_rows"] for count in counts.values()):
        raise ValueError("Gross ratio resolves to zero or all rows")
    ordered_counts = [counts[ratio] for ratio in sorted(ratios)]
    if any(a >= b for a, b in zip(ordered_counts, ordered_counts[1:])):
        raise ValueError("Ratios must resolve to strictly increasing counts")
    ten_indices = realization["permutation"][: counts[0.10]]
    fifteen_indices = realization["permutation"][: counts[0.15]]
    if not np.array_equal(ten_indices, fifteen_indices[: ten_indices.size]):
        raise AssertionError("10% gross indices are not the exact 15% prefix")
    if not set(ten_indices.tolist()).issubset(fifteen_indices.tolist()):
        raise AssertionError("10% gross indices are not a subset of 15%")

    background_scale = 0.1 * info["reference_mean_abs"]
    background = realization["raw_background"] * background_scale
    background_std = float(np.std(background, ddof=1))
    if not np.isfinite(background_std) or background_std <= 0.0:
        raise ValueError("Invalid realized background scale")
    clean_u = arrays["u_observed"].copy()
    clean_v = arrays["v_observed"].copy()
    outputs: list[dict[str, Any]] = []

    for ratio in ratios:
        n_gross = counts[ratio]
        gross_rows = realization["permutation"][:n_gross].copy()
        gross_offsets = np.zeros_like(background)
        gross_offsets[gross_rows] = (
            realization["gross_factors"][gross_rows] * background_std
        )
        if not np.all(gross_offsets[gross_rows] > 0.0):
            raise AssertionError("Gross offsets are not strictly positive")
        observed = info["reference"] + background + gross_offsets

        corrupted_u = clean_u.copy()
        corrupted_v = clean_v.copy()
        corrupted_u.reshape(info["n_frames"], -1)[:, info["train"]] = (
            observed[:, 0].reshape(info["n_frames"], info["n_sensors"])
            * info["velocity_scale"]
        ).astype(clean_u.dtype)
        corrupted_v.reshape(info["n_frames"], -1)[:, info["train"]] = (
            observed[:, 1].reshape(info["n_frames"], info["n_sensors"])
            * info["velocity_scale"]
        ).astype(clean_v.dtype)
        for corrupted, clean in (
            (corrupted_u, clean_u),
            (corrupted_v, clean_v),
        ):
            if not np.array_equal(
                corrupted.reshape(info["n_frames"], -1)[
                    :, info["non_training"]
                ],
                clean.reshape(info["n_frames"], -1)[
                    :, info["non_training"]
                ],
            ):
                raise AssertionError("Non-training values changed")
        gross_mask = np.zeros(
            (info["n_frames"], info["n_sensors"], 2), dtype=bool
        )
        gross_mask.reshape(-1, 2)[gross_rows] = True

        corruption = {
            "schema_version": 1,
            "kind": "exact_legacy_4g_positive_additive_gross_offsets",
            "corruption_seed": seed,
            "gross_row_ratio_requested": ratio,
            "gross_row_ratio_realized": n_gross / info["n_rows"],
            "n_training_vector_rows": info["n_rows"],
            "n_gross_vector_rows": n_gross,
            "raw_gmm_components_mean_std": COMPONENTS.tolist(),
            "raw_gmm_component_weights": [0.25] * 4,
            "legacy_background_scale_formula": (
                "0.1 * pooled mean(abs(pre-injection y/U_ref))"
            ),
            "resolved_background_scale_nondimensional": background_scale,
            "realized_background_sample_std_nondimensional": background_std,
            "gross_offset_factor_distribution": "Uniform(3,10)",
            "gross_offset_semantics": (
                "Strictly positive additive u/v offsets after background "
                "noise; observations are not replaced."
            ),
            "label_semantics": (
                "training_corruption_mask labels gross-offset components; "
                "false components still contain Legacy-4G background noise."
            ),
            "pairing": {
                "pairing_id": realization["pairing_id"],
                "parent_artifact_sha256": parent_hash,
                "same_realization_shared_across_ratios": True,
                "gross_sets_are_permutation_prefixes": True,
                "ten_percent_subset_of_fifteen_percent": True,
                "same_artifact_shared_by_all_methods": True,
                "ten_percent_row_count": int(ten_indices.size),
                "fifteen_percent_row_count": int(fifteen_indices.size),
                "ten_percent_exact_prefix_of_fifteen_percent": True,
                "ten_percent_gross_indices_sha256": sha256_array(ten_indices),
                "fifteen_percent_gross_indices_sha256": sha256_array(
                    fifteen_indices
                ),
            },
            "checksums": {
                **realization["checksums"],
                "background_noise_sha256": sha256_array(background),
                "gross_row_indices_sha256": sha256_array(gross_rows),
                "gross_offsets_sha256": sha256_array(gross_offsets),
            },
            "invariants": {
                "heldout_bitwise_unchanged": True,
                "all_non_training_values_bitwise_unchanged": True,
                "gross_offsets_strictly_positive": True,
                "ten_percent_subset_of_fifteen_percent": True,
            },
        }
        derived_metadata = json.loads(json.dumps(metadata))
        derived_metadata["corruption"] = corruption
        ratio_percent = int(round(100 * ratio))
        output = output_dir / (
            f"{parent.stem}_legacy4g_seed{seed}_r{ratio_percent}.npz"
        )
        manifest_path = output.with_suffix(".manifest.json")
        if output.exists() or manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output_arrays = {
            **arrays,
            "u_clean": clean_u,
            "v_clean": clean_v,
            "u_observed": corrupted_u,
            "v_observed": corrupted_v,
            "training_corruption_mask": gross_mask,
            "legacy4g_component_choices": realization["choices"],
            "legacy4g_standard_normal": realization["standard_normal"],
            "legacy4g_raw_background": realization["raw_background"],
            "legacy4g_background_noise_nondimensional": background,
            "legacy4g_gross_row_permutation": realization["permutation"],
            "legacy4g_positive_gross_factors": realization["gross_factors"],
            "legacy4g_gross_row_indices": gross_rows,
            "legacy4g_gross_offsets_nondimensional": gross_offsets,
            "metadata_json": np.asarray(
                json.dumps(derived_metadata, sort_keys=True)
            ),
        }
        _write_npz_exclusive(output, output_arrays)
        record = {
            "schema_version": 1,
            "dataset_family": metadata["dataset_family"],
            "sim_id": metadata["sim_id"],
            "artifact_path": str(output.resolve()),
            "artifact_sha256": sha256_file(output),
            "parent_artifact_path": str(parent),
            "parent_artifact_sha256": parent_hash,
            "corruption": corruption,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in output_arrays.items()
            },
        }
        try:
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except BaseException:
            output.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise
        outputs.append(record)
    if len(outputs) != 2:
        raise AssertionError("Frozen protocol requires exactly two ratio outputs")
    first, second = (record["corruption"] for record in outputs)
    if first["pairing"]["pairing_id"] != second["pairing"]["pairing_id"]:
        raise AssertionError("Paired artifacts have different pairing IDs")
    shared_checksum_keys = {
        "component_choices_sha256",
        "standard_normal_sha256",
        "raw_four_gaussian_sha256",
        "gross_row_permutation_sha256",
        "positive_gross_factors_sha256",
        "background_noise_sha256",
    }
    for key in shared_checksum_keys:
        if first["checksums"][key] != second["checksums"][key]:
            raise AssertionError(
                f"Paired artifacts differ in shared checksum {key}"
            )
    if (
        first["pairing"]["ten_percent_gross_indices_sha256"]
        != second["pairing"]["ten_percent_gross_indices_sha256"]
        or first["pairing"]["fifteen_percent_gross_indices_sha256"]
        != second["pairing"]["fifteen_percent_gross_indices_sha256"]
    ):
        raise AssertionError("Paired manifests disagree on nested index hashes")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--scenario",
        choices=["controlled_cylinder", "fsi", "foil", "all"],
        default="all",
    )
    parser.add_argument(
        "--seed", type=int, choices=list(SEEDS), action="append"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    prepared_manifest = json.loads(
        (root / "prepared_manifest.json").read_text(encoding="utf-8")
    )
    scenarios = (
        ["controlled_cylinder", "fsi", "foil"]
        if args.scenario == "all"
        else [args.scenario]
    )
    seeds = args.seed or list(SEEDS)
    parent_by_scenario = {
        artifact["dataset_family"]: Path(artifact["prepared_npz_path"])
        for artifact in prepared_manifest["artifacts"]
    }
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            records.extend(
                inject_parent(
                    parent_by_scenario[scenario],
                    root / "corrupted",
                    seed,
                )
            )
    manifest = {
        "schema_version": 1,
        "status": "corrupted_inputs_only_not_performance_evidence",
        "frozen_protocol": {
            "scenarios": ["controlled_cylinder", "fsi", "foil"],
            "seeds": list(SEEDS),
            "gross_row_ratios": list(RATIOS),
            "methods_share_each_artifact": True,
        },
        "artifacts": records,
    }
    path = root / "corruption_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} corruption artifacts")
    print(f"Manifest: {path}")


if __name__ == "__main__":
    main()
