#!/usr/bin/env python3
"""Inject the legacy four-Gaussian plus positive point-outlier PIV corruption.

The generator operates only on the fixed training measurements of an
unmodified prepared Cylinder artifact.  One seeded random realization is
shared across every requested gross-row ratio and ``(b, o)`` scale cell:

* ``b`` multiplies the submitted background scale ``0.1 * mean(abs(y))``;
* ``o`` multiplies the submitted positive gross-factor range ``U(3, 10)``;
* gross-row sets are prefixes of one permutation, so ratios are nested; and
* both velocity components of a gross row receive independent positive
  offsets based on the realized background standard deviation.

Every training component receives four-Gaussian background noise.  Therefore
``training_corruption_mask == False`` means background-only, not physically
clean.  Held-out and all other non-training spatial values remain bitwise
identical to the parent artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LEGACY_NOISE_SCALE = 0.1
LEGACY_GROSS_FACTOR_RANGE = (3.0, 10.0)
LEGACY_COMPONENTS = np.asarray(
    [
        [-9.0, 2.0],
        [-0.3, 4.0],
        [2.7, 0.6],
        [8.5, 1.0],
    ],
    dtype=np.float64,
)
DEFAULT_INPUT = Path(
    "analysis/results/runs/rebuttal_realpde/data/"
    "realpdebench_cylinder_10031_frames1000-1200_stride1.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "analysis/results/runs/rebuttal_realpde_legacy_4g/data"
)
CHECKSUM_SCHEME = "sha256(dtype.str + JSON(shape) + C-order array bytes)"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Return a dtype/shape-aware SHA-256 for a NumPy array."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _git_output(repo: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def generator_provenance() -> dict[str, Any]:
    """Capture the generator source and current repository state."""

    script = Path(__file__).resolve()
    repo = script.parents[2]
    commit = _git_output(repo, "rev-parse", "HEAD")
    status = _git_output(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    tracked = _git_output(
        repo,
        "ls-files",
        "--error-unmatch",
        str(script.relative_to(repo)),
    )
    return {
        "generator_path": str(script),
        "generator_sha256": sha256_file(script),
        "source_commit": commit,
        "source_dirty": status is None or bool(status),
        "source_status_porcelain": (
            status.splitlines() if status is not None else ["unavailable"]
        ),
        "generator_tracked_at_source_commit": tracked is not None,
    }


def _unique_sorted(values: Iterable[float], name: str) -> list[float]:
    resolved = sorted({float(value) for value in values})
    if not resolved:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(resolved).all():
        raise ValueError(f"{name} must contain only finite values")
    return resolved


def _validate_controls(
    ratios: Iterable[float],
    base_multipliers: Iterable[float],
    gross_multipliers: Iterable[float],
) -> tuple[list[float], list[float], list[float]]:
    ratio_values = _unique_sorted(ratios, "ratios")
    base_values = _unique_sorted(base_multipliers, "base_multipliers")
    gross_values = _unique_sorted(gross_multipliers, "gross_multipliers")
    if any(not 0.0 < value < 1.0 for value in ratio_values):
        raise ValueError("Every gross-row ratio must lie strictly in (0, 1)")
    if any(value <= 0.0 for value in base_values):
        raise ValueError("Every base-scale multiplier must be positive")
    if any(value <= 0.0 for value in gross_values):
        raise ValueError("Every gross-scale multiplier must be positive")
    return ratio_values, base_values, gross_values


def _number_token(value: float) -> str:
    token = f"{value:.12g}"
    return token.replace("-", "m").replace(".", "p")


def _ratio_token(ratio: float) -> str:
    percent = 100.0 * ratio
    rounded = round(percent)
    if np.isclose(percent, rounded, rtol=0.0, atol=1e-10):
        return str(int(rounded))
    return _number_token(percent)


def artifact_name(
    parent_stem: str,
    seed: int,
    ratio: float,
    base_multiplier: float,
    gross_multiplier: float,
) -> str:
    return (
        f"{parent_stem}_legacy4g_seed{seed}_r{_ratio_token(ratio)}"
        f"_b{_number_token(base_multiplier)}"
        f"_o{_number_token(gross_multiplier)}.npz"
    )


def _load_parent(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {
        "x_m",
        "y_m",
        "t_s",
        "u_mps",
        "v_mps",
        "fluid_mask",
        "train_sensor_flat_indices",
        "heldout_flat_indices",
        "metadata_json",
    }
    with np.load(path, allow_pickle=False) as raw:
        missing = required.difference(raw.files)
        if missing:
            raise ValueError(f"Parent artifact lacks arrays: {sorted(missing)}")
        arrays = {name: raw[name].copy() for name in raw.files}

    metadata = json.loads(str(arrays["metadata_json"].item()))
    if "corruption" in metadata:
        raise ValueError("Parent artifact already contains a corruption record")
    if "training_corruption_mask" in arrays:
        raise ValueError("Parent artifact already contains corruption labels")
    return arrays, metadata


def _validate_parent(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    clean_u = arrays["u_mps"]
    clean_v = arrays["v_mps"]
    if clean_u.shape != clean_v.shape or clean_u.ndim != 3:
        raise ValueError("u_mps and v_mps must share shape (frames, height, width)")
    if arrays["x_m"].shape != clean_u.shape[1:]:
        raise ValueError("x_m shape does not match the velocity grid")
    if arrays["y_m"].shape != clean_u.shape[1:]:
        raise ValueError("y_m shape does not match the velocity grid")
    if arrays["t_s"].size != clean_u.shape[0]:
        raise ValueError("t_s length does not match the velocity frame count")

    n_frames = clean_u.shape[0]
    grid_size = clean_u.shape[1] * clean_u.shape[2]
    train = arrays["train_sensor_flat_indices"].astype(np.int64, copy=False)
    heldout = arrays["heldout_flat_indices"].astype(np.int64, copy=False)
    if train.ndim != 1 or heldout.ndim != 1:
        raise ValueError("Training and held-out indices must be one-dimensional")
    if np.unique(train).size != train.size:
        raise ValueError("Training sensor indices are not unique")
    if np.unique(heldout).size != heldout.size:
        raise ValueError("Held-out indices are not unique")
    if (
        np.any(train < 0)
        or np.any(train >= grid_size)
        or np.any(heldout < 0)
        or np.any(heldout >= grid_size)
    ):
        raise ValueError("Training or held-out index is outside the spatial grid")
    if np.intersect1d(train, heldout, assume_unique=True).size:
        raise ValueError("Training and held-out spatial indices overlap")

    fluid_indices = np.flatnonzero(arrays["fluid_mask"].reshape(-1))
    split_indices = np.sort(np.concatenate((train, heldout)))
    if not np.array_equal(split_indices, fluid_indices):
        raise ValueError(
            "Training plus held-out indices do not exactly partition fluid_mask"
        )

    reynolds = float(metadata["reynolds_number"])
    diameter = float(metadata["cylinder_diameter_m"])
    viscosity = float(metadata["water_kinematic_viscosity_m2ps"])
    if reynolds <= 0.0 or diameter <= 0.0 or viscosity <= 0.0:
        raise ValueError("Invalid Reynolds number, diameter, or viscosity")
    velocity_scale = reynolds * viscosity / diameter
    recorded_scale = float(
        metadata.get("characteristic_velocity_mps", velocity_scale)
    )
    if not np.isclose(recorded_scale, velocity_scale, rtol=1e-10, atol=0.0):
        raise ValueError("Parent artifact has an inconsistent velocity scale")

    training_mps = np.stack(
        (
            clean_u.reshape(n_frames, -1)[:, train],
            clean_v.reshape(n_frames, -1)[:, train],
        ),
        axis=-1,
    )
    training_reference = training_mps.astype(np.float64) / velocity_scale
    if not np.isfinite(training_reference).all():
        raise ValueError("Training reference contains non-finite values")
    reference_mean_abs = float(np.mean(np.abs(training_reference)))
    reference_std = float(np.std(training_reference, ddof=1))
    if reference_mean_abs <= 0.0 or reference_std <= 0.0:
        raise ValueError("Training reference has a degenerate pooled scale")

    return {
        "n_frames": n_frames,
        "grid_size": grid_size,
        "train_indices": train,
        "heldout_indices": heldout,
        "non_training_indices": np.setdiff1d(
            np.arange(grid_size, dtype=np.int64),
            train,
            assume_unique=False,
        ),
        "n_sensors": int(train.size),
        "n_training_rows": int(n_frames * train.size),
        "velocity_scale_mps": velocity_scale,
        "training_reference_nondimensional": training_reference.reshape(-1, 2),
        "reference_mean_abs": reference_mean_abs,
        "reference_std": reference_std,
    }


def _distribution_diagnostics() -> dict[str, float]:
    means = LEGACY_COMPONENTS[:, 0]
    stds = LEGACY_COMPONENTS[:, 1]
    mixture_mean = float(means.mean())
    second_moment = float(np.mean(stds**2 + means**2))
    variance = second_moment - mixture_mean**2
    return {
        "raw_mixture_mean": mixture_mean,
        "raw_mixture_variance": variance,
        "raw_mixture_std": float(np.sqrt(variance)),
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    quantile_levels = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    quantiles = np.quantile(flattened, quantile_levels)
    return {
        "min": float(quantiles[0]),
        "median": float(quantiles[4]),
        "max": float(quantiles[-1]),
        "quantiles": {
            f"q{int(round(level * 100)):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
    }


def _pairing_realization(
    seed: int,
    n_rows: int,
    parent_sha256: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    component_choices = rng.integers(
        0,
        LEGACY_COMPONENTS.shape[0],
        size=(n_rows, 2),
        dtype=np.uint8,
    )
    gaussian_draws = rng.standard_normal(size=(n_rows, 2))
    raw_draws = (
        LEGACY_COMPONENTS[component_choices, 0]
        + LEGACY_COMPONENTS[component_choices, 1] * gaussian_draws
    )
    row_permutation = rng.permutation(n_rows).astype(np.int64)
    base_uniform_factors = rng.uniform(
        LEGACY_GROSS_FACTOR_RANGE[0],
        LEGACY_GROSS_FACTOR_RANGE[1],
        size=(n_rows, 2),
    )
    checksums = {
        "mixture_component_choices_sha256": sha256_array(component_choices),
        "standard_gaussian_draws_sha256": sha256_array(gaussian_draws),
        "raw_four_gaussian_draws_sha256": sha256_array(raw_draws),
        "gross_row_permutation_sha256": sha256_array(row_permutation),
        "base_uniform_factors_sha256": sha256_array(base_uniform_factors),
    }
    pairing_payload = {
        "parent_artifact_sha256": parent_sha256,
        "corruption_seed": int(seed),
        **checksums,
    }
    pairing_id = hashlib.sha256(
        json.dumps(pairing_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "component_choices": component_choices,
        "gaussian_draws": gaussian_draws,
        "raw_draws": raw_draws,
        "row_permutation": row_permutation,
        "base_uniform_factors": base_uniform_factors,
        "checksums": checksums,
        "pairing_id": pairing_id,
    }


def _target_paths(
    parent: Path,
    output_dir: Path,
    seed: int,
    ratios: Iterable[float],
    base_multipliers: Iterable[float],
    gross_multipliers: Iterable[float],
) -> list[tuple[float, float, float, Path, Path]]:
    targets = []
    for ratio in ratios:
        for base_multiplier in base_multipliers:
            for gross_multiplier in gross_multipliers:
                output = output_dir / artifact_name(
                    parent.stem,
                    seed,
                    ratio,
                    base_multiplier,
                    gross_multiplier,
                )
                manifest = output.with_suffix(".manifest.json")
                targets.append(
                    (
                        ratio,
                        base_multiplier,
                        gross_multiplier,
                        output,
                        manifest,
                    )
                )
    if len({target[3] for target in targets}) != len(targets):
        raise ValueError("Requested controls resolve to duplicate output names")
    collisions = [
        str(path)
        for target in targets
        for path in target[3:]
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing artifact(s): "
            + ", ".join(collisions)
        )
    return targets


def _write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def generate(
    *,
    input_path: Path,
    output_dir: Path,
    seed: int,
    ratios: Iterable[float],
    base_multipliers: Iterable[float],
    gross_multipliers: Iterable[float],
) -> list[tuple[Path, Path]]:
    """Generate every requested paired corruption artifact."""

    source = input_path.resolve()
    destination = output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    ratio_values, base_values, gross_multiplier_values = _validate_controls(
        ratios,
        base_multipliers,
        gross_multipliers,
    )
    targets = _target_paths(
        source,
        destination,
        seed,
        ratio_values,
        base_values,
        gross_multiplier_values,
    )
    arrays, metadata = _load_parent(source)
    parent_info = _validate_parent(arrays, metadata)
    parent_sha256 = sha256_file(source)
    provenance = generator_provenance()
    realization = _pairing_realization(
        seed,
        parent_info["n_training_rows"],
        parent_sha256,
    )
    reference = parent_info["training_reference_nondimensional"]
    row_permutation = realization["row_permutation"]
    base_uniform_factors = realization["base_uniform_factors"]
    requested_counts = {
        f"{ratio:.12g}": int(round(ratio * parent_info["n_training_rows"]))
        for ratio in ratio_values
    }
    if any(
        count <= 0 or count >= parent_info["n_training_rows"]
        for count in requested_counts.values()
    ):
        raise ValueError("A requested ratio resolves to zero or all training rows")
    nested = all(
        set(row_permutation[:smaller_count]).issubset(
            row_permutation[:larger_count]
        )
        for smaller_count, larger_count in zip(
            sorted(requested_counts.values()),
            sorted(requested_counts.values())[1:],
        )
    )
    if not nested:
        raise AssertionError("Gross-row prefix sets are not nested")

    clean_u = arrays["u_mps"].copy()
    clean_v = arrays["v_mps"].copy()
    train = parent_info["train_indices"]
    heldout = parent_info["heldout_indices"]
    non_training = parent_info["non_training_indices"]
    n_frames = parent_info["n_frames"]
    n_sensors = parent_info["n_sensors"]
    output_records: list[tuple[Path, Path]] = []
    destination.mkdir(parents=True, exist_ok=True)

    for ratio, base_multiplier, gross_multiplier, output, manifest_path in targets:
        n_gross = int(round(ratio * parent_info["n_training_rows"]))
        gross_rows = row_permutation[:n_gross].copy()
        background_scale = (
            LEGACY_NOISE_SCALE
            * base_multiplier
            * parent_info["reference_mean_abs"]
        )
        background = realization["raw_draws"] * background_scale
        background_mean = float(np.mean(background))
        background_std = float(np.std(background, ddof=1))
        if not np.isfinite(background_std) or background_std <= 0.0:
            raise ValueError("Realized background standard deviation is invalid")

        resolved_factors = base_uniform_factors * gross_multiplier
        gross_offsets = np.zeros_like(background)
        gross_offsets[gross_rows] = (
            resolved_factors[gross_rows] * background_std
        )
        observed = reference + background + gross_offsets

        corrupted_u = clean_u.copy()
        corrupted_v = clean_v.copy()
        corrupted_u.reshape(n_frames, -1)[:, train] = (
            observed[:, 0].reshape(n_frames, n_sensors)
            * parent_info["velocity_scale_mps"]
        ).astype(clean_u.dtype)
        corrupted_v.reshape(n_frames, -1)[:, train] = (
            observed[:, 1].reshape(n_frames, n_sensors)
            * parent_info["velocity_scale_mps"]
        ).astype(clean_v.dtype)

        if not np.array_equal(
            corrupted_u.reshape(n_frames, -1)[:, heldout],
            clean_u.reshape(n_frames, -1)[:, heldout],
        ) or not np.array_equal(
            corrupted_v.reshape(n_frames, -1)[:, heldout],
            clean_v.reshape(n_frames, -1)[:, heldout],
        ):
            raise AssertionError("Held-out PIV values were modified")
        if not np.array_equal(
            corrupted_u.reshape(n_frames, -1)[:, non_training],
            clean_u.reshape(n_frames, -1)[:, non_training],
        ) or not np.array_equal(
            corrupted_v.reshape(n_frames, -1)[:, non_training],
            clean_v.reshape(n_frames, -1)[:, non_training],
        ):
            raise AssertionError("A non-training spatial value was modified")

        gross_mask = np.zeros(
            (n_frames, n_sensors, 2),
            dtype=bool,
        )
        gross_mask.reshape(-1, 2)[gross_rows] = True
        if int(gross_mask.sum()) != 2 * n_gross:
            raise AssertionError("Gross-outlier mask count is inconsistent")

        gross_offset_values = gross_offsets[gross_rows]
        gross_background_units = gross_offset_values / background_std
        gross_reference_units = (
            gross_offset_values / parent_info["reference_std"]
        )
        ratio_key = f"{ratio:.12g}"
        ten_count = int(round(0.10 * parent_info["n_training_rows"]))
        fifteen_count = int(round(0.15 * parent_info["n_training_rows"]))
        ten_subset_fifteen = bool(
            set(row_permutation[:ten_count]).issubset(
                row_permutation[:fifteen_count]
            )
        )
        corruption = {
            "schema_version": 1,
            "kind": "legacy_four_gaussian_plus_positive_point_outliers",
            "seed": int(seed),
            "scope": (
                "All training components receive background 4G noise; gross "
                "positive offsets affect selected training vector rows only; "
                "held-out and other non-training spatial values are unchanged."
            ),
            "label_semantics": (
                "training_corruption_mask marks gross point-outlier scalar "
                "components only; false means background-only, not clean."
            ),
            "gross_offset_semantics": (
                "Independent positive additive u/v offsets are added to the "
                "background-corrupted values, not used as replacements."
            ),
            "scalar_sampling_order": (
                "Time-major training rows; u then v within each row."
            ),
            "raw_gmm_components_mean_std": LEGACY_COMPONENTS.tolist(),
            "raw_gmm_component_weights": [0.25, 0.25, 0.25, 0.25],
            **_distribution_diagnostics(),
            "n_training_rows": parent_info["n_training_rows"],
            "n_training_scalar_components": (
                2 * parent_info["n_training_rows"]
            ),
            "gross_row_ratio_requested": ratio,
            "gross_row_ratio_realized": (
                n_gross / parent_info["n_training_rows"]
            ),
            "n_gross_rows": n_gross,
            "n_gross_scalar_components": 2 * n_gross,
            "base_scale_multiplier": base_multiplier,
            "legacy_noise_scale": LEGACY_NOISE_SCALE,
            "resolved_background_scale_nondimensional": background_scale,
            "gross_scale_multiplier": gross_multiplier,
            "base_gross_factor_range": list(LEGACY_GROSS_FACTOR_RANGE),
            "resolved_gross_factor_range": [
                LEGACY_GROSS_FACTOR_RANGE[0] * gross_multiplier,
                LEGACY_GROSS_FACTOR_RANGE[1] * gross_multiplier,
            ],
            "reference_diagnostics": {
                "terminology": "pre-injection PIV training measurements",
                "characteristic_velocity_mps": (
                    parent_info["velocity_scale_mps"]
                ),
                "pooled_mean_abs_nondimensional": (
                    parent_info["reference_mean_abs"]
                ),
                "pooled_sample_std_nondimensional": (
                    parent_info["reference_std"]
                ),
            },
            "realized_background": {
                "mean_nondimensional": background_mean,
                "sample_std_nondimensional": background_std,
                "sample_std_over_reference_sample_std": (
                    background_std / parent_info["reference_std"]
                ),
            },
            "realized_gross_offsets": {
                "nondimensional": _summary(gross_offset_values),
                "background_std_units": _summary(gross_background_units),
                "reference_std_units": _summary(gross_reference_units),
            },
            "gross_row_indices": gross_rows.tolist(),
            "pairing": {
                "pairing_id": realization["pairing_id"],
                "requested_ratios": ratio_values,
                "requested_base_scale_multipliers": base_values,
                "requested_gross_scale_multipliers": (
                    gross_multiplier_values
                ),
                "requested_ratio_row_counts": requested_counts,
                "one_raw_background_realization_shared": True,
                "one_row_permutation_shared": True,
                "one_base_uniform_factor_tensor_shared": True,
                "requested_ratio_sets_are_nested": nested,
                "ten_percent_indices_subset_of_fifteen_percent": (
                    ten_subset_fifteen
                ),
            },
            "checksums": {
                "scheme": CHECKSUM_SCHEME,
                **realization["checksums"],
                "background_noise_tensor_sha256": sha256_array(background),
                "gross_row_indices_sha256": sha256_array(gross_rows),
                "resolved_gross_factors_sha256": sha256_array(
                    resolved_factors
                ),
                "gross_offset_tensor_sha256": sha256_array(gross_offsets),
            },
            "invariants": {
                "heldout_piv_bitwise_unchanged": True,
                "all_non_training_spatial_values_bitwise_unchanged": True,
                "background_applied_to_every_training_component": True,
                "gross_mask_marks_only_gross_offsets": True,
                "gross_offsets_strictly_positive": bool(
                    np.all(gross_offset_values > 0.0)
                ),
                "gross_row_component_count_matches_mask": True,
                "requested_ratio_sets_are_nested": nested,
                "ten_percent_indices_subset_of_fifteen_percent": (
                    ten_subset_fifteen
                ),
            },
            "parent_artifact_path": str(source),
            "parent_artifact_sha256": parent_sha256,
            "generator_provenance": provenance,
            "final_artifact_sha256_record": (
                "See the exclusive sidecar .manifest.json written after NPZ "
                "serialization; an NPZ cannot contain its own final checksum."
            ),
        }
        artifact_metadata = json.loads(json.dumps(metadata))
        artifact_metadata["corruption"] = corruption
        artifact_metadata["notes"] = (
            str(artifact_metadata.get("notes", ""))
            + " This derived artifact applies paired legacy four-Gaussian "
            "background noise and labeled positive point-outlier offsets only "
            "to training measurements."
        ).strip()
        output_arrays = {
            **arrays,
            "u_clean_mps": clean_u,
            "v_clean_mps": clean_v,
            "u_mps": corrupted_u,
            "v_mps": corrupted_v,
            "training_corruption_mask": gross_mask,
            "legacy4g_mixture_component_indices": realization[
                "component_choices"
            ],
            "legacy4g_standard_gaussian_draws": realization[
                "gaussian_draws"
            ],
            "legacy4g_raw_draws": realization["raw_draws"],
            "legacy4g_background_noise_nondimensional": background,
            "legacy4g_gross_row_permutation": row_permutation,
            "legacy4g_base_uniform_factors": base_uniform_factors,
            "legacy4g_resolved_gross_factors": resolved_factors,
            "legacy4g_gross_row_indices": gross_rows,
            "legacy4g_gross_offsets_nondimensional": gross_offsets,
            "metadata_json": np.asarray(
                json.dumps(artifact_metadata, sort_keys=True)
            ),
        }
        _write_npz_exclusive(output, output_arrays)
        artifact_sha256 = sha256_file(output)
        manifest = {
            "schema_version": 1,
            "artifact_path": str(output),
            "artifact_sha256": artifact_sha256,
            "parent_artifact_path": str(source),
            "parent_artifact_sha256": parent_sha256,
            "generator_provenance": provenance,
            "corruption": corruption,
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for name, value in output_arrays.items()
            },
        }
        try:
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except BaseException:
            output.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise
        output_records.append((output, manifest_path))
    return output_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.10, 0.15],
        help="Gross vector-row ratios; row sets are nested permutation prefixes.",
    )
    parser.add_argument(
        "--base-multipliers",
        nargs="+",
        type=float,
        default=[1.0],
        help="b values multiplying the legacy 0.1 background scale.",
    )
    parser.add_argument(
        "--gross-multipliers",
        nargs="+",
        type=float,
        default=[1.0],
        help="o values multiplying both endpoints of the legacy U(3,10) range.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = generate(
        input_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        ratios=args.ratios,
        base_multipliers=args.base_multipliers,
        gross_multipliers=args.gross_multipliers,
    )
    for output, manifest in records:
        print(f"Prepared: {output}")
        print(f"Manifest: {manifest}")
        print(f"NPZ SHA256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
