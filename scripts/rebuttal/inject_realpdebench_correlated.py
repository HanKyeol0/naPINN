#!/usr/bin/env python3
"""Create deterministic labeled correlated corruptions on real PIV sensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_clean(source: Path):
    with np.load(source, allow_pickle=False) as raw:
        arrays = {name: raw[name].copy() for name in raw.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if "corruption" in metadata:
        raise ValueError("Input artifact already contains a corruption record")
    return arrays, metadata


def common_scales(arrays):
    u = arrays["u_mps"].astype(np.float32, copy=True)
    v = arrays["v_mps"].astype(np.float32, copy=True)
    train = arrays["train_sensor_flat_indices"].astype(np.int64)
    n_frames = u.shape[0]
    clean_training = np.stack(
        (
            u.reshape(n_frames, -1)[:, train],
            v.reshape(n_frames, -1)[:, train],
        ),
        axis=-1,
    )
    scale = clean_training.std(axis=(0, 1), ddof=1).astype(np.float64)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"Invalid component scales: {scale}")
    return u, v, train, scale


def inject_ar1(arrays, metadata, args):
    clean_u, clean_v, train, scale = common_scales(arrays)
    corrupt_u, corrupt_v = clean_u.copy(), clean_v.copy()
    n_frames = clean_u.shape[0]
    n_sensors = train.size
    n_failed = max(1, round(args.failure_fraction * n_sensors))
    rng = np.random.default_rng(args.seed)
    failed_positions = np.sort(
        rng.choice(n_sensors, size=n_failed, replace=False)
    )
    target_std = args.ar1_scale_std * scale
    innovation_std = target_std * np.sqrt(1.0 - args.ar1_rho**2)
    process = np.empty((n_frames, n_failed, 2), dtype=np.float32)
    process[0] = rng.normal(
        0.0, target_std, size=(n_failed, 2)
    ).astype(np.float32)
    for frame in range(1, n_frames):
        process[frame] = (
            args.ar1_rho * process[frame - 1]
            + rng.normal(
                0.0, innovation_std, size=(n_failed, 2)
            ).astype(np.float32)
        )
    labels = np.zeros((n_frames, n_sensors, 2), dtype=bool)
    labels[:, failed_positions, :] = True
    for local_position, sensor_position in enumerate(failed_positions):
        flat = int(train[sensor_position])
        row, column = np.unravel_index(flat, clean_u.shape[1:])
        corrupt_u[:, row, column] += process[:, local_position, 0]
        corrupt_v[:, row, column] += process[:, local_position, 1]
    record = {
        "kind": "temporally_correlated_AR1_sensor_drift",
        "seed": args.seed,
        "scope": "training sensors only; held-out PIV remains unchanged",
        "failure_fraction_requested": args.failure_fraction,
        "failure_fraction_realized": n_failed / n_sensors,
        "n_failed_spatial_sensors": int(n_failed),
        "n_total_spatial_sensors": int(n_sensors),
        "failed_sensor_positions_in_training_split": failed_positions.tolist(),
        "ar1_rho": args.ar1_rho,
        "ar1_stationary_scale_component_std": args.ar1_scale_std,
        "clean_component_std_mps": scale.tolist(),
    }
    extra = {"ar1_process_mps": process}
    return clean_u, clean_v, corrupt_u, corrupt_v, labels, record, extra


def inject_spatial_burst(arrays, metadata, args):
    clean_u, clean_v, train, scale = common_scales(arrays)
    corrupt_u, corrupt_v = clean_u.copy(), clean_v.copy()
    n_frames = clean_u.shape[0]
    n_sensors = train.size
    n_failed = max(1, round(args.failure_fraction * n_sensors))
    x = arrays["x_m"].reshape(-1)[train]
    y = arrays["y_m"].reshape(-1)[train]
    rng = np.random.default_rng(args.seed)
    anchor = int(rng.integers(0, n_sensors))
    distance = np.square(x - x[anchor]) + np.square(y - y[anchor])
    failed_positions = np.sort(
        np.argsort(distance, kind="stable")[:n_failed]
    )
    total_burst_frames = max(1, round(args.burst_fraction * n_frames))
    first_length = total_burst_frames // 2
    second_length = total_burst_frames - first_length
    available = n_frames - total_burst_frames
    first_start = int(round(0.25 * available))
    second_start = int(round(0.75 * available)) + first_length
    second_start = min(second_start, n_frames - second_length)
    burst_frames = np.concatenate(
        (
            np.arange(first_start, first_start + first_length),
            np.arange(second_start, second_start + second_length),
        )
    )
    signs = rng.choice((-1.0, 1.0), size=(n_failed, 2))
    offsets = (
        args.burst_scale_std * scale[None, :] * signs
    ).astype(np.float32)
    labels = np.zeros((n_frames, n_sensors, 2), dtype=bool)
    labels[np.ix_(burst_frames, failed_positions, np.arange(2))] = True
    for local_position, sensor_position in enumerate(failed_positions):
        flat = int(train[sensor_position])
        row, column = np.unravel_index(flat, clean_u.shape[1:])
        corrupt_u[burst_frames, row, column] += offsets[local_position, 0]
        corrupt_v[burst_frames, row, column] += offsets[local_position, 1]
    record = {
        "kind": "spatially_correlated_two_burst_sensor_failure",
        "seed": args.seed,
        "scope": "training sensors and burst frames only; held-out PIV unchanged",
        "failure_fraction_requested": args.failure_fraction,
        "failure_fraction_realized": n_failed / n_sensors,
        "n_failed_spatial_sensors": int(n_failed),
        "n_total_spatial_sensors": int(n_sensors),
        "failed_sensor_positions_in_training_split": failed_positions.tolist(),
        "spatial_anchor_training_position": anchor,
        "burst_fraction_requested": args.burst_fraction,
        "burst_fraction_realized": burst_frames.size / n_frames,
        "burst_frames": burst_frames.tolist(),
        "burst_scale_component_std": args.burst_scale_std,
        "clean_component_std_mps": scale.tolist(),
    }
    extra = {
        "burst_frames": burst_frames.astype(np.int64),
        "sensor_burst_offsets_mps": offsets,
    }
    return clean_u, clean_v, corrupt_u, corrupt_v, labels, record, extra


def inject_heteroscedastic(arrays, metadata, args):
    """State-dependent measurement error whose scale grows with local speed.

    PIV displacement error is not homogeneous: it grows with the particle
    displacement, so faster regions carry larger absolute error. We therefore
    set each failed sensor's per-frame noise scale from the local velocity
    magnitude, which makes the variance vary jointly with position, time, and
    state instead of being a single constant.
    """
    clean_u, clean_v, train, scale = common_scales(arrays)
    corrupt_u, corrupt_v = clean_u.copy(), clean_v.copy()
    n_frames = clean_u.shape[0]
    n_sensors = train.size
    n_failed = max(1, round(args.failure_fraction * n_sensors))
    rng = np.random.default_rng(args.seed)
    failed_positions = np.sort(
        rng.choice(n_sensors, size=n_failed, replace=False)
    )
    flat_u = clean_u.reshape(n_frames, -1)[:, train]
    flat_v = clean_v.reshape(n_frames, -1)[:, train]
    speed = np.sqrt(np.square(flat_u) + np.square(flat_v))
    mean_speed = float(speed.mean())
    if not np.isfinite(mean_speed) or mean_speed <= 0:
        raise ValueError(f"Invalid mean speed: {mean_speed}")
    # sigma_i(t) = base * scale_component * (1 + gain * speed_i(t)/mean_speed)
    relative_speed = speed[:, failed_positions] / mean_speed
    gain = np.float32(1.0) + np.float32(args.hetero_gain) * relative_speed
    sigma = (
        np.float32(args.hetero_base_std)
        * scale.astype(np.float32)[None, None, :]
        * gain[:, :, None]
    )
    perturbation = rng.normal(0.0, 1.0, size=sigma.shape).astype(np.float32)
    perturbation = perturbation * sigma
    labels = np.zeros((n_frames, n_sensors, 2), dtype=bool)
    labels[:, failed_positions, :] = True
    for local_position, sensor_position in enumerate(failed_positions):
        flat = int(train[sensor_position])
        row, column = np.unravel_index(flat, clean_u.shape[1:])
        corrupt_u[:, row, column] += perturbation[:, local_position, 0]
        corrupt_v[:, row, column] += perturbation[:, local_position, 1]
    record = {
        "kind": "state_dependent_heteroscedastic_sensor_error",
        "seed": args.seed,
        "scope": "training sensors only; held-out PIV remains unchanged",
        "failure_fraction_requested": args.failure_fraction,
        "failure_fraction_realized": n_failed / n_sensors,
        "n_failed_spatial_sensors": int(n_failed),
        "n_total_spatial_sensors": int(n_sensors),
        "failed_sensor_positions_in_training_split": failed_positions.tolist(),
        "hetero_base_component_std": args.hetero_base_std,
        "hetero_speed_gain": args.hetero_gain,
        "scale_law": (
            "sigma_i(t) = hetero_base_component_std * clean_component_std * "
            "(1 + hetero_speed_gain * speed_i(t) / mean_training_speed)"
        ),
        "mean_training_speed_mps": mean_speed,
        "realized_sigma_min_mps": sigma.min(axis=(0, 1)).tolist(),
        "realized_sigma_max_mps": sigma.max(axis=(0, 1)).tolist(),
        "clean_component_std_mps": scale.tolist(),
    }
    extra = {"heteroscedastic_perturbation_mps": perturbation}
    return clean_u, clean_v, corrupt_u, corrupt_v, labels, record, extra


def inject_uv_correlated(arrays, metadata, args):
    """Measurement error correlated between the two velocity components.

    The existing gross-outlier and drift protocols draw the ``u`` and ``v``
    perturbations independently. A shared calibration or registration error in
    PIV biases both components together, so here the per-frame innovation is
    drawn from a bivariate normal with a fixed off-diagonal correlation and no
    temporal correlation, which isolates the cross-component effect from the
    AR(1) condition.
    """
    clean_u, clean_v, train, scale = common_scales(arrays)
    corrupt_u, corrupt_v = clean_u.copy(), clean_v.copy()
    n_frames = clean_u.shape[0]
    n_sensors = train.size
    n_failed = max(1, round(args.failure_fraction * n_sensors))
    rng = np.random.default_rng(args.seed)
    failed_positions = np.sort(
        rng.choice(n_sensors, size=n_failed, replace=False)
    )
    rho = float(args.uv_rho)
    if not -1.0 < rho < 1.0:
        raise ValueError("uv_rho must lie strictly inside (-1, 1)")
    correlation = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    factor = np.linalg.cholesky(correlation).astype(np.float32)
    standard = rng.normal(
        0.0, 1.0, size=(n_frames, n_failed, 2)
    ).astype(np.float32)
    correlated = standard @ factor.T
    perturbation = (
        np.float32(args.uv_scale_std)
        * scale.astype(np.float32)[None, None, :]
        * correlated
    )
    labels = np.zeros((n_frames, n_sensors, 2), dtype=bool)
    labels[:, failed_positions, :] = True
    for local_position, sensor_position in enumerate(failed_positions):
        flat = int(train[sensor_position])
        row, column = np.unravel_index(flat, clean_u.shape[1:])
        corrupt_u[:, row, column] += perturbation[:, local_position, 0]
        corrupt_v[:, row, column] += perturbation[:, local_position, 1]
    realized = float(
        np.corrcoef(
            perturbation[:, :, 0].reshape(-1),
            perturbation[:, :, 1].reshape(-1),
        )[0, 1]
    )
    record = {
        "kind": "cross_component_correlated_sensor_error",
        "seed": args.seed,
        "scope": "training sensors only; held-out PIV remains unchanged",
        "failure_fraction_requested": args.failure_fraction,
        "failure_fraction_realized": n_failed / n_sensors,
        "n_failed_spatial_sensors": int(n_failed),
        "n_total_spatial_sensors": int(n_sensors),
        "failed_sensor_positions_in_training_split": failed_positions.tolist(),
        "uv_correlation_requested": rho,
        "uv_correlation_realized": realized,
        "uv_scale_component_std": args.uv_scale_std,
        "temporal_correlation": "none; innovations are independent across frames",
        "clean_component_std_mps": scale.tolist(),
    }
    extra = {"uv_correlated_perturbation_mps": perturbation}
    return clean_u, clean_v, corrupt_u, corrupt_v, labels, record, extra


def inject(args):
    source = args.input.resolve()
    output = args.output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite derived artifact or sidecar: "
            f"{output}, {manifest_path}"
        )
    arrays, metadata = load_clean(source)
    if not 0.0 < args.failure_fraction < 1.0:
        raise ValueError("failure_fraction must be in (0,1)")
    injectors = {
        "ar1": inject_ar1,
        "spatial_burst": inject_spatial_burst,
        "heteroscedastic": inject_heteroscedastic,
        "uv_correlated": inject_uv_correlated,
    }
    values = injectors[args.kind](arrays, metadata, args)
    clean_u, clean_v, corrupt_u, corrupt_v, labels, record, extra = values
    heldout = arrays["heldout_flat_indices"].astype(np.int64)
    n_frames = clean_u.shape[0]
    if not np.array_equal(
        corrupt_u.reshape(n_frames, -1)[:, heldout],
        clean_u.reshape(n_frames, -1)[:, heldout],
    ) or not np.array_equal(
        corrupt_v.reshape(n_frames, -1)[:, heldout],
        clean_v.reshape(n_frames, -1)[:, heldout],
    ):
        raise AssertionError("Held-out PIV values changed")
    record["parent_clean_npz_path"] = str(source)
    record["parent_clean_npz_sha256"] = sha256(source)
    metadata["corruption"] = record
    arrays.update(
        {
            "u_clean_mps": clean_u,
            "v_clean_mps": clean_v,
            "u_mps": corrupt_u,
            "v_mps": corrupt_v,
            "training_corruption_mask": labels,
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            **extra,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest = {
        **metadata,
        "derived_npz_path": str(output),
        "derived_npz_sha256": sha256(output),
        "heldout_values_bitwise_equal_to_parent": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, manifest_path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde/data/"
            "realpdebench_cylinder_10031_frames1000-1200_stride1.npz"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=["ar1", "spatial_burst", "heteroscedastic", "uv_correlated"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--failure-fraction", type=float, default=0.15)
    parser.add_argument("--ar1-rho", type=float, default=0.98)
    parser.add_argument("--ar1-scale-std", type=float, default=3.0)
    parser.add_argument("--burst-fraction", type=float, default=0.20)
    parser.add_argument("--burst-scale-std", type=float, default=5.0)
    parser.add_argument("--hetero-base-std", type=float, default=1.0)
    parser.add_argument("--hetero-gain", type=float, default=4.0)
    parser.add_argument("--uv-rho", type=float, default=0.8)
    parser.add_argument("--uv-scale-std", type=float, default=3.0)
    return parser


def main():
    output, manifest = inject(build_parser().parse_args())
    print(f"Prepared: {output}")
    print(f"Manifest: {manifest}")
    print(f"NPZ SHA256: {sha256(output)}")


if __name__ == "__main__":
    main()
