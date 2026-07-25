#!/usr/bin/env python3
"""Create a deterministic structured sensor-failure RealPDEBench artifact.

Only selected training-sensor values are changed. Clean full-field arrays,
held-out PIV labels, the original split, corruption labels, per-component
biases/drifts, and parent-artifact provenance are retained in the output.
"""

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


def inject(args: argparse.Namespace) -> tuple[Path, Path]:
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not 0.0 < args.failure_fraction < 1.0:
        raise ValueError("failure_fraction must lie strictly between 0 and 1")
    if args.bias_scale_std <= 0.0 or args.drift_end_scale_std <= 0.0:
        raise ValueError("bias and drift scales must be positive")

    with np.load(source, allow_pickle=False) as raw:
        arrays = {name: raw[name].copy() for name in raw.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if "corruption" in metadata:
        raise ValueError("Input artifact already contains a corruption record")

    clean_u = arrays["u_mps"].astype(np.float32, copy=True)
    clean_v = arrays["v_mps"].astype(np.float32, copy=True)
    corrupted_u = clean_u.copy()
    corrupted_v = clean_v.copy()
    train_indices = arrays["train_sensor_flat_indices"].astype(np.int64)
    heldout_indices = arrays["heldout_flat_indices"].astype(np.int64)
    n_frames = clean_u.shape[0]
    n_sensors = train_indices.size
    n_failed = max(1, int(round(args.failure_fraction * n_sensors)))

    clean_training = np.stack(
        (
            clean_u.reshape(n_frames, -1)[:, train_indices],
            clean_v.reshape(n_frames, -1)[:, train_indices],
        ),
        axis=-1,
    )
    component_std = clean_training.std(axis=(0, 1), ddof=1).astype(np.float64)
    if not np.isfinite(component_std).all() or np.any(component_std <= 0):
        raise ValueError(f"Invalid clean component scales: {component_std}")

    rng = np.random.default_rng(args.seed)
    failed_positions = np.sort(
        rng.choice(n_sensors, size=n_failed, replace=False)
    ).astype(np.int64)
    bias_signs = rng.choice((-1.0, 1.0), size=(n_failed, 2))
    drift_signs = rng.choice((-1.0, 1.0), size=(n_failed, 2))
    failed_bias_mps = (
        args.bias_scale_std * component_std[None, :] * bias_signs
    ).astype(np.float32)
    failed_drift_end_mps = (
        args.drift_end_scale_std * component_std[None, :] * drift_signs
    ).astype(np.float32)

    sensor_bias_mps = np.zeros((n_sensors, 2), dtype=np.float32)
    sensor_drift_end_mps = np.zeros((n_sensors, 2), dtype=np.float32)
    sensor_bias_mps[failed_positions] = failed_bias_mps
    sensor_drift_end_mps[failed_positions] = failed_drift_end_mps
    failed_sensor_mask = np.zeros(n_sensors, dtype=bool)
    failed_sensor_mask[failed_positions] = True
    corruption_mask = np.broadcast_to(
        failed_sensor_mask[None, :, None],
        (n_frames, n_sensors, 2),
    ).copy()

    time_fraction = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    for sensor_position in failed_positions:
        flat_index = int(train_indices[sensor_position])
        row, column = np.unravel_index(flat_index, clean_u.shape[1:])
        delta = (
            sensor_bias_mps[sensor_position][None, :]
            + time_fraction[:, None]
            * sensor_drift_end_mps[sensor_position][None, :]
        )
        corrupted_u[:, row, column] += delta[:, 0]
        corrupted_v[:, row, column] += delta[:, 1]

    # The held-out label set must remain bitwise identical to the clean source.
    if not np.array_equal(
        corrupted_u.reshape(n_frames, -1)[:, heldout_indices],
        clean_u.reshape(n_frames, -1)[:, heldout_indices],
    ) or not np.array_equal(
        corrupted_v.reshape(n_frames, -1)[:, heldout_indices],
        clean_v.reshape(n_frames, -1)[:, heldout_indices],
    ):
        raise AssertionError("Held-out PIV values were modified")
    clean_sensor_mask = ~failed_sensor_mask
    clean_sensor_indices = train_indices[clean_sensor_mask]
    if not np.array_equal(
        corrupted_u.reshape(n_frames, -1)[:, clean_sensor_indices],
        clean_u.reshape(n_frames, -1)[:, clean_sensor_indices],
    ) or not np.array_equal(
        corrupted_v.reshape(n_frames, -1)[:, clean_sensor_indices],
        clean_v.reshape(n_frames, -1)[:, clean_sensor_indices],
    ):
        raise AssertionError("A clean training sensor was modified")

    parent_hash = sha256(source)
    failed_grid_indices = train_indices[failed_positions]
    metadata["corruption"] = {
        "kind": "persistent_sensor_component_bias_plus_linear_drift",
        "scope": "training sensors only; held-out PIV remains clean",
        "seed": args.seed,
        "failure_fraction_requested": args.failure_fraction,
        "failure_fraction_realized": n_failed / n_sensors,
        "n_failed_spatial_sensors": n_failed,
        "n_total_spatial_sensors": int(n_sensors),
        "failed_sensor_positions_in_training_split": failed_positions.tolist(),
        "failed_sensor_flat_grid_indices": failed_grid_indices.tolist(),
        "clean_component_std_mps": component_std.tolist(),
        "bias_scale_component_std": args.bias_scale_std,
        "drift_end_scale_component_std": args.drift_end_scale_std,
        "time_profile": (
            "delta_c(t)=bias_c+tau(t)*drift_end_c, with tau linearly "
            "increasing from 0 at the first frame to 1 at the final frame"
        ),
        "component_sign_policy": (
            "Independent deterministic Rademacher signs for each failed "
            "sensor and velocity component, generated after sensor selection"
        ),
        "parent_clean_npz_path": str(source),
        "parent_clean_npz_sha256": parent_hash,
        "labels": (
            "training_corruption_mask has shape (frames,sensors,components); "
            "both u and v scalars of a failed identity are labeled at all frames"
        ),
    }
    metadata["notes"] = (
        str(metadata.get("notes", ""))
        + " This derived variant preserves u_clean_mps/v_clean_mps and corrupts "
        "only labeled training-sensor identities."
    ).strip()

    arrays.update(
        {
            "u_clean_mps": clean_u,
            "v_clean_mps": clean_v,
            "u_mps": corrupted_u,
            "v_mps": corrupted_v,
            "failed_sensor_mask": failed_sensor_mask,
            "failed_sensor_positions": failed_positions,
            "failed_sensor_flat_indices": failed_grid_indices,
            "training_corruption_mask": corruption_mask,
            "sensor_bias_mps": sensor_bias_mps,
            "sensor_drift_end_mps": sensor_drift_end_mps,
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output_hash = sha256(output)
    manifest = {
        **metadata,
        "derived_npz_path": str(output),
        "derived_npz_sha256": output_hash,
        "heldout_values_bitwise_equal_to_parent": True,
        "clean_training_sensors_bitwise_equal_to_parent": True,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde/data/"
            "realpdebench_cylinder_10031_frames1000-1200_stride1.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde/data/"
            "realpdebench_cylinder_10031_frames1000-1200_stride1_"
            "sensor_drift10pct.npz"
        ),
    )
    parser.add_argument("--failure-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--bias-scale-std", type=float, default=3.0)
    parser.add_argument("--drift-end-scale-std", type=float, default=2.0)
    return parser


def main() -> None:
    output, manifest = inject(build_parser().parse_args())
    print(f"Prepared: {output}")
    print(f"Manifest: {manifest}")
    print(f"NPZ SHA256: {sha256(output)}")


if __name__ == "__main__":
    main()
