#!/usr/bin/env python3
"""Extract a compact, auditable RealPDEBench Cylinder PIV trajectory window.

The source Arrow files are Hugging Face ``Dataset.save_to_disk`` stream files.
Each row contains one complete physical trajectory. This script deliberately
decodes only the selected row, retains the raw SI coordinates/velocities, and
creates one deterministic spatial sensor split shared by every benchmark
method.

RealPDEBench is distributed under CC BY-NC 4.0. The prepared artifact is for
non-commercial research use and remains subject to that license.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASET_REPO = "AI4Science-WestlakeU/RealPDEBench"
DATASET_VERSION = "2.0.0"
DATASET_LICENSE = "CC BY-NC 4.0"
DATASET_CITATION = (
    "Hu et al., RealPDEBench: A Benchmark for Complex Physical Systems "
    "with Real-World Data, ICLR 2026 Oral."
)
DEFAULT_SOURCE_PATH = (
    "cylinder/hf_dataset/real/data-00000-of-00073.arrow"
)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(row: dict[str, Any], key: str, dtype, shape) -> np.ndarray:
    raw = row[key]
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise ValueError(
            f"{key} has {len(raw)} bytes, expected {expected} for shape {shape}"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def load_arrow_row(path: Path, sim_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError(
            "Preparing RealPDEBench data requires pyarrow (`pip install pyarrow`)."
        ) from exc

    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        schema_metadata = {
            key.decode("utf-8", errors="replace"): value.decode(
                "utf-8", errors="replace"
            )
            for key, value in (reader.schema.metadata or {}).items()
        }
        for batch in reader:
            sim_column = batch.column(batch.schema.get_field_index("sim_id"))
            for row_index in range(batch.num_rows):
                if sim_column[row_index].as_py() != sim_id:
                    continue
                row = {
                    name: batch.column(column_index)[row_index].as_py()
                    for column_index, name in enumerate(batch.schema.names)
                }
                return row, schema_metadata
    raise KeyError(f"sim_id {sim_id!r} was not found in {path}")


def prepare(args: argparse.Namespace) -> tuple[Path, Path]:
    source = args.arrow.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.frame_start < 0 or args.frame_stop <= args.frame_start:
        raise ValueError("Require 0 <= frame_start < frame_stop")
    if args.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if args.n_sensors <= 0:
        raise ValueError("n_sensors must be positive")

    source_hash = sha256(source)
    row, arrow_metadata = load_arrow_row(source, args.sim_id)
    full_shape = (
        int(row["shape_t"]),
        int(row["shape_h"]),
        int(row["shape_w"]),
    )
    x_shape = (int(row["x_shape_h"]), int(row["x_shape_w"]))
    y_shape = (int(row["y_shape_h"]), int(row["y_shape_w"]))
    if x_shape != full_shape[1:] or y_shape != full_shape[1:]:
        raise ValueError(
            f"Coordinate shapes {x_shape}, {y_shape} do not match {full_shape[1:]}"
        )

    u_full = _decode(row, "u", np.float32, full_shape)
    v_full = _decode(row, "v", np.float32, full_shape)
    x_m = _decode(row, "x", np.float64, x_shape).copy()
    y_m = _decode(row, "y", np.float64, y_shape).copy()
    t_s_full = _decode(row, "t", np.float64, (int(row["t_shape"]),))

    if args.frame_stop > full_shape[0]:
        raise ValueError(
            f"frame_stop={args.frame_stop} exceeds trajectory length {full_shape[0]}"
        )
    frame_indices = np.arange(
        args.frame_start, args.frame_stop, args.frame_stride, dtype=np.int64
    )
    t_s = t_s_full[frame_indices].copy()
    u_mps = u_full[frame_indices].copy()
    v_mps = v_full[frame_indices].copy()

    if not all(
        np.isfinite(array).all() for array in (x_m, y_m, t_s, u_mps, v_mps)
    ):
        raise ValueError("Selected trajectory window contains non-finite values")
    if not np.all(np.diff(t_s) > 0):
        raise ValueError("Selected time stamps are not strictly increasing")

    rows, columns = np.indices(full_shape[1:])
    radius_px = 0.5 * float(args.cylinder_diameter_px)
    cylinder_mask = (
        (columns - float(args.cylinder_center_x_px)) ** 2
        + (rows - float(args.cylinder_center_y_px)) ** 2
        <= radius_px**2
    )
    fluid_mask = ~cylinder_mask
    fluid_flat_indices = np.flatnonzero(fluid_mask.reshape(-1))
    if args.n_sensors >= fluid_flat_indices.size:
        raise ValueError(
            f"n_sensors={args.n_sensors} must be smaller than "
            f"{fluid_flat_indices.size} fluid grid points"
        )

    rng = np.random.default_rng(args.sensor_seed)
    train_sensor_flat_indices = np.sort(
        rng.choice(fluid_flat_indices, size=args.n_sensors, replace=False)
    ).astype(np.int64)
    heldout_flat_indices = np.setdiff1d(
        fluid_flat_indices,
        train_sensor_flat_indices,
        assume_unique=True,
    ).astype(np.int64)
    if np.intersect1d(
        train_sensor_flat_indices, heldout_flat_indices, assume_unique=True
    ).size:
        raise AssertionError("Training sensors overlap held-out evaluation points")

    reynolds = int(Path(args.sim_id).stem)
    characteristic_velocity_mps = (
        reynolds * args.kinematic_viscosity_m2ps / args.cylinder_diameter_m
    )
    metadata = {
        "schema_version": 1,
        "dataset": "RealPDEBench Cylinder real PIV",
        "dataset_repo": DATASET_REPO,
        "dataset_version": DATASET_VERSION,
        "dataset_license": DATASET_LICENSE,
        "dataset_citation": DATASET_CITATION,
        "source_hf_path": args.source_hf_path,
        "source_local_path": str(source),
        "source_sha256": source_hash,
        "source_arrow_schema_metadata": arrow_metadata,
        "sim_id": args.sim_id,
        "reynolds_number": reynolds,
        "cylinder_diameter_m": args.cylinder_diameter_m,
        "water_kinematic_viscosity_m2ps": args.kinematic_viscosity_m2ps,
        "characteristic_velocity_mps": characteristic_velocity_mps,
        "nondimensionalization": {
            "x": "x/D",
            "y": "y/D",
            "t": "t*U/D",
            "u": "u/U",
            "v": "v/U",
            "U": "Re*nu/D",
        },
        "frame_start_inclusive": args.frame_start,
        "frame_stop_exclusive": args.frame_stop,
        "frame_stride": args.frame_stride,
        "n_frames": int(frame_indices.size),
        "raw_full_shape_t_h_w": list(full_shape),
        "selected_shape_t_h_w": list(u_mps.shape),
        "sensor_seed": args.sensor_seed,
        "n_spatial_sensors": args.n_sensors,
        "n_training_measurements": int(args.n_sensors * frame_indices.size),
        "n_heldout_measurements": int(
            heldout_flat_indices.size * frame_indices.size
        ),
        "split_policy": (
            "One fixed irregular spatial sensor subset is used at every selected "
            "frame. Held-out PIV labels are every remaining fluid-grid location; "
            "training sensor locations are excluded exactly."
        ),
        "cylinder_mask": {
            "definition": (
                "RealPDEBench reference Cylinder geometry in released image-grid "
                "pixels (sub_s_real=1)"
            ),
            "center_x_px": args.cylinder_center_x_px,
            "center_y_px": args.cylinder_center_y_px,
            "diameter_px": args.cylinder_diameter_px,
            "source_code_repository": (
                "https://github.com/AI4Science-WestlakeU/RealPDEBench"
            ),
            "source_code_commit_audited": (
                "62f4c80ab17f78933d046f2b038531dbc6a478a0"
            ),
            "source_code_location": (
                "realpdebench/data/fluid_dataset.py::Cylinder"
            ),
        },
        "notes": (
            "Arrays x_m, y_m, t_s, u_mps, and v_mps retain source physical "
            "units. The tracked ns_data.npz is not used."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x_m=x_m,
        y_m=y_m,
        t_s=t_s,
        u_mps=u_mps,
        v_mps=v_mps,
        frame_indices=frame_indices,
        fluid_mask=fluid_mask,
        train_sensor_flat_indices=train_sensor_flat_indices,
        heldout_flat_indices=heldout_flat_indices,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    output_hash = sha256(output)
    manifest = {
        **metadata,
        "prepared_npz_path": str(output),
        "prepared_npz_sha256": output_hash,
        "arrays": {
            "x_m": {"shape": list(x_m.shape), "dtype": str(x_m.dtype)},
            "y_m": {"shape": list(y_m.shape), "dtype": str(y_m.dtype)},
            "t_s": {"shape": list(t_s.shape), "dtype": str(t_s.dtype)},
            "u_mps": {"shape": list(u_mps.shape), "dtype": str(u_mps.dtype)},
            "v_mps": {"shape": list(v_mps.shape), "dtype": str(v_mps.dtype)},
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--sim-id", default="10031.h5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/runs/rebuttal_realpde/data/"
            "realpdebench_cylinder_10031_frames1000-1200_stride1.npz"
        ),
    )
    parser.add_argument("--frame-start", type=int, default=1000)
    parser.add_argument("--frame-stop", type=int, default=1200)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--n-sensors", type=int, default=192)
    parser.add_argument("--sensor-seed", type=int, default=20260724)
    parser.add_argument("--cylinder-diameter-m", type=float, default=0.03)
    parser.add_argument(
        "--kinematic-viscosity-m2ps", type=float, default=1.0e-6
    )
    parser.add_argument("--cylinder-center-x-px", type=float, default=32.0)
    parser.add_argument("--cylinder-center-y-px", type=float, default=32.0)
    parser.add_argument("--cylinder-diameter-px", type=float, default=18.0)
    parser.add_argument("--source-hf-path", default=DEFAULT_SOURCE_PATH)
    return parser


def main() -> None:
    output, manifest = prepare(build_parser().parse_args())
    print(f"Prepared: {output}")
    print(f"Manifest: {manifest}")
    print(f"NPZ SHA256: {sha256(output)}")


if __name__ == "__main__":
    main()
