#!/usr/bin/env python3
"""Prepare the frozen non-Cylinder RealPDEBench real-PIV window.

The source files are Hugging Face Arrow stream shards with one complete
trajectory per row.  This script selects the predeclared trajectory and
frames, constructs one deterministic fixed-sensor split, and records both the
source and prepared-artifact checksums.  It never infers a body mask from
zero/non-finite velocities: the released real arrays contain neither.

Geometry is intentionally conservative and explicit.  Controlled Cylinder
uses the official reference disk.  FSI uses only the official *static
reference* disk because the released measurement row has no moving-body
coordinates; it does not claim to exclude both moving bodies at every frame.
Foil uses an axis-aligned square envelope around the official reference
centre/length because the body is a foil, not a disk.  These exclusions are
provenance-bearing approximations for a nominal 2-D Navier--Stokes
model-discrepancy stress test, not recovered experimental boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_COMMIT = "62f4c80ab17f78933d046f2b038531dbc6a478a0"
DATASET_REPO = "AI4Science-WestlakeU/RealPDEBench"
DATASET_LICENSE = "CC BY-NC 4.0"
DEFAULT_ROOT = Path("outputs/rebuttal/realpdebench_multidataset")


@dataclass(frozen=True)
class Scenario:
    name: str
    source_name: str
    source_hf_path: str
    sim_id: str
    geometry_kind: str
    diameter_px: float
    center_x_px: float
    center_y_px: float
    geometry_limitation: str


SCENARIOS = {
    "controlled_cylinder": Scenario(
        name="controlled_cylinder",
        source_name="controlled_cylinder_real_00000.arrow",
        source_hf_path=(
            "controlled_cylinder/hf_dataset/real/"
            "data-00000-of-00051.arrow"
        ),
        sim_id="1781_0.5.h5",
        geometry_kind="official_reference_disk",
        diameter_px=18.0,
        center_x_px=16.0,
        center_y_px=32.0,
        geometry_limitation=(
            "Static disk from the official real-data wrapper. The nominal "
            "2-D PDE does not model the cylinder-control actuation."
        ),
    ),
    "fsi": Scenario(
        name="fsi",
        source_name="fsi_real_00000.arrow",
        source_hf_path="fsi/hf_dataset/real/data-00000-of-00051.arrow",
        sim_id="3272_18.2_0.8_1.h5",
        geometry_kind="official_static_reference_disk_only",
        diameter_px=24.0,
        center_x_px=66.0,
        center_y_px=64.0,
        geometry_limitation=(
            "The official wrapper identifies FSI as a double-cylinder case "
            "but exposes only d=24 and reference centre=(66,64). The released "
            "measurement row contains no time-varying body coordinates. This "
            "static disk exclusion therefore does not guarantee exclusion of "
            "the second or moving body at every frame."
        ),
    ),
    "foil": Scenario(
        name="foil",
        source_name="foil_real_00000.arrow",
        source_hf_path="foil/hf_dataset/real/data-00000-of-00098.arrow",
        sim_id="10000_0.0.h5",
        geometry_kind="official_reference_square_envelope",
        diameter_px=62.0,
        center_x_px=30.0,
        center_y_px=64.0,
        geometry_limitation=(
            "The official wrapper records d=62 and centre=(30,64), but the "
            "body is a foil rather than a disk and no real-measurement body "
            "mask is released. We conservatively exclude the enclosing "
            "axis-aligned d-by-d reference square. This is an approximation, "
            "not an exact foil boundary."
        ),
    ),
}


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


def _decode(
    row: dict[str, Any], key: str, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    raw = row[key]
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if raw is None or len(raw) != expected:
        actual = None if raw is None else len(raw)
        raise ValueError(
            f"{key} has {actual} bytes; expected {expected} for shape {shape}"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def load_arrow_row(
    path: Path, sim_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("Preparation requires pyarrow") from exc

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
                return (
                    {
                        name: batch.column(column_index)[row_index].as_py()
                        for column_index, name in enumerate(batch.schema.names)
                    },
                    schema_metadata,
                )
    raise KeyError(f"{sim_id!r} was not found in {path}")


def parse_nominal_reynolds(sim_id: str) -> float:
    match = re.match(r"^(\d+(?:\.\d+)?)", Path(sim_id).stem)
    if match is None:
        raise ValueError(f"Cannot parse nominal Reynolds number from {sim_id!r}")
    value = float(match.group(1))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid nominal Reynolds number in {sim_id!r}")
    return value


def geometry_mask(
    scenario: Scenario, height: int, width: int
) -> np.ndarray:
    rows, columns = np.indices((height, width))
    half = 0.5 * scenario.diameter_px
    if scenario.geometry_kind in {
        "official_reference_disk",
        "official_static_reference_disk_only",
    }:
        return (
            (columns - scenario.center_x_px) ** 2
            + (rows - scenario.center_y_px) ** 2
            <= half**2
        )
    if scenario.geometry_kind == "official_reference_square_envelope":
        return (
            (np.abs(columns - scenario.center_x_px) <= half)
            & (np.abs(rows - scenario.center_y_px) <= half)
        )
    raise ValueError(f"Unsupported geometry kind: {scenario.geometry_kind}")


def _reference_scales(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid_mask: np.ndarray,
    diameter_px: float,
) -> tuple[float, float, dict[str, Any]]:
    """Return explicit data-derived scales without claiming SI calibration."""

    neighbor_distances = np.concatenate(
        (
            np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)).reshape(-1),
            np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)).reshape(-1),
        )
    )
    neighbor_distances = neighbor_distances[
        np.isfinite(neighbor_distances) & (neighbor_distances > 0.0)
    ]
    if neighbor_distances.size == 0:
        raise ValueError("Cannot derive a positive spatial reference spacing")
    spacing = float(np.median(neighbor_distances))
    length_scale = diameter_px * spacing

    valid_u = u[:, valid_mask].astype(np.float64)
    valid_v = v[:, valid_mask].astype(np.float64)
    velocity_scale = float(np.sqrt(np.mean(valid_u**2 + valid_v**2)))
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("Derived characteristic length is invalid")
    if not np.isfinite(velocity_scale) or velocity_scale <= 0.0:
        raise ValueError("Derived characteristic velocity is invalid")
    return (
        length_scale,
        velocity_scale,
        {
            "spatial_spacing_statistic": "median positive 4-neighbour distance",
            "spatial_spacing_source_units": spacing,
            "length_formula": "official_reference_d_px * median_spacing",
            "velocity_statistic": "pooled RMS speed over valid selected window",
            "scale_limitation": (
                "Arrow metadata do not identify calibrated characteristic "
                "length/velocity fields for these rows. These deterministic "
                "data-derived scales make the nominal PDE stress reproducible "
                "but are not asserted to reproduce the experimental Re "
                "nondimensionalisation."
            ),
        },
    )


def prepare_scenario(
    scenario: Scenario,
    source_root: Path,
    prepared_root: Path,
    frame_start: int,
    frame_stop: int,
    n_sensors: int,
    sensor_seed: int,
) -> dict[str, Any]:
    source = (source_root / scenario.source_name).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    row, arrow_metadata = load_arrow_row(source, scenario.sim_id)
    full_shape = (
        int(row["shape_t"]),
        int(row["shape_h"]),
        int(row["shape_w"]),
    )
    if not 0 <= frame_start < frame_stop <= full_shape[0]:
        raise ValueError(
            f"Invalid [{frame_start}, {frame_stop}) for {full_shape[0]} frames"
        )
    spatial_shape = full_shape[1:]
    if (
        (int(row["x_shape_h"]), int(row["x_shape_w"])) != spatial_shape
        or (int(row["y_shape_h"]), int(row["y_shape_w"])) != spatial_shape
    ):
        raise ValueError("Coordinate shapes do not match velocity grid")

    frame_indices = np.arange(frame_start, frame_stop, dtype=np.int64)
    u = _decode(row, "u", np.float32, full_shape)[frame_indices].copy()
    v = _decode(row, "v", np.float32, full_shape)[frame_indices].copy()
    x = _decode(row, "x", np.float64, spatial_shape).copy()
    y = _decode(row, "y", np.float64, spatial_shape).copy()
    t = _decode(row, "t", np.float64, (int(row["t_shape"]),))[
        frame_indices
    ].copy()
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("Selected timestamps are not strictly increasing")

    finite_coordinates = np.isfinite(x) & np.isfinite(y)
    finite_all_frames = (
        np.isfinite(u).all(axis=0) & np.isfinite(v).all(axis=0)
    )
    body_exclusion = geometry_mask(scenario, *spatial_shape)
    valid_mask = finite_coordinates & finite_all_frames & ~body_exclusion
    valid_indices = np.flatnonzero(valid_mask.reshape(-1))
    if valid_indices.size <= n_sensors:
        raise ValueError(
            f"{scenario.name}: only {valid_indices.size} valid points for "
            f"{n_sensors} sensors"
        )

    rng = np.random.default_rng(sensor_seed)
    train_indices = np.sort(
        rng.choice(valid_indices, size=n_sensors, replace=False)
    ).astype(np.int64)
    heldout_indices = np.setdiff1d(
        valid_indices, train_indices, assume_unique=True
    ).astype(np.int64)
    if np.intersect1d(train_indices, heldout_indices).size:
        raise AssertionError("Training and held-out indices overlap")
    if not np.isfinite(u.reshape(u.shape[0], -1)[:, train_indices]).all():
        raise AssertionError("Selected u training observations are not finite")
    if not np.isfinite(v.reshape(v.shape[0], -1)[:, train_indices]).all():
        raise AssertionError("Selected v training observations are not finite")
    if not np.isfinite(u.reshape(u.shape[0], -1)[:, heldout_indices]).all():
        raise AssertionError("Selected u held-out observations are not finite")
    if not np.isfinite(v.reshape(v.shape[0], -1)[:, heldout_indices]).all():
        raise AssertionError("Selected v held-out observations are not finite")

    length_scale, velocity_scale, scale_metadata = _reference_scales(
        x, y, u, v, valid_mask, scenario.diameter_px
    )
    nominal_reynolds = parse_nominal_reynolds(scenario.sim_id)
    metadata = {
        "schema_version": 2,
        "dataset": f"RealPDEBench {scenario.name} real PIV",
        "dataset_family": scenario.name,
        "dataset_repo": DATASET_REPO,
        "dataset_license": DATASET_LICENSE,
        "source_hf_path": scenario.source_hf_path,
        "source_local_path": str(source),
        "source_sha256": sha256_file(source),
        "source_arrow_schema_metadata": arrow_metadata,
        "sim_id": scenario.sim_id,
        "reynolds_number": nominal_reynolds,
        "reynolds_role": (
            "Nominal metadata parsed from the leading simulation-ID field; "
            "not treated as noise-free inverse-coefficient ground truth."
        ),
        "characteristic_length_source_units": length_scale,
        "characteristic_velocity_source_units": velocity_scale,
        "released_unit_labels": {
            "x_y": "released_coordinate_unit (not asserted to be metres)",
            "u_v": "released_velocity_unit (not asserted to be m/s)",
            "t": "released_time_coordinate (not asserted to be seconds)",
        },
        "reference_scale_provenance": scale_metadata,
        "nondimensionalization": {
            "x": "x/L_ref",
            "y": "y/L_ref",
            "t": "t*U_ref/L_ref",
            "u": "u/U_ref",
            "v": "v/U_ref",
            "nominal_viscosity_coefficient": "1/Re_metadata",
            "unit_assumption": (
                "Released coordinate, velocity, and time values are treated "
                "as one internally consistent numerical unit system solely "
                "for this nominal-PDE stress. Physical SI calibration is not "
                "claimed, especially for Foil coordinates."
            ),
        },
        "selected_window_time_diagnostics": {
            "minimum_dt_released_time": float(np.min(np.diff(t))),
            "maximum_dt_released_time": float(np.max(np.diff(t))),
            "median_dt_released_time": float(np.median(np.diff(t))),
            "median_U_dt_over_L": float(
                velocity_scale * np.median(np.diff(t)) / length_scale
            ),
        },
        "frame_start_inclusive": frame_start,
        "frame_stop_exclusive": frame_stop,
        "frame_stride": 1,
        "n_frames": int(frame_indices.size),
        "raw_full_shape_t_h_w": list(full_shape),
        "selected_shape_t_h_w": list(u.shape),
        "sensor_seed": sensor_seed,
        "n_spatial_sensors": n_sensors,
        "n_training_measurements": int(frame_indices.size * n_sensors),
        "n_heldout_measurements": int(
            frame_indices.size * heldout_indices.size
        ),
        "split_policy": (
            "The same fixed irregular spatial sensors are used in all 200 "
            "frames. Held-out points are the disjoint remainder of the valid "
            "finite reference-mask domain."
        ),
        "validity_policy": (
            "A point must have finite x/y and finite u/v in every selected "
            "frame. Zero velocity is not used as a body or invalid-data mask."
        ),
        "validity_diagnostics": {
            "total_spatial_points": int(np.prod(spatial_shape)),
            "finite_coordinate_points": int(finite_coordinates.sum()),
            "finite_uv_all_200_frames_points": int(finite_all_frames.sum()),
            "official_reference_excluded_points": int(body_exclusion.sum()),
            "valid_points": int(valid_mask.sum()),
            "training_points": int(train_indices.size),
            "heldout_points": int(heldout_indices.size),
            "spatial_overlap": 0,
            "training_finite_all_frames": True,
            "heldout_finite_all_frames": True,
            "source_exact_zero_u_count": int((u == 0.0).sum()),
            "source_exact_zero_v_count": int((v == 0.0).sum()),
            "source_nonfinite_u_count": int((~np.isfinite(u)).sum()),
            "source_nonfinite_v_count": int((~np.isfinite(v)).sum()),
        },
        "geometry_exclusion": {
            "kind": scenario.geometry_kind,
            "center_x_px": scenario.center_x_px,
            "center_y_px": scenario.center_y_px,
            "reference_d_px": scenario.diameter_px,
            "source_code_repository": (
                "https://github.com/AI4Science-WestlakeU/RealPDEBench"
            ),
            "source_code_commit_audited": OFFICIAL_COMMIT,
            "source_code_location": (
                "realpdebench/data/fluid_hf_dataset.py::"
                + {
                    "controlled_cylinder": "ControlledCylinderHFDataset",
                    "fsi": "FSIHFDataset",
                    "foil": "FoilHFDataset",
                }.get(scenario.name, "test_fixture")
            ),
            "limitation": scenario.geometry_limitation,
        },
        "physics_interpretation": (
            "Nominal pressure-latent 2-D incompressible Navier--Stokes "
            "model-discrepancy stress. Observation corruption and physics "
            "model discrepancy are not separately identifiable."
        ),
    }

    output = prepared_root / (
        f"realpdebench_{scenario.name}_{Path(scenario.sim_id).stem}_"
        f"frames{frame_start}-{frame_stop}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x_coord=x,
        y_coord=y,
        t_coord=t,
        u_observed=u,
        v_observed=v,
        frame_indices=frame_indices,
        fluid_mask=valid_mask,
        body_exclusion_mask=body_exclusion,
        train_sensor_flat_indices=train_indices,
        heldout_flat_indices=heldout_indices,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    manifest = {
        **metadata,
        "prepared_npz_path": str(output.resolve()),
        "prepared_npz_sha256": sha256_file(output),
        "array_sha256": {
            "x_coord": sha256_array(x),
            "y_coord": sha256_array(y),
            "t_coord": sha256_array(t),
            "u_observed": sha256_array(u),
            "v_observed": sha256_array(v),
            "fluid_mask": sha256_array(valid_mask),
            "train_sensor_flat_indices": sha256_array(train_indices),
            "heldout_flat_indices": sha256_array(heldout_indices),
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Campaign root containing source/ and receiving prepared/.",
    )
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all"],
        default="all",
    )
    parser.add_argument("--frame-start", type=int, default=1000)
    parser.add_argument("--frame-stop", type=int, default=1200)
    parser.add_argument("--n-sensors", type=int, default=192)
    parser.add_argument("--sensor-seed", type=int, default=20260724)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    if args.frame_start != 1000 or args.frame_stop != 1200:
        raise ValueError("Frozen protocol requires frames [1000, 1200)")
    if args.n_sensors != 192 or args.sensor_seed != 20260724:
        raise ValueError(
            "Frozen protocol requires 192 sensors and seed 20260724"
        )
    root = args.root.resolve()
    manifests = [
        prepare_scenario(
            SCENARIOS[name],
            root / "source",
            root / "prepared",
            args.frame_start,
            args.frame_stop,
            args.n_sensors,
            args.sensor_seed,
        )
        for name in names
    ]
    campaign_manifest = {
        "schema_version": 1,
        "status": "prepared_inputs_only_not_performance_evidence",
        "frozen_protocol": {
            "scenarios": list(SCENARIOS),
            "frames": [1000, 1200],
            "fixed_spatial_sensors": 192,
            "sensor_seed": 20260724,
        },
        "official_geometry_commit": OFFICIAL_COMMIT,
        "scenario_specifications": {
            name: asdict(specification)
            for name, specification in SCENARIOS.items()
        },
        "artifacts": manifests,
    }
    manifest_path = root / "prepared_manifest.json"
    manifest_path.write_text(
        json.dumps(campaign_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for manifest in manifests:
        print(
            f"{manifest['dataset_family']}: "
            f"{manifest['prepared_npz_path']} "
            f"{manifest['prepared_npz_sha256']}"
        )
    print(f"Campaign manifest: {manifest_path}")


if __name__ == "__main__":
    main()
