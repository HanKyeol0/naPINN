from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.inject_realpdebench_multidataset import inject_parent
from scripts.rebuttal.prepare_realpdebench_multidataset import (
    Scenario,
    prepare_scenario,
)
from scripts.rebuttal.run_realpdebench_multidataset_queue import (
    build_jobs,
    load_protocol,
    shard_assignments,
)


def _write_tiny_arrow(path: Path) -> Scenario:
    shape = (1200, 16, 16)
    rows, columns = np.indices(shape[1:])
    x = columns.astype(np.float64) * 0.1
    y = rows.astype(np.float64) * 0.1
    t = np.arange(shape[0], dtype=np.float64) * 0.01
    phase = np.arange(shape[0], dtype=np.float32)[:, None, None]
    u = 1.0 + 0.001 * phase + 0.01 * columns[None]
    v = -0.5 + 0.0005 * phase + 0.01 * rows[None]
    table = pa.table(
        {
            "sim_id": ["123_test.h5"],
            "u": [np.asarray(u, dtype=np.float32).tobytes()],
            "v": [np.asarray(v, dtype=np.float32).tobytes()],
            "shape_t": [shape[0]],
            "shape_h": [shape[1]],
            "shape_w": [shape[2]],
            "vo": [None],
            "x": [x.tobytes()],
            "x_shape_h": [shape[1]],
            "x_shape_w": [shape[2]],
            "y": [y.tobytes()],
            "y_shape_h": [shape[1]],
            "y_shape_w": [shape[2]],
            "t": [t.tobytes()],
            "t_shape": [shape[0]],
        }
    )
    with path.open("wb") as stream:
        with ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table)
    return Scenario(
        name="tiny",
        source_name=path.name,
        source_hf_path="tiny/real/data.arrow",
        sim_id="123_test.h5",
        geometry_kind="official_reference_disk",
        diameter_px=2.0,
        center_x_px=8.0,
        center_y_px=8.0,
        geometry_limitation="Synthetic test fixture only.",
    )


def test_tiny_prepare_and_paired_nested_injection(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    scenario = _write_tiny_arrow(source_root / "tiny.arrow")
    prepared = prepare_scenario(
        scenario,
        source_root,
        tmp_path / "prepared",
        frame_start=1000,
        frame_stop=1200,
        n_sensors=192,
        sensor_seed=20260724,
    )
    parent = Path(prepared["prepared_npz_path"])
    with np.load(parent, allow_pickle=False) as raw:
        train = raw["train_sensor_flat_indices"]
        heldout = raw["heldout_flat_indices"]
        assert np.intersect1d(train, heldout).size == 0
        assert np.isfinite(raw["u_observed"][:, train // 16, train % 16]).all()
        assert np.isfinite(
            raw["v_observed"][:, heldout // 16, heldout % 16]
        ).all()

    records = inject_parent(parent, tmp_path / "corrupted", seed=40)
    assert len(records) == 2
    first, second = records
    assert (
        first["corruption"]["pairing"]["pairing_id"]
        == second["corruption"]["pairing"]["pairing_id"]
    )
    for key in (
        "component_choices_sha256",
        "standard_normal_sha256",
        "raw_four_gaussian_sha256",
        "gross_row_permutation_sha256",
        "positive_gross_factors_sha256",
        "background_noise_sha256",
    ):
        assert (
            first["corruption"]["checksums"][key]
            == second["corruption"]["checksums"][key]
        )
    with np.load(first["artifact_path"], allow_pickle=False) as ten, np.load(
        second["artifact_path"], allow_pickle=False
    ) as fifteen:
        ten_indices = ten["legacy4g_gross_row_indices"]
        fifteen_indices = fifteen["legacy4g_gross_row_indices"]
        assert np.array_equal(
            ten_indices, fifteen_indices[: ten_indices.size]
        )
        assert set(ten_indices.tolist()).issubset(fifteen_indices.tolist())
        assert np.array_equal(
            ten["legacy4g_background_noise_nondimensional"],
            fifteen["legacy4g_background_noise_nondimensional"],
        )
        assert "u_observed" in ten.files and "v_observed" in ten.files
        assert "training_corruption_mask" in ten.files


def test_frozen_matrix_has_exact_unique_eight_shard_coverage() -> None:
    protocol = load_protocol()
    jobs = build_jobs(protocol)
    assert len(jobs) == 144
    assert len({job.key for job in jobs}) == 144
    assignments = shard_assignments(jobs, 8)
    assert [len(shard) for shard in assignments] == [18] * 8
    flattened = [job.key for shard in assignments for job in shard]
    assert len(flattened) == len(set(flattened)) == 144


def test_combustion_is_one_channel_applicability_limitation() -> None:
    payload = json.loads(
        Path(
            "outputs/rebuttal/realpdebench_multidataset/"
            "combustion_applicability.json"
        ).read_text(encoding="utf-8")
    )
    audit = payload["observed_data_audit"]
    compatibility = payload["current_method_compatibility"]
    assert audit["physical_observation"] == "real OH* chemiluminescence intensity"
    assert audit["observed_channels"] == 1
    assert not audit["velocity_u_channel_present"]
    assert not audit["velocity_v_channel_present"]
    assert not compatibility["applicable_to_current_pressure_latent_velocity_pinn"]
    assert compatibility["arbitrary_pde_application_forbidden"]
