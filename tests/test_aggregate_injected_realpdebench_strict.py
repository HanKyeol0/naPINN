import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregate_injected_realpdebench import (
    EXPECTED_GROUPS,
    EXPECTED_RUNS,
    EXPECTED_TAG_METHOD,
    HELD_OUT_SEEDS,
    main,
    sha256_file,
)


def _write_complete_matrix(root: Path) -> None:
    for tag, method in EXPECTED_TAG_METHOD.items():
        input_path = root / "inputs" / f"{tag}.npz"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(f"input:{tag}".encode())
        input_hash = sha256_file(input_path)
        inverse = "_inverse_re_" in tag
        for seed in sorted(HELD_OUT_SEEDS):
            run_dir = (
                root
                / "cylinder_real_piv"
                / tag
                / f"heldout_seed_{seed}"
            )
            run_dir.mkdir(parents=True)
            estimator_steps = 5000 if method in {"pinn_ebm", "napinn"} else 0
            metrics = {
                "status": "complete",
                "evidence_status": "full_run_complete_unaggregated",
                "smoke_test": False,
                "seed": seed,
                "method": method,
                "tag": tag,
                "pinn_update_steps": 30000,
                "estimator_init_steps": estimator_steps,
                "benchmark": "RealPDEBench/2D_Cylinder_Flow",
                "sim_id": "10031_5.0.h5",
                "source_sha256": "source-hash",
                "sensor_seed": 20260320,
                "n_spatial_sensors": 192,
                "n_frames": 200,
                "n_heldout_measurements": 10,
                "corruption_kind": tag,
                "input_artifact_path": str(input_path.resolve()),
                "input_artifact_sha256": input_hash,
                "learned_reynolds": inverse,
                "rMAE": 0.1 + seed / 1000,
                "rMSE": 0.2 + seed / 1000,
            }
            if method == "mad_pinn":
                metrics.update(
                    {
                        "mad_stage1_pinn_update_steps": 30000,
                        "mad_stage2_pinn_update_steps": 30000,
                        "mad_total_pipeline_pinn_update_steps": 60000,
                    }
                )
                (run_dir / "mad_screening.npz").write_bytes(b"screen")
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            config = {
                "seed": seed,
                "tag": tag,
                "method": {"kind": method},
                "data": {
                    "path": str(input_path.resolve()),
                    "input_artifact_sha256": input_hash,
                },
                "effective_schedule": {
                    "pinn_update_steps": 30000,
                    "estimator_init_steps": estimator_steps,
                },
            }
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            metadata = {
                "status": "complete",
                "evidence_status": "full_run_complete_unaggregated",
                "smoke_test": False,
                "seed": seed,
                "input_artifact_path": str(input_path.resolve()),
                "input_artifact_sha256": input_hash,
            }
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (run_dir / "final.pt").write_bytes(b"checkpoint")


def test_frozen_matrix_dimensions() -> None:
    assert len(EXPECTED_TAG_METHOD) == EXPECTED_GROUPS == 34
    assert EXPECTED_GROUPS * len(HELD_OUT_SEEDS) == EXPECTED_RUNS == 102


def test_strict_aggregation_succeeds_only_for_complete_matrix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    _write_complete_matrix(root)
    output = tmp_path / "aggregation.json"
    args = argparse.Namespace(root=root, output=output)
    main(args)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "strict_complete"
    assert payload["included_run_count"] == 102
    assert payload["three_seed_group_count"] == 34
    assert payload["policy"]["expected_runs"] == 102
    assert payload["policy"]["expected_groups"] == 34
    assert len(payload["groups"]) == 34
    assert output.with_suffix(".csv").is_file()

    one_metrics = next(root.rglob("metrics.json"))
    one_metrics.unlink()
    second_output = tmp_path / "partial.json"
    args = argparse.Namespace(root=root, output=second_output)
    try:
        main(args)
    except RuntimeError as error:
        assert "exactly 102" in str(error)
    else:
        raise AssertionError("Partial matrix was incorrectly aggregated")
    assert not second_output.exists()
    assert not second_output.with_suffix(".csv").exists()


def test_strict_aggregation_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    args = argparse.Namespace(root=tmp_path / "missing", output=output)
    try:
        main(args)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing strict aggregation was overwritten")
    assert output.read_text(encoding="utf-8") == "keep"
