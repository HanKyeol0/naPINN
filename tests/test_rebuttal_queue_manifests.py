from __future__ import annotations

import sys
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.run_synthetic_pinn_ebm_weight_queue import (
    jobs as weight_jobs,
)
from scripts.rebuttal.run_synthetic_supplement_queue import (
    jobs as supplement_jobs,
)
from scripts.rebuttal.launch_reviewer_recovery import (
    CAMPAIGNS,
    build_worker_commands,
    load_manifest,
    validate_dependencies,
)
from scripts.rebuttal.monitor_reviewer_recovery import (
    aggregate_status,
    build_summary,
    structured_track_count,
)
from scripts.rebuttal.finalize_reviewer_recovery import (
    CAMPAIGNS as FINALIZE_CAMPAIGNS,
    build_command as build_finalize_command,
)


def test_supplement_matrix_has_108_unique_run_directories() -> None:
    jobs = supplement_jobs()
    keys = {
        (
            job["experiment"],
            job["method"],
            job["seed"],
            job["noise"],
            job["ratio"],
            job["momentum"],
            job["rejection"],
        )
        for job in jobs
    }
    assert len(jobs) == len(keys) == 108


def test_weight_calibration_and_heldout_are_disjoint() -> None:
    calibration = weight_jobs("calibration")
    heldout = weight_jobs("heldout")
    assert len(calibration) == 9
    assert len(heldout) == 18
    assert {job[0] for job in calibration} == {39}
    assert {job[0] for job in heldout} == {40, 41, 42}
    assert set(calibration).isdisjoint(heldout)
    assert weight_jobs("all") == calibration + heldout


def test_reviewer_recovery_manifest_has_exact_declared_dimensions() -> None:
    path = ROOT / "configs/rebuttal/reviewer_recovery_manifest.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    queued = manifest["queued_after_primary_gpu_release"]
    assert manifest["active"]["realpdebench_multidataset"][
        "napinn_rejection_cost"
    ] == 0.5
    expected = {
        "synthetic_supplement": (108, 36, 3),
        "synthetic_pinn_ebm_weight_calibration": (9, 9, 1),
        "synthetic_pinn_ebm_weight_heldout": (18, 6, 3),
        "synthetic_compute_35k": (27, 9, 3),
        "synthetic_mad": (9, 3, 3),
        "structured_piv_fixed_re": (75, 25, 3),
        "structured_piv_inverse_re": (12, 4, 3),
        "structured_piv_mad": (15, 5, 3),
        "structured_piv_combined_strict_aggregate": (102, 34, 3),
    }
    for name, (runs, groups, seeds) in expected.items():
        record = queued[name]
        assert record["expected_runs"] == runs
        assert record["expected_groups"] == groups
        assert len(record["required_seeds"]) == seeds
        assert runs == groups * seeds

    calibration_root = queued[
        "synthetic_pinn_ebm_weight_calibration"
    ]["output_root"]
    heldout_root = queued["synthetic_pinn_ebm_weight_heldout"][
        "output_root"
    ]
    assert calibration_root != heldout_root
    assert queued["synthetic_mad"]["stage1_root"] != queued[
        "synthetic_mad"
    ]["output_root"]
    assert (
        queued["structured_piv_fixed_re"]["expected_runs"]
        + queued["structured_piv_inverse_re"]["expected_runs"]
        + queued["structured_piv_mad"]["expected_runs"]
        == queued["structured_piv_combined_strict_aggregate"][
            "expected_runs"
        ]
    )
    for name in ("structured_piv_fixed_re", "structured_piv_inverse_re"):
        assert queued[name]["napinn_rejection_cost"] == 0.10
        assert queued[name]["napinn_config"].endswith(
            "realpdebench_cylinder_napinn_rejection_01.yaml"
        )


def test_recovery_launcher_builds_exact_isolated_worker_commands() -> None:
    manifest = load_manifest(
        ROOT / "configs/rebuttal/reviewer_recovery_manifest.yaml"
    )
    for campaign in CAMPAIGNS:
        record, workers = build_worker_commands(
            manifest=manifest,
            campaign=campaign,
            gpus=[2, 3],
            num_shards=8,
            python_executable="python-test",
        )
        assert len(workers) == 8
        assert [worker["gpu"] for worker in workers] == [2, 3] * 4
        assert [worker["shard_index"] for worker in workers] == list(range(8))
        commands = [worker["command"] for worker in workers]
        for index, command in enumerate(commands):
            assert command[0] == "python-test"
            assert command[command.index("--shard-index") + 1] == str(index)
            assert command[command.index("--num-shards") + 1] == "8"
            assert command[command.index("--output-root") + 1] == record[
                "output_root"
            ]
            assert command[command.index("--status-root") + 1] == record[
                "status_root"
            ]
            assert "--fail-fast" in command

    calibration = build_worker_commands(
        manifest=manifest,
        campaign="synthetic_pinn_ebm_weight_calibration",
        gpus=[2, 3],
        num_shards=8,
    )[1][0]["command"]
    heldout = build_worker_commands(
        manifest=manifest,
        campaign="synthetic_pinn_ebm_weight_heldout",
        gpus=[2, 3],
        num_shards=8,
    )[1][0]["command"]
    assert calibration[calibration.index("--phase") + 1] == "calibration"
    assert heldout[heldout.index("--phase") + 1] == "heldout"
    mad = build_worker_commands(
        manifest=manifest,
        campaign="synthetic_mad",
        gpus=[2, 3],
        num_shards=8,
    )[1][0]["command"]
    assert "--stage1-root" in mad
    assert mad[mad.index("--stage1-root") + 1] != mad[
        mad.index("--output-root") + 1
    ]
    for campaign in (
        "structured_piv_fixed_re",
        "structured_piv_inverse_re",
    ):
        command = build_worker_commands(
            manifest=manifest,
            campaign=campaign,
            gpus=[2, 3],
            num_shards=8,
        )[1][0]["command"]
        assert command[command.index("--napinn-config") + 1].endswith(
            "realpdebench_cylinder_napinn_rejection_01.yaml"
        )


def test_recovery_launcher_validates_all_synthetic_mad_dependencies(
    tmp_path: Path,
) -> None:
    for experiment in ("allencahn2d", "burgers2d", "lambdaomega2d"):
        for seed in (40, 41, 42):
            run_dir = (
                tmp_path
                / experiment
                / "4G"
                / "15pct"
                / "lad"
                / "ema_0.05_reject_0.5_pde_1_data_1"
                / f"seed_{seed}"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "final.pt").write_bytes(b"checkpoint")
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "method": "lad",
                        "pinn_update_steps": 30000,
                    }
                ),
                encoding="utf-8",
            )
    result = validate_dependencies(
        "synthetic_mad",
        {
            "stage1_root": str(tmp_path),
            "output_root": str(tmp_path / "mad"),
        },
    )
    assert result["validated_checkpoint_count"] == 9
    assert len(result["checkpoints"]) == 9

    next(iter(tmp_path.rglob("final.pt"))).unlink()
    try:
        validate_dependencies(
            "synthetic_mad",
            {
                "stage1_root": str(tmp_path),
                "output_root": str(tmp_path / "mad"),
            },
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing MAD stage-one checkpoint was accepted")


def test_recovery_monitor_separates_shared_structured_tracks() -> None:
    records = [
        {"tag": "mse_sensor_drift10"},
        {"tag": "pinn_ebm_pilar_inverse_re_sensor_drift30"},
        {"tag": "mad_pinn_ar1"},
    ]
    assert structured_track_count("structured_piv_fixed_re", records) == 1
    assert structured_track_count("structured_piv_inverse_re", records) == 1
    assert structured_track_count("structured_piv_mad", records) == 1
    assert (
        structured_track_count(
            "structured_piv_combined_strict_aggregate", records
        )
        == 3
    )


def test_recovery_monitor_contains_progress_not_performance_values() -> None:
    summary = build_summary(
        ROOT / "configs/rebuttal/reviewer_recovery_manifest.yaml"
    )
    assert "not performance evidence" in summary["semantics"]
    assert set(summary["campaigns"]) >= {
        "synthetic_core",
        "realpdebench_multidataset",
        "pinn_ebm_upstream_a_b",
        "allen_cahn_rejection_cost_1",
        "synthetic_supplement",
        "structured_piv_combined_strict_aggregate",
    }
    serialized = json.dumps(summary)
    for forbidden in ("field_rMAE", "field_rMSE", "learned_reynolds"):
        assert forbidden not in serialized


def test_recovery_monitor_rejects_existing_but_invalid_aggregate(
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "aggregation.json"
    csv_path = tmp_path / "aggregation.csv"
    aggregate.write_text(
        json.dumps(
            {
                "status": "strict_complete",
                "included_run_count": 2,
                "groups": [{}, {}],
                "required_seeds": [40, 41, 42],
            }
        ),
        encoding="utf-8",
    )
    csv_path.write_text("header\n", encoding="utf-8")
    record = {
        "expected_runs": 3,
        "expected_groups": 1,
        "required_seeds": [40, 41, 42],
        "strict_aggregate": {
            "json": str(aggregate),
            "csv": str(csv_path),
        },
    }
    status = aggregate_status("synthetic_test", record)
    assert status["json_exists"]
    assert status["csv_exists"]
    assert not status["evidence_ready"]
    assert set(status["aggregate_validation_errors"]) == {
        "included_run_count_mismatch",
        "group_count_mismatch",
    }


def test_recovery_finalizer_builds_all_strict_commands_from_manifest() -> None:
    manifest = load_manifest(
        ROOT / "configs/rebuttal/reviewer_recovery_manifest.yaml"
    )
    for campaign in FINALIZE_CAMPAIGNS:
        command = build_finalize_command(
            manifest, campaign, python_executable="python-test"
        )
        assert command[0] == "python-test"
        assert "--output" in command or "--output-json" in command
        assert "--strict" in command or campaign in {
            "realpdebench_multidataset",
            "pinn_ebm_upstream_a_b",
            "structured_piv_combined_strict_aggregate",
        }

    calibration = build_finalize_command(
        manifest,
        "synthetic_pinn_ebm_weight_calibration",
    )
    assert calibration[
        calibration.index("--required-seeds") + 1 : calibration.index(
            "--strict"
        )
    ] == ["39"]
    assert calibration[calibration.index("--expected-runs") + 1] == "9"
    heldout = build_finalize_command(
        manifest, "synthetic_pinn_ebm_weight_heldout"
    )
    assert heldout[
        heldout.index("--required-seeds") + 1 : heldout.index("--strict")
    ] == ["40", "41", "42"]
    compute = build_finalize_command(manifest, "synthetic_compute_35k")
    assert compute[compute.index("--expected-pinn-steps") + 1] == "35000"
    mad = build_finalize_command(manifest, "synthetic_mad")
    assert mad[mad.index("--expected-pinn-steps") + 1] == "60000"
