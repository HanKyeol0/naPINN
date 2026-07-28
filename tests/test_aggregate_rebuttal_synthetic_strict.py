from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregate_rebuttal_synthetic import main


def _write_run(root: Path, seed: int) -> None:
    run_dir = root / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    metrics = {
        "status": "complete",
        "experiment_name": "allencahn2d",
        "noise_kind": "4G",
        "outlier_ratio": 0.15,
        "method": "napinn",
        "pde_loss_weight": 1.0,
        "data_loss_weight": 1.0,
        "field_rMAE": 0.1 + seed / 10000,
        "field_rMSE": 0.2 + seed / 10000,
        "pde_parameter_learned": 0.3,
        "pde_parameter_absolute_error": 0.0,
        "training_seconds": 1.0,
        "pinn_update_steps": 30000,
        "estimator_only_steps": 5000,
        "gpu_peak_memory_allocated_bytes": 1,
        "seed": seed,
    }
    config = {
        "run": {
            "method": "napinn",
            "seed": seed,
            "noise_kind": "4G",
            "outlier_ratio": 0.15,
            "ema_momentum": 0.05,
            "rejection_cost": 1.0,
            "pde_weight": 1.0,
            "data_weight": 1.0,
        }
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    (run_dir / "final.pt").write_bytes(b"test checkpoint")


def _args(root: Path, output_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_root=root,
        output_json=output_root.with_suffix(".json"),
        output_csv=output_root.with_suffix(".csv"),
        required_seeds=[40, 41, 42],
        strict=True,
        expected_runs=3,
        expected_groups=1,
    )


def test_strict_aggregate_writes_only_exact_complete_matrix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    for seed in (40, 41, 42):
        _write_run(root, seed)
    args = _args(root, tmp_path / "aggregate")

    main(args)

    payload = json.loads(args.output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "strict_complete"
    assert payload["included_run_count"] == 3
    assert len(payload["groups"]) == 1
    assert payload["groups"][0]["seeds"] == [40, 41, 42]
    assert args.output_csv.is_file()


def test_strict_aggregate_refuses_partial_matrix_without_writing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    for seed in (40, 41):
        _write_run(root, seed)
    args = _args(root, tmp_path / "aggregate")

    with pytest.raises(RuntimeError, match="found 2/3 required runs"):
        main(args)

    assert not args.output_json.exists()
    assert not args.output_csv.exists()


def test_strict_aggregate_accepts_explicit_35000_baseline_budget(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    for seed in (40, 41, 42):
        _write_run(root, seed)
        metrics_path = root / f"seed_{seed}" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["pinn_update_steps"] = 35000
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    args = _args(root, tmp_path / "aggregate_35k")
    args.expected_pinn_steps = 35000

    main(args)

    payload = json.loads(args.output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "strict_complete"
    assert payload["expected_pinn_steps_per_included_run"] == 35000
