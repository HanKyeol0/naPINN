from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis.aggregate_pinn_ebm_upstream as upstream_aggregate


def _write_run(
    root: Path,
    config_path: Path,
    variant: str,
    offset: float,
) -> tuple[Path, dict]:
    run_dir = root / variant
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "stdout_stderr.log").write_text("complete\n", encoding="utf-8")
    result_path = run_dir / "source/results/reproduction/result.dat"
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(f"official {variant}".encode())
    result_info = {
        "path": str(result_path.relative_to(run_dir)),
        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "size_bytes": result_path.stat().st_size,
    }
    records = []
    for index in range(5):
        value = offset + index
        records.append(
            {
                "run_index": index,
                "lambda1": value,
                "lambda2": value + 0.1,
                "validation_mean_absolute_error_named_rmse_upstream": value
                + 0.2,
                "validation_nll": value + 0.3,
                "validation_pde_mean_squared_residual": value + 0.4,
                "training_seconds": value + 0.5,
                "ebm_initialization_seconds": value + 0.6,
            }
        )
    aggregate = {}
    for key in set(records[0]) - {"run_index"}:
        values = np.asarray([record[key] for record in records])
        aggregate[key] = {
            "mean": float(values.mean()),
            "std_sample": float(values.std(ddof=1)),
        }
    metrics = {
        "evidence_status": upstream_aggregate.EXPECTED_EVIDENCE,
        "variant_id": variant,
        "seed": 0,
        "nrun": 5,
        "model_index": 3,
        "records": records,
        "aggregate": aggregate,
        "official_result": result_info,
        "metric_name_warning": "test",
    }
    dataset = {
        "path": str((root / "dataset.mat").resolve()),
        "size_bytes": 1,
        "sha256": "test",
        "source_url": "test",
    }
    metadata = {
        "status": "complete",
        "evidence_status": upstream_aggregate.EXPECTED_EVIDENCE,
        "variant_id": variant,
        "official_result": result_info,
        "dataset": dataset,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return run_dir, dataset


def test_strict_upstream_aggregate_validates_both_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a_dir, a_dataset = _write_run(
        tmp_path,
        ROOT / "configs/rebuttal/pinn_ebm_upstream_active.yaml",
        upstream_aggregate.EXPECTED_VARIANTS["A"],
        1.0,
    )
    b_dir, b_dataset = _write_run(
        tmp_path,
        ROOT / "configs/rebuttal/pinn_ebm_paper_spec.yaml",
        upstream_aggregate.EXPECTED_VARIANTS["B"],
        2.0,
    )

    def fake_dataset(path: Path, config: dict) -> dict:
        return a_dataset if path == Path(a_dataset["path"]) else b_dataset

    monkeypatch.setattr(upstream_aggregate, "verify_dataset", fake_dataset)
    monkeypatch.setattr(
        upstream_aggregate,
        "expected_source_status",
        lambda config, source: {
            "path": str(source.resolve()),
            "commit": config["source"]["commit"],
            "tracked_status": "",
            "diff_sha256": "test",
            "diff": "",
        },
    )
    output = tmp_path / "aggregation.json"
    args = SimpleNamespace(
        run_dir_a=a_dir,
        run_dir_b=b_dir,
        output=output,
    )

    payload = upstream_aggregate.aggregate(args)

    assert payload["status"] == "strict_complete"
    assert payload["variant_count"] == 2
    assert payload["official_sequential_runs_per_variant"] == 5
    assert payload["comparison"]["lambda1"]["B_minus_A"] == pytest.approx(1.0)
    assert output.is_file()


def test_strict_upstream_aggregate_refuses_incomplete_run_without_output(
    tmp_path: Path,
) -> None:
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    output = tmp_path / "aggregation.json"
    args = SimpleNamespace(
        run_dir_a=a_dir,
        run_dir_b=b_dir,
        output=output,
    )

    with pytest.raises(FileNotFoundError, match="Incomplete upstream run"):
        upstream_aggregate.aggregate(args)

    assert not output.exists()
