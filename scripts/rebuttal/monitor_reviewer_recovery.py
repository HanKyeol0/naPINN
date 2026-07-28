#!/usr/bin/env python3
"""Write a non-performance progress summary for all rebuttal recoveries."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml


MANIFEST = Path("configs/rebuttal/reviewer_recovery_manifest.yaml")
OUTPUT = Path("outputs/status/reviewer_recovery_summary.json")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def complete_metric_records(root: Path) -> list[dict[str, Any]]:
    records = []
    if not root.is_dir():
        return records
    for path in sorted(root.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "complete":
            continue
        if bool(payload.get("smoke_test", False)):
            continue
        payload["_metrics_path"] = str(path.resolve())
        records.append(payload)
    return records


def validate_aggregate_payload(
    name: str,
    record: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    errors = []
    if payload.get("status") != "strict_complete":
        errors.append("status_not_strict_complete")
    expected_runs = record.get("expected_runs")
    expected_groups = record.get("expected_groups")
    if name == "pinn_ebm_upstream_a_b":
        if int(payload.get("variant_count", -1)) != 2:
            errors.append("variant_count_not_2")
        if int(payload.get("official_sequential_runs_per_variant", -1)) != 5:
            errors.append("official_runs_per_variant_not_5")
        if set(payload.get("variants", {})) != {"A", "B"}:
            errors.append("variant_keys_not_A_B")
        return errors
    if expected_runs is not None:
        observed = payload.get("included_run_count")
        if int(observed if observed is not None else -1) != int(expected_runs):
            errors.append("included_run_count_mismatch")
    if expected_groups is not None:
        observed = payload.get("three_seed_group_count")
        if observed is None and isinstance(payload.get("groups"), list):
            observed = len(payload["groups"])
        if int(observed if observed is not None else -1) != int(expected_groups):
            errors.append("group_count_mismatch")
    required_seeds = record.get("required_seeds")
    if required_seeds is not None and payload.get("required_seeds") is not None:
        if list(payload["required_seeds"]) != list(required_seeds):
            errors.append("required_seeds_mismatch")
    return errors


def aggregate_status(name: str, record: dict[str, Any]) -> dict[str, Any]:
    aggregate = record.get("strict_aggregate")
    if not isinstance(aggregate, dict):
        return {"strict_aggregate_declared": False}
    output: dict[str, Any] = {
        "strict_aggregate_declared": True,
        "aggregate_validation_errors": [],
    }
    payload = None
    for kind in ("json", "csv"):
        value = aggregate.get(kind)
        if not value:
            continue
        path = Path(str(value))
        output[f"{kind}_path"] = str(path)
        output[f"{kind}_exists"] = path.is_file()
        if path.is_file():
            output[f"{kind}_sha256"] = sha256_file(path)
            output[f"{kind}_size_bytes"] = path.stat().st_size
            if kind == "json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    output["aggregate_validation_errors"].append(
                        "aggregate_json_unreadable"
                    )
    if payload is not None:
        if not isinstance(payload, dict):
            output["aggregate_validation_errors"].append(
                "aggregate_json_not_mapping"
            )
        else:
            output["aggregate_validation_errors"].extend(
                validate_aggregate_payload(name, record, payload)
            )
    csv_required = "csv" in aggregate
    output["evidence_ready"] = bool(
        output.get("json_exists", False)
        and (not csv_required or output.get("csv_exists", False))
        and not output["aggregate_validation_errors"]
    )
    return output


def structured_track_count(
    track: str, records: list[dict[str, Any]]
) -> int:
    tags = [str(record.get("tag", "")) for record in records]
    if track == "structured_piv_fixed_re":
        return sum(
            "_inverse_re_" not in tag and not tag.startswith("mad_pinn_")
            for tag in tags
        )
    if track == "structured_piv_inverse_re":
        return sum("_inverse_re_" in tag for tag in tags)
    if track == "structured_piv_mad":
        return sum(tag.startswith("mad_pinn_") for tag in tags)
    if track == "structured_piv_combined_strict_aggregate":
        return len(tags)
    raise ValueError(track)


def campaign_progress(
    name: str,
    record: dict[str, Any],
    *,
    shared_structured_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output = {
        "manifest_status": record.get("status"),
        "expected_runs": record.get("expected_runs"),
        "expected_groups": record.get("expected_groups"),
        "required_seeds": record.get("required_seeds"),
    }
    root_value = record.get("output_root") or record.get("input_root")
    if root_value:
        output["output_root"] = str(root_value)
    if name.startswith("structured_piv_"):
        records = shared_structured_records or []
        complete = structured_track_count(name, records)
    elif root_value:
        complete = len(complete_metric_records(Path(str(root_value))))
    else:
        complete = 0
    output["discovered_complete_non_smoke_metrics"] = complete
    expected = record.get("expected_runs")
    output["discovered_run_fraction"] = (
        complete / int(expected) if expected else None
    )
    output.update(aggregate_status(name, record))
    return output


def build_summary(manifest_path: Path) -> dict[str, Any]:
    manifest = load_yaml(manifest_path)
    active = manifest["active"]
    complete = manifest["complete"]
    queued = manifest["queued_after_primary_gpu_release"]
    structured_root = Path(
        str(queued["structured_piv_combined_strict_aggregate"]["input_root"])
    )
    structured_records = complete_metric_records(structured_root)
    campaigns: dict[str, Any] = {}
    for section_name, section in (
        ("active", active),
        ("complete", complete),
        ("queued_after_primary_gpu_release", queued),
    ):
        for name, record in section.items():
            if name == "pinn_ebm_upstream_a_b":
                monitor_runs: dict[str, dict[str, Any]] = {}
                monitor_path_value = record.get("monitor_json")
                if monitor_path_value:
                    monitor_path = Path(str(monitor_path_value))
                    if monitor_path.is_file():
                        try:
                            monitor_payload = json.loads(
                                monitor_path.read_text(encoding="utf-8")
                            )
                        except json.JSONDecodeError:
                            monitor_payload = {}
                        for run in monitor_payload.get("runs", []):
                            monitor_runs[str(run.get("variant_id"))] = run
                variant_status = {}
                for variant, path_value in record["variants"].items():
                    run_dir = Path(str(path_value))
                    metadata_path = run_dir / "run_metadata.json"
                    metadata = {}
                    if metadata_path.is_file():
                        try:
                            metadata = json.loads(
                                metadata_path.read_text(encoding="utf-8")
                            )
                        except json.JSONDecodeError:
                            metadata = {"status": "unreadable"}
                    progress = monitor_runs.get(variant, {})
                    variant_status[variant] = {
                        "run_dir": str(run_dir),
                        "metadata_status": metadata.get("status", "absent"),
                        "official_metrics_exists": (
                            run_dir / "metrics.json"
                        ).is_file(),
                        "completed_update_equivalents": progress.get(
                            "completed_update_equivalents"
                        ),
                        "total_update_equivalents": progress.get(
                            "total_update_equivalents"
                        ),
                        "progress_fraction": progress.get(
                            "progress_fraction"
                        ),
                        "naive_eta_utc": progress.get("naive_eta_utc"),
                        "tail_retry_summary": progress.get(
                            "tail_retry_summary"
                        ),
                    }
                campaigns[name] = {
                    "section": section_name,
                    "manifest_status": record.get("status"),
                    "variants": variant_status,
                    **aggregate_status(name, record),
                }
                continue
            campaigns[name] = {
                "section": section_name,
                **campaign_progress(
                    name,
                    record,
                    shared_structured_records=structured_records,
                ),
            }
    return {
        "schema_version": 1,
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "semantics": (
            "Progress metadata only. Counts and partial artifacts are not "
            "performance evidence. evidence_ready is true only when the "
            "declared strict aggregate JSON exists."
        ),
        "campaigns": campaigns,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    payload = build_summary(args.manifest)
    write_json_atomic(args.output, payload)
    for name, record in payload["campaigns"].items():
        if "discovered_complete_non_smoke_metrics" in record:
            print(
                f"{name}: "
                f"{record['discovered_complete_non_smoke_metrics']}/"
                f"{record.get('expected_runs')} "
                f"evidence_ready={record.get('evidence_ready', False)}"
            )
        else:
            print(
                f"{name}: variants={len(record.get('variants', {}))} "
                f"evidence_ready={record.get('evidence_ready', False)}"
            )
    print(f"Status: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
