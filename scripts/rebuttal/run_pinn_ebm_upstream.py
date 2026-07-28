#!/usr/bin/env python3
"""Prepare, audit, smoke-test, or run the official PINN-EBM code.

This runner deliberately separates:

* variant A: the active upstream source at commit 0b74f6f (4x30 Net_NS), and
* variant B: the same source with one declared 8x20 architecture patch.

The official training files are executed from an archived source checkout.
Fresh artifacts are written only below ``outputs/``. Component smoke output is
always marked non-evidentiary and must not be reported as a reproduction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/rebuttal/pinn_ebm_upstream"
DEFAULT_DATASET = DEFAULT_OUTPUT_ROOT / "data/cylinder_nektar_wake.mat"
ALLOWED_VARIANTS = {
    "A_upstream_active_0b74f6f",
    "B_paper_spec_8x20",
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def git_output(source_dir: Path, *args: str) -> str:
    return run_command(["git", *args], cwd=source_dir).stdout.strip()


def validate_config(config: dict[str, Any], config_path: Path) -> None:
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema in {config_path}")
    variant = config.get("variant_id")
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(f"Unsupported variant_id {variant!r}")

    source = config.get("source")
    dataset = config.get("dataset")
    execution = config.get("execution")
    expected = config.get("expected_active_source")
    for name, value in (
        ("source", source),
        ("dataset", dataset),
        ("execution", execution),
        ("expected_active_source", expected),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")

    if source["commit"] != "0b74f6f9209d68c79ecb9608b71755977d08f578":
        raise ValueError("The upstream commit is frozen and may not be changed")
    if execution["x_opt"] != 101 or execution["noise"] != "3G":
        raise ValueError("The faithful variants must use Navier--Stokes with 3G noise")
    if execution["prop_noise"] != 0:
        raise ValueError("The disclosed upstream runnability choice is additive noise")
    if execution["model_indices"] != [0, 3, 2]:
        raise ValueError("The official comparison order [0,3,2] must be preserved")
    if execution["pinn_ebm_model_index"] != 3:
        raise ValueError("Official no-offset PINN-EBM is model index 3")
    if execution["pinn_updates"] != 100000:
        raise ValueError("The faithful Navier--Stokes budget is 100,000 updates")
    if execution["ebm_init_after_pinn_update"] != 10000:
        raise ValueError("The faithful EBM initialization point is update 10,000")
    if execution["ebm_init_updates"] != 2000:
        raise ValueError("The faithful EBM-only initialization uses 2,000 updates")
    if execution["scheduler_update"] != 80000:
        raise ValueError("The faithful learning-rate decay occurs at update 80,000")

    patch = source.get("patch")
    if variant.startswith("A_") and patch is not None:
        raise ValueError("Variant A must not apply a source patch")
    if variant.startswith("B_") and not patch:
        raise ValueError("Variant B requires its declared architecture patch")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_dataset(path: Path, dataset_config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset {path}. Re-run with --download-dataset."
        )
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != int(dataset_config["size_bytes"]):
        raise ValueError(
            f"Dataset size mismatch: expected {dataset_config['size_bytes']}, got {size}"
        )
    if digest != dataset_config["sha256"]:
        raise ValueError(
            f"Dataset SHA-256 mismatch: expected {dataset_config['sha256']}, got {digest}"
        )
    return {
        "path": str(path.resolve()),
        "size_bytes": size,
        "sha256": digest,
        "source_url": dataset_config["source_url"],
    }


def download_dataset(path: Path, dataset_config: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return verify_dataset(path, dataset_config)
    temporary = path.with_name(f".{path.name}.download-{os.getpid()}")
    try:
        with urllib.request.urlopen(dataset_config["source_url"]) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        verify_dataset(temporary, dataset_config)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return verify_dataset(path, dataset_config)


def expected_source_status(config: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    commit = git_output(source_dir, "rev-parse", "HEAD")
    if commit != config["source"]["commit"]:
        raise ValueError(f"Source commit mismatch in {source_dir}: {commit}")

    status = git_output(source_dir, "status", "--short", "--untracked-files=no")
    diff = git_output(source_dir, "diff", "--no-ext-diff", "--binary")
    variant = config["variant_id"]
    if variant.startswith("A_") and status:
        raise ValueError(f"Variant A source is unexpectedly modified:\n{status}")
    if variant.startswith("B_"):
        changed = git_output(source_dir, "diff", "--name-only").splitlines()
        if changed != ["code/pebm/utils_navier_stokes.py"]:
            raise ValueError(f"Variant B changed unexpected files: {changed}")
        run_command(["git", "diff", "--check"], cwd=source_dir)
        if not diff:
            raise ValueError("Variant B architecture patch is not present")
    return {
        "path": str(source_dir.resolve()),
        "commit": commit,
        "tracked_status": status,
        "diff_sha256": sha256_text(diff),
        "diff": diff,
    }


def stage_source(config: dict[str, Any], output_root: Path) -> tuple[Path, dict[str, Any]]:
    source_root = output_root / "_sources"
    source_root.mkdir(parents=True, exist_ok=True)
    source_dir = source_root / config["variant_id"]
    if source_dir.exists():
        return source_dir, expected_source_status(config, source_dir)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{config['variant_id']}-",
            dir=source_root,
        )
    )
    try:
        run_command(
            [
                "git",
                "clone",
                "--filter=blob:none",
                config["source"]["repository"],
                str(temporary),
            ]
        )
        run_command(
            ["git", "checkout", "--detach", config["source"]["commit"]],
            cwd=temporary,
        )
        patch = config["source"].get("patch")
        if patch:
            patch_path = resolve_project_path(patch)
            run_command(["git", "apply", "--check", str(patch_path)], cwd=temporary)
            run_command(["git", "apply", str(patch_path)], cwd=temporary)
        metadata = expected_source_status(config, temporary)
        temporary.replace(source_dir)
        metadata["path"] = str(source_dir.resolve())
        write_json(source_dir / "codex_source_manifest.json", metadata)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return source_dir, expected_source_status(config, source_dir)


def render_upstream_input(config: dict[str, Any]) -> str:
    execution = config["execution"]
    return textwrap.dedent(
        f"""\
        # Generated by scripts/rebuttal/run_pinn_ebm_upstream.py.
        # This uses the official input.py extension mechanism; training source is unchanged.
        if not 'pars' in locals():
            pars = dict()

        set_par(pars, 'x_opt', {execution['x_opt']})
        set_par(pars, 'n_opt', {execution['noise']!r})
        set_par(pars, 'Npinn', {execution['pinn_updates']})
        set_par(pars, 'Nrun', {execution['nrun']})
        set_par(pars, 'jmodel_vec', {execution['model_indices']!r})
        set_par(pars, 'i_init_ebm', {execution['ebm_init_after_pinn_update']})
        set_par(pars, 'i_sched', {execution['scheduler_update']})
        set_par(pars, 'itest', {execution['test_interval']})
        set_par(pars, 'iplot', {execution['plot_interval']})
        set_par(pars, 'fnoise', {execution['fnoise']})
        set_par(pars, 'prop_noise', {execution['prop_noise']})
        set_par(pars, 'lf_fac2', {execution['joint_pde_weight']})
        set_par(pars, 'lf_fac2_alt', {execution['non_ebm_pde_weight']})
        set_par(pars, 'ebm_ubound', {execution['ebm_initial_upper_bound']})
        ds = get_ds(pars)
        """
    )


def _component_smoke_program(expected_layers: int, expected_width: int) -> str:
    return textwrap.dedent(
        f"""\
        import json
        import math
        import torch
        from pebm.datasets import get_ds
        from pebm.ebm import EBM
        from pebm.utils_navier_stokes import Net_NS, get_aux, load_ns_data

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pars = {{"x_opt": 101, "fnoise": 1, "prop_noise": 0}}
        ds = get_ds(pars)
        model = Net_NS(device).to(device)
        ebm = EBM(pars | {{"ebm_ubound": 1}}, torch.tensor(2.0), device)

        generator = torch.Generator(device=device)
        generator.manual_seed(1234)
        inputs = torch.rand((8, 3), generator=generator, device=device)
        inputs[:, 0] = 1.0 + 7.0 * inputs[:, 0]
        inputs[:, 1] = -2.0 + 4.0 * inputs[:, 1]
        inputs[:, 2] = 19.9 * inputs[:, 2]
        inputs.requires_grad_(True)

        outputs = model(inputs)
        u, v, pressure, f, g = get_aux(
            inputs, outputs, model.dpar[0], model.dpar[1]
        )
        prediction = torch.stack((u, v), dim=1)
        target = prediction.detach() + torch.linspace(
            -0.1, 0.1, prediction.numel(), device=device
        ).reshape_as(prediction)
        residual = (target - prediction).flatten()
        nll = ebm.get_mean_NLL(residual)
        pde = (f.square() + g.square()).mean()
        loss = nll + 50.0 * pde
        model.zero_grad()
        ebm.optimizer_ebm.zero_grad()
        loss.backward()

        model_grads = [
            parameter.grad for parameter in model.parameters()
            if parameter.grad is not None
        ]
        ebm_grads = [
            parameter.grad for parameter in ebm.net_ebm.parameters()
            if parameter.grad is not None
        ]
        result = {{
            "evidence_status": "non_evidentiary_component_smoke",
            "device": str(device),
            "hidden_layers": len(model.Wlist) - 1,
            "width": int(model.Wlist[0].shape[1]),
            "output_width": int(model.Wlist[-1].shape[1]),
            "trainable_dpar_count": int(model.dpar.numel()),
            "used_pde_parameter_indices": [0, 1],
            "declared_pinn_layers": len(pars["Uvec_pinn"]),
            "declared_pinn_width": int(pars["Uvec_pinn"][0]),
            "ebm_hidden_layers": len(ebm.net_ebm.layers) - 1,
            "ebm_width": int(ebm.net_ebm.layers[0].out_features),
            "ebm_dropout": float(ebm.net_ebm.dr.p),
            "adaptive_grid_points": int(ebm.get_rvec(residual).numel()),
            "data_batch_size": int(pars["bs_train"]),
            "collocation_batch_size": int(pars["bs_coll"]),
            "pinn_learning_rate": float(pars["lr_pinn"]),
            "ebm_learning_rate": float(pars["lr_ebm"]),
            "ebm_init_updates": int(pars["Nebm"]),
            "load_ns_data_defaults": list(load_ns_data.__defaults__),
            "loss": float(loss.detach().cpu()),
            "nll": float(nll.detach().cpu()),
            "pde_loss": float(pde.detach().cpu()),
            "pinn_gradient_finite": bool(
                model_grads and all(torch.isfinite(x).all() for x in model_grads)
            ),
            "ebm_gradient_finite": bool(
                ebm_grads and all(torch.isfinite(x).all() for x in ebm_grads)
            ),
        }}
        assert result["hidden_layers"] == {expected_layers}
        assert result["width"] == {expected_width}
        assert result["output_width"] == 2
        assert result["ebm_hidden_layers"] == 3
        assert result["ebm_width"] == 5
        assert result["ebm_dropout"] == 0.5
        assert result["adaptive_grid_points"] == 1001
        assert result["data_batch_size"] == 200
        assert result["collocation_batch_size"] == 100
        assert result["pinn_learning_rate"] == 0.002
        assert result["ebm_learning_rate"] == 0.002
        assert result["ebm_init_updates"] == 2000
        assert result["load_ns_data_defaults"] == [4000, 1000]
        assert math.isfinite(result["loss"])
        assert result["pinn_gradient_finite"]
        assert result["ebm_gradient_finite"]
        print(json.dumps(result, sort_keys=True))
        """
    )


def device_environment(device: str) -> dict[str, str]:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        index = int(device)
        if index < 0:
            raise ValueError("--device must be 'cpu' or a non-negative GPU index")
        env["CUDA_VISIBLE_DEVICES"] = str(index)
    return env


def run_component_smoke(
    config: dict[str, Any],
    source_dir: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    expected_pinn = config["expected_active_source"]["pinn"]
    program = _component_smoke_program(
        int(expected_pinn["hidden_layers"]),
        int(expected_pinn["width"]),
    )
    result = run_command(
        [sys.executable, "-c", program],
        cwd=source_dir / "code",
        env=device_environment(device),
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    smoke_dir = output_root / "smoke" / config["variant_id"]
    smoke_dir.mkdir(parents=True, exist_ok=True)
    payload.update(
        {
            "variant_id": config["variant_id"],
            "source_commit": config["source"]["commit"],
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": "component-level forward/backward; no training reproduction",
        }
    )
    write_json(smoke_dir / "metrics.json", payload)
    return payload


def audit_source_semantics(config: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    ns_text = (source_dir / "code/pebm/utils_navier_stokes.py").read_text(
        encoding="utf-8"
    )
    ebm_text = (source_dir / "code/pebm/ebm.py").read_text(encoding="utf-8")
    train_text = (source_dir / "code/pebm/utils_train.py").read_text(
        encoding="utf-8"
    )
    required = {
        "psi_pressure_outputs": "psi = outputs_nn[:,0]" in ns_text
        and "p = outputs_nn[:,1]" in ns_text,
        "velocity_from_streamfunction": "u = torch.autograd.grad(psi.sum()" in ns_text
        and "v = -torch.autograd.grad(psi.sum()" in ns_text,
        "lambda1_lambda2_path": "get_aux(t, x, dpar[0], dpar[1])" in (
            source_dir / "code/pebm/datasets.py"
        ).read_text(encoding="utf-8"),
        "adaptive_grid_1001": "torch.linspace(lb, ub, 1001" in ebm_text,
        "tail_retry": "while (True):" in train_text
        and "if Ntry == 3" in train_text
        and "ebm.Nebm *= 2" in train_text,
        "direct_nll": "loss_d0 = ebm.get_mean_NLL(residuals)" in train_text,
        "joint_backward": "loss.backward()" in (
            source_dir / "code/pebm.py"
        ).read_text(encoding="utf-8"),
    }
    if not all(required.values()):
        failed = [key for key, value in required.items() if not value]
        raise ValueError(f"Source semantic audit failed: {failed}")
    return required


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
    }
    for package in ("torch", "numpy", "scipy", "matplotlib", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def current_repository_metadata() -> dict[str, Any]:
    commit = run_command(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
    ).stdout.strip()
    status = run_command(
        ["git", "status", "--short"], cwd=PROJECT_ROOT
    ).stdout.strip()
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status,
    }


def hardware_metadata(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "requested_device": device,
        "packages": package_versions(),
    }
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    payload["nvidia_smi"] = (
        query.stdout.strip().splitlines() if query.returncode == 0 else []
    )
    return payload


def extract_official_metrics(
    run_source: Path,
    official_result: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    sys.path.insert(0, str(run_source / "code"))
    try:
        import pickle

        with official_result.open("rb") as stream:
            result = pickle.load(stream)
    finally:
        sys.path.pop(0)

    model_index = int(config["execution"]["pinn_ebm_model_index"])
    nrun = int(config["execution"]["nrun"])
    records = []
    for run_index in range(nrun):
        log_likelihood = float(result.logLebm_gesges[model_index, 1, run_index, -1])
        records.append(
            {
                "run_index": run_index,
                "lambda1": float(
                    result.dpargesges[model_index, 0, run_index, -1]
                ),
                "lambda2": float(
                    result.dpargesges[model_index, 1, run_index, -1]
                ),
                "validation_mean_absolute_error_named_rmse_upstream": float(
                    result.rmse_gesges[model_index, 1, run_index, -1]
                ),
                "validation_nll": -log_likelihood,
                "validation_pde_mean_squared_residual": float(
                    result.fl_gesges[model_index, 1, run_index, -1]
                ),
                "training_seconds": float(
                    result.tm_gesges[model_index, run_index]
                ),
                "ebm_initialization_seconds": float(
                    result.tebm_gesges[model_index, run_index]
                ),
            }
        )

    aggregate: dict[str, dict[str, float]] = {}
    for key in records[0]:
        if key == "run_index":
            continue
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std_sample": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return {
        "evidence_status": "complete_full_upstream_execution",
        "variant_id": config["variant_id"],
        "seed": config["execution"]["seed"],
        "nrun": nrun,
        "model_index": model_index,
        "records": records,
        "aggregate": aggregate,
        "metric_name_warning": (
            "Upstream get_test_error computes mean absolute error but stores it "
            "under rmse_gesges; this artifact uses an explicit non-RMSE name."
        ),
    }


def create_driver(seed: int) -> str:
    return textwrap.dedent(
        f"""\
        from pebm.utils import init_random_seeds

        rand_init, s = init_random_seeds(s={seed})
        input_path = "../results/reproduction/"
        with open("pebm.py", "rb") as stream:
            source = compile(stream.read(), "pebm.py", "exec")
        exec(source)
        """
    )


def run_full_experiment(
    config: dict[str, Any],
    config_path: Path,
    source_dir: Path,
    source_metadata: dict[str, Any],
    dataset_metadata: dict[str, Any],
    output_root: Path,
    device: str,
    run_id: str | None,
) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc)
    suffix = run_id or timestamp.strftime("%Y%m%dT%H%M%SZ")
    execution = config["execution"]
    run_dir = (
        output_root
        / "runs"
        / config["variant_id"]
        / f"seed{int(execution['seed']):03d}_nrun{int(execution['nrun'])}_{suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    archived_source = run_dir / "source"
    shutil.copytree(source_dir, archived_source, symlinks=True)
    archived_data = archived_source / "data"
    archived_data.mkdir(exist_ok=True)
    dataset_link = archived_data / config["dataset"]["filename"]
    dataset_link.symlink_to(Path(dataset_metadata["path"]))

    result_dir = archived_source / "results/reproduction"
    result_dir.mkdir(parents=True)
    (result_dir / "input.py").write_text(
        render_upstream_input(config),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    driver = archived_source / "code/codex_run_reproduction.py"
    driver.write_text(create_driver(int(execution["seed"])), encoding="utf-8")

    metadata = {
        "status": "running",
        "evidence_status": "full_run_pending",
        "variant_id": config["variant_id"],
        "started_at_utc": timestamp.isoformat(),
        "command": sys.argv,
        "current_repository": current_repository_metadata(),
        "official_source": source_metadata,
        "dataset": dataset_metadata,
        "hardware": hardware_metadata(device),
        "seed": execution["seed"],
        "nrun": execution["nrun"],
        "seed_semantics": execution["seed_semantics"],
        "official_source_modified_by_training": False,
        "source_semantics": audit_source_semantics(config, archived_source),
    }
    write_json(run_dir / "run_metadata.json", metadata)

    log_path = run_dir / "stdout_stderr.log"
    environment = device_environment(device)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, driver.name],
                cwd=archived_source / "code",
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Official training exited {completed.returncode}; see {log_path}"
            )

        official_result = (
            result_dir
            / f"x101n{execution['noise']}r{int(execution['nrun'])}.dat"
        )
        if not official_result.is_file():
            raise FileNotFoundError(f"Missing official result pickle {official_result}")
        metrics = extract_official_metrics(archived_source, official_result, config)
        metrics["official_result"] = {
            "path": str(official_result.relative_to(run_dir)),
            "sha256": sha256_file(official_result),
            "size_bytes": official_result.stat().st_size,
        }
        write_json(run_dir / "metrics.json", metrics)
        metadata.update(
            {
                "status": "complete",
                "evidence_status": "complete_full_upstream_execution",
                "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "official_result": metrics["official_result"],
                "checkpoint_limitation": (
                    "The official source stores result histories but no model "
                    "state_dict. No training hook was added because that would "
                    "modify the source execution path."
                ),
            }
        )
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "evidence_status": "not_evidence",
                "failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "error": repr(error),
            }
        )
        write_json(run_dir / "run_metadata.json", metadata)
        raise
    write_json(run_dir / "run_metadata.json", metadata)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run separately labeled active-upstream or paper-architecture "
            "PINN-EBM Navier--Stokes reproductions."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download the frozen Raissi dataset if --dataset is absent.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stage and audit source/data without starting training.",
    )
    action.add_argument(
        "--component-smoke",
        action="store_true",
        help="Run a non-evidentiary finite forward/backward component smoke.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="GPU index exposed to official code, or 'cpu'.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional caller-provided suffix; the run directory must not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    validate_config(config, config_path)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset.resolve()
    if args.download_dataset:
        dataset_metadata = download_dataset(dataset_path, config["dataset"])
    else:
        dataset_metadata = verify_dataset(dataset_path, config["dataset"])

    source_dir, source_metadata = stage_source(config, output_root)
    source_audit = audit_source_semantics(config, source_dir)
    audit_payload = {
        "variant_id": config["variant_id"],
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": source_metadata,
        "dataset": dataset_metadata,
        "source_semantics": source_audit,
        "rendered_input_sha256": sha256_text(render_upstream_input(config)),
    }
    audit_dir = output_root / "audits" / config["variant_id"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_dir / "audit.json", audit_payload)

    if args.prepare_only:
        print(json.dumps(audit_payload, indent=2, sort_keys=True))
        return
    if args.component_smoke:
        smoke = run_component_smoke(config, source_dir, output_root, args.device)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return

    run_dir = run_full_experiment(
        config,
        config_path,
        source_dir,
        source_metadata,
        dataset_metadata,
        output_root,
        args.device,
        args.run_id,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
