#!/usr/bin/env python3
"""Hamiltonian Monte Carlo B-PINN baseline.

Reviewer 6XZg correctly observed that the submitted Bayesian baseline is a
mean-field variational B-PINN, not the HMC posterior sampler of the original
B-PINN work, and that comparing against the weaker approximation is not a fair
test. This runner supplies the HMC version.

The posterior is the one the original formulation uses for an inverse problem:

    U(theta, lambda) = ||u_theta(X_d) - y_d||^2 / (2 sigma_d^2)
                     + ||f_theta,lambda(X_c)||^2 / (2 sigma_f^2)
                     + ||theta||^2 / (2 sigma_p^2)
                     + lambda^2 / (2 sigma_l^2)

with the unknown PDE coefficient sampled jointly with the network weights.

Three properties matter for this to be an honest baseline rather than a
strawman:

* The collocation set and the measurement set are drawn once and held fixed.
  Resampling them each step would make the potential stochastic and silently
  break detailed balance, which is the most common way an HMC baseline is
  quietly made worse than it should be.
* The leapfrog step size is adapted by dual averaging during burn-in only, to
  a target acceptance rate, and then frozen. Adaptation never touches the
  retained samples.
* Every run reports acceptance rate, split R-hat and effective sample size.
  If those diagnostics fail, the error numbers must be reported as an
  inconclusive reproduction attempt rather than as HMC's performance.

The whole configuration is fixed by the CLI defaults below before any result
is inspected.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from pinnlab.registry import get_experiment, get_model
from pinnlab.utils.seed import seed_everything


TRUE_PARAMETER = {
    "allencahn2d": ("eps", 0.3),
    "burgers2d": ("nu", 0.01),
    "lambdaomega2d": ("beta", 1.0),
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def git_metadata() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--short")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status)}


class Potential:
    """Negative log posterior over network weights and the PDE coefficient."""

    def __init__(self, model, experiment, batch, args):
        self.model = model
        self.experiment = experiment
        self.batch = batch
        self.sigma_d = args.sigma_data
        self.sigma_f = args.sigma_pde
        self.sigma_p = args.sigma_prior
        self.sigma_l = args.sigma_parameter
        self.params = list(model.parameters()) + list(experiment.extra_params())

    def state(self) -> torch.Tensor:
        return parameters_to_vector(self.params).detach().clone()

    def load(self, vector: torch.Tensor) -> None:
        vector_to_parameters(vector, self.params)

    def energy_and_gradient(
        self, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.load(vector)
        for parameter in self.params:
            if parameter.grad is not None:
                parameter.grad = None
        pde = self.experiment.pde_residual_loss(self.model, self.batch).sum()
        residual = self.batch["y_d"] - self.model(self.batch["X_d"])
        data = residual.square().sum()
        weights = parameters_to_vector(self.model.parameters())
        prior = weights.square().sum()
        extra = self.experiment.extra_params()
        coefficient = (
            torch.stack([p.reshape(-1).square().sum() for p in extra]).sum()
            if extra
            else torch.zeros((), device=weights.device)
        )
        energy = (
            data / (2.0 * self.sigma_d**2)
            + pde / (2.0 * self.sigma_f**2)
            + prior / (2.0 * self.sigma_p**2)
            + coefficient / (2.0 * self.sigma_l**2)
        )
        gradient = torch.autograd.grad(energy, self.params, allow_unused=True)
        flat = torch.cat(
            [
                (
                    torch.zeros_like(parameter).reshape(-1)
                    if piece is None
                    else piece.reshape(-1)
                )
                for parameter, piece in zip(self.params, gradient)
            ]
        )
        return energy.detach(), flat.detach()


def leapfrog(potential, position, momentum, step_size, n_steps):
    energy, gradient = potential.energy_and_gradient(position)
    momentum = momentum - 0.5 * step_size * gradient
    for index in range(n_steps):
        position = position + step_size * momentum
        energy, gradient = potential.energy_and_gradient(position)
        if not torch.isfinite(energy) or not torch.isfinite(gradient).all():
            return position, momentum, torch.tensor(float("inf"))
        if index < n_steps - 1:
            momentum = momentum - step_size * gradient
    momentum = momentum - 0.5 * step_size * gradient
    return position, -momentum, energy


def run_chain(potential, position, args, generator, chain_index):
    """One HMC chain with dual-averaging step-size adaptation during burn-in."""
    step_size = args.step_size
    log_step = float(np.log(step_size))
    log_step_bar = 0.0
    h_bar = 0.0
    mu = float(np.log(10.0 * step_size))
    gamma, t0, kappa = 0.05, 10.0, 0.75

    accepted = 0
    samples: list[torch.Tensor] = []
    parameter_trace: list[float] = []
    total = args.burn_in + args.n_samples
    for iteration in range(total):
        momentum = torch.randn(
            position.shape, generator=generator, device=position.device
        )
        current_energy, _ = potential.energy_and_gradient(position)
        current_hamiltonian = current_energy + 0.5 * momentum.square().sum()
        proposal, proposal_momentum, proposal_energy = leapfrog(
            potential, position, momentum, step_size, args.n_leapfrog
        )
        proposal_hamiltonian = (
            proposal_energy + 0.5 * proposal_momentum.square().sum()
        )
        log_accept = float(current_hamiltonian - proposal_hamiltonian)
        accept_probability = float(np.exp(min(0.0, log_accept)))
        if not np.isfinite(accept_probability):
            accept_probability = 0.0
        if float(torch.rand((), generator=generator, device=position.device)) < (
            accept_probability
        ):
            position = proposal
            if iteration >= args.burn_in:
                accepted += 1

        if iteration < args.burn_in:
            # Dual averaging. Confined to burn-in so retained samples come
            # from a fixed-kernel chain.
            step = iteration + 1
            h_bar = (1.0 - 1.0 / (step + t0)) * h_bar + (
                args.target_accept - accept_probability
            ) / (step + t0)
            log_step = mu - np.sqrt(step) / gamma * h_bar
            eta = step ** (-kappa)
            log_step_bar = eta * log_step + (1.0 - eta) * log_step_bar
            step_size = float(np.exp(log_step))
            if iteration == args.burn_in - 1:
                step_size = float(np.exp(log_step_bar))
        else:
            if (iteration - args.burn_in) % args.thin == 0:
                samples.append(position.detach().clone())
            potential.load(position)
            extra = potential.experiment.extra_params()
            if extra:
                parameter_trace.append(float(extra[0].detach().reshape(-1)[0]))
        if (iteration + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "chain": chain_index,
                        "iteration": iteration + 1,
                        "step_size": step_size,
                        "accept_prob": accept_probability,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "samples": samples,
        "acceptance_rate": accepted / max(args.n_samples, 1),
        "final_step_size": step_size,
        "parameter_trace": parameter_trace,
    }


def split_r_hat(chains: list[np.ndarray]) -> float:
    """Split R-hat over scalar traces; >1.1 conventionally signals failure."""
    pieces = []
    for chain in chains:
        half = len(chain) // 2
        if half < 2:
            return float("nan")
        pieces.append(np.asarray(chain[:half], dtype=np.float64))
        pieces.append(np.asarray(chain[half : 2 * half], dtype=np.float64))
    m, n = len(pieces), len(pieces[0])
    means = np.array([piece.mean() for piece in pieces])
    variances = np.array([piece.var(ddof=1) for piece in pieces])
    w = variances.mean()
    b = n * means.var(ddof=1)
    if w <= 0:
        return float("nan")
    var_hat = (n - 1) / n * w + b / n
    return float(np.sqrt(var_hat / w))


def effective_sample_size(chain: np.ndarray) -> float:
    chain = np.asarray(chain, dtype=np.float64)
    n = len(chain)
    if n < 4:
        return float("nan")
    centered = chain - chain.mean()
    denominator = np.dot(centered, centered)
    if denominator <= 0:
        return float("nan")
    total = 0.0
    for lag in range(1, min(n - 1, 1000)):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        if rho < 0.05:
            break
        total += rho
    return float(n / (1.0 + 2.0 * total))


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HMC B-PINN requires CUDA.")
    common = load_yaml(args.common_config)
    model_config = load_yaml(args.model_config)
    experiment_config = load_yaml(args.experiment_config)
    experiment_config["device"] = args.device
    experiment_config["noise"]["extra_noise"]["n_points"] = args.outlier_points
    experiment_config["ebm"]["enabled"] = False
    experiment_config["data_loss_balancer"]["use_loss_balancer"] = False
    experiment_config["phase"]["enabled"] = False
    model_config["in_features"] = experiment_config["in_features"]
    model_config["out_features"] = experiment_config["out_features"]

    output_dir = (
        args.output_root
        / args.experiment_name
        / f"outliers_{args.outlier_points}"
        / f"seed_{args.seed}"
    )
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"Completed output already exists: {metrics_path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    experiment = get_experiment(args.experiment_name)(experiment_config, device)
    seed_everything(args.seed)
    model = get_model(args.model_name)(model_config).to(device)

    # Fixed design: HMC requires a deterministic potential, so the collocation
    # points and the measurement set are drawn once here and never resampled.
    fixed = experiment.sample_batch(n_f=args.n_collocation)
    fixed["X_d"] = experiment.X_data
    fixed["y_d"] = experiment.y_data

    potential = Potential(model, experiment, fixed, args)
    start = time.perf_counter()
    chains = []
    for chain_index in range(args.n_chains):
        seed_everything(args.seed * 1000 + chain_index)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed * 1000 + chain_index)
        # Each chain starts from an independently initialised network, which
        # is what makes the between-chain R-hat meaningful.
        fresh = get_model(args.model_name)(model_config).to(device)
        model.load_state_dict(fresh.state_dict())
        with torch.no_grad():
            for parameter in experiment.extra_params():
                parameter.fill_(float(experiment_config["pde"]["init_eps"]))
        chains.append(
            run_chain(potential, potential.state(), args, generator, chain_index)
        )
    sampling_seconds = time.perf_counter() - start

    traces = [np.asarray(chain["parameter_trace"]) for chain in chains]
    r_hat = split_r_hat(traces) if len(traces) > 1 else float("nan")
    ess = float(np.mean([effective_sample_size(trace) for trace in traces]))
    acceptance = float(np.mean([chain["acceptance_rate"] for chain in chains]))

    # Posterior predictive mean over all retained samples from all chains.
    grid = common["eval"]["grid"]
    accumulated_rmae, accumulated_rmse, parameter_values = [], [], []
    all_samples = [s for chain in chains for s in chain["samples"]]
    if not all_samples:
        raise RuntimeError("No retained posterior samples.")
    model.eval()
    with torch.no_grad():
        for sample in all_samples:
            potential.load(sample)
            evaluation = experiment.eval_on_grid(model, grid)
            accumulated_rmae.append(float(evaluation["rMAE"]))
            accumulated_rmse.append(float(evaluation["rMSE"]))
            extra = experiment.extra_params()
            if extra:
                parameter_values.append(float(extra[0].reshape(-1)[0]))

    parameter_name, true_value = TRUE_PARAMETER[args.experiment_name]
    learned = float(np.mean(parameter_values)) if parameter_values else float("nan")
    diagnostics_pass = bool(
        acceptance >= args.min_acceptance
        and (not np.isfinite(r_hat) or r_hat <= args.max_r_hat)
    )
    result = {
        "status": "complete",
        "method": "bpinn_hmc",
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "outlier_points": args.outlier_points,
        "field_rMAE_posterior_mean_of_draws": float(np.mean(accumulated_rmae)),
        "field_rMSE_posterior_mean_of_draws": float(np.mean(accumulated_rmse)),
        "field_rMAE_best_draw": float(np.min(accumulated_rmae)),
        "pde_parameter_name": parameter_name,
        "pde_parameter_true": true_value,
        "pde_parameter_posterior_mean": learned,
        "pde_parameter_absolute_error": abs(learned - true_value),
        "diagnostics": {
            "acceptance_rate": acceptance,
            "split_r_hat_on_pde_parameter": r_hat,
            "effective_sample_size_on_pde_parameter": ess,
            "n_chains": args.n_chains,
            "burn_in": args.burn_in,
            "n_samples": args.n_samples,
            "n_leapfrog": args.n_leapfrog,
            "thin": args.thin,
            "final_step_sizes": [c["final_step_size"] for c in chains],
            "passes_predeclared_thresholds": diagnostics_pass,
            "threshold_min_acceptance": args.min_acceptance,
            "threshold_max_r_hat": args.max_r_hat,
        },
        "claim_boundary": (
            "Report as an HMC B-PINN reproduction under a fixed, predeclared "
            "sampling budget. If passes_predeclared_thresholds is false, the "
            "error values must be reported as an inconclusive reproduction "
            "attempt and must not be presented as the performance of HMC "
            "B-PINN."
        ),
        "gradient_evaluations": args.n_chains
        * (args.burn_in + args.n_samples)
        * (args.n_leapfrog + 1),
        "sampling_seconds": sampling_seconds,
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_properties(device).name,
        },
        "git": git_metadata(),
    }
    metrics_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name", default="allencahn2d", choices=sorted(TRUE_PARAMETER)
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--outlier-points", type=int, default=3375)
    parser.add_argument("--n-collocation", type=int, default=5000)
    # Posterior definition.
    parser.add_argument("--sigma-data", type=float, default=0.1)
    parser.add_argument("--sigma-pde", type=float, default=0.1)
    parser.add_argument("--sigma-prior", type=float, default=1.0)
    parser.add_argument("--sigma-parameter", type=float, default=1.0)
    # Sampler budget, frozen before any result is inspected.
    parser.add_argument("--n-chains", type=int, default=2)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-leapfrog", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=1.0e-4)
    parser.add_argument("--target-accept", type=float, default=0.7)
    parser.add_argument("--thin", type=int, default=10)
    # Predeclared acceptance criteria for calling the reproduction usable.
    parser.add_argument("--min-acceptance", type=float, default=0.4)
    parser.add_argument("--max-r-hat", type=float, default=1.1)
    parser.add_argument(
        "--common-config", type=Path, default=Path("configs/rebuttal/common.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/rebuttal/model_mlp.yaml")
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/rebuttal/allencahn2d.yaml"),
    )
    parser.add_argument("--model-name", default="mlp")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rebuttal/hmc_bpinn_20260728"),
    )
    main(parser.parse_args())
