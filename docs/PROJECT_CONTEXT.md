# naPINN project context

Last reviewed: 2026-07-24

## Research objective

naPINN is a measurement-driven inverse physics-informed neural network for
recovering latent PDE solutions and unknown physical parameters from sparse
measurements corrupted by unknown non-Gaussian noise and gross outliers.

The paper's central claim is not simply that a robust loss helps. naPINN
learns a residual distribution, converts residual likelihood into
measurement-level reliability, and trains with a gate that downweights
unreliable observations. A rejection penalty discourages the degenerate
solution in which the gate rejects nearly every measurement.

## Method in one pass

1. A PINN maps `(x, y, t)` to one scalar field for Allen-Cahn or two fields
   `(u, v)` for Burgers and lambda-omega reaction diffusion.
2. Phase 1 performs 5,000 PINN warm-up steps using PDE residual and
   measurement data losses.
3. The current measurement residuals are normalized by a running standard
   deviation. The main paper estimator is a scalar EBM; GMM and KDE are
   estimator ablations.
4. Trainable estimators receive 5,000 estimator-only initialization updates.
   These updates are additional to, not part of, the 30,000 PINN-step budget.
5. Phase 2 performs 25,000 joint steps. Standardized estimator scores enter a
   trainable sigmoid gate. The objective combines PDE loss, gate-weighted data
   loss, and rejection cost.
6. For two-output PDEs, residual components are flattened and modeled as
   scalar samples.

The paper also evaluates estimator-free quantile and learnable
residual-threshold gates, robust data losses, B-PINN, alternative backbones,
staged training, running-standard-deviation normalization, rejection-cost
sensitivity, parameter recovery, and training cost.

## Benchmarks and evaluation

| Benchmark | State | Unknown parameter in code | Reference solution |
| --- | --- | --- | --- |
| 2D Allen-Cahn | `u(x,y,t)` | interface width `eps` | analytical field |
| 2D Burgers | `(u,v)` | viscosity `nu` | generated numerical simulation |
| 2D lambda-omega RD | `(u,v)` | reaction parameter `beta` | generated numerical simulation |

The main corruption uses a four-component Gaussian mixture scaled relative to
the solution magnitude, plus uniformly sampled high-magnitude outliers.
Reported outlier ratios are 5%, 10%, and 15%. Evaluation uses rMAE and rMSE on
a dense 120 by 120 spatial grid. Main-table values are reported over ten
seeds; some ablations use five seeds.

## Repository execution flow

```text
scripts/{mlp,bpinn}/*.sh
        |
        v
configs/common_config.yaml + model YAML + experiment YAML
        |
        v
pinnlab.train
        |
        +--> pinnlab.registry --> model + PDE experiment
        +--> phase 1 warm-up
        +--> estimator initialization
        +--> phase 2 joint training
        +--> checkpoints, metrics, videos
        |
        v
analysis/results/runs/<experiment>/<run>/
```

Post-training evaluation uses `python -m analysis.evaluate_checkpoint`.
Standalone figure scripts live under `analysis/plots/`. The current manuscript
is `paper/neurips_2026.tex`; its consumed figures live in `paper/figures/`.

## Important interpretation boundaries

- The reported setting is measurement-driven. The paper introduces the
  general PINN objective with PDE, boundary, initial, and data terms, but the
  current training implementation uses PDE plus measurement data losses.
- `data_loss_balancer` is a historical configuration name for per-measurement
  reliability gating. It is not the auxiliary cross-loss balancing method
  excluded by the paper.
- An experiment filename does not identify a paper condition. Always inspect
  the YAML fields and tag.
- The paper tables are embedded in TeX. This checkout currently has no single
  result aggregation script that regenerates every reported table.
- Lightweight compilation and smoke tests establish code health, not
  scientific reproduction.
