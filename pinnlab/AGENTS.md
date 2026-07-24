# Implementation guidance

Read the root `AGENTS.md` and `docs/CODE_PAPER_MAP.md` first.

## Architecture

- `train.py` owns phase orchestration, optimization, checkpointing, periodic
  evaluation, performance logging, and optional analysis hooks.
- `experiments/` owns PDE definitions, noisy measurement construction,
  residual/data losses, estimator initialization, gating, metrics, and
  experiment state.
- `utils/density.py`, `utils/ebm.py`, and `utils/kde.py` implement the common
  scalar residual-density interface.
- `data/noise.py` implements the supported Gaussian, four-component Gaussian
  mixture, Laplace, and Student-t corruption families.
- `models/` contains the shared MLP and B-PINN baseline.

## Stable experiment interface

Preserve these behaviors unless the requested change explicitly revises them:

- `sample_batch(n_f)` returns collocation and measurement batches.
- `pde_residual_loss`, `data_loss`, and `eval_on_grid` remain experiment
  entry points.
- `initialize_EBM` initializes whichever residual estimator is configured;
  KDE refreshes its buffer and has no trainable optimizer step.
- `extra_params`, `state_dict`, and `load_state_dict` include trainable PDE,
  estimator, and gate state needed by checkpoints.
- Burgers and lambda-omega residual components are flattened into scalar
  samples before density estimation and gating.

## Verification

For changes to a shared estimator, gate, noise model, or trainer, test all
three experiment classes and every affected YAML family. Use tiny synthetic
fixtures for smoke tests; never treat those fixtures as paper evidence.
