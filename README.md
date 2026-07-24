# naPINN

Implementation of the noise-adaptive physics-informed neural network described
in `paper/neurips_2026.tex`. The repository contains the three paper benchmarks
(2D Burgers, 2D Allen–Cahn, and 2D lambda–omega reaction diffusion), MLP and
Bayesian-PINN models, and the EBM/GMM/KDE residual-density variants.

Future Codex sessions should begin with [AGENTS.md](AGENTS.md), then read
[project context](docs/PROJECT_CONTEXT.md), the
[code-to-paper map](docs/CODE_PAPER_MAP.md), and
[current progress](docs/PROGRESS.md).

## Layout

```text
analysis/                  post-training evaluation and paper plots
  evaluate_checkpoint.py   reload a run and produce metrics/videos
  plots/                   standalone paper-figure scripts
  results/
    data/                  small inputs used by plotting scripts
    figures/               generated figure artifacts
    runs/                  ignored training outputs
configs/                   common, model, and experiment YAML files
paper/                     NeurIPS paper source
pinnlab/
  data/                    geometry and corruption models
  experiments/             PDE definitions and training objectives
  models/                  MLP and Bayesian PINN
  simulation/              reference-solution generators
  utils/                   density estimators, losses, and logging
  train.py                 training entry point
scripts/                   experiment and simulation launchers
```

## Setup

```bash
pip install -r requirements.txt
```

Generate the required reference data when needed:

```bash
bash scripts/simulation/burgers_3.sh
bash scripts/simulation/lambdaomega.sh
```

## Training

The launchers under `scripts/mlp/` and `scripts/bpinn/` select the corresponding
model and experiment configurations. For example:

```bash
bash scripts/mlp/allencahn2d.sh
```

Runs are written to `analysis/results/runs/`. W&B logging is controlled by
`configs/common_config.yaml`.

## Analysis

Reload a checkpoint for evaluation or video generation:

```bash
bash analysis/scripts/evaluate_checkpoint.sh
```

Generate a standalone paper figure:

```bash
python -m analysis.plots.plot_sigmoid
```

See `analysis/README.md` for the analysis directory conventions.

## Rebuttal workflow

Reviewer intake, evidence tracking, and response drafting live under
`rebuttal/`. The operating procedure is documented in
`docs/REBUTTAL_PLAYBOOK.md`, with project-scoped Codex roles configured under
`.codex/agents/`.
