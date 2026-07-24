# Analysis

This directory contains code and artifacts used after or alongside training:

- `evaluate_checkpoint.py` reloads a saved run for metrics, gate analysis, and
  video generation.
- `plotting.py` contains reusable plotting helpers.
- `plots/` contains standalone paper-figure scripts.
- `results/data/` contains the small tabular inputs used by those scripts.
- `results/figures/` contains generated paper-figure artifacts.
- `results/runs/` is the ignored destination for training checkpoints, logs,
  evaluations, and videos.

Run commands from the repository root:

```bash
python -m analysis.plots.plot_sigmoid
bash analysis/scripts/evaluate_checkpoint.sh
```

Training writes new runs under `analysis/results/runs/`, as configured in
`configs/common_config.yaml`.
