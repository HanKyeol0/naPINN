# Experiment outputs

`outputs/` is the canonical destination for every new training run. Large
checkpoints and generated measurements are intentionally excluded from Git, but
the directory structure and the metadata stored inside each run make results
auditable on the experiment server.

## Directory layout

```text
outputs/
├── training/                         # General `python -m pinnlab.train` runs
├── rebuttal/
│   ├── synthetic/                   # Synthetic rebuttal runs
│   ├── realpde/                     # RealPDEBench/PIV runs
│   ├── realpde_legacy_4g/           # Legacy-4G exact-center runs
│   ├── realpde_legacy_4g_scale/     # Legacy-4G development grid
│   ├── realpde_legacy_4g_candidates/# Fresh-seed candidate confirmation
│   ├── natural_piv/                 # naPINN-MSE/L1/q2.9 on unmodified PIV
│   ├── natural_piv_baselines/       # MSE/LAD/OrPINN and direct-NLL weights
│   ├── synthetic_background_only/   # Four-Gaussian noise with zero gross rows
│   └── pinn_ebm_upstream/           # Official-source and paper-architecture runs
├── audits/                          # Read-only data/protocol audit artifacts
└── status/                           # Queue status and per-job logs
```

The exact method, condition, and seed appear below these roots. Rebuttal run
directories contain the following files when applicable:

- `config.yaml`: fully resolved configuration and effective schedule.
- `run_metadata.json`: command, Git state, input checksum, software, hardware,
  timestamps, and completion status.
- `train_history.jsonl`: step-level training losses for real-PIV runs.
- `step_*.pt`: periodic model/estimator checkpoints.
- `final.pt`: final model, experiment state, estimator, and gate state.
- `metrics.json`: final evaluation metrics and evidence status.
- `mad_screening.npz`: MAD screening details for two-stage MAD-PINN runs.
- `aggregation.json` and `aggregation.csv`: complete-only multi-seed summaries
  at a campaign root after strict validation.

The source-faithful PINN-EBM runner is a special case. The unmodified official
code saves its result-history pickle but does not save a PyTorch `state_dict`.
Its run directory therefore stores the archived source, frozen config,
dataset/source audit, complete stdout/stderr, official result pickle,
`metrics.json`, and `run_metadata.json`. Adding a checkpoint hook would change
the source execution path, so no model checkpoint is claimed for this exact
source run.

Queue launchers pass their `--output-root` explicitly to child processes, so
completion checks and newly written artifacts use the same root. A direct
real-PIV run defaults to `outputs/rebuttal/realpde`; a direct synthetic rebuttal
run defaults to `outputs/rebuttal/synthetic`.

## Historical artifacts

Completed experiments produced before this policy change remain under
`analysis/results/runs/`. They are not moved because their resolved configs,
checksums, aggregation files, and reports record those paths. Moving them would
break the existing evidence chain. All fresh runs use `outputs/`.
