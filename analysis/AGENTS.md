# Analysis guidance

Read the root `AGENTS.md` and `analysis/README.md` first.

- Put reusable plotting code in `analysis/` and standalone paper-figure
  generators in `analysis/plots/`.
- Put small plotting inputs in `analysis/results/data/`, generated analysis
  figures in `analysis/results/figures/`, and run artifacts in the ignored
  `analysis/results/runs/`.
- Paper-ready figures consumed by LaTeX belong in `paper/figures/`; copying or
  promoting an analysis figure there must be an explicit, reviewed action.
- Make plotting scripts path-independent by resolving paths from `__file__`.
- Do not overwrite or relabel a reported figure without recording its source
  data and generation command.
