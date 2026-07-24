# Project progress

Last updated: 2026-07-24

## Completed

- Reviewed the current NeurIPS manuscript and mapped its method, experiments,
  ablations, metrics, and implementation claims to the codebase.
- Removed high-confidence dead or paper-irrelevant code, including unused
  dynamic loss balancing, early stopping, dormant offset/weight-network/NLL
  branches, unused noise families, duplicate training blocks, and broken
  simulation launchers.
- Simplified the residual estimators and fixed EBM NLL scaling, the
  three-component GMM default, and best-checkpoint restoration.
- Kept all three PDEs, MLP/B-PINN, MSE/L1/q-Gaussian, EBM/GMM/KDE,
  likelihood/quantile/threshold gates, evaluation, checkpointing, and
  supplementary visualization paths.
- Moved standalone plotting, result inputs, generated analysis figures,
  checkpoint evaluation, and run outputs under `analysis/`.
- Added durable repository guidance, code-paper mapping, rebuttal documents,
  and project-scoped rebuttal agents.

## Validation completed

- Python compilation passes for `pinnlab` and `analysis`.
- Every YAML file parses.
- Shell syntax passes for surviving launchers.
- Training and checkpoint-analysis CLI imports pass.
- Component smoke tests pass for supported noise families, EBM/GMM/KDE, and
  all three gate types.
- Tiny forward/backward smoke tests previously passed for every experiment
  YAML using synthetic fixtures.
- Project Codex TOML and all four custom-agent files parse; Codex Doctor loads
  the project configuration, and prompt inspection confirms root `AGENTS.md`
  is included in model-visible project guidance.
- `git diff --check` passes.

## Open blockers and risks

1. The generated Burgers and lambda-omega datasets referenced by current
   configs are absent:
   - `pinnlab/simulation/simulation_result/Burgers2D_3-5/data.npz`
   - `pinnlab/simulation/simulation_result/LambdaOmega_Spiral_2/data.npz`
2. Full 30,000-step experiments have not been rerun after cleanup.
3. There is no canonical immutable config/seed manifest connecting every
   reported table cell to a run directory.
4. Several current default-named YAMLs select ablations or baselines rather
   than the paper's main EBM-gated setup. See `docs/CODE_PAPER_MAP.md`.
5. Paper table values are embedded in TeX; a complete result aggregation and
   table-generation pipeline is not present.
6. The paper/config alignment still needs resolution for the EMA convention,
   the reported Allen-Cahn rejection cost, and likelihood-gate initializer
   fields that are present in YAML but not passed to the likelihood gate.
7. No reviewer comments or venue rebuttal word/character limit have been
   added yet.
8. The worktree contains the user's ongoing changes. The pre-existing deletion
   of `.claude/settings.json` was not created or restored during cleanup.
9. `latexmk` and `pdflatex` are unavailable in the current environment, so the
   manuscript has not been recompiled here.

## Recommended next actions

1. Add reviewer comments verbatim to `rebuttal/reviewer_comments.md` and record
   the venue's response limit.
2. Build a canonical reproduction manifest before running any new rebuttal
   experiments.
3. Generate or recover the missing simulation datasets and run targeted tests
   or experiments required by reviewer concerns.
4. Record each new artifact, command, seed set, and result location in
   `rebuttal/response_matrix.md`.
5. Compile the revised paper and perform a final claim/code/config audit before
   submitting the rebuttal.

## Update protocol

After substantive work, update this file with:

- the change and its motivation;
- exact validation performed;
- new evidence or artifact locations;
- unresolved blockers;
- any code-paper-config discrepancy introduced or resolved.
