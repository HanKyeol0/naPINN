# naPINN repository guidance

## Read first

Before changing code, experiments, results, or the paper, read:

1. `docs/PROJECT_CONTEXT.md` for the scientific and repository overview.
2. `docs/CODE_PAPER_MAP.md` for claim-to-implementation links and known caveats.
3. `docs/PROGRESS.md` for completed work, validation status, and blockers.

For rebuttal work, also read `docs/REBUTTAL_PLAYBOOK.md` and everything under
`rebuttal/`.

## Sources of truth

- `paper/neurips_2026.tex` is the current paper. `paper/old.tex` and
  `paper/old_icml2026_ver.tex` are historical references only.
- `pinnlab/` is the implementation; `configs/` contains individual run
  recipes.
- The current experiment YAML filenames are not a canonical reproduction
  manifest. Inspect their actual `tag`, `noise`, `phase`, `ebm`,
  `data_loss`, and `data_loss_balancer` fields before drawing conclusions.
- `paper/figures/` contains figures consumed by the paper.
  `analysis/results/figures/` contains analysis artifacts, and
  `analysis/results/runs/` is the ignored run-output directory.
- Numerical claims in the paper are authoritative only as reported claims;
  do not imply that they were regenerated in the current checkout unless
  the corresponding run artifacts and aggregation evidence are present.

## Integrity rules

- Never invent results, reviewer concerns, citations, run histories, or
  experimental evidence.
- Keep code, configuration, equations, captions, tables, and rebuttal claims
  consistent. When they disagree, surface the discrepancy before editing.
- Do not silently replace paper results with smoke-test output.
- Preserve unrelated and pre-existing worktree changes. Do not reset or
  discard a dirty worktree.
- Keep training logic in `pinnlab/`, post-training analysis in `analysis/`,
  paper sources in `paper/`, and rebuttal working documents in `rebuttal/`.
- Update `docs/PROGRESS.md` after a substantive verified change. Record what
  changed, what was validated, and what remains blocked.

## Implementation expectations

- Prefer the smallest change that addresses the stated scientific or
  reviewer concern.
- Preserve the common scalar residual-estimator interface: multi-output
  residuals are flattened before EBM/GMM/KDE scoring.
- Preserve the measurement-driven training objective unless a paper or
  reviewer change explicitly requires boundary/initial-condition terms.
- Treat the 5,000-step estimator initialization as estimator-only work. The
  paper's 30,000 PINN-step budget is 5,000 warm-up plus 25,000 joint steps.
- Do not add auxiliary dynamic PDE/data loss balancing; the paper explicitly
  excludes it from the reported comparison.

## Validation

Run checks proportional to the change. The default lightweight suite is:

```bash
python -m compileall -f -q pinnlab analysis
python -m pinnlab.train --help
python -m analysis.evaluate_checkpoint --help
python - <<'PY'
from pathlib import Path
import yaml
for path in Path("configs").rglob("*.yaml"):
    yaml.safe_load(path.read_text())
print("YAML: OK")
PY
for script in $(find scripts analysis/scripts -name '*.sh' -type f); do
  bash -n "$script"
done
git diff --check
```

For mathematical or training changes, also run focused forward/backward smoke
tests for every affected experiment configuration. Do not claim full
reproduction unless the generated Burgers and lambda-omega datasets exist and
the requested full runs completed.

For paper changes, compile from `paper/` when a TeX engine is available and
inspect warnings, references, tables, and affected figures.

## Rebuttal orchestration

For a rebuttal request with multiple independent tracks, use the project
subagents in `.codex/agents/`:

- `rebuttal_strategist` triages comments and defines the evidence plan.
- `rebuttal_code` implements and validates code/config changes.
- `rebuttal_paper` edits the manuscript from verified evidence.
- `rebuttal_writer` produces the final point-by-point response after the
  implementation and paper edits are verified.

The primary agent owns `rebuttal/response_matrix.md` and
`docs/PROGRESS.md` to avoid concurrent edit conflicts. Run code and paper work
in parallel only when their file scopes and evidence dependencies are
independent. Strategy comes before implementation; final response writing
comes after verification.
