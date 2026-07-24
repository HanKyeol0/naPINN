# Paper guidance

Read the root `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, and
`docs/CODE_PAPER_MAP.md` before editing.

- Edit `neurips_2026.tex`. Treat `old.tex` and `old_icml2026_ver.tex` as
  historical references, not alternate sources of truth.
- Do not change a numerical result, uncertainty, sample count, training
  budget, hardware claim, or comparison unless its evidence is identified in
  `rebuttal/response_matrix.md` or supplied run artifacts.
- Never add a citation from memory. Verify the bibliographic entry and that
  the cited work supports the sentence.
- Keep the method convention consistent: larger residual unreliability scores
  reduce gate weights; the implementation may equivalently operate on
  standardized log density with the sign reversed.
- Distinguish the 30,000 PINN optimization steps from the additional 5,000
  estimator-only initialization steps.
- Keep the general PINN formulation with BC/IC distinct from the reported
  measurement-driven implementation, which optimizes PDE and data terms.
- Check every changed label, reference, table value, caption, and figure path.
- Compile the paper when a TeX engine is available; otherwise report that
  limitation explicitly.
