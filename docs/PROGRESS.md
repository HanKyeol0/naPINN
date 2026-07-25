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
- Ingested the NeurIPS 2026 meta-review and three reviewer reports verbatim,
  recorded the official response constraints, and triaged every concern in
  `rebuttal/response_matrix.md`.
- A manuscript revision was drafted during initial rebuttal work, but the
  author subsequently rolled back `paper/neurips_2026.tex` and explicitly
  prohibited further paper-source edits. All remaining positioning,
  clarification, and additional evidence must therefore stay in rebuttal
  working documents rather than the TeX source.
- Audited independently verifiable manuscript issues for reviewer
  communication without editing the rolled-back TeX: the implemented EMA
  convention uses update weight \(0.05\) (old-state decay \(0.95\)); training
  uses 30,000 PINN updates plus 5,000 estimator-only updates; the displayed
  timing values imply 11.5%, 13.0%, and 3.1% overheads; and the Baydin
  publication year is 2018. These are response findings, not current-paper
  corrections.
- Audited two real RealPDEBench Cylinder PIV trajectory shards for the planned
  rebuttal experiment. The primary audited trajectory is `10031.h5`; a second
  trajectory, `5062.h5`, is available for replication after the primary
  implementation passes its smoke test.
- Added an auditable RealPDEBench Cylinder preparation pipeline, a
  pressure-latent \(u,v,p\) incompressible Navier--Stokes experiment, canonical
  MSE/LAD/PINN-EBM/naPINN configurations and runner, and a result aggregator.
  The PINN-EBM path applies the learned EBM NLL directly and backpropagates
  once through both the PINN and EBM without a reliability gate.
- Added two explicitly separated PINN-EBM variants after auditing the primary
  source: an equal-weight no-gate ablation and a closest-prior configuration
  that switches the Navier--Stokes PDE weight from \(\omega=1\) before EBM
  initialization to \(\omega=50\) during joint training, as reported by Pilar
  and Wahlström.
- Completed three paired full real-PIV runs for MSE-PINN and LAD-PINN. LAD
  has lower discrepancy against held-out PIV measurements, while MSE has
  substantially lower nominal momentum and continuity residuals; exact
  aggregates and artifact paths are recorded in
  `rebuttal/response_matrix.md`. Held-out real PIV is an independent
  measurement set, not noise-free ground truth.
- Added a deterministic structured sensor-failure generator over the same real
  PIV split. It corrupts 19/192 fixed training sensor identities across every
  frame with componentwise persistent bias plus linear drift, stores complete
  labels/provenance, and leaves held-out PIV and all clean training sensors
  bitwise unchanged.
- Added a faithful Peng et al. two-stage MAD-PINN path: a completed
  30,000-update LAD checkpoint supplies the published
  \(\operatorname{median}(|r|)/1.6777\), \(k=3\) scalar screen; an additional
  30,000-update MSE stage is initialized from that checkpoint and trains on
  retained components. Its 60,000-update pipeline and non-compute-matched
  status are recorded explicitly.

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
- The manuscript source passes brace/environment/label/citation-key checks;
  all 51 cited keys resolve among 64 unique bibliography entries.
- The corrected timing percentages were recomputed directly from the displayed
  per-iteration values.
- The RealPDEBench experiment passed an analytic PDE derivative check
  (maximum absolute residual \(9.54\times10^{-7}\)), direct-NLL gradient
  checks for both PINN and EBM, split-leakage checks, and finite five-step and
  200-step smoke tests. Smoke artifacts are explicitly marked as
  non-evidentiary.
- Three full MSE and three full LAD unmodified real-PIV seeds are complete and
  aggregated as historical diagnostics, but the author has excluded all
  natural/unmodified-PIV results from rebuttal reporting. Only controlled
  injected-corruption real-PIV results are eligible for the new evidence
  tables.
  The original naPINN rejection cost \(0.5\) produced an all-accept gate in its
  completed seed and is preserved as a negative sensitivity result. Full
  scale-calibrated naPINN runs at \(0.005\), sensitivity runs at \(0.01\), and
  all targeted structured comparisons are complete.
- Three paired full closest-prior PINN-EBM seeds are complete using the
  primary-source-audited \(\omega=50\) joint Navier--Stokes weight. The exact
  aggregate is recorded in `rebuttal/response_matrix.md`; it is materially
  worse in held-out field error than the completed calibrated naPINN seeds.
- Three paired full naPINN real-PIV seeds at rejection cost \(0.005\) are
  complete. They improve held-out-PIV discrepancy and nominal-physics
  residuals over the closest PINN-EBM baseline, but do not beat LAD on
  held-out-PIV discrepancy and show substantial gate-retention variability
  across seeds.
- The three-seed \(0.01\) sensitivity is complete and selected for the
  unmodified real-PIV comparison: it has similar mean held-out-PIV discrepancy
  to \(0.005\) but much lower between-seed variance in field, physics, and
  retention metrics. The original \(0.5\) all-accept result and the unstable
  \(0.005\) retention sweep remain part of the evidence record.
- The structured sensor-failure artifact passed checksum, split, label,
  bitwise-invariance, loader, finite-gradient, and gate-metric smoke checks.
  Its smoke metrics are explicitly non-evidentiary; paired three-seed MSE,
  LAD, closest-prior PINN-EBM, primary naPINN, and naPINN sensitivity runs
  are complete and aggregated.
- The MAD-PINN implementation passed stage-one checkpoint/config/seed
  validation, exact screening-formula and per-component-mask checks, a finite
  stage-two smoke, CLI/YAML checks, and provenance recording. Three paired
  full natural-PIV runs are complete: rMAE
  \(0.14810\pm0.00103\), rMSE \(0.22537\pm0.00095\), momentum RMS
  \(0.01399\pm0.00019\), and continuity RMS
  \(0.01349\pm0.00125\). The recorded two-stage pipeline uses 60,000 PINN
  updates and remains explicitly non-compute-matched.
- Three paired full MSE and primary naPINN structured sensor-failure runs are
  complete. At rejection cost \(0.005\), naPINN reduces held-out rMAE by
  13.9% and nominal momentum-residual RMS by 91.2% relative to MSE-PINN, and
  ranks the persistent failed measurements with AUROC
  \(0.96234\pm0.00086\). It also rejects
  \(47.51\%\pm10.63\%\) of known-clean measurements, an important negative
  selectivity result that is recorded alongside the positive detection
  metrics. Three paired LAD runs are also complete and give the best
  held-out-PIV agreement in this condition, while naPINN gives the smallest
  nominal-physics residuals and explicit failure scores. A separate
  three-seed rejection-cost-\(0.01\) sensitivity is also complete: it lowers
  clean-scalar rejection to \(33.68\%\pm7.62\%\) and improves held-out field
  discrepancy relative to \(0.005\), while retaining 98.86% of failed-scalar
  rejections and accepting larger nominal-physics residuals. It remains
  labeled as sensitivity rather than replacing the manifest-recorded primary
  condition.
- The three-seed structured closest-prior PINN-EBM/no-gate aggregate is also
  complete and shows high variability
  (rMAE \(0.26545\pm0.08838\)). The frozen RealPDEBench aggregate is
  `analysis/results/runs/rebuttal_realpde/aggregation.json`; the canonical
  protocol and checksums are recorded in
  `rebuttal/realpde_reproduction_manifest.yaml`.
- Archived the earlier natural-PIV response draft after the author excluded
  unmodified-PIV results. The current `rebuttal/response_draft.md` is marked
  “do not submit yet” and contains explicit placeholders for the active
  three-seed synthetic and injected-PIV aggregates.
- Replaced `rebuttal/rebuttal_report_ko.md` with a Korean internal report that
  excludes natural-PIV evidence, translates and answers every reviewer,
  records the verified 10% injected sensor-drift sensitivity, includes the
  complete three-seed severe synthetic table, and preserves nuance-bearing
  terminology in English. Per user instruction,
  `paper/neurips_2026.tex` was not changed and must remain untouched during
  author response.
- Generated the previously missing Burgers and lambda--omega numerical
  datasets without visualization rendering. Added immutable rebuttal
  experiment configs under `configs/rebuttal/`, a frozen matrix in
  `rebuttal/experiment_plan.yaml`, and a CUDA-only synthetic runner that
  records field error, PDE-parameter error, gate diagnostics, exact phase
  timings, update counts, peak memory, config, seed, and Git provenance.
- The synthetic runner passed finite CUDA smoke tests for MSE, naPINN, and
  direct-NLL PINN-EBM on all three PDEs. The direct baseline backpropagates
  EBM NLL through the PINN; naPINN instead fits the estimator on detached
  residuals and sends gate-weighted MSE to the PINN, matching the distinct
  optimization roles in the submitted implementation.
- Froze and started the three-seed synthetic core queue on CUDA devices 0--6.
  Device 0 later dropped to roughly 1.4 GiB free when another shared-server
  job allocated memory, so its first attempt was preserved as incomplete.
  The missing shard was restarted only after device 0 returned to roughly
  81 GiB free. Shared workloads later changed free memory again; no new
  independent queue is launched on a device below the 4 GiB threshold.
- Added labeled real-PIV corruption artifacts for 20% and 30% persistent
  sensor bias plus drift, 15% temporally correlated AR(1) drift, and a
  spatially correlated two-burst failure. Held-out PIV values remain bitwise
  unchanged. Added an OrPINN q=2.9 path to the real-PIV runner and passed a
  finite CUDA schema smoke test on the AR(1) artifact.
- Added post-training raw-EBM surprise metrics for both PINN-EBM and naPINN,
  plus a checkpoint-hashed backfill path for older completed staged runs.
  This prevents the trainable gate from receiving an asymmetric detection
  metric when the no-gate EBM can also rank corruptions.
- Added a separately tagged inverse-Re extension for 30% injected sensor
  drift. It initializes positive log-Re at 8000 and reports recovery relative
  to metadata Re 10031, with an explicit caveat that nominal 2D model
  discrepancy can make the learned value an effective rather than literal
  physical coefficient.
- Added a faithful synthetic two-stage MAD-PINN runner and synthetic result
  aggregator. MAD stage two is intentionally recorded as additional work
  (30,000 LAD plus 30,000 retained-scalar MSE updates), not compute matched.
- Completed all 54 severe 15%-four-Gaussian core cells (three PDEs, six
  methods, seeds 40--42) and aggregated them. Direct PINN-EBM has lower field
  error than naPINN on all three PDEs and lower Burgers/lambda--omega
  parameter error; its raw EBM score also has slightly higher AUROC. The
  Korean report and response draft explicitly retain this adverse
  closest-prior result.
- Completed all five seed-39 real-PIV rejection-cost calibration cells.
  Cost 0.10 is the only value satisfying failed rejection at least 90% and
  clean rejection at most 40% (93.79% and 18.56%). The reproducible selection
  record is
  `analysis/results/runs/rebuttal_realpde/piv_rejection_calibration_selection.json`;
  selected-cost held-out runs are active on seeds 40--42. The report also
  records that earlier 10% cost-0.01 held-out cells predated the 0.05/0.10
  grid refinement, so calibration is not called fully held-out-blind.
- A newly added post-training selector diagnostic initially passed an
  \(N\times2\) lambda--omega residual directly to the scalar EBM after
  training, causing two 5% naPINN cells to lack final metrics. The diagnostic
  now flattens residuals to the common scalar interface and passed CUDA
  naPINN/fixed-quantile/learnable-threshold smoke checks. The original
  artifacts are preserved below `rebuttal_synthetic_incomplete`; they are
  excluded from aggregation and scheduled for clean reruns.
- An attempted live repartition of the two-shard PIV baseline queue terminated
  two incomplete seed-41 children through terminal hangup. Neither produced
  metrics. Their configs, histories, and checkpoints are preserved below
  `rebuttal_realpde_incomplete/queue_repartition_terminal_hangup`, and a clean
  five-shard queue now reruns those cells without treating partial checkpoints
  as evidence.
- Completed the nine-cell severe synthetic MAD-PINN matrix. Three-seed rMAE
  is 0.32482, 0.24696, and 0.13008 for Allen--Cahn, Burgers, and
  lambda--omega. The published screen rejects 98.4--98.6% of known outliers
  and 10.8--17.8% of clean scalars. Its 60,000-update field error is lower
  than LAD but higher than both direct PINN-EBM and naPINN on every PDE.
- Extended the injected-PIV aggregator to retain warm-up, estimator-only,
  joint-training, and evaluation phase times together with metadata/effective
  Reynolds numbers and relative recovery error. Compilation, partial
  aggregation, `git diff --check`, and the frozen-paper check pass.
- Completed all three selected-cost naPINN held-out seeds for the 30%
  persistent injected-PIV failure. The aggregate is rMAE
  \(0.21194\pm0.01783\), rMSE \(0.30034\pm0.02341\), failure AUROC
  \(0.93445\pm0.01306\), failed-scalar rejection
  \(93.22\%\pm3.26\%\), and clean-scalar rejection
  \(28.64\%\pm13.07\%\). Matched baseline seeds are still active, so no final
  method ranking is recorded yet. Two inverse-Re shards have started with
  the frozen initial/reference values 8000/10031 and selected rejection cost
  0.10.
- Completed the matched 30% injected-PIV MSE, LAD, and closest-prior
  PINN-EBM groups. Relative to MSE/LAD, selected-cost naPINN lowers rMAE by
  42.9%/10.0% and rMSE by 38.8%/16.2%. Direct PINN-EBM has 4.1% lower rMAE
  than naPINN, whereas naPINN has 3.3% lower rMSE and substantially lower
  nominal momentum/continuity residuals. Direct raw-EBM AUROC is higher than
  gate AUROC. The matched OrPINN group is also complete
  (rMAE \(0.29745\pm0.00073\), rMSE
  \(0.41328\pm0.00164\)), so the full severe real-PIV primary table is
  verified and recorded in the Korean report and response draft.
- Completed all 54 five-percent four-Gaussian synthetic core cells after
  clean recovery of four missing artifacts. Direct PINN-EBM has the lowest
  field error on all three PDEs; naPINN is second on all three. PDE-parameter
  error rankings are mixed across methods. The refreshed synthetic aggregate
  contains 143 completed runs in 56 groups; the 10% core remains active.
- Started all seven shards of the frozen synthetic supplement matrix after
  rechecking per-device free memory. This matrix covers EMA update weight,
  rejection cost, Gaussian/Laplace/Student-t background noise, naPINN data
  loss variants, fixed-quantile selection, and the learnable residual
  threshold ablation. These were separate from the then-active 10% core.
- Completed the three-seed 30% injected-PIV MAD-PINN comparison. Its
  60,000-update rMAE/rMSE is
  \(0.25127\pm0.00817/0.37205\pm0.00738\), worse than both its LAD
  stage-one reference and naPINN. The screen rejects
  \(80.45\%\pm1.66\%\) of failed and \(20.30\%\pm0.54\%\) of clean
  scalars. The Korean report and response draft include the adverse result,
  exact compute status, and shared-GPU timing caveat.
- Completed the requested three-seed EMA sensitivity on severe Allen--Cahn.
  For implementation update weights \(m=0.01/0.05/0.10\), rMAE is
  \(0.11397/0.10060/0.10898\) and rMSE is
  \(0.11187/0.10038/0.10613\). Variability overlaps, so the report states
  modest tested-range sensitivity rather than universal optimality and
  explicitly maps decay to \(\rho=1-m\).
- Completed the remaining two 10%-four-Gaussian recovery cells. The full
  synthetic core is now complete at every 5%/10%/15% ratio: 162 held-out
  runs across three PDEs, six methods, and seeds 40--42. Direct PINN-EBM has
  the lowest field rMAE in all nine PDE/ratio conditions and naPINN is second
  in all nine; parameter-error rankings are mixed. The Korean report now
  records the complete field and parameter tables.
- Fixed a synthetic aggregation integrity issue exposed when seed-39
  closest-prior weight-calibration artifacts entered the same directory as
  held-out runs. When `--required-seeds 40 41 42` is used, the aggregator now
  excludes all other seeds and records their source paths in explicit
  metadata. Re-aggregation verifies 54/54 complete held-out core groups
  without calibration-seed contamination.
- Completed the five-value Allen--Cahn severe rejection-cost sensitivity.
  Costs 0.10/0.30/0.50/0.70 give rMAE
  0.09814/0.11383/0.10060/0.09564, whereas the submitted-appendix value
  1.00 degrades to 0.87859 and rejects only 2.21% of known outliers. The
  Korean report and response draft disclose the adverse paper-aligned result
  and keep the initial 0.5 core unchanged.
- Completed all 12 held-out inverse-Re PIV runs. Relative to metadata
  Re 10031, learned Re is \(1309\pm155\) for MSE,
  \(4133\pm506\) for LAD, \(4651\pm2561\) for direct PINN-EBM, and
  \(2284\pm533\) for naPINN. naPINN has the lowest nominal-physics residual
  but worse coefficient recovery than LAD/direct. The report treats this as
  an adverse identifiability/model-discrepancy result rather than successful
  parameter recovery.
- Completed the full direct PINN-EBM PDE-weight grid on the severe synthetic
  setting. Weights 1/10/50 give rMAE
  0.08334/0.08226/0.78955 on Allen--Cahn,
  0.04783/0.04776/0.04122 on Burgers, and
  0.03976/0.03708/0.04551 on lambda--omega. The report retains the
  catastrophic Allen weight-50 result and does not post-hoc select a single
  favorable weight.
- Completed the full calibration-selected cost-0.10 PIV comparison at 10%
  persistent failure. naPINN rMAE/rMSE is
  \(0.15501\pm0.00119/0.23226\pm0.00463\), below MSE, OrPINN, and direct
  PINN-EBM but above LAD. Its gate rejects
  \(84.97\%\pm1.90\%\) of corrupted and \(3.63\%\pm0.89\%\) of clean
  scalars, so the report explicitly states that the 30%-severity
  calibration constraint does not transfer perfectly to 10% severity.
- Completed the full 20% persistent-failure PIV comparison. naPINN
  rMAE/rMSE is \(0.16889\pm0.00337/0.24869\pm0.00871\), about 1.8%
  below LAD on both field metrics and lower than MSE, OrPINN, and direct
  PINN-EBM. The gate rejects 91.25% of corrupted and 9.16% of clean
  scalars; direct raw-EBM AUROC remains higher than gate AUROC.
- Completed the lambda--omega Gaussian/Laplace/Student-t/four-Gaussian
  matrix for MSE, LAD, OrPINN, direct PINN-EBM, and three naPINN
  reconstruction backbones. naPINN-MSE is best for Gaussian,
  naPINN-LAD for Laplace, and direct PINN-EBM for Student-t and
  four-Gaussian. Base naPINN is worse than MSE on Student-t and the alternate
  naPINN backbones degrade sharply on four-Gaussian noise, so the report
  claims broader stress coverage rather than noise-family invariance.
- Completed and aggregated all 27 conservative 35,000-full-PINN-update
  MSE/LAD/OrPINN cells. Their rMAE is
  0.88440/0.34645/0.30771 on Allen--Cahn,
  0.59215/0.24871/0.19923 on Burgers, and
  0.44750/0.18358/0.14191 on lambda--omega. Every cell remains worse than
  both direct PINN-EBM and naPINN, while the 30k-to-35k change itself is
  mixed. The report labels this a conservative update-count check, not a
  wall-clock match or proof that extra steps always improve performance.
  Aggregate:
  `analysis/results/runs/rebuttal_synthetic_compute35_aggregation.json`.
- Completed all 18 estimator-free selector cells. Fixed-quantile rMAE is
  0.54862/0.37395/0.23718 on Allen--Cahn/Burgers/lambda--omega and rejects
  only about one third of known outliers because it removes a fixed 5% under
  15% corruption. The learnable-threshold gate gives
  0.22245/0.08493/0.06644: worse than naPINN on Allen--Cahn, close on
  Burgers, and slightly better on lambda--omega, while direct PINN-EBM
  remains best on all three. The report preserves this mixed result and
  distinguishes the trainable threshold from fixed preprocessing.
- Completed the nine-cell paper-aligned Allen--Cahn rejection-cost-1.0
  extension. Its 5%/10%/15% rMAE is 0.25878/0.56180/0.87859, versus
  0.09621/0.10159/0.10060 for the initial cost-0.5 core. Cost 1.0 rejects
  only 2--3% of known outliers at every ratio. The report discloses this
  adverse configuration discrepancy instead of silently replacing the
  submitted value.
- Started the dependency-satisfied PIV MAD stages after verifying every
  required seed/variant-matched LAD checkpoint. No held-out result is used
  to alter the frozen matrix.
- Completed the full AR(1) and spatial-burst real-PIV baseline matrices.
  LAD has the lowest field error in both. naPINN has the lowest nominal
  momentum/continuity residuals but rejects only 64.97% of AR(1)-corrupted
  scalars, versus 100% for spatial burst. The report preserves this mixed
  result as structured-noise coverage and incomplete calibration transfer,
  not universal superiority. Aggregate:
  `analysis/results/runs/rebuttal_realpde/injected_piv_aggregation.json`.
- Completed the three-seed 20% persistent-failure MAD-PINN extension.
  rMAE/rMSE is 0.17087/0.25418, nearly matching LAD and slightly above
  naPINN. Its fixed screen rejects 91.60% of corrupted and 25.42% of clean
  scalars. It remains explicitly labeled a 60,000-update,
  non-compute-matched reference.
- Completed the three-seed 10% persistent-failure MAD-PINN extension.
  rMAE/rMSE is 0.15850/0.24077, slightly above naPINN and above LAD.
  MAD has lower nominal-physics residuals than naPINN but rejects 30.30% of
  clean scalars, versus 3.63% for naPINN. The report preserves the
  field/physics/retention trade-off.
- Completed the three-seed AR(1) MAD-PINN extension. rMAE/rMSE is
  0.15266/0.23266, above naPINN and LAD. MAD has lower nominal-physics
  residuals and rejects 94.34% of corrupted scalars, but also rejects
  27.60% of clean scalars; naPINN rejects 64.97%/3.86%. The report
  preserves this recall/retention/field trade-off.
- Completed the three-seed spatial-burst MAD-PINN extension and therefore
  the full frozen CUDA experiment matrix. MAD rMAE/rMSE is
  0.14967/0.22770: rMAE slightly below naPINN, rMSE slightly above, and
  LAD best on both. MAD has lower nominal-physics residuals but rejects
  33.60% of clean scalars, versus 5.53% for naPINN; both reject all
  corrupted scalars. The final injected-PIV aggregate contains 108 eligible
  runs in 36 complete groups.
- Final validation passes: Python compilation, training/evaluation CLI
  imports, all 43 YAML files, shell syntax, and `git diff --check`.
  Synthetic aggregation has 100/100 complete groups, the conservative 35k
  aggregation has 9/9, injected PIV has 36/36, and all 15 required injected
  PIV MAD artifacts exist. Relevant queues have no failures or unfinished
  status files, no tmux experiment sessions remain, and
  `paper/neurips_2026.tex` has no diff.
- `git diff --check` passes.

## Open blockers and risks

1. The frozen additional-experiment matrix is complete and aggregated.
   Remaining risk is response selection and wording: positive, mixed, and
   adverse results must remain visible when compressing the evidence into
   the per-review character limits.
2. Broad submitted synthetic tables still have not been regenerated after
   cleanup.
3. There is no canonical immutable config/seed manifest connecting every
   reported table cell to a run directory.
4. Several current default-named YAMLs select ablations or baselines rather
   than the paper's main EBM-gated setup. See `docs/CODE_PAPER_MAP.md`.
5. Paper table values are embedded in TeX; a complete result aggregation and
   table-generation pipeline is not present.
6. The paper/config alignment still needs resolution for the reported
   Allen-Cahn rejection cost and likelihood-gate initializer fields that are
   present in YAML but not passed to the likelihood gate.
7. The exact OpenReview timestamp for the pre-July-27 author-response
   deadline must still be confirmed by the submitting author. The venue limit
   is 10,000 characters per review.
8. The worktree contains the user's ongoing changes. The pre-existing deletion
   of `.claude/settings.json` was not created or restored during cleanup.
9. `latexmk` and `pdflatex` are unavailable in the current environment, so the
   manuscript has not been recompiled here.
10. Full MAD-PINN natural-PIV results are complete. The implemented method
    requires 60,000 total PINN updates (30,000 LAD plus 30,000 MSE), so it
    must not be presented as compute-matched to the 30,000-update methods.

## Recommended next actions

1. Convert the verified evidence draft into concise review-specific
   responses under the venue character limit without hiding adverse cells.
2. Confirm the exact OpenReview response deadline and distribute shared
   evidence consistently across the AC and three reviewer responses.
3. Perform a final claim/code/config audit of the rebuttal documents only;
   do not modify or compile the paper source unless the author explicitly
   reverses the current instruction.

## Update protocol

After substantive work, update this file with:

- the change and its motivation;
- exact validation performed;
- new evidence or artifact locations;
- unresolved blockers;
- any code-paper-config discrepancy introduced or resolved.
