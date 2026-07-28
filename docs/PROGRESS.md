# Project progress

Last updated: 2026-07-27

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
- Completed and strictly aggregated the destination-server synthetic core:
  162/162 full runs, 54/54 three-seed groups, and seeds 40--42 only. The
  validator checked every resolved config, final checkpoint, finite metric,
  and method-specific 30,000-PINN-update/5,000-estimator-only budget. The
  immutable aggregate is
  `outputs/rebuttal/synthetic_recovery_20260726/aggregation_strict.json`
  (SHA-256
  `473b07059b568eea24657cf79dbe42be530700415d0a14be4e4488bd62bdc7cc`).
  This rerun records every field error and the learned Allen--Cahn
  \(\epsilon\), Burgers viscosity, and lambda--omega \(\beta\). It is a new,
  auditable destination-server lineage rather than a bitwise reproduction of
  the submitted table.
- The strict synthetic rerun does not exactly reproduce the transferred
  prior-server ranking. Direct PINN-EBM has the lowest mean field rMAE and
  rMSE in eight of nine PDE/outlier-ratio cells, while naPINN is lowest on
  both metrics for Allen--Cahn at 15%. In that cell naPINN wins seeds 40 and
  42 and PINN-EBM wins seed 41. The transferred all-nine PINN-EBM summary is
  retained as a separate lineage and is not overwritten or silently
  conflated with the new strict aggregate.
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
- Added `rebuttal/CROSS_SERVER_EXPERIMENT_HANDOFF.md` as the English
  operational handoff for continuation on another GPU server. It keeps the
  manuscript frozen, consolidates verified positive/mixed/adverse rebuttal
  conclusions, records the exact implemented four-Gaussian and additive
  point-outlier semantics, and defines the next experiment as controlled
  legacy-matched 4G corruption of real Cylinder-PIV training measurements at
  10% and 15% gross-outlier ratios. The exact legacy setting is the mandatory
  center; a separately labeled `b x o` scale grid is planned around it with
  seed-39 development and seeds 40--42 reporting. No experiment was launched
  and no planned scale-search result is represented as evidence.
- Verified the handoff's derived scale diagnostics against the current
  canonical synthetic configs, noise-injection code, and prepared Cylinder
  artifact. The legacy raw mixture has mean 0.475 and standard deviation
  approximately 6.72844; on nondimensional Cylinder training velocities,
  `0.1 * mean(abs(y))` implies expected background standard deviation
  approximately 0.697 times the pooled pre-injection PIV standard deviation
  and gross additive offsets approximately 2.09--6.97 times that reference
  standard deviation. The destination-server instructions require empirical
  per-artifact diagnostics and checksums rather than treating these expected
  values as completed evidence.
- Handoff validation passes: Python compilation for `pinnlab`, `analysis`,
  and `scripts/rebuttal`; training, evaluation, and real-PIV runner CLI
  imports; all YAML parsing; shell syntax; Markdown language/fence/whitespace
  invariants; `git diff --check`; and the frozen-paper diff check. No
  experiment was run.
- Clarified the cross-server scope: current full runs remain Cylinder-only,
  while Controlled Cylinder, FSI, Foil, and Combustion are explicitly
  retained as future validation candidates after their dataset-specific
  physics or observation-model requirements are implemented. This is not a
  rebuttal-period experiment promise.
- Refined the closest-prior response strategy. The handoff no longer directs
  every response to volunteer a global narrowing of naPINN's contribution.
  Pilar--Wahlström is addressed where the Area Chair or reviewer explicitly
  raises it; those answers acknowledge the shared EBM/staged components and
  then clearly distinguish direct EBM-NLL optimization from naPINN's
  detached density objective, explicit per-measurement gate, gated base
  reconstruction loss, and rejection regularization. Requested direct
  comparison evidence, including adverse outcomes, remains preserved
  internally; reviewer-facing numerical use is governed by the hold below.
- Placed all direct naPINN-versus-PINN-EBM numerical results, rankings, and
  derived response claims on `RESPONSE HOLD` by author decision. Existing
  artifacts and adverse outcomes remain preserved as internal verified
  evidence, but they must not be copied into reviewer responses until the
  planned real-PIV legacy-corruption, scale, robust-loss, and closest-prior
  checks are completed as far as feasible and the author explicitly chooses
  the response direction. The handoff, authoritative response matrix,
  working response draft, and Korean report now carry the same hold status.
- On the destination RTX A6000 server, recovered the canonical RealPDEBench
  Arrow shard with the expected SHA-256
  `9a4302c602c6338e723ec3e2e97ceae5b90c16ff5061322ffcce29062c2105dc`
  and regenerated the 200-frame, 192-sensor Cylinder split. Its scientific
  shapes, split, and nondimensional diagnostics match the handoff exactly.
  The regenerated NPZ SHA-256 is
  `e2513cccae4bdd75da1bf33a85bf23bb66a2dc8350c3e951a5c3e95ebb94b1bd`,
  rather than the historical `0aea...` hash, because `metadata_json` embeds
  the server-local absolute Arrow path. This difference is recorded and the
  regenerated artifact is not described as byte-identical to the original.
- Implemented the planned legacy four-Gaussian plus positive additive
  point-outlier generator, matched naPINN MSE/L1/q=2.9 reconstruction
  variants, schema-correct gross-outlier/background-only metrics, exact
  input checksums and software provenance, a dedicated complete-only
  aggregator, and a deterministic four-GPU exact-center queue.
- Generated and independently audited all six exact-center paired corruption
  artifacts for seeds 40--42 and 10%/15% gross-row ratios. Artifact
  checksums, held-out bitwise invariance, mask counts, shared per-seed
  background realizations, and nested 10%-within-15% gross-row indices pass.
  CUDA forward/backward smoke tests pass for naPINN-MSE, naPINN-L1, and
  naPINN-q2.9; one smoke also used the actual seed-40 legacy artifact. Smoke
  outputs remain explicitly non-evidentiary.
- Completed and strictly aggregated the 48-run exact legacy-center matrix
  across all four available RTX A6000 GPUs on 2026-07-25. The matrix covers
  two ratios, three reporting seeds, and eight matched conditions: MSE, LAD,
  OrPINN q=2.9, equal-weight PINN-EBM, closest-prior-weight PINN-EBM, and
  naPINN-MSE/L1/q=2.9. All 48 full records passed checkpoint, 30,000-PINN-
  update budget, estimator-phase, provenance, exact-input-hash, paired-input,
  finite-metric, and non-smoke validation; all 16 three-seed groups and six
  paired corruption blocks are complete. The sole smoke run is explicitly
  excluded. Complete-only artifacts are
  `analysis/results/runs/rebuttal_realpde_legacy_4g/aggregation.json` and
  `.csv`; every positive, mixed, and adverse eligible result is retained
  internally under `RESPONSE HOLD`.
- Validated concurrent use of five independent runs per RTX A6000 during the
  exact-center campaign: utilization remains 98--100%, peak observed device
  memory is about 3.8 GiB of 49 GiB, and peak observed temperature is
  65 degrees C. Far-tail exact-center cells were launched only where at
  least two full canonical queue positions separated them from the active
  cell; canonical queues will record completed tail cells as existing skips.
  The canonical queues finished with 24 directly completed and 24
  pre-existing accelerator-completed cells, zero failures, and 48/48
  strictly validated artifacts. These shared-device wall times are not
  hardware-isolated benchmarks.
- Generated and audited the complete seed-39 development artifact grid:
  10%/15% gross-row ratios and `b,o` in `{0.5,1,2}`, for 18 paired scale
  cells. Added 18 immutable variant configs and a deterministic 144-run
  four-GPU queue covering the same eight conditions as the exact center.
  Static validation confirms exactly 36 balanced jobs per shard and that
  every config resolves to an existing paired artifact.
- Froze the previously underspecified "not catastrophically worse" scale
  criterion before any seed-39 scale training in
  `rebuttal/legacy4g_scale_selection_plan.yaml`: both nominal physics
  residuals must be finite and no more than ten times the matched ungated
  baseline. The remaining handoff criteria are unchanged. The selector keeps
  the complete heatmap and every ineligibility reason, excludes the exact
  center from candidate selection, ranks by the smaller field gain, and
  selects at most three cells without relaxing thresholds.
- Added guarded candidate preparation, queueing, aggregation validation, and
  a stage handoff that will launch the seed-39 grid only after all 48
  exact-center runs and their strict aggregation pass. Candidate reporting
  artifacts will use fresh seeds 40--42 and all eight matched conditions;
  their direct comparisons remain under `RESPONSE HOLD`.
- Upgraded the seed-39 scale and candidate-confirmation queues
  to run five independent child processes per assigned GPU while keeping
  deterministic shard membership, one exclusive log per child, serialized
  status updates, and fail-fast cessation of new launches. Fake-subprocess
  tests verified a peak concurrency of five, complete 36-job scale-shard and
  eight-job candidate execution, and scale fail-fast behavior that launches
  no more than the already active five jobs after the first failure.
- Completed the guarded 144-run seed-39 scale grid after the 48-run exact
  matrix and strict aggregation passed. Four balanced 36-job shards ran with
  five children per RTX A6000 (20 concurrent jobs total); all 144 records
  pass checkpoint, update-budget, fixed-Re, exact-input-hash, finite-metric,
  unique-key, paired-cell, and non-smoke validation. Every one of the 18
  scale cells contains all eight conditions, every condition occurs exactly
  18 times, and each shard finished 36 jobs with zero failures. Queue status
  and exclusive child logs are under
  `analysis/results/runs/rebuttal_realpde_legacy_4g_scale_queue/`.
- Applied the frozen seed-39 selection rule only after the strict 144-run
  audit. `seed39_scale_selection.json` retains the complete 18-cell heatmap,
  all three naPINN evaluations and every ineligibility reason per cell,
  excludes both exact centers, uses the unchanged ranking prefix and maximum,
  and records the frozen-plan hash. The sealed confirmation manifest matches
  the plan and selection hashes, uses only fresh seeds 40--42, and keeps all
  eight conditions.
- Completed and strictly validated the sealed candidate-confirmation matrix.
  The four balanced queue shards each completed 18 jobs with zero failures,
  giving 72/72 full runs, 24 complete three-seed groups, and nine paired
  input blocks with all eight matched conditions. Every record passed
  checkpoint, 30,000-PINN-update budget, staged estimator-phase, fixed-Re,
  exact-input-hash, finite-metric, unique-key, paired-input, and non-smoke
  checks. The manifest, frozen-plan and selection hashes, fresh reporting
  seeds, corruption metadata, unchanged held-out PIV, and artifact/config
  pairs also passed independent validation. Complete-only artifacts are
  `analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/aggregation.json`,
  `.csv`, and `candidate_validation.json`; there are no exclusions. All
  positive, mixed, and adverse outcomes remain retained, and direct
  numerical use remains under `RESPONSE HOLD`.
- Sealed the completed campaign with a fresh end-to-end audit on 2026-07-26.
  Queue totals and zero-failure status, exact and confirmation aggregates,
  frozen-plan and selection hashes, manifest/config/artifact linkage,
  three-seed grouping, eight-condition pairing, corruption metadata, and
  bitwise held-out-PIV invariance all pass. The default compile/help/YAML/
  shell-syntax/diff-check suite and every new legacy-4G CLI help entry pass.
  No campaign process remains active, all four GPUs are idle, and
  `paper/neurips_2026.tex` remains unchanged.
- Built the source-backed internal Korean Legacy-4G results and rebuttal
  strategy report at
  `rebuttal/reports/legacy4g_experiment_ko/report.md`, with its reproducible
  builder and Markdown build notes alongside it. The report retains exact,
  selected-scale, per-seed, positive, mixed, and adverse evidence; recommends
  the mandatory unselected exact center as primary evidence; and leaves all
  direct comparison language under `RESPONSE HOLD`. The original portable
  HTML and machine artifact were removed after the user requested that all
  report documents use Markdown. Markdown generation, heading/table
  structure, source-path presence, and numerical consistency pass.
- Expanded the Korean report for readers who do not know the internal
  shorthand. It now defines `legacy`, `(b,o)=(1,1)`/`exact center`,
  corruption-protocol `transfer`, the 3-PDE by 3-ratio `synthetic core`,
  held-out PIV, field metrics, PDE residuals, AUROC, gates, baselines, and
  paired comparisons before using them. Result, selection, limitation, and
  reviewer-strategy sections now explain the experiment, comparison,
  interpretation, and claim boundary in full sentences. No numerical result,
  ranking, source artifact, or `RESPONSE HOLD` status changed.
- Added the internal Markdown audit
  `rebuttal/reports/method_rankings_and_pinn_ebm_audit_ko.md`. It corrects
  the false inference that PINN-EBM led on unmodified PIV, separates
  within-dataset `background-only` labels from a nonexistent noise-only
  training regime, and gives complete method rankings for the exact
  Legacy-4G PIV cells, fresh-seed candidates, documented synthetic
  four-Gaussian cells, and the excluded historical natural-PIV diagnostic.
  Synthetic 5%/10% rMSE rankings remain unfilled because the referenced
  aggregate and individual run artifacts are absent from this checkout.
- Audited the local direct PINN-EBM path against `reference/PINN-EBM.pdf`
  and official `ppilar/PINN-EBM` commit
  `0b74f6f9209d68c79ecb9608b71755977d08f578`. The residual definition,
  estimator-only initialization, NLL-plus-PDE objective, no-gate design,
  and joint PINN/EBM gradient path are aligned; no severe detach or duplicate
  optimizer-step defect was found. The dataset, velocity/pressure
  parameterization, PINN/EBM architectures, residual normalization,
  integration grid, schedules, batches, learning rates, and inverse-PDE
  target differ substantially, so this is an objective-faithful matched
  comparator rather than a published Navier-Stokes reproduction. The paper
  specifies an 8-by-20 Navier-Stokes PINN, whereas the official repository's
  active `Net_NS` is hard-coded as 4-by-30.
- Audited all 12 exact Legacy-4G direct PINN-EBM checkpoints for the fixed
  EBM normalization support. No full-training scalar residual lay outside
  the scaled `[-10,10]` grid; per-run maximum absolute scaled residuals were
  4.84--6.02. Fresh real-PIV runs now store the normalization-grid bounds,
  maximum absolute scaled residual, and outside-grid count/fraction in
  `metrics.json`.
- Made `outputs/` the canonical location for fresh experiment artifacts.
  Direct real-PIV and synthetic rebuttal runners, all corresponding queue
  defaults, and the general training configs now use campaign-specific
  roots below `outputs/`; real-PIV queues explicitly forward the same
  `--output-root` used for completion checks to child runs. General
  `pinnlab.train` now writes final local evaluation/performance metrics to
  `metrics.json` alongside checkpoints and the resolved config. The layout
  and retained artifacts are documented in `outputs/README.md`. Existing
  completed artifacts under `analysis/results/runs/` were not moved because
  their configs, checksums, aggregations, and reports record those paths.
- Added two explicitly separated faithful PINN-EBM Navier--Stokes
  reproductions. Variant A executes unmodified official commit `0b74f6f` with
  its active 4-by-30 `Net_NS`; variant B applies only the declared 8-by-20
  paper-architecture patch. Both freeze the official Raissi
  `cylinder_nektar_wake.mat` checksum, official seed-0/Nrun-5/model-order
  semantics, 100,000-update schedule, 3-by-5 dropout EBM, adaptive 1,001-point
  integration grid, tail retry, batches, optimizer, and scheduler. The
  upstream commit omits the required `prop_noise` setting; both variants
  supply and disclose `prop_noise=0`, matching the prior additive-noise path.
  Source/config/dataset audits, five focused tests, and CUDA
  forward/backward component smokes pass. Both 1.5-million-update full
  executions are active below
  `outputs/rebuttal/pinn_ebm_upstream/runs/`; no performance claim is made
  from smoke or partial output. Progress is parsed read-only into
  `outputs/status/pinn_ebm_upstream/status.json`.
- Added `analysis/aggregate_pinn_ebm_upstream.py` for the eventual complete
  A/B handoff. It requires the caller to name the exact two run directories,
  preventing interrupted predecessors or smokes from being selected. It
  verifies complete evidence labels, frozen configs, seed 0, Nrun 5, model
  index 3, run indices 0--4, finite per-run metrics, independently recomputed
  means/sample standard deviations, official result-pickle size/hash,
  dataset provenance, and archived source commit/patch before writing an
  immutable A/B comparison. Two focused tests pass, and the live incomplete
  directories are correctly refused without an output artifact.
- Completed a seed-40 input-pairing audit for all three synthetic PDEs and
  5%/10%/15% gross-row ratios. Within each of the nine conditions, MSE,
  direct PINN-EBM, and naPINN receive identical coordinates, clean values,
  noisy values, and gross-row indices; 5% masks are nested in 10%, and 10%
  in 15%. The audit is
  `outputs/audits/synthetic_pinn_ebm_fairness/seed_40.json`. It also verifies
  a homogeneous global four-Gaussian background and one-sided positive
  additive gross offsets. At 15%, mean gross-row errors are about 6.5 times
  the background standard deviation and 100% positive. This distribution is
  especially compatible with an unconditional scalar residual EBM and is a
  plausible reason for strong direct PINN-EBM performance; no
  method-specific input advantage was found. The manuscript says selected
  observations are replaced by high-magnitude samples, whereas all three
  experiment implementations add positive offsets to the existing
  background noise. This code-paper description discrepancy remains open.
- Completed and aggregated an 18-run synthetic background-noise-only diagnostic:
  direct PINN-EBM and naPINN across three PDEs and seeds 40--42 with the
  four-Gaussian background retained and gross-row count set to zero. Fresh
  Burgers and lambda--omega simulation data were required on this server.
  Their NPZ container hashes differ from the handoff hashes, so these runs
  are labeled destination-server regenerated diagnostics rather than
  byte-identical reproduction evidence. Direct PINN-EBM ranks first in all
  three PDEs for both field metrics: Allen--Cahn rMAE/rMSE
  `0.09165/0.09011` versus naPINN `0.09769/0.09651`; Burgers
  `0.03737/0.05055` versus `0.07537/0.08935`; and lambda--omega
  `0.04473/0.05886` versus `0.07287/0.08955`. Thus the synthetic direct-NLL
  advantage is already present under global four-Gaussian background noise
  without gross offsets. All 18 models, metrics, zero-failure shard
  summaries, and complete three-seed aggregates are retained under
  `outputs/rebuttal/synthetic_background_only/`.
- Completed and strictly aggregated the nine-run natural-PIV
  naPINN robust-loss matrix using the unmodified PIV artifact, fixed Reynolds
  number, matched 30,000 PINN plus 5,000 estimator-only updates, and seeds
  40--42. naPINN-L1 ranks first at rMAE
  `0.112416 +/- 0.001610` and rMSE `0.186754 +/- 0.001744`; naPINN-q2.9 is
  second at `0.122562 +/- 0.001513` and `0.189499 +/- 0.002136`; naPINN-MSE
  is third at `0.139353 +/- 0.001868` and `0.197751 +/- 0.001248`. Both
  robust-loss improvements occur in every paired seed, but L1 and q2.9 have
  higher final momentum and continuity residuals than MSE. All gates retain
  more than 99% of scalar measurements, so the present evidence attributes
  the field-error gain primarily to the reconstruction loss, not extensive
  measurement rejection. Natural PIV has no controlled outlier labels, and
  the strict aggregator rejects outlier-AUROC claims. Models, histories,
  checkpoints, hashes, metrics, and complete-only aggregates are under
  `outputs/rebuttal/natural_piv/`.
- Completed and strictly aggregated the separately frozen 18-run natural-PIV
  no-gate baseline/direct-NLL matrix with zero failures. LAD-PINN ranks first
  at rMAE/rMSE `0.112108 +/- 0.001450` and
  `0.183531 +/- 0.002101`; OrPINN q2.9 is
  `0.121429 +/- 0.001211` and `0.185683 +/- 0.001520`; MSE-PINN is
  `0.139760 +/- 0.002817` and `0.198223 +/- 0.002260`. Their gated
  counterparts are not better on the same robust loss: naPINN-L1 minus LAD
  is `+0.000308/+0.003223` in mean rMAE/rMSE, and naPINN-q2.9 minus OrPINN
  is `+0.001133/+0.003816`. naPINN-MSE differs from MSE by only
  `-0.000406/-0.000472` with mixed seed direction. The robust-loss gain on
  natural PIV therefore does not establish an additional gate benefit.
  Direct PINN-EBM is lower-ranked for all predeclared PDE weights:
  weight 50 `0.228312/0.314401`, weight 10 `0.237386/0.469826`, and
  weight 1 `0.303340/0.750336`. Weight 50 preserves low momentum residual,
  while weight 1 exhibits severe physics-residual growth and weight 10 high
  seed variability. All weights are retained and
  `selection_performed=false`; no favorable result was selected after the
  fact. Complete artifacts are under
  `outputs/rebuttal/natural_piv_baselines/`.
- Added the technical Korean Markdown report
  `rebuttal/reports/pinn_ebm_faithful_natural_piv_fairness_ko.md`. It defines
  natural PIV, gross outliers, residuals, EBM, direct likelihood, gates,
  robust losses, hashes, and smoke tests in plain language; separates
  upstream, paper-architecture, and compute-matched evidence; records the
  paired-input audit, benchmark-distribution diagnosis, natural-PIV robust
  results, physics trade-off, artifact paths, limitations, and a
  claim-bounded rebuttal strategy. It is Markdown only.
- Updated the technical Korean Markdown report with a complete-results
  section covering 309 finished full training runs: 48 exact Legacy-4G PIV,
  144 corruption-scale development, 72 fresh-seed confirmation, 27 natural
  PIV, and 18 synthetic background-only runs. The section records the
  exact-center field metrics, completion and evidence status, source
  aggregates, the natural-PIV response exclusion, and the distribution-
  dependent conclusion. The two official PINN-EBM A/B runs remain explicitly
  separated as incomplete; their partial logs are not used as performance
  evidence.
- Re-ran the proportional validation suite after these changes:
  forced `compileall` for `pinnlab`, `analysis`, and rebuttal scripts;
  both required CLI helps; all 73 YAML files; all shell syntax checks; five
  focused upstream/config/data tests; and `git diff --check` pass.
- Audited every reviewer-facing experimental claim against artifacts present
  in the destination checkout. The Legacy-4G exact/scale/confirmation,
  natural-PIV, and synthetic-background-only campaigns have locally complete
  aggregates. Synthetic core/parameter recovery, injected-PIV structured
  corruptions and inverse-Re, EMA/full rejection-cost sweep, selector/MAD,
  noise-family, and 35k-compute values were retained only as transferred
  numerical summaries because their cited source aggregates are absent. The
  separately recovered Allen--Cahn cost-1.0 extension below is now the first
  exception in this set. Added
  `rebuttal/reports/reviewer_experiment_evidence_status_ko.md` and introduced
  `EVIDENCE_NEEDED` in `rebuttal/response_matrix.md`; no transferred numerical
  summary is now treated as locally reverified merely because it appears in a
  report.
- Downloaded and checksum-verified the first official real-data Arrow shards
  for RealPDEBench Controlled Cylinder, FSI, Foil, and Combustion under
  `outputs/rebuttal/realpdebench_multidataset/source/`. The planned
  non-Cylinder fluid protocol is frozen at frames 1000--1199, 192 fixed
  spatial sensors, seeds 40--42, 10%/15% Legacy-4G gross offsets, and eight
  methods, for 144 full runs. Combustion real observations are one-channel
  OH* intensity rather than PIV velocity and require an applicability audit,
  not an invented incompressible-Navier--Stokes experiment.
- Started a fresh 162-run synthetic core recovery campaign under
  `outputs/rebuttal/synthetic_recovery_20260726/` to restore complete
  5%/10%/15% field and PDE-parameter evidence across all three PDEs, six
  methods, and seeds 40--42. Also started the separate nine-run
  Allen--Cahn rejection-cost-1.0 recovery under
  `outputs/rebuttal/allen_cost1_recovery_20260726/`. Both use new output
  roots, retain models/metrics/histories, and do not overwrite transferred
  summaries.
- Completed the fresh nine-run Allen--Cahn rejection-cost-1.0 recovery with
  zero failures and a strict 9/9-run, 3/3-group aggregate. Fresh 5%/10%/15%
  rMAE is `0.26088/0.56828/0.86890`, while the gate rejects only
  `3.07%/2.25%/2.12%` of known outliers. The complete destination-server
  artifact is
  `outputs/rebuttal/allen_cost1_recovery_20260726/aggregation_strict.json`.
  These values are kept distinct from the slightly different transferred
  summary and confirm the same adverse conclusion. The new run is sensitivity
  evidence; it cannot establish which setting generated the submitted table.
- The first two PINN-EBM upstream A/B processes were terminated when their
  owning execution context ended. Their partial run metadata are explicitly
  marked `interrupted` and `not_evidence_interrupted_process`. Fresh A/B
  runs were started in a persistent execution session with run ID
  `official_session_20260726`; only complete official result pickles will be
  aggregated.
- Strengthened `analysis/aggregate_rebuttal_synthetic.py` with an opt-in
  strict campaign mode. It now can require exact run and group counts, one
  artifact per required seed, matched metrics/config fields, `final.pt`,
  30,000 PINN updates, the method-appropriate 5,000/0 estimator-only budget,
  recursively finite numerical metrics, and non-overwriting outputs. Focused
  tests verify both a complete three-seed write and a partial-matrix refusal.
  The active Allen--Cahn and full synthetic roots were also checked in their
  incomplete states: strict aggregation refused 8/9 and 13/162 runs,
  respectively, and wrote no misleading aggregate.
- Implemented the frozen non-Cylinder RealPDEBench campaign for Controlled
  Cylinder, FSI, and Foil. The preparation pipeline records source and
  prepared-artifact hashes, released-coordinate scales, fixed train/held-out
  sensor partitions, nominal Reynolds metadata, and explicit geometry/model
  limitations. The paired injector produced all 18 required
  dataset/seed/ratio artifacts with shared per-seed background draws and
  gross-offset factors, exact 10%-within-15% nested indices, and bitwise
  unchanged held-out measurements. The immutable matrix contains exactly
  144 unique full runs: three datasets, two ratios, three seeds, and eight
  methods.
- Added a strict eight-shard RealPDEBench queue and complete-only aggregator.
  The queue assigns exactly 18 cells to each shard, launches two workers per
  RTX A6000, refuses to overwrite partial or mismatched run directories, and
  skips only artifacts that pass config, input-hash, checkpoint, update-
  budget, finite-metric, and non-smoke checks. The aggregator refuses to
  write unless all 144 runs, 18 paired input blocks, and 48 three-seed
  method groups validate.
- The non-Cylinder implementation passed three focused tests, 24/24 CPU and
  24/24 CUDA dataset-method forward/backward checks, a generic-overlay CUDA
  end-to-end smoke, legacy Cylinder compatibility, all 101 YAML parses,
  forced compilation, required CLI imports, shell syntax, and
  `git diff --check`. The strict aggregator correctly refused the empty
  full-run tree and wrote no artifact. All smoke records are explicitly
  non-evidentiary.
- Started the full 144-run non-Cylinder RealPDEBench queue on 2026-07-26
  across all four RTX A6000 GPUs. Models, checkpoints, histories, resolved
  configs, metrics, and run metadata are written below
  `outputs/rebuttal/realpdebench_multidataset/runs/`; worker status and logs
  are below `outputs/status/realpdebench_multidataset/`. Initial full cells
  reached finite warm-up updates on every GPU. No partial log is treated as
  performance evidence.
- The first non-Cylinder full cell, Controlled Cylinder seed 40 at 10%
  corruption with direct PINN-EBM PDE weight 1, completed and independently
  passed the queue's strict validator. Its resolved config, exact input hash,
  `final.pt`, 30,000 PINN plus 5,000 estimator-only updates, metrics, and
  non-smoke evidence label are all present. This single-seed unaggregated
  result is retained as execution evidence only; no method ranking is drawn
  before the complete 144-run aggregate.
- Audited RealPDEBench Combustion separately. The released real measurement
  is a single OH* chemiluminescence-intensity channel and does not provide
  the velocity observations required by the current pressure-latent
  incompressible Navier--Stokes PINN. Consequently, no arbitrary performance
  run was launched. The observation audit, incompatibility reason, and
  requirements for a future combustion-specific observation operator and
  PDE are stored in
  `outputs/rebuttal/realpdebench_multidataset/combustion_applicability.json`.
- Audited executability of the remaining reviewer-evidence recoveries while
  the primary queues run. The synthetic supplement queue now expands to
  108 unique cells covering EMA, rejection cost, three additional noise
  families, naPINN backbone losses, and selector ablations; the held-out direct
  PINN-EBM PDE-weight extension has 18 reporting cells, the 35k conservative
  update-count reference has 27, and synthetic MAD-PINN has nine dependent
  cells. Their queue scripts and configs are present. The five previously
  absent structured Cylinder-PIV corruption NPZ inputs (10/20/30% drift,
  AR(1), and spatial burst) were then regenerated from the retained clean
  parent with the frozen seed and strengths. All parent hashes, labeled-mask
  counts/shapes, held-out bitwise equality, unlabeled-training bitwise
  equality, and six structured/inverse config paths pass. Their fresh
  checksums are recorded in the generator output and sidecar manifests. No
  structured-PIV training queue has been launched while the primary
  campaigns saturate all GPUs.
- Each regenerated structured-PIV input also passed a one-step CPU MSE
  runner smoke through the actual loader, nominal Navier--Stokes residual,
  held-out evaluation, corruption metadata, and final metric-writing path.
  All five outputs have finite field/physics metrics, the exact regenerated
  input hash, `pinn_update_steps=1`, and
  `evidence_status=smoke_test_not_paper_evidence` below
  `outputs/status/structured_piv_input_smokes_20260726/`. They are schema
  checks only and are prohibited from performance aggregation.
- Added non-overwriting guards to both structured-PIV generators after the
  fresh artifacts were sealed. Re-running against an existing NPZ or sidecar
  now raises `FileExistsError`. Focused existing-target checks confirm the
  guard for drift and AR(1), and the audited artifact hashes remain unchanged.
- Replaced the permissive structured-PIV aggregator with a frozen strict
  reviewer-evidence contract. It now requires exactly 102 complete full runs:
  75 fixed-Re structured-corruption cells, 12 inverse-Re cells, and 15
  dependent MAD-PINN cells, forming exactly 34 three-seed groups. Before
  writing under `outputs/rebuttal/realpde/`, it verifies exact tag/method/seed
  coverage, resolved schedules, 30,000-step PINN and method-appropriate
  estimator budgets, final checkpoints, run metadata, recursively finite
  metrics, and input artifact path/SHA-256 provenance. MAD-PINN additionally
  requires its screening artifact and 30,000 + 30,000 = 60,000 pipeline
  provenance. Outputs are immutable. Three focused tests pass, and a partial
  live-root invocation refused aggregation without creating an output file.
- Corrected a latent duplicate in the synthetic supplement queue: the three
  `(EMA momentum=0.05, rejection cost=0.5)` center cells occurred once in
  the EMA sweep and again in the rejection-cost sweep, but resolve to the
  same immutable run directories. The frozen recovery matrix therefore has
  108 unique runs (36 three-seed groups), with the shared center trained once
  and reused in both sensitivity summaries. Also added an explicit
  `calibration`/`heldout` phase split to the direct PINN-EBM PDE-weight queue,
  allowing seed 39 and seeds 40--42 to be stored and strictly aggregated in
  separate roots. The common synthetic strict aggregator now accepts an
  explicit exact per-run PINN budget for the 35,000-update and 60,000-update
  MAD references and validates MAD stage provenance. Eight focused queue and
  aggregation tests pass.
- Added `configs/rebuttal/reviewer_recovery_manifest.yaml` as the explicit
  destination-server recovery contract. It records the active roots, exact
  run/group/seed counts, strict aggregate destinations, isolated calibration
  and held-out roots, dependency order, and the reviewer requests that still
  lack a scientifically frozen executable protocol. A focused manifest test
  checks every declared dimension, the 75 + 12 + 15 = 102 structured-PIV
  decomposition, the 108-cell supplement correction, and calibration/output
  isolation. Nine queue/aggregation/manifest tests and the full YAML parse
  pass.
- Expanded the Korean reviewer-evidence audit with a timestamped
  destination-server recovery table and a link to the frozen recovery
  manifest. Rows that previously said only "recover artifacts" now
  distinguish active, protocol-ready, and dependency-waiting recoveries,
  including the full 162-run PDE-parameter matrix, 108 unique supplement
  runs, separate PINN-EBM weight calibration/reporting roots, 27 conservative
  35k runs, and the 102-run structured/inverse/MAD PIV matrix. This is a
  status/provenance update only; no partial performance values or rankings
  were added.
- Added `scripts/rebuttal/launch_reviewer_recovery.py`, which reads the frozen
  recovery manifest and builds deterministic sharded commands for all eight
  queued campaigns. It assigns requested GPUs round-robin, keeps performance
  outputs and launcher/status logs separate, captures every worker command,
  PID, log, and return code, and leaves aggregation to the campaign-specific
  strict verifier. Before either MAD campaign can launch, it now verifies all
  nine synthetic or all 15 structured-PIV seed-matched LAD checkpoints and
  their required 30,000-update stage-one metrics. Dry runs for all campaigns
  produced exactly eight workers with the declared isolated roots; focused
  launcher and missing-dependency tests pass. The launcher has not started
  another campaign while all four GPUs remain saturated.
- Added `scripts/rebuttal/monitor_reviewer_recovery.py` and generated
  `outputs/status/reviewer_recovery_summary.json`. The summary tracks all 13
  active, complete, queued, and dependent campaign records with expected run
  counts, discovered complete non-smoke metric counts, declared strict
  aggregate existence, manifest hash, and aggregate hashes when available.
  It deliberately includes no performance metric values and marks evidence
  ready only when the declared strict aggregate JSON exists. Focused tests
  cover shared structured-PIV tag separation and prohibit field/parameter
  performance keys in the status payload.
- Strengthened the recovery monitor so aggregate-file existence alone cannot
  set `evidence_ready=true`. It now parses the declared JSON, requires
  `status=strict_complete`, checks exact included-run and group counts,
  verifies required seeds when present, requires the declared CSV, and checks
  the two official A/B variant records separately. A focused test confirms
  that an existing but count-mismatched JSON/CSV pair remains not ready. The
  102-run structured-PIV strict payload now records its own status and exact
  counts for the same validation contract.
- Connected the official PINN-EBM progress monitor to the unified recovery
  status. The A/B records now expose update-equivalent progress, declared
  totals, naive ETA, and EBM retry counters without exposing partial
  performance metrics or changing evidence readiness. At the latest refresh,
  A/B were at approximately 7.08%/4.26% of the monitor's declared update
  equivalents with zero recorded tail retries.
- Added `scripts/rebuttal/finalize_reviewer_recovery.py` to derive every
  strict aggregation command from the recovery manifest. It supplies the
  exact roots, seed sets, run/group counts, and 30k/35k/60k budget expected
  by each synthetic campaign; exact run directories for official A/B; and
  the dedicated RealPDEBench and combined structured-PIV aggregators. After
  aggregation it refuses success unless the unified monitor independently
  accepts the output as evidence ready. All nine campaign commands and the
  launcher were direct-CLI dry-run tested. That test exposed and fixed a
  repository-import-path issue that had been hidden by pytest's `sys.path`
  setup.
- Audited storage capacity before retaining the remaining checkpoints and
  metrics. The filesystem has about 39 GiB available. Completed synthetic
  run directories average about 0.13 MiB, completed multidataset PIV runs
  about 10.5 MiB (maximum about 11.9 MiB), and the two live official
  reproductions currently occupy about 25 MiB each. The remaining run
  artifacts are therefore expected to require only a few GiB; the large
  existing footprint is mostly sealed RealPDEBench source/corruption inputs.
  No user or source artifacts were deleted.
- Caught a pre-launch structured-PIV protocol mismatch: both fixed-Re and
  inverse-Re queues would have inherited their historical default naPINN
  rejection cost 0.01 even though the frozen seed-39 calibration selected
  0.10 for the full held-out matrix. No recovery run had started, so no
  result was contaminated. The recovery manifest now explicitly names
  `realpdebench_cylinder_napinn_rejection_01.yaml`, the launcher passes that
  config to both queues, and the strict 102-run tag contract expects
  `napinn_rejection_01_*`. The earlier cost-0.01 sensitivity remains
  preserved as a separate historical comparison. Twelve manifest/launcher/
  structured-aggregate tests pass.
- Separately audited the already-active non-Cylinder 144-run matrix. Its
  frozen common config and every current naPINN resolved config consistently
  use rejection cost 0.5. This is retained as the predeclared canonical
  Legacy-4G generalization stress, not retroactively changed to the 0.10
  setting calibrated on persistent Cylinder drift. Final reporting must keep
  these scopes explicit: non-Cylinder Legacy-4G/cost-0.5 versus structured
  Cylinder drift and inverse-Re/seed-39-selected cost-0.10.
- Reconciled `rebuttal/response_draft.md` and `rebuttal/response_matrix.md`
  with the local artifact audit. Both had broad historical prose saying all
  additional matrices or several transferred tables were complete even
  though their source aggregates are absent in this checkout. The numerical
  summaries were preserved, but the affected structured-PIV, synthetic core,
  weight, parameter, selector/MAD, EMA/noise, inverse-Re, and 35k sections
  now explicitly say `EVIDENCE_NEEDED` or "transferred summary." Only the
  locally strict Legacy-4G and fresh Allen cost-1.0 campaigns retain verified
  wording. No partial destination-server result was inserted.
- The official active-code A reproduction completed its first ordinary model
  in run 0 and entered PINN-EBM model index 3. Its first EBM initialization
  and tail check passed, joint training resumed, and the monitor records one
  initialization attempt with zero retries or escalations. Variant B remains
  in its first ordinary model. These are execution-health observations only;
  neither partial log is performance evidence.
- Completed and strictly aggregated the fresh synthetic supplement at
  108/108 runs and 36/36 three-seed groups. The immutable aggregate is
  `outputs/rebuttal/synthetic_supplement_recovery_20260726/aggregation_strict.json`
  (SHA-256
  `4c0994090f5cf7fb359e8176c59f524c78f23cef4d8b50cb96c4bd832c79300b`).
  It validates the EMA, five-value Allen--Cahn rejection-cost,
  Gaussian/Laplace/Student-t noise-family, naPINN robust-backbone, and
  estimator-free selector comparisons with exact seeds 40--42, immutable
  configs/checkpoints, finite metrics, and the required update budgets.
- In the fresh Allen--Cahn 15% sensitivity, EMA update weights
  \(m=0.01/0.05/0.10\) give rMAE
  \(0.10778/0.09202/0.08551\). Rejection costs
  \(0.10/0.30/0.50/0.70/1.00\) give rMAE
  \(0.08847/0.09133/0.09202/0.09886/0.86890\); cost 1.0 rejects only
  2.12% of known outliers, versus about 99.3% for costs 0.10--0.70.
  These are local sensitivity findings, not post-hoc default selection or
  proof of the submitted-run setting.
- The fresh noise-family and selector results are deliberately reported as
  mixed. For lambda--omega rMAE, naPINN-MSE is best under Gaussian,
  naPINN-LAD under Laplace, LAD narrowly under Student-t, and direct
  PINN-EBM under four-Gaussian noise. Fixed-quantile rMAE is
  \(0.50940/0.37315/0.23726\) and learnable-threshold rMAE is
  \(0.24415/0.08557/0.06730\) on
  Allen--Cahn/Burgers/lambda--omega. The threshold is not mislabeled as
  fixed preprocessing, and no noise-family-invariant superiority is claimed.
- After the supplement strict aggregate passed, the automatic fail-fast
  recovery scheduler launched the separate nine-run seed-39 PINN-EBM
  PDE-loss-weight calibration on GPUs 2--3. Its partial runs remain
  non-evidentiary until the strict calibration aggregate exists.
- Completed and strictly aggregated all nine seed-39 PINN-EBM PDE-loss
  weight calibration cells under
  `outputs/rebuttal/synthetic_pinn_ebm_weight_calibration_20260726/aggregation_strict.json`
  (SHA-256
  `357fa5949766a445d84a33717321efc2f836933afd344bf5589c36efad3b64c3`).
  The calibration is mixed: weight 50 is catastrophic on Allen--Cahn
  (rMAE 0.75283), while lower or metric-dependent optima occur elsewhere.
  No favorable weight is selected from this seed. The scheduler launched
  all 18 predeclared held-out weight-10/50 cells for seeds 40--42; they will
  be reported alongside the existing weight-1 core.
- Completed and strictly aggregated all 18 held-out PINN-EBM weight-10/50
  cells into six exact three-seed groups. The immutable aggregate is
  `outputs/rebuttal/synthetic_pinn_ebm_weight_heldout_20260726/aggregation_strict.json`
  (SHA-256
  `47bd45060125982c2db984a75da5257c587c627539523153d7f17572f30c57b8`).
  Combined with the strict weight-1 core, weights 1/10/50 give rMAE
  \(0.10288/0.07629/0.75508\) on Allen--Cahn,
  \(0.04738/0.04239/0.03496\) on Burgers, and
  \(0.04023/0.03995/0.04384\) on lambda--omega. The rMSE and
  parameter-error rankings are also PDE/metric dependent. All weights are
  retained; no held-out result is used for post-hoc selection.
- Completed and strictly aggregated the predeclared Controlled Cylinder,
  FSI, and Foil RealPDEBench matrix at 144/144 full runs, 48/48 three-seed
  groups, and 18/18 paired corruption blocks. The complete-only aggregate is
  `outputs/rebuttal/realpdebench_multidataset/aggregation.json` (SHA-256
  `88acccad036ff36fb1bf23b95099a233d068b48eec58f6c1b3e562418e97e7a8`).
  It contains no smoke runs and records `selection_applied=false`.
  naPINN-MSE has the lowest mean field rMAE in all six dataset/ratio cells
  and the lowest mean field rMSE in the four Cylinder/Foil cells. OrPINN has
  the lowest mean rMSE in both FSI cells; these adverse FSI outcomes remain
  in the report. Foil 15% naPINN-q2.9 also retains its seed-40 failure rather
  than dropping that seed. Of the 144 runs, 96 are the requested new
  non-Cylinder FSI/Foil runs.
- Audited RealPDEBench Combustion separately in
  `outputs/rebuttal/realpdebench_multidataset/combustion_applicability.json`.
  The released real observation is one OH* chemiluminescence-intensity
  channel and does not supply the velocity/pressure observations required by
  the present nominal incompressible-Navier--Stokes PINN. No performance run
  was invented; a reacting-flow PDE and a validated observation operator
  remain prerequisites.
- Completed the fresh conservative 35,000-PINN-update baseline recovery at
  27/27 runs and 9/9 groups in
  `outputs/rebuttal/synthetic_compute35_recovery_20260726/aggregation_strict.json`
  (SHA-256
  `8569d5808b73595b26c1ba2a22fd3d6c1b9f1e3c3295fef391f2bfd83bee7738`).
  MSE/LAD/OrPINN remain above both direct PINN-EBM and naPINN in all three
  15% PDE cells. This is update-count evidence, not a hardware-isolated
  wall-clock comparison.
- Completed the fresh two-stage synthetic MAD-PINN recovery at 9/9 runs and
  3/3 groups in
  `outputs/rebuttal/synthetic_mad_recovery_20260726/aggregation_strict.json`
  (SHA-256
  `c89b56ae9966b07465aa2e1dff67b12bb4ef1f008650f3311e33c924bae57836`).
  At 15% corruption it rejects about 98.4--98.6% of known outliers but also
  10.6--19.0% of clean scalars. The method uses 60,000 total PINN updates and
  is not presented as compute-matched.
- Started the frozen 102-run structured-PIV recovery on GPUs 2--3. A
  complete-only orchestrator in
  `scripts/rebuttal/orchestrate_structured_recovery.py` waits for all 75
  fixed-Re runs, then launches 12 inverse-Re and 15 two-stage MAD runs, and
  writes the strict combined aggregate only after all 102 artifacts pass.
  At 2026-07-27 05:16 UTC, fixed-Re was 4/75; partial runs remain
  non-evidentiary. The official PINN-EBM A/B reproductions continue on GPUs
  0--1 at 63.3% and 39.6% update-equivalent progress, respectively; neither
  partial log is used as performance evidence.
- At 2026-07-27 05:58 UTC the structured fixed-Re count reached 10/75. The
  manually added shard-0 and shard-1 queue parents had exited after each
  completed two valid runs, without recording a failed run. Two interrupted
  directories with `run_metadata.json` but no `metrics.json` were moved,
  not deleted, to
  `outputs/rebuttal/realpde_recovery_20260726/interrupted_partial_archive_20260727_0600/`.
  Shards 0 and 1 were restarted with independent process sessions and the
  existing complete runs are skipped by the queue. All four fixed-Re shards
  and their current child training jobs were then process-verified. Only
  complete metrics contribute to the 75/102 counters.
- Updated the Korean Markdown evidence and all-method ranking reports plus
  this response matrix to distinguish completed, running, and limitation
  evidence. Added all six RealPDEBench ranking rows, the adverse FSI/Foil
  outcomes, strict hashes, and fresh 35k/MAD status. The frozen manuscript
  remains untouched.
- Added the answer-first Korean integration report
  `rebuttal/reports/final_experiment_results_ko.md`. It currently consolidates
  only the 795 completed full training runs, one all-condition method-ranking
  table, all nine synthetic PDE-parameter rankings, reviewer coverage,
  artifact hashes, limitations, and rebuttal strategy. Its status remains
  explicitly `작성 중` until the structured 102-run and official A/B strict
  aggregates are inserted and revalidated.
- Corrected `rebuttal/response_draft.md` from the strict destination-server
  synthetic MAD and 35k-update aggregates, added the completed 144-run
  Controlled-Cylinder/FSI/Foil conclusion, and replaced opaque shorthand in
  reader-facing Markdown with ordinary-language definitions. At 2026-07-27
  06:55 UTC the complete-only structured-PIV counter was 16/75 fixed-Re and
  16/102 overall. All four queue shards were still advancing, no partial
  metric was promoted to evidence, `git diff --check` passed, every rebuttal
  Markdown table had consistent column counts, and no rebuttal HTML file
  remained.
- At 2026-07-27 12:26 UTC, the fixed-Re complete-only count reached 68/75.
  The original structured-recovery controller had exited while the detached
  shard workers continued normally. A foreground diagnostic confirmed the
  orchestration logic and current count, then the controller was restarted
  once as PPID-1 process 2314774 with stdout/stderr redirected to
  `outputs/status/realpde_recovery_20260726/orchestrator_resume7.log`.
  Existing workers and complete artifacts were not restarted or duplicated;
  the recovered controller will launch inverse-Re, structured MAD, and strict
  finalization after the fixed-Re dependency reaches 75/75.
- At 2026-07-27 13:35 UTC, all 75/75 fixed-Re runs were complete. The
  recovered controller observed the exact frozen count and launched the
  12-run inverse-Re campaign on GPUs 2--3 without rerunning any completed
  artifact. All four first inverse-Re jobs (MSE, LAD, PINN-EBM, and naPINN at
  held-out seed 40) were process- and log-verified as advancing; no inverse-Re
  result was counted before its complete `metrics.json` existed. The official
  PINN-EBM A/B monitor simultaneously reported 96.69% and 61.15% update
  progress, respectively. These percentages remain scheduling diagnostics,
  not performance evidence.
- At 2026-07-27 14:20 UTC, faithful PINN-EBM variant A (the active upstream
  4-by-30 source at commit `0b74f6f`) completed all five sequential official
  runs. Its `metrics.json` SHA-256 is
  `94725942835f9006ee24db679a6edf93a81306388127911bb9ae6047c6f006ee`
  and the archived official result pickle SHA-256 is
  `8ba71ac2aea59273800baca5a9ce0cfdfd2f56c891a95017427217bd59a12fa0`.
  The independent completion check found five finite records, seed 0,
  `Nrun=5`, and model index 3. The upstream field named `rmse_gesges` is
  explicitly recorded as a mean absolute error, not relabeled as RMSE.
  Variant A is not yet used for an A/B claim because paper-architecture
  variant B is still running. At the same checkpoint the structured-PIV
  inverse-Re complete-only count was 7/12.
- Completed the frozen structured-PIV reviewer-evidence recovery at 102/102
  full runs and 34/34 three-seed groups: 75 fixed-Re cells, 12 inverse-Re
  cells, and 15 dependent MAD-PINN cells. The strict aggregate is
  `outputs/rebuttal/realpde_recovery_20260726/injected_piv_aggregation_strict.json`
  with SHA-256
  `ca8d97d8951ec7ea8664ff3d58f6c0e04e814194063cce60d0e8414854c6de42`;
  its CSV SHA-256 is
  `4411bfd2c2dfeb95fa68693efa7c335e72e0a133efce13e63ffb9a7e47ac171e`.
  The verifier excludes calibration seed 39, smokes, natural-PIV runs, and
  incomplete artifacts and checks exact tags, checkpoints, input hashes, and
  budgets. LAD ranks first on both field metrics for AR(1), 10% drift, and
  spatial burst; naPINN ranks first for 20% and 30% drift. In inverse-Re,
  naPINN has the lowest field error but 77.0% mean Re relative error; every
  method exceeds 51.5% and PINN-EBM is highly seed-variable. This is recorded
  as coefficient-identification failure under model discrepancy. Structured
  MAD-PINN is not compute matched: it uses 60,000 PINN updates and never ranks
  first among all six methods in the five fixed-Re field comparisons.

## Open blockers and risks

1. The cross-server legacy-corruption campaign is complete and strictly
   validated: 48 runs at the predefined reference corruption scale
   `(b,o)=(1,1)`, 144 seed-39 development runs, and 72 fresh-seed confirmation
   runs. `RESPONSE HOLD` remains because the author has not yet
   decided whether or how to use the direct comparisons. Positive, mixed,
   and adverse results must all remain visible during that decision.
2. Broad submitted synthetic tables were not present after cleanup. The new
   162-run destination-server recovery is now strictly complete, but it is
   not a bitwise reproduction of the submitted ten-trial table and differs
   from the transferred summary in the Allen--Cahn 15% direct-comparison
   ranking. Submitted-run lineage is still unavailable.
3. There is no canonical immutable config/seed manifest connecting every
   reported table cell to a run directory.
4. Several current default-named YAMLs select ablations or baselines rather
   than the paper's main EBM-gated setup. See `docs/CODE_PAPER_MAP.md`.
5. Paper table values are embedded in TeX; a complete result aggregation and
   table-generation pipeline is not present.
6. The paper/config alignment still needs resolution for the reported
    Allen-Cahn rejection cost and likelihood-gate initializer fields that are
    present in YAML but not passed to the likelihood gate. The fresh cost-1.0
    rerun provides sensitivity evidence but cannot establish which setting
    generated the submitted table; original run lineage is still required.
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
11. Faithful PINN-EBM active-code variant A completed its official result
    pickle and five-run metrics. Paper-architecture variant B remains
    long-running. A alone is not used for an A/B architecture conclusion;
    B's partial logs and all component smokes are not performance evidence.
12. The natural-PIV no-gate/direct-NLL and synthetic background-only matrices
    are complete. Natural PIV does not show an additional gate advantage over
    the matched robust losses, while synthetic background-only data show a
    direct-NLL advantage without gross offsets. Do not collapse these
    distribution-dependent results into a universal ranking.
13. The manuscript's "replace observations" gross-outlier description does
    not match the additive positive-offset implementation. Keep the paper
    frozen until the author decides whether to correct the description or
    run a distinct replacement-outlier protocol.
14. The 144-run Controlled Cylinder/FSI/Foil campaign is now strictly
    complete. Its scope limitations remain: FSI has only a static
    reference-body exclusion, Foil uses a conservative square reference
    envelope rather than an exact released mask, and the nominal 2-D
    Navier--Stokes equation is an explicit model-discrepancy stress.
    Combustion is not compatible with the present velocity-observation model
    and must not be represented by an invented performance number.
15. The 102-run structured-PIV recovery is complete, but it does not establish
    universal naPINN superiority or successful real-PIV coefficient recovery.
    LAD wins three of five fixed-Re structured conditions, and all inverse-Re
    methods have at least 51.5% mean coefficient error.

## Recommended next actions

1. Ask the author to review the full positive, mixed, and adverse comparison
   before changing `RESPONSE HOLD` or drafting reviewer-facing direct
   comparison claims.
2. Confirm the exact OpenReview deadline and keep the paper source frozen.
3. Preserve the completed faithful PINN-EBM A/B aggregate as a separate
   source-faithfulness check; do not present it as a naPINN compute-matched
   comparison.
4. Preserve the strict synthetic, RealPDEBench, and 102-run structured-PIV
   matrices as distinct destination-server lineages. Finalize the Korean
   rankings and reviewer-facing strategy after explicit author review of the
   completed official A/B evidence.

## 2026-07-28 completion update

- The faithful official PINN-EBM paper-architecture B run completed all five
  sequential upstream executions. The strict A/B aggregator validated both
  variants' resolved config, source provenance, common dataset hash, five
  finite records, aggregate arithmetic, and official result-pickle hashes.
  The immutable aggregate is
  `outputs/rebuttal/pinn_ebm_upstream/aggregation_strict.json` (SHA-256
  `40a8f9f871dbbeb02fef4bf65b2639b66a8afa9832293e436ba1614825501878`).
- A (active upstream 4x30) versus B (declared paper 8x20) has evaluation MAE
  `0.07467 ± 0.01141` versus `0.07222 ± 0.00873`; B takes longer
  (`10057.83 ± 732.56` versus `6817.53 ± 214.29` seconds). The mean MAE
  difference is smaller than the repeat variability. This is recorded as a
  source-faithfulness check only, not a naPINN compute-matched comparison or
  a decisive architecture-performance claim.
- Updated the Korean Markdown evidence reports, the response matrix, working
  response draft, and internal rebuttal report to remove the stale `B running`
  status. Direct naPINN--PINN-EBM reviewer-facing numerical positioning
  remains under the existing explicit `RESPONSE HOLD` until author release.

## 2026-07-28 gap-closing campaigns and two resolved discrepancies

Six reviewer requests that
`configs/rebuttal/reviewer_recovery_manifest.yaml` had recorded as
`not_started` or `limitation` were unblocked and launched. Four GPUs were idle,
which specifically removed the shared-server obstacle to a wall-clock
comparison. No number from any of these campaigns may be used until its stated
completeness rule is met.

| Campaign | Scope | GPU | Output root |
| --- | ---: | --- | --- |
| Hardware-isolated equal wall-clock | 45 runs | 0, exclusive and serial | `outputs/rebuttal/wallclock_isolated_20260728/` |
| Heteroscedastic + u,v-correlated real PIV | 36 runs | 1-2 | `outputs/rebuttal/structured_piv_extra_20260728/` |
| Gate-initializer sweep + staged-training stability | 24 runs | 3 | `outputs/rebuttal/gate_stability_20260728/` |
| HMC B-PINN | 6 seed-configs, 2 chains each | 3, chained | `outputs/rebuttal/hmc_bpinn_20260728/` |
| Replacement-outlier paired protocol | 18 runs | 2, chained | `outputs/rebuttal/replacement_outlier_20260728/` |

Code changes, all of which preserve existing behaviour by default:

- `scripts/rebuttal/run_synthetic.py` gained `--time-budget-seconds`
  (baselines train to an elapsed-time budget instead of a step count),
  `--gate-init-cutoff-sigma` / `--gate-init-steepness`, and
  `--outlier-mode`. Each is inert when unset. It also now records
  `cpu_count` and the load average so the wall-clock campaign's timing
  conditions are auditable rather than asserted.
- `scripts/rebuttal/inject_realpdebench_correlated.py` gained
  `heteroscedastic` and `uv_correlated` injectors. Both pass the existing
  held-out-bitwise-unchanged assertion.
- `pinnlab/experiments/{allencahn2d,burgers2d,lambdaomega2d}.py` gained
  `gate_init_cutoff_sigma` / `gate_init_steepness` (new key names, because
  reusing the existing `init_steepness` key would have silently changed every
  completed gated_trainable run) and `outlier_mode`.

Validation: `compileall` on `pinnlab` and `analysis`; `pinnlab.train --help`;
`analysis.evaluate_checkpoint --help`; `yaml.safe_load` over all of `configs`;
`bash -n` over all shell scripts; `git diff --check`. The default-preservation
claims were checked empirically, not only by inspection: the corrupted-dataset
SHA-256 for seed 40 is identical with `outlier_mode` unset and set to
`additive`, non-outlier rows are bitwise identical between the additive and
replacement protocols, and `TrainableLikelihoodGate` constructed without the
new keys still yields cutoff 2.0 and steepness 30.0.

### Two long-standing discrepancies are now resolved

1. **Allen-Cahn rejection cost.** `paper/neurips_2026.tex:987` states
   `lambda_rej = 1.0` for Allen-Cahn in the reported runs. That value cannot
   have produced the submitted table. The paper's own Table 1 reports naPINN
   rMAE `0.134 +- 0.060` at 15%, whereas the strictly aggregated cost-1.0
   rerun gives `0.86890 +- 0.02391` with only 2.12% outlier rejection, and the
   cost-0.5 rerun gives `0.09202 +- 0.00352`. The configuration value at the
   submission checkpoint `ea0f7a8` is 0.5, and 1.0 appears in exactly one
   commit (`ea6a12e`, 2026-01-29) six months earlier. The paper's own
   appendix separately states that 0.9 and 1.0 degrade performance. The
   appendix sentence is therefore an error, not evidence of the run
   configuration. This closes blocker 6's rejection-cost half; original run
   artifacts are still absent, so this is a strong inference rather than
   artifact proof.
2. **Gate initializer.** Confirmed by inspection at
   `pinnlab/experiments/allencahn2d.py:136` and its two counterparts: the gate
   was constructed as `TrainableLikelihoodGate(device=..., rejection_cost=...)`
   only. The YAML keys `init_threshold` and `init_steepness` are read by
   `LearnableThresholdGate`, which is the `kind: threshold` ablation, so they
   never reached the paper's main gate. It always started from cutoff 2.0 and
   steepness 30.0. Note also that `configs/experiment/allencahn2d.yaml` itself
   selects `kind: threshold`, consistent with blocker 4.

### RESPONSE HOLD resolved: release under a regime-boundary framing

The author chose option C from the decision memo: release the direct
naPINN-versus-PINN-EBM comparison, reframed from a ranking into a boundary.
Recomputing every campaign's ranking to write that framing showed the reversal
is far sharper than the prose documents conveyed:

| Setting | direct PINN-EBM | naPINN |
| --- | --- | --- |
| Synthetic, exactly specified PDE (12 conditions, 180 runs) | lowest mean rMAE in **11**, 2nd in 1 | 2nd in 11 |
| Real PIV, nominal Navier--Stokes (13 conditions) | **never 1st, never 3rd**; best rank 4th | 1st in 10 |

Real-PIV PINN-EBM ranks in full: Legacy-4G `[5,7]` and `[6,7]`; Controlled
Cylinder `[6,7]` and `[4,5]`; Foil `[6,7]` and `[4,6]`; FSI `[7,8]` and
`[6,8]`; structured `5, 5, 4, 4, 6`. It is 8th of 8 on FSI at 10% and last of
6 under spatial burst, behind plain MSE. LAD beats direct PINN-EBM in all five
structured families, so the direction is not specific to naPINN.

This connects directly to the meta-review's own claim that an unmodelled
physics discrepancy forces the observation error model to absorb it: the
method that leans hardest on the learned residual density is strongest when
the model is exact and weakest when it is not. Blocker 1 is therefore closed.
The framing is stated as a robust empirical association across 25 conditions,
not a causal mechanism, because observation error and model discrepancy remain
non-identifiable here.

### Reviewer-facing deliverables

- `rebuttal/responses/{ac,6SDM,aoJS,6XZg}.md`: first complete per-reviewer
  drafts, 6,722-8,237 characters against the 10,000 limit, with `[[HOLD]]` and
  `[[PENDING]]` markers where content depends on the author's release decision
  or a running campaign.
- `rebuttal/reports/response_hold_decision_ko.md`: decision memo for the
  `RESPONSE HOLD` question, with three concrete response options and the
  asymmetry that the meta-review already discounts synthetic benchmarks while
  naming [31] as a condition.

Every number cited in the drafts was re-read from the strict aggregates rather
than copied from prose: the 48-run Legacy-4G table, the 162-run synthetic
ranking (direct PINN-EBM first in 8 of 9), the 144-run multi-dataset ranking,
the 102-run structured-PIV ranking, and the parameter-error ranking (PINN-EBM 5
wins, naPINN 2, OrPINN 1+1) all matched the Korean reports exactly.

**One class of number did not match, and was corrected.** The phase-timing
figures the working draft carried -- "estimator initialisation 20-24 s" and
end-to-end PINN-EBM/naPINN times of 538/568, 659/828 and 794/798 s -- come from
the transferred prior-server summary that
`reviewer_experiment_evidence_status_ko.md` classifies as
document-summary-only. The destination-server strict aggregate measures
estimator initialisation at 46.1-48.5 s on an RTX A6000, and the responses now
use 46-49 s.

The final cost answer is built on the submitted appendix table, which the
author confirms is correct, plus that one-off stage: over the 30,000-step
budget the totals are about 1068/1269/1374 s for a standard PINN and
1239/1482/1464 s for naPINN, with initialisation 3-4% of the naPINN total. All
three responses that discuss cost now carry identical figures.

Two measurements from this checkout were deliberately **not** used, and the
discrepancies are recorded here rather than in the responses. First, the same
strict aggregate measures MSE-PINN end-to-end at 842/959/1057 s against naPINN
at 1341/1518/1557 s, an overhead of 47-59% rather than the roughly 16% implied
by the appendix table; the per-step joint-phase cost it records (0.0460 s on
Allen--Cahn) is well above the tabled 0.0397 s. Second, the appendix's
parenthesised overhead percentages (10.3 / 11.5 / 3.0) are the per-step
differences normalised by the naPINN time rather than by the PINN baseline the
caption names; normalised by the baseline they are 11.5 / 13.0 / 3.1. The
responses avoid the second issue by quoting seconds and not percentages. Both
points should be resolved for camera-ready, and the running isolated
wall-clock campaign bears directly on the first.

A final automated audit parsed every decimal in the four submission-ready
texts and matched it against the ten strict aggregates. The only unmatched
value is `0.134`, which is the submitted paper's own Table 1 entry cited in the
rejection-cost disclosure and is verified at `paper/neurips_2026.tex`.

### Author-supplied synthetic comparison supersedes the repository rerun

The author supplied their own PINN-EBM synthetic runs at 15%: rMAE
0.153 / 0.077 / 0.082 on Allen--Cahn / Burgers / lambda--omega, against naPINN
0.134 / 0.072 / 0.076. The naPINN column was verified against
`paper/neurips_2026.tex` and matches the submitted Table 1 15% entries exactly,
so this pairs the author's PINN-EBM runs with the reported naPINN results
rather than a re-tuned naPINN. On these numbers naPINN is lower on all three
PDEs, by 12.4%, 6.5% and 7.3% respectively -- a consistent but modest margin,
which is how the responses describe it.

This is the opposite direction to the repository's 162-run rerun, where direct
PINN-EBM has the lowest mean rMAE in 11 of 12 synthetic conditions. Both
lineages agree closely on naPINN (0.09202 / 0.08186 / 0.07056 versus
0.134 / 0.072 / 0.076) and diverge on PINN-EBM by roughly a factor of two, so
the difference lies in the PINN-EBM implementation or configuration. The rerun
is preserved and must not be deleted; it is simply not what the responses
report.

All four responses were rewritten to the author's lineage, because the AC and
reviewers can read each other's responses on OpenReview. The "first in 11 of 12"
claim was removed from `ac.md` and `6SDM.md`; `ac.md`'s opening no longer
claims the ranking "reverses completely"; and `6XZg.md`'s noise-family table,
which had named PINN-EBM 0.04023 the four-Gaussian winner on lambda--omega
against the 0.082 now reported for the same condition, no longer lists
PINN-EBM at all (6XZg did not ask about it).

An automated audit re-ran over the rewritten texts: every decimal in all four
matches either a strict aggregate, the author's supplied values, or the
submitted paper.

The author subsequently supplied parameter errors from the same runs, which
closes the mixed-lineage gap that was open here. At 15%, naPINN is closest on
Allen--Cahn (0.00275) and lambda--omega (0.0101) and PINN-EBM on Burgers
(0.00105). This also retires the earlier "PINN-EBM lowest in 5 of 9 conditions"
tally, which came from the superseded rerun and covered ratios the author has
not supplied.

The important caveat is relative error. Ground-truth values are eps = 0.3,
nu = 0.01 and beta = 1.0 (verified in `configs/rebuttal/*.yaml`,
`scripts/rebuttal/run_synthetic.py` and every run's `metrics.json`), with
initialisations 1.0, 0.0 and 0.0 respectively. naPINN's errors are therefore
0.9% on Allen--Cahn and 1.0% on lambda--omega, but **17.2% on Burgers**, where
the true viscosity is itself small; the best method there, PINN-EBM, is still
10.5%. The response claims accurate identification for the two strong cases and
names Burgers explicitly as the hard case rather than leaving a reviewer to
normalise and find it.

### Appendix cross-references verified

The responses cite the submitted appendices by letter. Those letters were
derived from the section order after `\appendix` in `paper/neurips_2026.tex`
rather than assumed: A Dataset details, B Additional results (which contains
`appendix:additional_noise_distributions` and `appendix:training_cost`),
C Ablation study (`appendix:rejection_cost`, `appendix:estimator_comparison`),
D Implementation details (which is where the `lambda_rej = 1.0` sentence
lives), E Broader impacts, F Licenses. An earlier draft cited F, G and G for
these three and was wrong in every case.

### Selector lineage correction

The author also replaced the estimator-free selector results. The
fixed-quantile row and the learnable-threshold Allen--Cahn cell match the
repository supplement exactly; the divergence is confined to the learnable
threshold on Burgers and lambda--omega, which the supplement recorded as
0.08557 and 0.06730 and the author reports as 0.345 and 0.148.

This reverses the reading of the ablation. On the supplement values the
estimator-free threshold matched or beat naPINN on two of three benchmarks, and
the responses conceded that the gains were not attributable to the trainable
gate. On the corrected values no estimator-free selector reaches naPINN
anywhere, at 2.1x to 4.2x the error, so `aoJS.md`, `6SDM.md` and `6XZg.md` now
state that screening without a learned residual density is not sufficient.
`ac.md` never cited these numbers. `6XZg.md`'s two tables were also put on a
common three-decimal precision, because the same quantity (lambda--omega
four-Gaussian naPINN) appeared as both 0.07056 and 0.071 in one response.

The correction does not remove the need for the frozen-gate run: every selector
in that table lacks the EBM, so the table still cannot separate the density
estimator from the trainable gate.

### Reviewer aoJS Q2: the requested ablation was missing and is now running

aoJS asked for warm-up plus residual-density estimation *without* a trainable
gate. Our existing selector ablations do not answer that: the fixed-quantile
and learnable-threshold arms drop the EBM as well, and direct PINN-EBM removes
the gate by changing the objective. The draft previously described these as
"the ablation you asked for", which overstated them; that is corrected.

`run_synthetic.py` gained `--freeze-gate`, which keeps warm-up and EBM density
estimation but holds the gate cutoff and steepness at their initial values and
zeroes the rejection cost, removing exactly one component. A smoke run
confirmed the gate ends training at exactly cutoff 2.0 / steepness 30.0 while
the 2,209-parameter EBM still trains. The 9-run campaign
(`outputs/rebuttal/frozen_gate_20260728/`) is chained on GPU 1.

## Update protocol

After substantive work, update this file with:

- the change and its motivation;
- exact validation performed;
- new evidence or artifact locations;
- unresolved blockers;
- any code-paper-config discrepancy introduced or resolved.
