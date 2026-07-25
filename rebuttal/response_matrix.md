# NeurIPS 2026 rebuttal evidence matrix

This tracker covers reviewer communication and new experiments only.
`paper/neurips_2026.tex` is frozen by user instruction and must not be edited.
Natural/unmodified PIV results are excluded from the rebuttal evidence tables.

Status vocabulary:

- `VERIFIED`: a complete non-smoke artifact exists and was aggregated.
- `RUNNING`: a frozen held-out run is active.
- `READY`: implementation and smoke validation are complete, but the full run
  has not started.
- `LIMITATION`: the response will acknowledge the point without claiming an
  experiment that was not completed.
- `RESPONSE HOLD`: evidence is verified internally, but numerical results,
  rankings, and the resulting response position are not approved for reviewer
  communication pending additional experiments and an explicit author
  decision.

## Strategy decisions

1. Do not claim that no public noisy PDE data exist. RealPDEBench explicitly
   describes measurement noise in real PIV. The defensible gap is narrower:
   we did not identify a public inverse-PDE benchmark that jointly supplies
   known corruption labels, a clean paired physical reference, governing PDE,
   and unknown coefficients.
2. Report only controlled corruptions injected into real PIV training
   measurements. The unmodified held-out PIV measurements are used as an
   independent reference, not called noise-free ground truth.
3. Use PIV seed 39 for calibration and seeds 40--42 for final reporting.
   Retain every completed held-out cell and do not choose by test
   performance. Reviewer-facing use of direct naPINN-versus-PINN-EBM cells is
   subject to `RESPONSE HOLD`.
4. Discuss Pilar--Wahlström only where the Area Chair or reviewer explicitly
   raises it. In that context, accurately acknowledge the shared EBM and
   staged components, then clearly distinguish direct EBM-NLL optimization
   from naPINN's detached density objective, explicit per-measurement gate,
   gated base reconstruction loss, and rejection regularization. Do not
   volunteer a global reduction of the contribution claim elsewhere.
5. All direct naPINN-versus-PINN-EBM numbers, rankings, and resulting
   response claims are on `RESPONSE HOLD`. Preserve the verified evidence
   internally and complete the planned PIV legacy-corruption, scale,
   robust-loss, and closest-prior checks before asking the author to choose
   the response direction. If later released, do not say PINN-EBM failed or
   omit adverse cells.
6. Describe the submitted Bayesian baseline as B-PINN-VI, not HMC. A rushed
   full-network HMC reimplementation is outside defensible rebuttal evidence.

## Reviewer-to-evidence tracker

| ID | Concern | Planned evidence or response | Status |
| --- | --- | --- | --- |
| AC.1 / 6SDM.1 | No real or semi-realistic experiment | RealPDEBench Cylinder PIV with injected persistent sensor drift, AR(1) temporal corruption, and spatial burst corruption; seeds 40--42 | Persistent-severity and correlated baseline matrices `VERIFIED` |
| AC.2 / 6SDM.2 | Observation error plus physics-model discrepancy | Apply nominal incompressible Navier--Stokes residual to real PIV; explicitly state that observation and model discrepancy are not separately identifiable | `VERIFIED` as a stress test; no discrepancy-identification claim |
| AC.3 / aoJS.1 | Pilar--Wahlström is the closest prior | Complete further direct no-gate comparisons, preserve existing verified evidence, and decide final response wording only after author review | Existing matrices `VERIFIED`; numerical response use `RESPONSE HOLD` |
| 6SDM.3 | MAD-PINN, learned-noise EBM, tuned robust losses | MSE, LAD, OrPINN q=1.9/2.9, PINN-EBM, and naPINN core; faithful two-stage MAD-PINN as a 60k-update non-compute-matched reference | Severe synthetic and all injected-PIV MAD variants `VERIFIED` |
| 6SDM.4 | EMA coefficient and beta/rho confusion | State exact convention: implementation uses update weight `m`, so decay is `1-m`; run `m={0.01,0.05,0.10}` | `VERIFIED` |
| 6SDM.4b / 6XZg.4 | Allen rejection-cost paper/config mismatch | Run submitted-appendix value 1.0 for every ratio/seed, retain initial 0.5 cells as sensitivity, and disclose that appendix sensitivity text also calls 0.9/1.0 degrading | Five-value severe sweep and 5/10/15% paper-aligned extension `VERIFIED` |
| 6SDM.5 / aoJS.4 / 6XZg.6 | End-to-end cost | Record 5k warm-up + 5k estimator-only + 25k joint, phase times, total time, and peak GPU memory; call shared-A100 wall time observational | Phase accounting and 35k conservative update-count matrix `VERIFIED` |
| aoJS.2 | No-gate and fixed-screening ablations | Direct PINN-EBM, estimator-free fixed-quantile and learnable-threshold gates, and published fixed-screen MAD-PINN | Ablations `VERIFIED`; direct-comparison response use `RESPONSE HOLD` |
| aoJS.3 | PDE parameters for all benchmarks | Record Allen--Cahn epsilon, Burgers viscosity, and lambda--omega beta in every synthetic full run | 5/10/15% core `VERIFIED` |
| AC.1 / aoJS.3b | Real example is otherwise reconstruction-only | Separate 30% injected-PIV inverse-Re extension, initialized at 8000 with metadata reference 10031; disclose effective-coefficient/model-mismatch caveat | `VERIFIED` (adverse result) |
| aoJS.5 / 6XZg.7 | Baydin reference year | Acknowledge the verified 1989/2018 typo in the response; do not edit frozen paper source | `LIMITATION` |
| 6XZg.1 | Mean-field B-PINN is weaker than HMC | Concede; identify the submitted implementation as B-PINN-VI and avoid an HMC-equivalence claim | `LIMITATION` |
| 6XZg.2 | Bayesian UQ discussion is overstated | Clarify in response that Bayesian inference and adaptive selection are complementary, not mutually exclusive | `LIMITATION` |
| 6XZg.3 | Basic statistical preprocessing | Fixed-quantile gate and faithful fixed-screen MAD-PINN pipeline; separately label the residual-threshold gate as trainable | `VERIFIED` |
| 6XZg.4 | Stability and hyperparameter optimization | Report staged-training evidence already in submission; new EMA and rejection-cost sweeps; do not claim exhaustive optimization | EMA and severe rejection-cost sweeps `VERIFIED` |
| 6XZg.5 | Same noise distribution | New Gaussian, Laplace, Student-t, four-Gaussian, sensor drift, AR(1), and spatial burst matrix | `VERIFIED` |

## Verified evidence so far

**Author decision:** all direct naPINN-versus-PINN-EBM numerical comparisons
in this section are retained as internal verified evidence under
`RESPONSE HOLD`. Do not copy their values, rankings, or derived
contribution/novelty conclusions into a reviewer response until the planned
additional comparison campaign is complete and the author explicitly
releases them.

For 10% persistent bias plus linear drift injected into real PIV training
sensors, seeds 40--42 are complete for MSE, LAD, no-gate PINN-EBM, and the
earlier naPINN rejection-cost-0.01 sensitivity:

| Method | held-out rMAE | held-out rMSE | momentum RMS | failure AUROC |
| --- | ---: | ---: | ---: | ---: |
| MSE | 0.21182 ± 0.00261 | 0.28836 ± 0.00888 | 0.04915 ± 0.00101 | -- |
| LAD | **0.13212 ± 0.00251** | **0.20881 ± 0.00157** | 0.05127 ± 0.00064 | -- |
| PINN-EBM | 0.26545 ± 0.08838 | 0.35567 ± 0.08225 | 0.04847 ± 0.01452 | -- |
| naPINN | 0.16928 ± 0.00131 | 0.26378 ± 0.00335 | **0.00832 ± 0.00082** | 0.96673 ± 0.00149 |

naPINN rejects 98.86% ± 0.38% of corrupted scalars but also 33.68% ±
7.62% of clean scalars. LAD has the best held-out field agreement. Natural
PIV results are excluded.

The calibration-selected cost-0.10 rerun is also complete at 10% failure.
naPINN rMAE/rMSE is
\(0.15501\pm0.00119/0.23226\pm0.00463\), below MSE, OrPINN, and direct
PINN-EBM but above LAD. Its gate rejects
\(84.97\%\pm1.90\%\) of corrupted and \(3.63\%\pm0.89\%\) of clean
scalars. The 30%-calibration failure-recall constraint therefore does not
transfer perfectly across severity.

At 20% persistent failure, the complete rMAE/rMSE comparison is
\(0.29806/0.37994\) MSE, \(0.17195/0.25329\) LAD,
\(0.23768/0.32710\) OrPINN, \(0.20534/0.29317\) direct PINN-EBM, and
\(0.16889/0.24869\) naPINN. naPINN is about 1.8% below LAD on both field
metrics and has substantially lower nominal-physics residuals. Its gate
rejects 91.25% corrupted and 9.16% clean scalars; direct raw-EBM AUROC is
higher than gate AUROC.

The complete seed-39 30% calibration selected rejection cost 0.10. It was the
only candidate satisfying failed rejection at least 90% and clean rejection
at most 40% (93.79% and 18.56%). The full held-out 30% comparison is now
complete. naPINN rMAE/rMSE is 0.21194 ± 0.01783 /
0.30034 ± 0.02341 versus MSE 0.37100 / 0.49065, LAD
0.23558 / 0.35840, OrPINN-q2.9 0.29745 / 0.41328, and direct PINN-EBM
0.20329 / 0.31052. Thus the closest-prior result is mixed: direct has lower
rMAE, while naPINN has lower rMSE and much lower nominal-physics residuals.
naPINN gate AUROC is 0.93445 ± 0.01306, with 93.22% corrupted and 28.64%
clean-scalar rejection; direct raw-EBM AUROC is higher at
0.95046 ± 0.00695. The record therefore claims a field/physics trade-off,
not universal superiority. The earlier 10% cost-0.01 cells predate the
0.05/0.10 grid refinement, so the calibration is not called fully
held-out-blind.

The decisive three-seed 15% comparison is complete for PINN-EBM and naPINN:

| PDE | PINN-EBM rMAE | naPINN rMAE | PINN-EBM parameter error | naPINN parameter error |
| --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | **0.08334 ± 0.01430** | 0.10060 ± 0.02587 | 0.00272 ± 0.00113 | 0.00273 ± 0.00027 |
| Burgers | **0.04783 ± 0.01230** | 0.08179 ± 0.00335 | **0.00080 ± 0.00028** | 0.00271 ± 0.00024 |
| lambda--omega | **0.03976 ± 0.00158** | 0.06989 ± 0.00132 | **0.00966 ± 0.00366** | 0.01340 ± 0.00057 |

The prior also has slightly higher raw-estimator AUROC on all three PDEs.
Thus the response must not imply that only the trainable gate can rank
corruptions. The narrower distinction is an explicit learned inclusion
decision with rejection regularization. This adverse result is final for
these six groups and must be disclosed.

The complete four-Gaussian core now contains every 5%, 10%, and 15% cell:
three PDEs, six methods, and seeds 40--42 (162 held-out runs). Direct
PINN-EBM has the lowest field rMAE in all nine PDE/ratio conditions, with
naPINN second in all nine. Parameter-error rankings are mixed. The synthetic
aggregator now excludes calibration seed 39 whenever required held-out seeds
40--42 are requested, while retaining the excluded source paths in metadata.
The full held-out PINN-EBM PDE-weight grid is also complete. Weights 1/10/50
give rMAE \(0.08334/0.08226/0.78955\) on Allen--Cahn,
\(0.04783/0.04776/0.04122\) on Burgers, and
\(0.03976/0.03708/0.04551\) on lambda--omega. The response retains the
catastrophic Allen weight-50 cell and does not select a single favorable
weight post hoc.

The lambda--omega Gaussian/Laplace/Student-t/four-Gaussian matrix is
complete for MSE, LAD, OrPINN, PINN-EBM, and three naPINN backbone losses.
naPINN-MSE is best for Gaussian, naPINN-LAD for Laplace, and direct
PINN-EBM for Student-t and four-Gaussian. Base naPINN is worse than MSE on
Student-t, and the LAD/q2.9 naPINN backbones degrade sharply on
four-Gaussian noise. The response claims broader coverage, not
noise-family invariance.

The 30% injected-PIV inverse-Re extension is complete. Relative to metadata
Re 10031, learned Re is \(1309\pm155\) for MSE,
\(4133\pm506\) for LAD, \(4651\pm2561\) for direct PINN-EBM, and
\(2284\pm533\) for naPINN. naPINN has the lowest nominal-physics residual
but worse coefficient recovery than LAD/direct. This adverse result is
reported as observation/model-discrepancy confounding, not successful
metadata-parameter identification.

## Compute accounting

- MSE/LAD/OrPINN: 30,000 PINN updates.
- PINN-EBM and naPINN: 5,000 MSE warm-up PINN updates, 5,000
  estimator-only updates, then 25,000 joint PINN updates. This is 30,000 PINN
  updates plus 5,000 estimator-only updates.
- MAD-PINN: 30,000 LAD updates, screening, then 30,000 retained-data MSE
  updates. It is explicitly a 60,000-update, non-compute-matched reference.
- All wall times are measured on a shared server and are not
  hardware-isolated benchmarks.

For the 30% injected-real-PIV condition, the complete MAD-PINN aggregate is
rMAE \(0.25127\pm0.00817\) and rMSE \(0.37205\pm0.00738\), versus LAD
\(0.23558\pm0.00678/0.35840\pm0.00590\) and naPINN
\(0.21194\pm0.01783/0.30034\pm0.02341\). Its fixed screen rejects 80.45%
of known corrupted and 20.30% of clean scalars. It uses 60,000 PINN updates.

The conservative 35,000-full-PINN-update matrix is complete for
MSE/LAD/OrPINN-\(q=2.9\). Their rMAE is
\(0.88440/0.34645/0.30771\) on Allen--Cahn,
\(0.59215/0.24871/0.19923\) on Burgers, and
\(0.44750/0.18358/0.14191\) on lambda--omega. Every value remains above
both direct PINN-EBM and naPINN at 30,000 PINN updates plus 5,000
estimator-only updates. The artifact is
`analysis/results/runs/rebuttal_synthetic_compute35_aggregation.json`.

The estimator-free fixed-quantile rMAE is
\(0.54862/0.37395/0.23718\) on Allen--Cahn/Burgers/lambda--omega. The
learnable-threshold gate gives \(0.22245/0.08493/0.06644\), compared with
naPINN \(0.10060/0.08179/0.06989\) and direct PINN-EBM
\(0.08334/0.04783/0.03976\). The mixed threshold result is retained; it is
not described as a fixed statistical rule.

The paper-aligned Allen--Cahn rejection-cost-1.0 extension is complete at
all corruption ratios. rMAE is \(0.25878/0.56180/0.87859\) at
5%/10%/15%, compared with \(0.09621/0.10159/0.10060\) for the initial
cost-0.5 core. Cost 1.0 rejects only 2--3% of known outliers at every ratio.
This adverse configuration discrepancy is explicitly disclosed.

The complete correlated-PIV baseline matrix is mixed. AR(1)
MSE/LAD/OrPINN/PINN-EBM/naPINN rMAE is
\(0.22047/0.12330/0.14541/0.20891/0.14997\), while spatial-burst rMAE is
\(0.15981/0.11358/0.12634/0.23979/0.15167\). LAD is best on both field
comparisons; naPINN has the lowest nominal physics residuals. The naPINN
gate rejects 64.97% of AR(1) corrupted scalars and 100% of spatial-burst
corrupted scalars, showing incomplete calibration transfer.

At 20% persistent failure, MAD-PINN rMAE/rMSE is
\(0.17087/0.25418\), nearly matching LAD \(0.17195/0.25329\) and
slightly above naPINN \(0.16889/0.24869\). Its fixed screen rejects 91.60%
of corrupted and 25.42% of clean scalars.

At 10% persistent failure, MAD-PINN rMAE/rMSE is
\(0.15850/0.24077\), slightly above naPINN and above LAD. It has lower
nominal-physics residuals than naPINN but rejects 30.30% of clean scalars,
versus 3.63% for naPINN.

Under AR(1), MAD-PINN rMAE/rMSE is \(0.15266/0.23266\), above naPINN and
LAD. It has lower nominal-physics residuals and 94.34% corrupted-scalar
rejection, but rejects 27.60% of clean scalars; naPINN rejects
64.97%/3.86%.

Under spatial burst, MAD-PINN rMAE/rMSE is \(0.14967/0.22770\). Its rMAE
is slightly below naPINN and its rMSE slightly above; LAD remains best.
MAD has lower nominal-physics residuals but rejects 33.60% of clean scalars,
versus 5.53% for naPINN. Both reject all corrupted scalars.

## Reproduction caveat

The original Burgers and lambda--omega run artifacts and a canonical
submission manifest were absent. Their numerical datasets were regenerated
from the current checked-out generators. New full runs are therefore
traceable additional experiments, not claimed bitwise reproductions or
replacements of the submitted tables.

## Artifact locations

- Frozen matrix: `rebuttal/experiment_plan.yaml`
- Synthetic runner: `scripts/rebuttal/run_synthetic.py`
- Synthetic queue: `scripts/rebuttal/run_synthetic_queue.py`
- Synthetic MAD: `scripts/rebuttal/run_synthetic_mad.py`
- Synthetic aggregation: `analysis/aggregate_rebuttal_synthetic.py`
- PIV runner: `scripts/rebuttal/run_realpdebench.py`
- PIV held-out queue: `scripts/rebuttal/run_realpdebench_queue.py`
- PIV calibration selection:
  `analysis/results/runs/rebuttal_realpde/piv_rejection_calibration_selection.json`
- PIV MAD dependency queue:
  `scripts/rebuttal/run_realpdebench_mad_queue.py`
- Synthetic artifacts: `analysis/results/runs/rebuttal_synthetic/`
- Injected-PIV artifacts: `analysis/results/runs/rebuttal_realpde/`
