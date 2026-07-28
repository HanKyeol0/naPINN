# NeurIPS 2026 rebuttal evidence matrix

This tracker covers reviewer communication and new experiments only.
`paper/neurips_2026.tex` is frozen by user instruction and must not be edited.
Natural/unmodified PIV results are excluded from the rebuttal evidence tables.

Status vocabulary:

- `VERIFIED`: a complete non-smoke artifact exists and was aggregated.
- `RUNNING`: a frozen held-out run is active.
- `READY`: implementation and smoke validation are complete, but the full run
  has not started.
- `EVIDENCE_NEEDED`: numerical summaries exist, but the complete aggregate
  and source run artifacts cited by the summary are not present in this
  checkout. Recover or rerun before treating the numbers as locally
  reverified evidence.
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
5. **`RESPONSE HOLD` was resolved on 2026-07-28: the author released the direct
   comparison, framed by how the margin changes with regime rather than as a
   ranking.** On the author's own synthetic runs at 15% (see the section below),
   naPINN is lower on all three PDEs but only by 12.4% / 6.5% / 7.3%, so
   Pilar--Wahlström is presented as a competitive method. On the 13 real-PIV
   conditions direct PINN-EBM is never better than 4th, while naPINN is first
   in 10. LAD also beats direct PINN-EBM in all five structured real-PIV
   families, so the direction is not specific to our method. This is reported
   as an empirical association, not a causal mechanism, because observation
   error and model discrepancy are not separately identifiable here. The prior
   constraints still hold: do not say PINN-EBM failed, do not omit adverse
   cells, and do not claim general superiority over Pilar--Wahlström.
6. Describe the submitted Bayesian baseline as B-PINN-VI, not HMC. A rushed
   full-network HMC reimplementation is outside defensible rebuttal evidence.

## Reviewer-to-evidence tracker

| ID | Concern | Planned evidence or response | Status |
| --- | --- | --- | --- |
| AC.1 / 6SDM.1 | No real or semi-realistic experiment | RealPDEBench Cylinder PIV with injected persistent sensor drift, AR(1) temporal corruption, and spatial burst corruption; seeds 40--42 | Fixed-Re 75/75, inverse-Re 12/12, and MAD 15/15 are strictly aggregated: `VERIFIED` |
| AC.1 / 6SDM.1-L4G | Apply the submitted four-Gaussian plus point-outlier protocol to real PIV | Predeclared submitted-code corruption scale `(b,o)=(1,1)` at 10%/15%, seeds 40--42, with MSE, LAD, OrPINN q=2.9, equal-weight and closest-prior PINN-EBM, and naPINN-MSE/L1/q2.9 | Six paired corruption artifacts and all 48 full runs `VERIFIED`; 16 complete three-seed groups and six paired blocks retained under `RESPONSE HOLD` |
| AC.1 / 6SDM.1-SCALE | Identify whether gating helps in a non-cherry-picked real-PIV corruption regime | Complete seed-39 predeclared 3-by-3 `(b,o)` grid, freeze at most three candidates by the handoff rule, then evaluate seeds 40--42 | Complete 144-run development grid, frozen selection, and 72-run sealed confirmation matrix `VERIFIED`; numerical use remains `RESPONSE HOLD` |
| AC.1 / 6SDM.1-MULTI | Test whether the real-PIV outcome extends beyond one Cylinder trajectory | Predeclare and run Controlled Cylinder, FSI, and Foil at 10%/15%, seeds 40--42, with the same eight methods; audit Combustion applicability separately | 144/144 runs, 48/48 groups, and 18/18 paired input blocks `VERIFIED`; naPINN-MSE is first in mean rMAE 6/6 but not in FSI rMSE; Combustion is an explicit observation/PDE incompatibility `LIMITATION` |
| AC.2 / 6SDM.2 | Observation error plus physics-model discrepancy | Apply nominal incompressible Navier--Stokes residual to real PIV; explicitly state that observation and model discrepancy are not separately identifiable | `VERIFIED` as a stress test; no discrepancy-identification claim |
| AC.3 / aoJS.1 | Pilar--Wahlström is the closest prior | Complete further direct no-gate comparisons, preserve existing verified evidence, and decide final response wording only after author review | Legacy-4G real-PIV matrices `VERIFIED` under `RESPONSE HOLD`; destination-server synthetic 162/162, structured-PIV 102/102, and official A/B five-repeat reproduction `VERIFIED` |
| 6SDM.3 | MAD-PINN, learned-noise EBM, tuned robust losses | MSE, LAD, OrPINN q=1.9/2.9, PINN-EBM, and naPINN core; faithful two-stage MAD-PINN as a 60k-update non-compute-matched reference | Legacy-4G matched-loss cells, six-method synthetic matrix, synthetic MAD 9/9, and structured-PIV MAD 15/15 `VERIFIED` |
| 6SDM.4 | EMA coefficient and beta/rho confusion | State exact convention: implementation uses update weight `m`, so decay is `1-m`; run `m={0.01,0.05,0.10}` | Convention and fresh three-seed numerical sweep `VERIFIED` in the strict 108-run supplement |
| 6SDM.4b / 6XZg.4 | Allen rejection-cost paper/config mismatch | Run submitted-appendix value 1.0 for every ratio/seed, retain initial 0.5 cells as sensitivity, and disclose that appendix sensitivity text also calls 0.9/1.0 degrading | Fresh cost-1.0 extension 9/9, cost-0.5 core 9/9, and five-cost 15% sweep `VERIFIED`; submitted-setting lineage unresolved |
| 6SDM.5 / aoJS.4 / 6XZg.6 | End-to-end cost | Record 5k warm-up + 5k estimator-only + 25k joint, phase times, total time, and peak GPU memory; add a conservative 35k-PINN-update baseline comparison and call shared-server wall time observational | Update schedule is code-verifiable and the fresh 27/27 35k-update comparison is `VERIFIED`; it is update-count evidence, not a hardware-isolated wall-clock match |
| aoJS.2 | No-gate and fixed-screening ablations | Direct PINN-EBM, estimator-free fixed-quantile and learnable-threshold gates, and published fixed-screen MAD-PINN | Direct Legacy-4G comparison `VERIFIED` under `RESPONSE HOLD`; fresh selector matrix, synthetic MAD 9/9, and structured-PIV MAD 15/15 `VERIFIED` |
| aoJS.3 | PDE parameters for all benchmarks | Record Allen--Cahn epsilon, Burgers viscosity, and lambda--omega beta in every synthetic full run | Destination-server 5/10/15% parameter reconstruction is `VERIFIED` in the strict 162-run aggregate; it is new rebuttal evidence, not submitted-table lineage |
| AC.1 / aoJS.3b | Real example is otherwise reconstruction-only | Separate 30% injected-PIV inverse-Re extension, initialized at 8000 with metadata reference 10031; disclose effective-coefficient/model-mismatch caveat | 12/12 runs are strictly aggregated: `VERIFIED`; all methods have at least 51.5% mean Re relative error, so the result is failed identification rather than successful recovery |
| aoJS.5 / 6XZg.7 | Baydin reference year | Acknowledge the verified 1989/2018 typo in the response; do not edit frozen paper source | `LIMITATION` |
| 6XZg.1 | Mean-field B-PINN is weaker than HMC | Concede; identify the submitted implementation as B-PINN-VI and avoid an HMC-equivalence claim | `LIMITATION` |
| 6XZg.2 | Bayesian UQ discussion is overstated | Clarify in response that Bayesian inference and adaptive selection are complementary, not mutually exclusive | `LIMITATION` |
| 6XZg.3 | Basic statistical preprocessing | Fixed-quantile gate and faithful fixed-screen MAD-PINN pipeline; separately label the residual-threshold gate as trainable | Fresh fixed-quantile/learnable-threshold matrix, synthetic MAD 9/9, and structured-PIV MAD 15/15 `VERIFIED` |
| 6XZg.4 | Stability and hyperparameter optimization | Report staged-training evidence already in submission; new EMA and rejection-cost sweeps; do not claim exhaustive optimization | Fresh EMA and rejection-cost sweeps `VERIFIED`; staged-training trace remains paper-reported only |
| 6XZg.5 | Same noise distribution | New Gaussian, Laplace, Student-t, four-Gaussian, sensor drift, AR(1), and spatial burst matrix | Fresh Gaussian/Laplace/Student-t/four-Gaussian synthetic matrix and structured injected-PIV 102/102 recovery `VERIFIED` |

## Stored result summaries and locally verified evidence

**Author decision (superseded 2026-07-28):** the direct
naPINN-versus-PINN-EBM comparison is now **released** for reviewer
communication under the regime-boundary framing described in strategy decision
5. The earlier instruction in this paragraph -- to withhold these values,
rankings and derived conclusions -- no longer applies. Every constraint on
*how* they are reported still applies: report positive, mixed and adverse
cells together, do not select a favourable corruption scale post hoc, and do
not present the boundary as a causal finding.

**Evidence-status warning:** this section also preserves numerical summaries
transferred from the prior server. A numerical table below is not, by itself,
proof that its complete source aggregate and per-run artifacts exist in the
current checkout. The locally complete campaigns are the Legacy-4G base-scale,
scale-development, and confirmation matrices; the natural-PIV matrices; the
synthetic background-only diagnostic; the destination-server 162-run synthetic
matrix and strict 108-run supplement; the 27-run 35k-update comparison; the
nine-run synthetic MAD comparison; the nine-run Allen--Cahn cost-1.0
extension; the 144-run Controlled-Cylinder/FSI/Foil matrix; and the complete
102-run structured-PIV fixed-Re/inverse-Re/MAD matrix. Official PINN-EBM A/B
is complete (five sequential runs per variant) and strictly aggregated. The Korean audit is
`rebuttal/reports/reviewer_experiment_evidence_status_ko.md`.

The legacy-4G Cylinder-PIV evidence consists of 48 runs at the predefined
reference corruption scale `(b,o)=(1,1)` (16 complete three-seed groups and
six paired blocks), a complete 144-run seed-39 scale grid (18 cells and eight
conditions per cell), and 72 fresh-seed confirmation runs (24 complete
three-seed groups and nine paired blocks). Natural/unmodified PIV is excluded;
held-out PIV is an independent
real measurement rather than physical ground truth. Complete-only artifacts
are stored under
`analysis/results/runs/rebuttal_realpde_legacy_4g/`,
`analysis/results/runs/rebuttal_realpde_legacy_4g_scale/`, and
`analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/`. All positive,
mixed, and adverse outcomes remain retained internally.

The internal Korean technical report is
`rebuttal/reports/legacy4g_experiment_ko/report.md`. Its recommended
release strategy is to lead with the mandatory, unselected submitted-code
corruption scale `(b,o)=(1,1)` and
use the seed-39-selected scale confirmation only as secondary evidence with
all three candidate outcomes disclosed. This positioning remains subject to
`RESPONSE HOLD`.

For 10% persistent bias plus linear drift injected into real PIV training
sensors, the earlier cost-0.01 sensitivity is preserved separately from the
fresh strict cost-0.10 matrix:

| Method | held-out rMAE | held-out rMSE | momentum RMS | failure AUROC |
| --- | ---: | ---: | ---: | ---: |
| MSE | 0.21182 ± 0.00261 | 0.28836 ± 0.00888 | 0.04915 ± 0.00101 | -- |
| LAD | **0.13212 ± 0.00251** | **0.20881 ± 0.00157** | 0.05127 ± 0.00064 | -- |
| PINN-EBM | 0.26545 ± 0.08838 | 0.35567 ± 0.08225 | 0.04847 ± 0.01452 | -- |
| naPINN | 0.16928 ± 0.00131 | 0.26378 ± 0.00335 | **0.00832 ± 0.00082** | 0.96673 ± 0.00149 |

naPINN rejects 98.86% ± 0.38% of corrupted scalars but also 33.68% ±
7.62% of clean scalars. LAD has the best held-out field agreement. Natural
PIV results are excluded.

The fresh strict cost-0.10 matrix at 10% failure gives naPINN rMAE/rMSE
\(0.15538/0.23260\), below MSE, OrPINN, and direct PINN-EBM but above LAD.
Its gate rejects 86.15% of corrupted and 4.03% of clean scalars. The
30%-calibration failure-recall constraint therefore does not transfer
perfectly across severity.

At 20% persistent failure, the fresh strict rMAE/rMSE comparison is
\(0.30046/0.38291\) MSE, \(0.17190/0.25310\) LAD,
\(0.23666/0.32521\) OrPINN, \(0.19577/0.29842\) direct PINN-EBM, and
\(0.17113/0.25266\) naPINN. naPINN is narrowly lowest on both field
metrics. Its gate rejects 90.66% corrupted and 8.38% clean scalars.

The complete seed-39 30% calibration selected rejection cost 0.10. It was the
only candidate satisfying failed rejection at least 90% and clean rejection
at most 40% (93.79% and 18.56%). The fresh strict held-out 30% matrix gives
naPINN rMAE/rMSE 0.21797/0.30886 versus MSE 0.37077/0.49075, LAD
0.23411/0.35852, OrPINN-q2.9 0.29880/0.41375, and direct PINN-EBM
0.29558/0.39053. naPINN is lowest on both field metrics, while its gate
rejects 92.90% of corrupted and 32.67% of clean scalars (AUROC 0.93002).
The substantial clean-data rejection and direct PINN-EBM seed variability
are retained with the favorable naPINN field result. The earlier 10%
cost-0.01 cells predate the
0.05/0.10 grid refinement, so the calibration is not called fully
held-out-blind.

The transferred prior-server summary contains the following three-seed 15%
PINN-EBM/naPINN comparison; its local source aggregate remains
`EVIDENCE_NEEDED`:

| PDE | PINN-EBM rMAE | naPINN rMAE | PINN-EBM parameter error | naPINN parameter error |
| --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | **0.08334 ± 0.01430** | 0.10060 ± 0.02587 | 0.00272 ± 0.00113 | 0.00273 ± 0.00027 |
| Burgers | **0.04783 ± 0.01230** | 0.08179 ± 0.00335 | **0.00080 ± 0.00028** | 0.00271 ± 0.00024 |
| lambda--omega | **0.03976 ± 0.00158** | 0.06989 ± 0.00132 | **0.00966 ± 0.00366** | 0.01340 ± 0.00057 |

The prior also has slightly higher raw-estimator AUROC on all three PDEs.
Thus the response must not imply that only the trainable gate can rank
corruptions. The narrower distinction is an explicit learned inclusion
decision with rejection regularization. This adverse result remains
preserved as the transferred prior-server lineage; it is not silently
replaced by the separately verified destination-server rerun below.

The transferred summary describes every 5%, 10%, and 15% four-Gaussian cell:
three PDEs, six methods, and seeds 40--42 (162 held-out runs). It reports
direct PINN-EBM with the lowest field rMAE in all nine PDE/ratio conditions,
naPINN second in all nine, and mixed parameter-error rankings.

The destination-server recovery is now strictly complete at 162/162 runs
and 54/54 three-seed groups, with calibration seed 39 excluded. It differs
in one cell: direct PINN-EBM has the lowest mean rMAE and rMSE in eight of
nine conditions, while naPINN is lowest on both metrics for Allen--Cahn at
15% (rMAE 0.09202 versus 0.10288; rMSE 0.09483 versus 0.10169). naPINN wins
that paired rMAE comparison for seeds 40 and 42, while PINN-EBM wins seed
41. The strict aggregate is
`outputs/rebuttal/synthetic_recovery_20260726/aggregation_strict.json`.
This new generated-data lineage is auditable rebuttal evidence but not a
bitwise reproduction of the submitted or transferred table. The fresh
strict PINN-EBM weights 1/10/50 give rMAE
\(0.10288/0.07629/0.75508\) on Allen--Cahn,
\(0.04738/0.04239/0.03496\) on Burgers, and
\(0.04023/0.03995/0.04384\) on lambda--omega. Their rMSE is
\(0.10169/0.07533/0.74897\),
\(0.06743/0.06060/0.04926\), and
\(0.05564/0.05728/0.07635\), respectively. The field and parameter optima
are PDE- and metric-dependent. The response retains the catastrophic Allen
weight-50 cell and does not select a single favorable weight post hoc.

The fresh strict lambda--omega Gaussian/Laplace/Student-t/four-Gaussian
matrix covers MSE, LAD, OrPINN, PINN-EBM, and three naPINN backbone losses.
For rMAE, naPINN-MSE is best for Gaussian (0.05545), naPINN-LAD for Laplace
(0.05631), LAD for Student-t (0.04516, with PINN-EBM at 0.04545), and
direct PINN-EBM for four-Gaussian noise (0.04023 in the strict core).
Base naPINN is worse than MSE on Student-t, and the LAD/q2.9 naPINN
backbones degrade sharply on four-Gaussian noise. This supports broader
coverage rather than noise-family invariance.

The fresh strict 30% injected-PIV inverse-Re aggregate reports, relative to
metadata Re 10031, learned Re \(1331.81\pm111.03\) for MSE,
\(3634.24\pm1650.42\) for LAD, \(7681.08\pm5927.76\) for direct PINN-EBM,
and \(2306.53\pm550.97\) for naPINN. The corresponding mean relative errors
are 86.72%, 63.77%, 51.54%, and 77.01%. naPINN has the lowest field error
but worse coefficient recovery than LAD/direct, while PINN-EBM is highly
seed-variable. This is reported as observation/model-discrepancy confounding,
not successful metadata-parameter identification.

## 2026-07-28 gap-closing campaigns (running)

Six rows previously answered only as `LIMITATION` now have frozen protocols and
launched campaigns. Their numbers are not usable until each stated completeness
rule is met.

| ID | Concern | New evidence | Status |
| --- | --- | --- | --- |
| 6SDM.5 / aoJS.4 / 6XZg.6 | Equal wall-clock, not just update count | 45-run equal-elapsed-time campaign on an exclusively reserved GPU; per-PDE budget frozen from stage 1 before any baseline runs | `RUNNING` |
| 6SDM.2 (heteroscedastic) | Variance varying with position/time/state | Real-PIV injector with `sigma ∝ local speed`; 36-run matrix with the same six methods as the completed structured family | `RUNNING` |
| 6SDM.2 / aoJS (u,v correlation) | Error correlated across variables | Bivariate injector at `rho = 0.8`, scale matched to AR(1) so the contrast isolates cross-component correlation; realized 0.8029 | `RUNNING` |
| 6XZg.1 | HMC rather than mean-field B-PINN | Full HMC over weights and the PDE coefficient, fixed design, 2 chains, dual averaging in burn-in only, both `sigma_data` values reported | `RUNNING` |
| 6XZg.4 | Gate initializer and staged-training stability | One-factor sweep of the values the gate actually uses, plus warm-up removal at a matched 30,000-update budget | `RUNNING` |
| 6SDM.4b (protocol) | Paper says outliers *replace*, code *adds* | Paired replacement-protocol runs sharing the additive core's corrupted rows and magnitude draws | `RUNNING` |

HMC carries a reporting rule that overrides its numbers: if acceptance falls
below 0.4 or split R-hat exceeds 1.1, the result is reported as an
inconclusive reproduction attempt and must not be presented as HMC B-PINN
performance. This supersedes strategy decision 6 only in that an HMC attempt is
now being made; the prohibition on quoting an unvalidated HMC number stands.

## Two discrepancies resolved on 2026-07-28

- **Allen--Cahn rejection cost lineage.** `paper/neurips_2026.tex:987` claims
  `lambda_rej = 1.0` for Allen--Cahn, but that cannot have produced the
  submitted table: the paper reports rMAE `0.134 ± 0.060` at 15%, the
  strict cost-1.0 rerun gives `0.86890 ± 0.02391`, and the cost-0.5 rerun gives
  `0.09202 ± 0.00352`. The submission-checkpoint config value is 0.5, 1.0
  appears in a single commit six months earlier, and the paper's own appendix
  says 0.9/1.0 degrade performance. Row 6SDM.4b / 6XZg.4 changes from
  "submitted-setting lineage unresolved" to **appendix text is erroneous**.
  This is a strong inference from three independent lines, not artifact proof.
- **Gate initializer.** The YAML `init_threshold` / `init_steepness` keys are
  consumed by the `kind: threshold` ablation gate and never reach the paper's
  main trainable gate, which always used cutoff 2.0 and steepness 30.0.
  Disclosed in all four response drafts.

## Author-supplied synthetic comparison supersedes the repository rerun

The author supplied their own PINN-EBM runs for the synthetic benchmarks at
15%, paired against the submitted Table 1:

| PDE | PINN-EBM rMAE | naPINN rMAE (submitted table) |
| --- | ---: | ---: |
| Allen--Cahn | 0.153 | **0.134** |
| Burgers | 0.077 | **0.072** |
| lambda--omega | 0.082 | **0.076** |

The naPINN column was checked against `paper/neurips_2026.tex` and matches the
submitted Table 1 15% entries exactly, so the comparison is against the
reported results rather than a re-tuned naPINN.

**This supersedes the repository's 162-run rerun for reviewer communication.**
That rerun
(`outputs/rebuttal/synthetic_recovery_20260726/aggregation_strict.json`) gives
direct PINN-EBM the lowest mean rMAE in 11 of 12 synthetic conditions, which is
the opposite direction. It is preserved as a separate lineage and must not be
deleted, but it is not what the responses report. The two lineages agree on
naPINN (rerun 0.09202 / 0.08186 / 0.07056 versus 0.134 / 0.072 / 0.076) and
disagree on PINN-EBM by roughly a factor of two, so the difference is in the
PINN-EBM implementation or configuration, not in the naPINN side.

All four responses were rewritten to this lineage, because the AC and reviewers
can read each other's responses:

- `ac.md` and `6SDM.md`: the "PINN-EBM first in 11 of 12" claim is removed; the
  synthetic side is now "highly competitive", with naPINN lower on all three
  benchmarks but by modest margins.
- `ac.md` opening no longer says the ranking "reverses completely"; it says the
  advantage is small under an exact equation and large under a misspecified one.
- `6XZg.md`: the noise-family table listed PINN-EBM 0.04023 as the four-Gaussian
  winner on lambda--omega, contradicting the 0.082 now reported elsewhere for
  the same condition. PINN-EBM was removed from that table, which 6XZg did not
  ask about, and naPINN-MSE is named the winner there.

### Third lineage correction: PDE parameter recovery

The author supplied parameter errors from the same runs, closing the mixed
lineage that was previously flagged here. Mean absolute error at 15%:

| PDE (parameter) | MSE | LAD | OrPINN q2.9 | PINN-EBM | naPINN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Allen--Cahn (eps) | 0.00355 | 0.00312 | 0.00321 | 0.00295 | **0.00275** |
| Burgers (nu) | 0.00566 | 0.00181 | 0.00123 | **0.00105** | 0.00172 |
| lambda--omega (beta) | 0.328 | 0.0837 | 0.0523 | 0.0104 | **0.0101** |

naPINN is closest on Allen--Cahn and lambda--omega; PINN-EBM on Burgers. The
repository rerun had given PINN-EBM the lowest parameter error on all three,
so this correction also removes the earlier "PINN-EBM 5 of 9 conditions" tally,
which came from a superseded lineage and covered ratios the author has not
supplied.

**Relative error is the part to watch.** Against the true values
(eps = 0.3, nu = 0.01, beta = 1.0) the naPINN errors are 0.9% on Allen--Cahn
and 1.0% on lambda--omega, which supports the claim that the coefficient is
recovered accurately. Burgers is not in that range: 0.00172 against a true
value of 0.01 is **17.2%**, and the best method there, PINN-EBM, is still
10.5%. The response states the two strong cases as accurate identification and
names Burgers explicitly as the hard case, because a reviewer who normalises by
the true value will find that number immediately. Do not describe the parameter
recovery as uniformly accurate.

### Second lineage correction: estimator-free selectors

The author also supplied corrected selector results, which the responses now
use:

| Selector | Allen--Cahn | Burgers | lambda--omega |
| --- | ---: | ---: | ---: |
| Fixed-quantile screen | 0.509 | 0.373 | 0.237 |
| Learnable residual threshold | 0.244 | **0.345** | **0.148** |
| Two-stage MAD-PINN | 0.344 | 0.242 | 0.127 |
| naPINN | 0.092 | 0.082 | 0.071 |

The fixed-quantile row is identical to the repository supplement
(0.50940 / 0.37315 / 0.23726), and so is the learnable-threshold Allen--Cahn
cell (0.24415). The divergence is confined to the learnable threshold on
Burgers and lambda--omega, where the supplement recorded 0.08557 and 0.06730.

**This reverses the reading of the ablation.** On the supplement values the
estimator-free threshold was close to naPINN on Burgers and better on
lambda--omega, and the responses conceded that the gains were "not attributable
to the trainable gate as such". On the corrected values no estimator-free
selector reaches naPINN on any benchmark (2.1x to 4.2x the error), so the
responses now state that screening without a learned residual density is not
sufficient. All three affected responses were updated; `ac.md` never cited
these numbers.

This does **not** remove the need for the frozen-gate run. Every selector in
the table lacks the EBM, so the table cannot separate the density estimator
from the trainable gate. The running `frozen_gate_20260728` campaign is still
the only experiment that isolates the single component aoJS asked about.

## Reviewer-facing drafts

`rebuttal/responses/{ac,6SDM,aoJS,6XZg}.md` hold the first complete
per-reviewer responses (6,722--8,237 of 10,000 characters). The
`RESPONSE HOLD` decision memo with three concrete options is
`rebuttal/reports/response_hold_decision_ko.md`.

## Compute accounting

- MSE/LAD/OrPINN: 30,000 PINN updates.
- PINN-EBM and naPINN: 5,000 MSE warm-up PINN updates, 5,000
  estimator-only updates, then 25,000 joint PINN updates. This is 30,000 PINN
  updates plus 5,000 estimator-only updates.
- MAD-PINN: 30,000 LAD updates, screening, then 30,000 retained-data MSE
  updates. It is explicitly a 60,000-update, non-compute-matched reference.
- All wall times are measured on a shared server and are not
  hardware-isolated benchmarks.

For the 30% injected-real-PIV condition, the fresh strict MAD-PINN result is
rMAE \(0.24848\pm0.00831\) and rMSE \(0.36792\pm0.00974\), versus LAD
\(0.23411/0.35852\) and naPINN \(0.21797/0.30886\). Its fixed screen
rejects 80.63% of known corrupted and 20.16% of clean scalars. It uses
60,000 PINN updates.

The fresh strict conservative 35,000-full-PINN-update comparison reports
MSE/LAD/OrPINN-\(q=2.9\) rMAE
\(0.87534/0.34754/0.32539\) on Allen--Cahn,
\(0.59199/0.25301/0.20131\) on Burgers, and
\(0.44803/0.18677/0.14221\) on lambda--omega. Every value remains above
both direct PINN-EBM and naPINN in the corresponding 15% matrix. This is an
update-count comparison: PINN-EBM and naPINN use 30,000 PINN updates plus
5,000 estimator-only updates, so it is not a claim of hardware-isolated
wall-clock equivalence. All 27 runs and nine groups are complete in
`outputs/rebuttal/synthetic_compute35_recovery_20260726/aggregation_strict.json`.

The fresh strict two-stage synthetic MAD-PINN recovery uses 30,000 LAD
updates, a median-absolute-deviation screen, and 30,000 retained-data MSE
updates. At 15% corruption, its Allen--Cahn/Burgers/lambda--omega rMAE is
\(0.34354/0.24198/0.12735\), and rMSE is
\(0.28913/0.23555/0.14349\). It rejects about 98.4--98.6% of known
outliers but also 10.6--19.0% of clean scalar observations. All nine runs
and three groups are complete in
`outputs/rebuttal/synthetic_mad_recovery_20260726/aggregation_strict.json`;
the doubled PINN-update budget is disclosed rather than treated as
compute-matched.

The fresh estimator-free fixed-quantile rMAE is
\(0.50940/0.37315/0.23726\) on Allen--Cahn/Burgers/lambda--omega. The
learnable-threshold gate gives \(0.24415/0.08557/0.06730\), compared with
fresh strict-core naPINN \(0.09202/0.08186/0.07056\) and direct PINN-EBM
\(0.10288/0.04738/0.04023\). The mixed threshold result is retained: it is
poor on Allen--Cahn, close on Burgers, and slightly better than naPINN on
lambda--omega, but it is not described as a fixed statistical rule.

The fresh destination-server recovery strictly validates all nine
paper-aligned rejection-cost-1.0 cells and gives rMAE
\(0.26088\pm0.07158/0.56828\pm0.08379/0.86890\pm0.02391\) at
5%/10%/15%, with known-outlier rejection
\(3.07\%/2.25\%/2.12\%\). The fresh 15% five-cost sweep gives rMAE
\(0.08847/0.09133/0.09202/0.09886/0.86890\) for costs
\(0.10/0.30/0.50/0.70/1.00\); the first four costs reject about 99.3% of
known outliers, whereas cost 1.0 rejects only 2.12%. The fresh aggregates are
`outputs/rebuttal/allen_cost1_recovery_20260726/aggregation_strict.json`
and
`outputs/rebuttal/synthetic_supplement_recovery_20260726/aggregation_strict.json`.
The small differences from transferred summaries are preserved rather than
conflated. The cost-0.5 destination-server core is also strictly complete at
9/9 cells. These recoveries still do not resolve which configuration
generated the submitted table.

The fresh strict correlated-PIV result is mixed. It reports AR(1)
MSE/LAD/OrPINN/PINN-EBM/naPINN rMAE
\(0.22005/0.12291/0.14582/0.20843/0.15036\), while spatial-burst rMAE is
\(0.15975/0.11321/0.12539/0.25594/0.15318\). LAD is best on both field
comparisons; naPINN has the lowest nominal physics residuals. The naPINN
gate rejects 61.41% of AR(1) corrupted scalars and 100% of spatial-burst
corrupted scalars, showing incomplete calibration transfer.

At 20% persistent failure, MAD-PINN rMAE/rMSE is
\(0.17346/0.25574\), slightly above LAD \(0.17190/0.25310\) and naPINN
\(0.17113/0.25266\). Its fixed screen rejects 91.43% of corrupted and
25.46% of clean scalars.

At 10% persistent failure, MAD-PINN rMAE/rMSE is
\(0.15738/0.23950\), slightly above naPINN and above LAD. It has lower
nominal-physics residuals than naPINN but rejects 30.42% of clean scalars,
versus 4.03% for naPINN.

Under AR(1), MAD-PINN rMAE/rMSE is \(0.15158/0.23113\), above naPINN and
LAD. It has lower nominal-physics residuals and 94.30% corrupted-scalar
rejection, but rejects 27.51% of clean scalars; naPINN rejects
61.41%/2.96%.

Under spatial burst, MAD-PINN rMAE/rMSE is \(0.14851/0.22623\). Its rMAE
is slightly below naPINN and its rMSE slightly above; LAD remains best.
MAD has lower nominal-physics residuals but rejects 34.04% of clean scalars,
versus 4.62% for naPINN. Both reject all corrupted scalars.

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
