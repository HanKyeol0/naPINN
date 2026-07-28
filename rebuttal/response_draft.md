# NeurIPS 2026 author response — working draft, do not submit yet

> **AUTHOR DECISION — DIRECT COMPARISON RESPONSE HOLD**
>
> All direct naPINN-versus-PINN-EBM numbers, rankings, and response
> conclusions below are retained as internal evidence only. Do not submit
> them yet. The planned real-PIV legacy-corruption, scale, robust-loss, and
> closest-prior campaign is complete and strictly validated. An explicit
> author decision is still required before deciding whether or how to use
> the direct comparison. Existing adverse results must remain preserved
> during that decision.

This draft intentionally excludes natural/unmodified-PIV results. The
predefined Legacy-4G reference setting, its corruption-scale sweep, and its
fresh-seed confirmation matrices are locally verified. The destination-server
synthetic recovery, non-Cylinder RealPDEBench matrix, 35k-update check, and
synthetic MAD-PINN comparison are also strictly complete. The 102-run
structured-PIV recovery, including fixed-Re, inverse-Re, and MAD-PINN, is
strictly complete. Official PINN-EBM active-code A and paper-architecture B
are both complete: five official sequential runs per variant passed strict
aggregation. Their results remain a separate source-faithfulness check, not a
naPINN compute-matched comparison. The
authoritative evidence-status audit is
`rebuttal/reports/reviewer_experiment_evidence_status_ko.md`. This remains a
working evidence draft and must not be shortened into review-specific
responses until the cited strict aggregates exist.

## Internal legacy-4G Cylinder-PIV campaign status — do not submit

The controlled-corruption campaign is complete. Natural/unmodified PIV is
excluded, and the unchanged held-out PIV is an independent real measurement,
not physical ground truth. The predefined reference setting—meaning the
originally selected Legacy-4G corruption scale, rather than a value chosen
after seeing the results—contains 48 verified runs, 16 complete three-seed
groups, and six paired input blocks. The predeclared seed-39 scale grid
contains 144 verified runs across 18 cells and eight conditions per cell; the
frozen selection rule was then applied without threshold relaxation. The
fresh-seed confirmation stage contains 72 verified runs, 24 complete
three-seed groups, and nine paired input blocks.

The complete-only evidence is recorded in
`analysis/results/runs/rebuttal_realpde_legacy_4g/aggregation.json`,
`analysis/results/runs/rebuttal_realpde_legacy_4g_scale/seed39_scale_selection.json`,
`analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/aggregation.json`,
and
`analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/candidate_validation.json`.
All positive, mixed, and adverse outcomes are retained. Direct values,
rankings, selected-candidate identities, and derived response positioning
remain under `RESPONSE HOLD`.

## Shared opening

No shared closest-prior opening is currently approved. Address
Pilar--Wahlström only in responses where the Area Chair or reviewer
explicitly raises it. The eventual wording should acknowledge the shared EBM
and staged components, then distinguish direct EBM-NLL optimization from
naPINN's detached density-estimation objective, explicit per-measurement
gate, gated base reconstruction loss, and rejection regularization. Do not
voluntarily recast the overall contribution as "only a gate." Direct
comparison numbers and the final positioning remain on response hold.

## Real-data paragraph

> **VERIFIED:** The frozen destination-server structured-PIV recovery is
> strictly complete at 102/102 runs and 34/34 three-seed groups. The values
> below come from the new complete-only aggregate, not the prior-server
> transferred summary.

We evaluated controlled corruptions of real time-resolved PIV from the
RealPDEBench Cylinder dataset. We used 200 frames, 192 fixed irregular
training sensors (38,400 velocity vectors), and 1,549,400 held-out velocity
vectors at 7,747 spatial locations disjoint from training. Corruption was
applied only to training sensors; held-out PIV was unchanged and is treated
as an independent measurement reference, not noise-free ground truth. The
nominal pressure-latent incompressible Navier--Stokes model also introduces a
realistic model-discrepancy stress, but our method does not separately
identify observation and physics-model discrepancy.

At the calibration-selected rejection cost 0.10, the three-seed 30%
persistent-failure rMAE/rMSE is
\(0.37077/0.49075\) for MSE,
\(0.23411/0.35852\) for LAD,
\(0.29880/0.41375\) for OrPINN-\(q=2.9\),
\(0.29558/0.39053\) for direct PINN-EBM, and
\(0.21797/0.30886\) for naPINN. naPINN is lowest on both field metrics,
while its gate rejects 92.90% of corrupted and 32.67% of clean scalar
observations (AUROC 0.93002). We report the substantial clean-data rejection
and the large seed variability of direct PINN-EBM together with the favorable
naPINN field result, rather than claiming universal superiority.

For AR(1) temporal corruption, MSE/LAD/OrPINN/PINN-EBM/naPINN rMAE is
\(0.22005/0.12291/0.14582/0.20843/0.15036\); for spatial burst failure it
is \(0.15975/0.11321/0.12539/0.25594/0.15318\). LAD is best on both
field comparisons. naPINN rejects 61.41% of AR(1)-corrupted scalars and
2.96% of clean scalars, versus 100% and 4.62% under spatial burst. Gate
AUROC is 0.91307/0.98459.
Thus the persistent-failure calibration does not fully transfer to temporal
correlation, and we report structured-noise coverage rather than universal
superiority.

In the already completed cost-0.01 sensitivity at 10% persistent
bias-plus-drift, naPINN
reduces held-out rMAE from \(0.21182\pm0.00261\) for MSE-PINN to
\(0.16928\pm0.00131\), and reduces nominal momentum-residual RMS from
\(0.04915\pm0.00101\) to \(0.00832\pm0.00082\). Its gate obtains AUROC
\(0.96673\pm0.00149\) and rejects \(98.86\%\pm0.38\%\) of known corrupted
scalars, but it also rejects \(33.68\%\pm7.62\%\) of known-clean scalars.
LAD-PINN has the best held-out field agreement
(\(0.13212\pm0.00251\)); we report this mixed result rather than claiming
universal superiority.

Rerunning the same 10% condition with calibration-selected cost 0.10 gives
naPINN rMAE/rMSE \(0.15538/0.23260\), below MSE, OrPINN, and direct
PINN-EBM but still above LAD. Its gate AUROC is 0.92914, rejecting 86.15%
of corrupted and 4.03% of clean scalars. Thus the seed-39 30%-severity rejection constraint
does not transfer perfectly to 10% severity, although clean retention is
greatly improved relative to cost 0.01.

At 20% persistent failure, the complete rMAE/rMSE comparison is
\(0.30046/0.38291\) MSE, \(0.17190/0.25310\) LAD,
\(0.23666/0.32521\) OrPINN, \(0.19577/0.29842\) direct PINN-EBM, and
\(0.17113/0.25266\) naPINN. naPINN is narrowly lowest on both field
metrics. Its gate rejects 90.66% of corrupted and 8.38% of clean scalars
(AUROC 0.93683).

On seed-39 calibration at 30% failure, rejection cost 0.10 was the only
candidate satisfying both at least 90% failed-scalar rejection and at most
40% clean-scalar rejection (93.79% and 18.56%, respectively). We fixed 0.10
for the full held-out matrix. Because the earlier 10% cost-0.01 sensitivity
had already completed before refining the seed-39 grid, we do not describe
this process as fully held-out-blind.

We additionally completed a frozen 144-run RealPDEBench matrix covering
Controlled Cylinder, FSI, and Foil at 10% and 15% large-outlier injection,
with eight methods and seeds 40--42. naPINN-MSE has the lowest mean rMAE in
all six dataset/severity conditions and the lowest mean rMSE in four. On FSI,
OrPINN has the lowest rMSE at both 10% and 15%
(0.60933/0.61541 versus 0.64632/0.67973 for naPINN-MSE), while naPINN-MSE
has the lowest rMAE (0.39948/0.41764). On Foil, naPINN-MSE has the lowest
rMAE/rMSE at 10% (0.09968/0.15158) and 15% (0.11783/0.16361). We therefore
report a six-condition matrix rather than selecting only the favorable
Cylinder case. Combustion is not included: its available observation is a
single OH* channel, whereas the current solver assumes velocity and pressure
under incompressible Navier--Stokes, so assigning a performance number would
compare incompatible tasks.

## Closest-prior paragraph

> **EVIDENCE STATUS:** The destination-server three-PDE by three-outlier-ratio
> comparison is now strictly
> complete at 162/162 runs and 54/54 three-seed groups. The transferred
> prior-server values below are retained as a separate lineage because the
> new generated-data rerun is not a bitwise reproduction and differs in one
> direct-comparison ranking. The isolated PDE-weight calibration and
> held-out recoveries are now strictly verified.

We added a no-gate PINN-EBM baseline in which the learned EBM negative
log-likelihood is backpropagated directly to the PINN, EBM, and PDE
parameter. The prior studies homogeneous Gaussian, uniform, and Gaussian
mixture noise, whereas our target is sparse severe corruption with explicit
per-measurement decisions. We do not claim that the prior objective fails.

At 15% four-Gaussian corruption (three seeds), PINN-EBM versus naPINN rMAE
is \(0.08334\pm0.01430\) versus \(0.10060\pm0.02587\) on Allen--Cahn,
\(0.04783\pm0.01230\) versus \(0.08179\pm0.00335\) on Burgers, and
\(0.03976\pm0.00158\) versus \(0.06989\pm0.00132\) on lambda--omega.
PINN-EBM also gives lower viscosity and beta error; epsilon errors are nearly
equal. Raw EBM surprise has slightly higher corruption AUROC than the naPINN
gate on all three. This adverse comparison confirms that density learning,
staging, and anomaly ranking are not unique contributions of naPINN. We
therefore claim only the narrower mechanism-level increment above, not
synthetic accuracy superiority over the closest prior.

The transferred summary reports a full 5%/10%/15% four-Gaussian core (three
PDEs, six methods, and seeds 40--42), but its original source aggregate is
absent locally. It reports direct PINN-EBM with the lowest field rMAE in all
nine PDE/ratio conditions and naPINN second in all nine. We preserve that
adverse summary as prior-server evidence rather than silently overwriting it.

The strict destination-server rerun gives direct PINN-EBM the lowest mean
rMAE and rMSE in eight of nine conditions. The exception is Allen--Cahn at
15%, where naPINN gives rMAE/rMSE \(0.09202/0.09483\), versus
\(0.10288/0.10169\) for PINN-EBM. naPINN wins that paired rMAE comparison
for seeds 40 and 42, while PINN-EBM wins seed 41. This rerun therefore
supports a distribution- and seed-dependent comparison, not either a
universal naPINN advantage or the transferred claim that PINN-EBM ranked
first in all nine conditions.
Its strict aggregate is
`outputs/rebuttal/synthetic_recovery_20260726/aggregation_strict.json`.

Because equal-weight held-out results had already been inspected, we do not
select a favorable PINN-EBM PDE weight post hoc. The fresh strict
weight-\(1/10/50\) rMAE is
\(0.10288/0.07629/0.75508\) on Allen--Cahn,
\(0.04738/0.04239/0.03496\) on Burgers, and
\(0.04023/0.03995/0.04384\) on lambda--omega. The corresponding rMSE is
\(0.10169/0.07533/0.74897\),
\(0.06743/0.06060/0.04926\), and
\(0.05564/0.05728/0.07635\). Thus weight 10 is best on both Allen field
metrics but has substantial seed variability, weight 50 is best on both
Burgers field metrics, and lambda--omega changes order between rMAE and
rMSE. Weight 50 catastrophically degrades Allen--Cahn. Parameter-error
optima also differ by PDE. We therefore report all weights and do not
present a single tuned weight as universal. The weight-1 cells come from
the strict core and the weight-10/50 cells from
`outputs/rebuttal/synthetic_pinn_ebm_weight_heldout_20260726/aggregation_strict.json`.

## Parameter-recovery paragraph

> **EVIDENCE STATUS:** The destination-server synthetic parameter table is
> now strictly verified for 5%/10%/15%, all three PDEs, six methods, and
> seeds 40--42. The inverse-Re extension is also strictly complete at
> 12/12 runs. The submitted ten-trial
> parameter-table lineage is still unavailable.

At 15% four-Gaussian corruption, the three-seed absolute parameter errors
for MSE/LAD/OrPINN-\(q{=}2.9\)/PINN-EBM/naPINN are respectively:
Allen--Cahn epsilon \(0.00355/0.00312/0.00321/0.00215/0.00275\);
Burgers viscosity \(0.00566/0.00181/0.00123/0.00065/0.00272\); and
lambda--omega beta \(0.32785/0.08370/0.05234/0.01016/0.01359\).
Thus the additional evidence covers every unknown PDE parameter, while also
showing that PINN-EBM, not naPINN, is strongest in the severe Burgers and
lambda--omega settings.

We also ran an inverse-Re extension on the 30% injected-PIV condition,
initializing Re at 8000 and comparing to metadata Re 10031. Learned
Re/relative error is \(1331.81\pm111.03/86.72\%\) for MSE,
\(3634.24\pm1650.42/63.77\%\) for LAD,
\(7681.08\pm5927.76/51.54\%\) for PINN-EBM, and
\(2306.53\pm550.97/77.01\%\) for naPINN. naPINN has the lowest field
rMAE/rMSE (0.22825/0.31499), but its Re recovery is worse than LAD and
PINN-EBM. PINN-EBM's Re is highly seed-variable. Every method exceeds 51%
mean relative Re error, so this is a failed-identification result under
observation/model-discrepancy confounding, not coefficient-recovery success.

## Baselines and ablations paragraph

> **VERIFIED:** The synthetic selector matrix, synthetic MAD-PINN comparison,
> and all 15 dependent structured-PIV MAD runs are freshly verified by strict
> aggregates.

We added MSE, LAD, OrPINN \(q=1.9,2.9\), direct PINN-EBM, an
estimator-free fixed-quantile gate, an estimator-free learnable
residual-threshold gate, and the two-stage MAD-PINN pipeline. MAD-PINN uses
30,000 LAD updates followed by its published fixed MAD screen and 30,000
retained-data MSE updates, so we label it a 60,000-update,
non-compute-matched reference.

At 15% four-Gaussian corruption its strictly aggregated three-seed rMAE is
\(0.34354\pm0.04332\), \(0.24198\pm0.01160\), and
\(0.12735\pm0.00622\) on Allen--Cahn, Burgers, and lambda--omega. The MAD
screen rejects 98.52%, 98.58%, and 98.42% of known outliers, respectively,
but also rejects 18.95%, 16.67%, and 10.57% of clean scalars. It improves on
the matched 30k-update LAD field rMAE for every PDE, but remains worse than
both direct PINN-EBM and naPINN. This is a 60k-update, non-compute-matched
comparison, not evidence that MAD screening is intrinsically inefficient.

The fixed-quantile selector gives rMAE
\(0.50940/0.37315/0.23726\) on the same three PDEs; because it removes a
fixed 5% of scalars under 15% injected corruption, it rejects only about one
third of known outliers. The estimator-free learnable-threshold gate is much
stronger at \(0.24415/0.08557/0.06730\). It remains worse than naPINN on
Allen--Cahn, is close on Burgers, and is slightly better on lambda--omega;
direct PINN-EBM is lower than both selectors on all three, while naPINN is
the strict-core field winner on Allen--Cahn. Thus screening choice is
PDE-dependent, and neither a universal gate advantage nor sufficiency of a
fixed screen is supported. These fresh values come from
`outputs/rebuttal/synthetic_supplement_recovery_20260726/aggregation_strict.json`.

On the 30% injected-real-PIV condition, MAD-PINN gives
rMAE/rMSE \(0.24848/0.36792\), compared with
\(0.23411/0.35852\) for its LAD stage-one reference
and \(0.21797/0.30886\) for naPINN. The MAD screen
rejects 80.63% of known corrupted and 20.16% of clean scalars. It is worse
than both LAD and naPINN in the two field metrics. This remains a 60,000-update, non-compute-
matched preprocessing reference.

At 20% persistent failure, MAD-PINN gives rMAE/rMSE
\(0.17346/0.25574\), slightly above LAD \(0.17190/0.25310\) and naPINN
\(0.17113/0.25266\). Its fixed screen rejects 91.43% of corrupted and
25.46% of clean scalars.

At 10% persistent failure, MAD-PINN gives
\(0.15738/0.23950\), slightly above naPINN \(0.15538/0.23260\) and above
LAD. It rejects 91.57% of corrupted and 30.42% of clean scalars. naPINN
retains more clean data; MAD has higher failure recall.

Under AR(1) corruption, MAD-PINN gives rMAE/rMSE
\(0.15158/0.23113\), above naPINN \(0.15036/0.21544\) and LAD. MAD
rejects 94.30% of corrupted scalars, but also rejects 27.51% of clean
scalars; naPINN rejects 61.41%/2.96%. This is a failure-recall versus
clean-retention/field-agreement trade-off.

Under spatial-burst corruption, MAD-PINN gives
\(0.14851/0.22623\). Its rMAE is slightly below naPINN while its rMSE is
slightly above, and LAD remains best on both field metrics. MAD rejects
34.04% of clean scalars versus 4.62% for naPINN; both reject 100% of corrupted
scalars.

## EMA, stability, and noise paragraph

> **VERIFIED:** The 108-unique-run supplement and the separate 102-run
> structured-PIV recovery are strictly complete in the destination checkout.

The implementation uses the update convention
\(s_t=(1-m)s_{t-1}+m\hat{s}_t\); thus \(m\) is the new-batch weight and
\(\rho=1-m\) is the old-state decay. We test \(m=0.01,0.05,0.10\), rather
than treating beta and rho as the same number. We also test Gaussian,
Laplace, Student-t, four-Gaussian, persistent sensor drift, AR(1) corruption,
and spatial burst failure.

On Allen--Cahn at 15% four-Gaussian corruption, rMAE is
\(0.10778\pm0.03716\), \(0.09202\pm0.00352\), and
\(0.08551\pm0.00874\) for \(m=0.01,0.05,0.10\), respectively. The
corresponding rMSE is \(0.10931\pm0.03179\),
\(0.09483\pm0.00232\), and \(0.09235\pm0.00494\). Within this single severe
Allen--Cahn cell, \(m=0.10\) has the lowest mean field error and \(m=0.01\)
has the largest variability. We do not claim that 0.10 is universally
optimal; the sweep is a local sensitivity check and resolves the
update-weight/decay notation.

On lambda--omega with 15% extra outliers, rMAE for
MSE/LAD/OrPINN/PINN-EBM/naPINN-MSE/naPINN-LAD/naPINN-q2.9 is
\(0.21636/0.08686/0.07858/0.06853/0.05545/0.06058/0.06004\) for
Gaussian background,
\(0.31517/0.07447/0.07028/0.06921/0.05826/0.05631/0.05954\) for
Laplace,
\(0.09437/0.04516/0.06048/0.04545/0.09773/0.04578/0.06355\) for
Student-t, and
\(0.42187/0.19097/0.15263/0.04023/0.07056/0.13995/0.13801\) for
four-Gaussian noise. naPINN-MSE is best for Gaussian, naPINN-LAD for
Laplace, LAD is narrowly best in Student-t rMAE while direct PINN-EBM is
best in Student-t rMSE, and direct PINN-EBM is best for four-Gaussian noise.
Base naPINN is worse than MSE on Student-t, and the naPINN backbone variants
degrade sharply on four-Gaussian noise. We therefore claim broader coverage,
not noise-family invariance.

The fresh severe Allen--Cahn rejection-cost sweep gives rMAE
\(0.08847/0.09133/0.09202/0.09886/0.86890\) for costs
\(0.10/0.30/0.50/0.70/1.00\). Costs 0.10--0.70 reject about 99.3% of
known outliers, whereas cost 1.00 rejects only 2.12%. We disclose that the
initial rebuttal core used 0.50 while the appendix states 1.00; we do not
silently replace either result or use this sensitivity sweep for post-hoc
selection.

A fresh destination-server recovery strictly completes all nine
cost-1.0 cells. Its 5%/10%/15% rMAE is
\(0.26088\pm0.07158/0.56828\pm0.08379/0.86890\pm0.02391\), and the gate
rejects only 3.07%/2.25%/2.12% of known outliers. This independently
confirms that cost 1.0 accepts almost every outlier and degrades sharply as
corruption increases. The matched destination-server cost-0.5 core and the
five-cost sweep are now both strictly complete. These fresh sensitivity
results still do not establish which configuration generated the submitted
table.

## Compute paragraph

> **VERIFIED FOR UPDATE COUNT:** The isolated 27-run 35k-update recovery is
> strictly complete. The phase-time values below are observational measurements
> on a shared server and should not be interpreted as controlled wall-clock
> matching.

MSE/LAD/OrPINN use 30,000 PINN updates. PINN-EBM and naPINN use 5,000 MSE
warm-up PINN updates, 5,000 estimator-only updates, and 25,000 joint PINN
updates: 30,000 PINN updates plus 5,000 estimator-only updates. We report
phase and end-to-end times including initialization. We additionally give
ordinary baselines 35,000 full PINN updates as a conservative update-count
match. MAD-PINN uses 60,000 PINN updates. Because the A100 server is shared,
wall times are observational rather than hardware-isolated.

The strictly aggregated 35,000-update ordinary-baseline rMAE for
MSE/LAD/OrPINN is \(0.87534/0.34754/0.32539\) on Allen--Cahn,
\(0.59199/0.25301/0.20131\) on Burgers, and
\(0.44803/0.18677/0.14221\) on lambda--omega. Each remains above both
30,000-PINN-update direct PINN-EBM
(\(0.10288/0.04738/0.04023\)) and naPINN
(\(0.09202/0.08186/0.07056\)) from the matched destination-server core.
The ordinary baselines do not improve uniformly from 30k to 35k, so we
interpret this only as a conservative update-count check, not evidence that
extra steps always help or a wall-clock-matched comparison.

Across the severe synthetic runs, the 5,000 estimator-only updates take
20--24 seconds on average, or approximately 2.6--4.3% of total training
time. End-to-end PINN-EBM/naPINN time is
538/568 seconds on Allen--Cahn, 659/828 seconds on Burgers, and
794/798 seconds on lambda--omega. These measurements include warm-up,
estimator initialization, and joint training; joint time also includes
estimator updates.

## B-PINN and correction paragraph

The reviewer is correct that our submitted Bayesian baseline uses mean-field
variational inference rather than the stronger HMC option in the original
B-PINN paper. We refer to it as B-PINN-VI and do not claim equivalence to
HMC. A defensible HMC comparison would additionally require a specified prior,
trajectory/step-size schedule, burn-in, posterior sample budget, and
convergence diagnostics; we did not create an unvalidated HMC result during
the response period. Bayesian inference and adaptive measurement selection are complementary:
Bayesian methods can be robust to noise and anomalies, while naPINN learns
explicit inclusion weights in a point-estimate PINN. The Baydin reference year
should be 2018, not 1989; we acknowledge this typo.

## Limitations paragraph

The injected PIV corruptions are controlled stress tests over a real measured
field, not evidence of naturally occurring labeled sensor failures. Held-out
PIV is not noise-free truth, the nominal PDE can be misspecified, and our
method does not decompose model discrepancy from observation error. The
tested noise families broaden coverage but cannot represent every real
measurement process.
