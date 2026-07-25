# NeurIPS 2026 author response — working draft, do not submit yet

This draft intentionally excludes natural/unmodified-PIV results. All frozen
additional-experiment matrices have completed and the verified three-seed
results are populated below. The authoritative Korean analysis is
`rebuttal/rebuttal_report_ko.md`. This remains a working evidence draft and
must be shortened into review-specific responses before submission.

## Shared opening

Thank you for identifying the two central issues: real-data evidence and the
relationship to Pilar and Wahlström. We agree that the latter work is the
closest prior. It already introduced residual-distribution learning with an
EBM, MSE warm-up, estimator initialization, and direct EBM-NLL training. We
therefore narrow our claimed increment to an explicit per-measurement
reliability gate, rejection-cost regularization, and the separation of the
estimator's detached density objective from the PINN's gated reconstruction
objective.

## Real-data paragraph

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
\(0.37100\pm0.00190/0.49065\pm0.00544\) for MSE,
\(0.23558\pm0.00678/0.35840\pm0.00590\) for LAD,
\(0.29745\pm0.00073/0.41328\pm0.00164\) for OrPINN-\(q=2.9\),
\(0.20329\pm0.01529/0.31052\pm0.00543\) for direct PINN-EBM, and
\(0.21194\pm0.01783/0.30034\pm0.02341\) for naPINN. Thus naPINN lowers
rMAE/rMSE by 42.9%/38.8% versus MSE and 10.0%/16.2% versus LAD. The
closest-prior result is mixed: direct PINN-EBM has 4.1% lower rMAE, while
naPINN has 3.3% lower rMSE and 81.8%/92.0% lower nominal
momentum/continuity RMS. naPINN gate AUROC is
\(0.93445\pm0.01306\), with 93.22% corrupted and 28.64% clean-scalar
rejection; direct raw-EBM AUROC is higher
(\(0.95046\pm0.00695\)). We therefore report a field/physics trade-off,
not universal superiority.

For AR(1) temporal corruption, MSE/LAD/OrPINN/PINN-EBM/naPINN rMAE is
\(0.22047/0.12330/0.14541/0.20891/0.14997\); for spatial burst failure it
is \(0.15981/0.11358/0.12634/0.23979/0.15167\). LAD is best on both
field comparisons. naPINN has the lowest nominal momentum/continuity RMS
on both, but its gate rejects only 64.97% of AR(1)-corrupted scalars versus
100% of spatial-burst corrupted scalars. Gate AUROC is 0.91391/0.98303.
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
naPINN rMAE/rMSE \(0.15501\pm0.00119/0.23226\pm0.00463\), below
MSE, OrPINN, and direct PINN-EBM but still above LAD. Its gate AUROC is
\(0.92989\pm0.00134\), rejecting \(84.97\%\pm1.90\%\) of corrupted and
\(3.63\%\pm0.89\%\) of clean scalars; raw-EBM AUROC is higher at
\(0.94513\pm0.00082\). Thus the seed-39 30%-severity rejection constraint
does not transfer perfectly to 10% severity, although clean retention is
greatly improved relative to cost 0.01.

At 20% persistent failure, the complete rMAE/rMSE comparison is
\(0.29806/0.37994\) MSE, \(0.17195/0.25329\) LAD,
\(0.23768/0.32710\) OrPINN, \(0.20534/0.29317\) direct PINN-EBM, and
\(0.16889/0.24869\) naPINN. naPINN is 1.8% lower than LAD on both field
metrics and has 76.6%/81.8% lower nominal momentum/continuity RMS. Its gate
rejects \(91.25\%\pm1.72\%\) of corrupted and
\(9.16\%\pm1.03\%\) of clean scalars. Direct raw-EBM AUROC remains higher
than the naPINN gate (0.95991 versus 0.94192).

On seed-39 calibration at 30% failure, rejection cost 0.10 was the only
candidate satisfying both at least 90% failed-scalar rejection and at most
40% clean-scalar rejection (93.79% and 18.56%, respectively). We fixed 0.10
for the full held-out matrix. Because the earlier 10% cost-0.01 sensitivity
had already completed before refining the seed-39 grid, we do not describe
this process as fully held-out-blind.

## Closest-prior paragraph

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

The full 5%/10%/15% four-Gaussian core is now complete (three PDEs, six
methods, and seeds 40--42). Direct PINN-EBM has the lowest field rMAE in all
nine PDE/ratio conditions, with naPINN second in all nine. We retain that
adverse result and use the wider matrix to support evaluation completeness,
not a superiority claim over the closest prior.

Because equal-weight held-out results had already been inspected, we do not
select a favorable PINN-EBM PDE weight post hoc. We report the full
weight-\(1/10/50\) grid. rMAE is
\(0.08334/0.08226/0.78955\) on Allen--Cahn,
\(0.04783/0.04776/0.04122\) on Burgers, and
\(0.03976/0.03708/0.04551\) on lambda--omega. The optimum is
PDE-dependent and weight 50 catastrophically degrades Allen--Cahn; no
single tuned-weight result is presented as universal.

## Parameter-recovery paragraph

At 15% four-Gaussian corruption, the three-seed absolute parameter errors
for MSE/LAD/OrPINN-\(q{=}2.9\)/PINN-EBM/naPINN are respectively:
Allen--Cahn epsilon \(0.00367/0.00266/0.00313/0.00272/0.00273\);
Burgers viscosity \(0.00558/0.00178/0.00123/0.00080/0.00271\); and
lambda--omega beta \(0.32778/0.08441/0.05231/0.00966/0.01340\).
Thus the additional evidence covers every unknown PDE parameter, while also
showing that PINN-EBM, not naPINN, is strongest in the severe Burgers and
lambda--omega settings.

We also ran an inverse-Re extension on the 30% injected-PIV condition,
initializing Re at 8000 and comparing to metadata Re 10031. Learned
Re/relative error is \(1309\pm155/86.95\%\) for MSE,
\(4133\pm506/58.80\%\) for LAD,
\(4651\pm2561/53.63\%\) for PINN-EBM, and
\(2284\pm533/77.23\%\) for naPINN. Although naPINN has the lowest nominal
physics residual, its Re recovery is worse than LAD and PINN-EBM. We report
this adverse result as evidence that observation corruption and model
discrepancy remain confounded; low nominal-PDE residual does not establish
identification of the metadata coefficient.

## Baselines and ablations paragraph

We added MSE, LAD, OrPINN \(q=1.9,2.9\), direct PINN-EBM, an
estimator-free fixed-quantile gate, an estimator-free learnable
residual-threshold gate, and the two-stage MAD-PINN pipeline. MAD-PINN uses
30,000 LAD updates followed by its published fixed MAD screen and 30,000
retained-data MSE updates, so we label it a 60,000-update,
non-compute-matched reference.

At 15% four-Gaussian corruption its three-seed rMAE is
\(0.32482\pm0.03642\), \(0.24696\pm0.01307\), and
\(0.13008\pm0.00765\) on Allen--Cahn, Burgers, and lambda--omega. The MAD
screen rejects 98.4--98.6% of known outliers but also 10.8--17.8% of clean
scalars. It improves on LAD field error for every PDE, but remains worse than
both PINN-EBM and naPINN.

The fixed-quantile selector gives rMAE
\(0.54862/0.37395/0.23718\) on the same three PDEs; because it removes a
fixed 5% of scalars under 15% injected corruption, it rejects only about one
third of known outliers. The estimator-free learnable-threshold gate is much
stronger at \(0.22245/0.08493/0.06644\). It remains worse than naPINN on
Allen--Cahn, is close on Burgers, and is slightly better on lambda--omega;
direct PINN-EBM remains best on all three. Thus screening choice is
PDE-dependent, and neither a universal gate advantage nor sufficiency of a
fixed screen is supported.

On the 30% injected-real-PIV condition, MAD-PINN gives
rMAE/rMSE \(0.25127\pm0.00817/0.37205\pm0.00738\), compared with
\(0.23558\pm0.00678/0.35840\pm0.00590\) for its LAD stage-one reference
and \(0.21194\pm0.01783/0.30034\pm0.02341\) for naPINN. The MAD screen
rejects 80.45% of known corrupted and 20.30% of clean scalars. It lowers
nominal-physics residuals versus LAD, but is worse than naPINN in both field
metrics and physics residuals. This remains a 60,000-update, non-compute-
matched preprocessing reference.

At 20% persistent failure, MAD-PINN gives rMAE/rMSE
\(0.17087\pm0.00580/0.25418\pm0.00785\), nearly matching LAD
\(0.17195/0.25329\) but slightly above naPINN
\(0.16889/0.24869\). Its fixed screen rejects 91.60% of corrupted and
25.42% of clean scalars.

At 10% persistent failure, MAD-PINN gives
\(0.15850\pm0.00133/0.24077\pm0.00059\), slightly above naPINN
\(0.15501/0.23226\) and above LAD. It has lower nominal-physics residuals
than naPINN, while rejecting 91.38% of corrupted and 30.30% of clean
scalars. naPINN retains more clean data; MAD has higher failure recall.

Under AR(1) corruption, MAD-PINN gives rMAE/rMSE
\(0.15266\pm0.00119/0.23266\pm0.00322\), above naPINN
\(0.14997/0.21480\) and LAD. MAD has lower nominal-physics residuals than
naPINN and rejects 94.34% of corrupted scalars, but also rejects 27.60% of
clean scalars; naPINN rejects 64.97%/3.86%. This is a failure-recall versus
clean-retention/field-agreement trade-off.

Under spatial-burst corruption, MAD-PINN gives
\(0.14967\pm0.00075/0.22770\pm0.00030\). Its rMAE is slightly below
naPINN while its rMSE is slightly above, and LAD remains best on both field
metrics. MAD has lower nominal-physics residuals, but rejects 33.60% of
clean scalars versus 5.53% for naPINN; both reject 100% of corrupted
scalars.

## EMA, stability, and noise paragraph

The implementation uses the update convention
\(s_t=(1-m)s_{t-1}+m\hat{s}_t\); thus \(m\) is the new-batch weight and
\(\rho=1-m\) is the old-state decay. We test \(m=0.01,0.05,0.10\), rather
than treating beta and rho as the same number. We also test Gaussian,
Laplace, Student-t, four-Gaussian, persistent sensor drift, AR(1) corruption,
and spatial burst failure.

On Allen--Cahn at 15% four-Gaussian corruption, rMAE is
\(0.11397\pm0.04817\), \(0.10060\pm0.02587\), and
\(0.10898\pm0.04136\) for \(m=0.01,0.05,0.10\), respectively. The
corresponding rMSE is \(0.11187\pm0.03993\),
\(0.10038\pm0.01791\), and \(0.10613\pm0.03030\). The tested values have
overlapping variability; we do not claim that 0.05 is universally optimal,
but the result shows modest sensitivity over this range and resolves the
update-weight/decay notation.

On lambda--omega with 15% extra outliers, rMAE for
MSE/LAD/OrPINN/PINN-EBM/naPINN-MSE/naPINN-LAD/naPINN-q2.9 is
\(0.21689/0.08798/0.07908/0.06834/0.05422/0.06011/0.05954\) for
Gaussian background,
\(0.31439/0.07414/0.07108/0.06930/0.05800/0.05506/0.05900\) for
Laplace,
\(0.09335/0.04694/0.06184/0.04648/0.09641/0.04686/0.06550\) for
Student-t, and
\(0.42138/0.19194/0.15123/0.03976/0.06989/0.14179/0.13770\) for
four-Gaussian noise. naPINN-MSE is best only for Gaussian; direct PINN-EBM
is best for Student-t/four-Gaussian, and the naPINN backbone variants can
degrade sharply. We therefore claim broader coverage, not noise-family
invariance.

For Allen--Cahn rejection costs \(0.10/0.30/0.50/0.70/1.00\), rMAE is
\(0.09814/0.11383/0.10060/0.09564/0.87859\). The submitted-appendix value
1.00 accepts almost every observation (only 2.21% known-outlier rejection)
and degrades sharply, whereas 0.10--0.70 are similar. We disclose that the
initial rebuttal core used 0.50 while the appendix states 1.00; we do not
silently replace either result or use this sweep for post-hoc selection.
The paper-aligned cost-1.0 extension gives rMAE
\(0.25878/0.56180/0.87859\) at 5%/10%/15% corruption, versus
\(0.09621/0.10159/0.10060\) for cost 0.5. Cost 1.0 rejects only
3.27%/2.18%/2.21% of known outliers, so the discrepancy matters at every
tested ratio and is not confined to the severe cell.

## Compute paragraph

MSE/LAD/OrPINN use 30,000 PINN updates. PINN-EBM and naPINN use 5,000 MSE
warm-up PINN updates, 5,000 estimator-only updates, and 25,000 joint PINN
updates: 30,000 PINN updates plus 5,000 estimator-only updates. We report
phase and end-to-end times including initialization. We additionally give
ordinary baselines 35,000 full PINN updates as a conservative update-count
match. MAD-PINN uses 60,000 PINN updates. Because the A100 server is shared,
wall times are observational rather than hardware-isolated.

The 35,000-update ordinary-baseline rMAE for
MSE/LAD/OrPINN is \(0.88440/0.34645/0.30771\) on Allen--Cahn,
\(0.59215/0.24871/0.19923\) on Burgers, and
\(0.44750/0.18358/0.14191\) on lambda--omega. Each remains above both
30,000-PINN-update PINN-EBM
(\(0.08334/0.04783/0.03976\)) and naPINN
(\(0.10060/0.08179/0.06989\)). The ordinary baselines do not improve
uniformly from 30k to 35k, so we interpret this only as a conservative
update-count check, not evidence that extra steps always help.

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
