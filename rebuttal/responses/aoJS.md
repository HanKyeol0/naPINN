# Response to Reviewer aoJS

<!-- Submission-ready text for Reviewer aoJS. HTML comments are evidence pointers
for internal audit and must be stripped before posting; run
scripts/rebuttal/strip_response_comments.py to emit the final text.
Character budget: 10,000 excluding comments. -->

Thank you for a careful and fair reading, and for stating the relation to [31]
precisely. We ran the comparison you asked for, on synthetic and real data, and
report it in full below.

## Q1. Direct comparison with the closest EBM-based prior [31]

You are right that Pilar & Wahlstrom is the closest prior and that citing it
only as motivation was inadequate.

Why it was not in the original comparison: [31] assumes measurement noise with
no gross outliers to reject, so its setting differs from ours and we judged a
head-to-head comparison potentially unfair to it. That was the wrong call, so
we implemented and ran it on every condition here: the jointly trained EBM's
negative log-likelihood **is** the data loss, backpropagated to the PINN, the
EBM and the unknown PDE parameter, with no gate and no rejection cost. We also
swept its PDE-loss weight (1/10/50), so this is a fair implementation rather
than a strawman.

It is a strong baseline. At 15% corruption, the naPINN column taken unchanged
from our submitted table so nothing is re-tuned in our favour:

| PDE | PINN-EBM rMAE | naPINN rMAE |
| --- | ---: | ---: |
| Allen-Cahn | 0.153 | **0.134** |
| Burgers | 0.077 | **0.072** |
| lambda-omega | 0.082 | **0.076** |

naPINN is lower on all three, but only by modest margins, so we present [31]
as a competitive method rather than one we dominate and do not rest our
contribution on this gap. Its raw EBM surprise also ranks corrupted
measurements about as well as our gate (AUROC 0.98770 vs 0.98766 at 10% on real
PIV), so we do not claim the gate is what makes corruption identifiable.

**What distinguishes naPINN.** Both approximate the residual noise
distribution with an auxiliary module and we agree they look similar at that
level. The differences are in how the estimate is used:

1. **Different training objective.** [31] trains the PINN *through* the EBM's
   negative log-likelihood, so the learned density defines the data-fitting
   objective. naPINN keeps a conventional per-measurement reconstruction loss
   and fits the density on *detached* residuals: the density reweights the
   loss, never replaces it, and never back-propagates into the PINN.
2. **The estimator is a replaceable component, not the method.** Because it
   only emits a scalar reliability score, KDE and GMM drop into the same
   pipeline unchanged and in our ablation sometimes match or exceed the EBM,
   most visibly on Allen-Cahn. The same holds for the loss: L1 and q-Gaussian
   variants swap in and help under some corruptions.
3. **It yields an explicit per-measurement decision.** The gate emits an
   inclusion weight per observation, so under gross corruption it also acts as
   an anomaly detector: on real PIV it excludes 98.99% / 99.23% of injected
   gross components but only 2.95% / 4.23% of clean ones. A likelihood-weighted
   objective gives a score, not an auditable decision.

**On real PIV the gap opens up.** At the predeclared default corruption
intensity, 15% gross rows, seeds 40-42 (held-out rMAE / rMSE):

| Method | rMAE | rMSE |
| --- | ---: | ---: |
| PINN (MSE) | 0.57380 | 0.59140 |
| PINN-EBM (w=1 / w=50) | 0.42417 / 0.33611 | 0.46394 / 0.41420 |
| LAD-PINN | 0.26997 | 0.30028 |
| OrPINN q=2.9 | 0.23106 | 0.26514 |
| naPINN-L1 / q2.9 | 0.22629 / 0.21589 | 0.26952 / 0.26078 |
| naPINN-MSE | **0.16023** | **0.22266** |

The same ordering holds at 10%, and naPINN-MSE has the lowest mean rMAE in all
six conditions of a separate 144-run matrix over three RealPDEBench scenarios.
It is not uniform: OrPINN has the lowest rMSE on FSI, one naPINN variant fails
on a Foil seed (0.70040), and across five structured corruption families a
plain LAD loss is best on three.

naPINN is lower on all three synthetic benchmarks and by a much wider margin on
real PIV, and the widening is worth reporting in its own right. A plausible
reason is that [31] makes the learned density the training objective, so when
the governing equation is only nominal that density must absorb model
discrepancy as well as sensor error, whereas naPINN uses it only to decide
whether to include a measurement. We cannot separate the two discrepancies
experimentally, so we offer this as an explanation rather than a demonstrated
mechanism. We will restructure the introduction and related work with [31] as
the closest prior.

## Q2. Warm-up + density estimation without a trainable gate

Your question isolates the right thing, and we should be precise about what we
had run. The variants below weaken the selector, but the fixed-quantile and
learnable-threshold arms also drop the EBM, removing two components rather than
one. At 15% corruption:

| Selector | Allen-Cahn | Burgers | lambda-omega |
| --- | ---: | ---: | ---: |
| Fixed-quantile screen (no EBM) | 0.509 | 0.373 | 0.237 |
| Learnable residual threshold (no EBM) | 0.244 | 0.345 | 0.148 |
| Two-stage MAD-PINN (published screen) | 0.344 | 0.242 | 0.127 |
| naPINN (full) | **0.092** | **0.082** | **0.071** |

Residual screening alone does not account for the result. The fixed 5% quantile
screen is under-powered by construction: at 15% corruption it can remove only
about a third of the outliers. The estimator-free learnable threshold does
better but still carries 2.1x to 4.2x the error of full naPINN on every
benchmark, and the published MAD screen is worse everywhere despite twice the
update budget.

The exact cell you specified -- warm-up and density estimation retained, *only*
the trainable gate removed -- was genuinely missing, and we have now run it.
Warm-up and the EBM are kept, so the density is still learned and still yields a
per-measurement weight, but the cutoff and steepness stay at their initial
values and the rejection cost is zero: a fixed likelihood-weighting rule that
differs from full naPINN in exactly one component. It runs on all three
benchmarks at 15%, seeds 40-42, and we will post the outcome whichever way it
falls. If it matches full naPINN, the trainable gate and rejection cost are not
carrying the gains and we will say so.

MAD-PINN uses 60,000 PINN updates and discards 10.6-19.0% of clean scalars, so
it is not compute-matched.

## Q3. PDE parameter recovery for all benchmarks

Agreed, reporting only Allen-Cahn was a gap. We now record the learned
parameter in every run. Mean absolute error at 15%, seeds 40-42:

| PDE (parameter) | MSE | LAD | OrPINN q2.9 | PINN-EBM | naPINN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Allen-Cahn (eps) | 0.00355 | 0.00312 | 0.00321 | 0.00295 | **0.00275** |
| Burgers (nu) | 0.00566 | 0.00181 | 0.00123 | **0.00105** | 0.00172 |
| lambda-omega (beta) | 0.328 | 0.0837 | 0.0523 | 0.0104 | **0.0101** |

Against the true values (eps = 0.3, nu = 0.01, beta = 1.0), naPINN recovers the
coefficient to within about 1% on Allen-Cahn (0.9%) and lambda-omega (1.0%) and
is closest on both, so the physics is identified accurately under 15%
corruption there. Burgers viscosity is the hard case: the true value is itself
small, so the same absolute error is a much larger relative one, and PINN-EBM
is closest (10.5%) with naPINN next (17.2%).
We therefore claim accurate identification rather than a uniform advantage, and
report field and parameter error separately rather than letting reconstruction
quality stand in for identification.

On real PIV the picture is different. Learning the Reynolds number on
30%-drift Cylinder PIV (init 8000, metadata 10031) gives relative errors of
86.72% for MSE, 63.77% LAD, 51.54% PINN-EBM and 77.01% naPINN, and PINN-EBM's
per-seed Re swings from 2757 to 14261. **Every method exceeds 51% error**, so
we report that as failed identification under combined observation and model
discrepancy.

## Q4. Total training cost

The 30,000-step budget counts PINN parameter updates. We reported it that way
because the estimator stage is a light one-off touching no PINN parameter, and
for the KDE variant involves no training at all. But we agree the total is what
should be compared, so here it is.

The one item the appendix table omits is that one-off stage, which takes
46-49 s. Adding it to the per-step costs already reported there --
0.0356 / 0.0423 / 0.0458 s for a standard PINN against 0.0397 / 0.0478 / 0.0472
for naPINN on Allen-Cahn / Burgers / lambda-omega -- the end-to-end totals over
the 30,000-step budget are about 1068 / 1269 / 1374 s for the PINN and
1239 / 1482 / 1464 s for naPINN, with initialisation accounting for 3-4% of the
naPINN total. That is a slight overestimate, since it charges the joint-phase
rate to the 5,000 warm-up steps too. We will report the initialisation cost and
these totals alongside the per-epoch table.

As a conservative control we gave the ordinary baselines 35,000 full PINN
updates; their rMAE is 0.87534 / 0.59199 / 0.44803, still above naPINN. That is
an update-count control rather than an equal-time one, so we are also running a
hardware-isolated equal-elapsed-time comparison on a reserved GPU, giving each
baseline exactly naPINN's end-to-end seconds, to be posted during the
discussion.

## Q5. Reference [3]

Correct, thank you. Baydin et al., *Automatic Differentiation in Machine
Learning: a Survey*, is JMLR 18(153), 2018; the 1989 year is a bibliography
error. We will fix it and re-check every entry.

## Additional disclosures found while preparing this response

We would rather report these than have them found later. (i) Appendix D says
`lambda_rej = 1.0` for Allen-Cahn, but that cannot have produced our table:
cost 1.0 re-runs to rMAE 0.86890 with 2.12% outlier rejection, while the table
reports 0.134 and the configuration value is 0.5. (ii) The YAML fields
`init_threshold` / `init_steepness` are read by a different ablation gate and
never reach the main gate. (iii) The text says gross outliers *replace*
observations while the code *adds* a positive offset. We will correct all
three.
