# Meta Review / Area Chair

Thank you for identifying the two issues most likely to change the paper's
assessment: real-data evidence and positioning relative to Pilar and
Wahlström.

**Real PIV and model discrepancy.** We ran a new three-seed experiment on one
RealPDEBench Cylinder trajectory: 200 frames of real time-resolved PIV, with
192 fixed irregular training sensors (38,400 velocity vectors) and 1,549,400
held-out velocity vectors at spatial locations disjoint from training. All
methods use the same pressure-latent \(u,v,p\) network and nominal
two-dimensional incompressible Navier--Stokes residual, with Reynolds number
fixed from the trajectory metadata. The unmodified data contain natural
camera/PIV measurement effects, while disagreement with the nominal 2D model
also makes this a model-discrepancy stress test.

The result is informative but mixed. LAD-PINN has the best held-out-PIV
agreement, with rMAE \(0.11138\pm0.00122\), versus
\(0.16719\pm0.00105\) for naPINN. Held-out PIV is an independent real
measurement set, not noise-free truth. However, relative to a controlled
implementation of the closest no-gate PINN-EBM objective, naPINN lowers rMAE
from \(0.24688\pm0.02633\) to \(0.16719\pm0.00105\) (32.3%) and nominal
momentum-residual RMS from \(0.02354\pm0.00394\) to
\(0.00871\pm0.00070\) (63.0%); its continuity-residual RMS is also 82.6%
lower. Thus the real experiment supports the value of gating over learned
residual likelihood alone and improved nominal-physics fidelity, but not
universal superiority over a fixed robust loss.

We also tested a clearly separated, injected structured failure over the real
PIV: 19 of 192 sensor identities receive persistent bias plus linear drift in
both components for all 200 frames. This is not a naturally observed failure.
For the manifest-recorded primary setting, naPINN lowers rMAE from
\(0.21182\pm0.00261\) for MSE-PINN to \(0.18234\pm0.00711\), ranks failed
measurements with AUROC \(0.96234\pm0.00086\), and rejects 99.29% of failed
scalars. The important negative result is that it also rejects 47.51% of clean
scalars; LAD remains best in held-out agreement at
\(0.13212\pm0.00251\). A separately labeled rejection-cost sensitivity
improves rMAE to \(0.16928\pm0.00131\) and lowers clean rejection to 33.68%
while retaining 98.86% failed rejection, but it does not replace the primary
setting.

This experiment does not estimate an inverse coefficient, does not identify
observation and physics-model discrepancy separately, and covers only one
trajectory and one nominal model. We will state those limitations prominently
rather than interpreting gate rejections as identified observation errors.

**Closest prior work.** We agree that our original positioning understated
Pilar and Wahlström. Their PINN-EBM already contains residual EBM learning,
MSE warm-up, estimator-only initialization, and joint training with EBM
negative log likelihood. These are not our contributions. Our narrower
increment is the trainable per-measurement gate plus rejection regularization.
The direct no-gate result above isolates this increment under the same
backbone, data split, seeds, and 30,000 PINN-update budget, using their reported
Navier--Stokes joint-phase physics weight.

In a revision, we will make Pilar--Wahlström the closest prior in the
Introduction and Related Work, narrow the contribution statement, and add the
three-seed natural-PIV and structured-failure protocols and tables in
Additional Results. Main Results, Conclusion, and Broader Impacts will state
the LAD/naPINN tradeoff, clean-rejection limitation, fixed-Re setup, and
inability to separate the two discrepancy sources. We hope this directly
addresses the evidence and attribution issues underlying the current
recommendation.

# Reviewer 6SDM

Thank you. We ran the requested real-PIV comparison and a direct
Pilar--Wahlström-style no-gate PINN-EBM baseline, then separately tested
persistent sensor bias and drift.

**Real data, irregular sensors, and mismatch.** The new experiment uses 200
frames from one RealPDEBench Cylinder trajectory, 192 fixed irregular training
sensors, and a spatially disjoint dense held-out PIV set. A pressure-latent
\(u,v,p\) PINN is constrained by nominal 2D incompressible Navier--Stokes at
fixed metadata Reynolds number. Across three seeds, LAD-PINN gives the best
held-out-PIV rMAE, \(0.11138\pm0.00122\); naPINN gives
\(0.16719\pm0.00105\). Against the closest no-gate PINN-EBM implementation,
naPINN improves rMAE by 32.3% and nominal momentum-residual RMS by 63.0%.
This is a mixed result: held-out PIV is not noise-free truth, and the setup
does not separate natural observation error from model discrepancy or estimate
an inverse coefficient. We also did not quantify the PIV error's spatial or
temporal correlation structure, so we do not present this as a controlled
correlated-noise experiment.

For the structured test, 19 of 192 fixed sensor identities receive persistent
bias plus linear drift in both velocity components at every frame; held-out
PIV is unchanged. This is injected corruption over real PIV, not a natural
failure. The primary naPINN setting gives rMAE
\(0.18234\pm0.00711\), versus \(0.21182\pm0.00261\) for MSE-PINN, with
failure AUROC \(0.96234\pm0.00086\) and 99.29% failed-scalar rejection. It
also rejects 47.51% of clean scalars, while LAD has the best rMAE
\(0.13212\pm0.00251\). A separate \(0.01\) rejection-cost sensitivity
reduces clean rejection to 33.68% and gives rMAE
\(0.16928\pm0.00131\), while retaining 98.86% failed rejection; we report it
as sensitivity, not as a replacement for the primary \(0.005\) condition.

**Stronger baselines.** We added three-seed comparisons with LAD, the learned
EBM likelihood without a gate, and the published two-stage MAD-PINN screen.
The direct PINN-EBM result is rMAE \(0.24688\pm0.02633\), versus
\(0.16719\pm0.00105\) for naPINN. MAD-PINN uses the published
\(\widehat{\sigma}=\operatorname{median}(|r|)/1.6777\),
\(|r|\leq3\widehat{\sigma}\) screen followed by MSE retraining; it gives
rMAE \(0.14810\pm0.00103\), rMSE \(0.22537\pm0.00095\), momentum RMS
\(0.01399\pm0.00019\), and continuity RMS \(0.01349\pm0.00125\).
MAD is not compute matched: it uses 30,000 LAD plus 30,000 MSE updates.
The submission reports OrPINN at \(q\in\{1.9,2.9\}\), but we did not complete
a comprehensive robust-loss hyperparameter sweep and will not characterize
those settings as exhaustively tuned.

**EMA coefficient.** We agree that the notation was inconsistent. The actual
update is
\(\sigma_{\rm run}\leftarrow(1-\beta)\sigma_{\rm run}+\beta s_{\mathcal B}\)
with new-batch weight \(\beta=0.05\), equivalently old-state decay
\(\rho=0.95\), not 0.99. Before the update, the batch scale is lower-clamped
at \(10^{-6}\) and upper-clamped at ten times the current running scale. We
will use this single convention in the method and hyperparameter sections.
We did not complete a full EMA-coefficient sweep, so we cannot claim
coefficient insensitivity; the revision will preserve the submitted
EMA-versus-per-batch normalization ablation but explicitly limit what it
establishes.

**Compute.** MSE, LAD, PINN-EBM, and naPINN each use 30,000 PINN updates.
For naPINN this is 5,000 warm-up plus 25,000 joint updates; both EBM methods
add 5,000 estimator-only updates. Mean end-to-end observations on shared A100
GPUs were 1373.8 s (MSE), 1451.3 s (LAD), 1717.2 s (PINN-EBM), and
1714.4 s (naPINN). These are not hardware-isolated wall-clock matches. MAD's
60,000-update two-stage pipeline took 2993.6 s on average and is explicitly
not compute matched.

In a revision, we will add the natural-PIV and structured-failure tables,
direct PINN-EBM and MAD baselines, exact update accounting, and the corrected
EMA definition. We will also state the one-trajectory, fixed-Re,
model-discrepancy, clean-rejection, and uncontrolled-timing limitations.

# Reviewer aoJS

Thank you for identifying Pilar and Wahlström as the decisive comparison. We
agree that residual EBM learning and the staged schedule are prior work, and
we ran a direct no-gate comparison together with a new real-PIV evaluation.

**Closest prior and gate ablation.** Pilar and Wahlström already introduce MSE
warm-up, estimator-only EBM initialization, and joint optimization using the
learned EBM negative log likelihood as the data loss. We will not claim these
components as novel. Our narrower contribution is converting residual
likelihood into trainable per-measurement inclusion weights and regularizing
rejection.

We implemented the closest-prior objective without a gate or rejection cost,
using the same RealPDEBench Cylinder trajectory, backbone, irregular
train/held-out split, seeds, and 30,000 PINN updates as naPINN, including the
reported Navier--Stokes joint-phase physics weight. Across three seeds,
PINN-EBM gives rMAE \(0.24688\pm0.02633\), rMSE
\(0.33706\pm0.03480\), momentum-residual RMS
\(0.02354\pm0.00394\), and continuity-residual RMS
\(0.04654\pm0.00338\). naPINN gives \(0.16719\pm0.00105\),
\(0.26616\pm0.00124\), \(0.00871\pm0.00070\), and
\(0.00812\pm0.00139\), respectively: reductions of 32.3%, 21.0%, 63.0%,
and 82.6%. This directly isolates gating plus rejection regularization from
learned residual density alone.

**Real-data scope.** The experiment uses 200 frames of real time-resolved PIV,
192 fixed irregular training sensors, and 1,549,400 held-out velocity vectors
with zero spatial overlap. LAD-PINN nevertheless has the best held-out-PIV
rMAE, \(0.11138\pm0.00122\), so naPINN does not dominate robust losses.
Held-out PIV is an independent measurement set rather than noise-free truth.
The nominal 2D incompressible Navier--Stokes constraint at fixed Reynolds
number introduces a model-discrepancy stress test, but observation error and
model discrepancy are not separately identified.

As an additional fixed-screening comparison, the published two-stage
MAD-PINN procedure gives rMAE \(0.14810\pm0.00103\), rMSE
\(0.22537\pm0.00095\), momentum RMS \(0.01399\pm0.00019\), and
continuity RMS \(0.01349\pm0.00125\). It uses 60,000 PINN updates and is not
compute matched to the 30,000-update methods.

**PDE parameter recovery.** We agree that the submitted evidence supports
detailed parameter recovery only for Allen--Cahn. The required original
Burgers and lambda--omega run artifacts and generated datasets are
unavailable, so we cannot responsibly report viscosity or reaction-parameter
numbers in this response. The real-PIV experiment also fixes Reynolds number;
it is field reconstruction, not inverse-coefficient evidence. We will narrow
the general parameter-recovery claim and explicitly state that the detailed
coefficient evidence is limited to Allen--Cahn.

**Total cost.** All main methods use 30,000 PINN updates. naPINN uses 5,000
warm-up and 25,000 joint updates plus 5,000 estimator-only updates; PINN-EBM
has the same additional estimator-only count. Mean end-to-end observations on
shared A100 GPUs were 1373.8 s (MSE), 1451.3 s (LAD), 1717.2 s
(PINN-EBM), and 1714.4 s (naPINN). These timings include all stages and
evaluation, but are observational rather than a hardware-isolated,
wall-clock-matched benchmark.

**Reference.** The Baydin automatic-differentiation survey year is indeed
2018; 1989 was a bibliography error. We verified that all cited keys resolve
and that bibliography keys are unique; the revision will correct the year.

In a revision, the Introduction, Related Work, and contribution statement will
credit PINN-EBM as the closest prior and narrow novelty to gating and rejection
regularization. Additional Results will include the no-gate, MAD, and
natural-PIV three-seed comparisons; the parameter-recovery and training-cost
sections will receive the limitations and exact accounting above.

# Reviewer 6XZg

Thank you. We agree that the submitted B-PINN label and comparison were too
broad. We also ran the more relevant direct Pilar--Wahlström no-gate baseline
and a real-PIV evaluation, which sharpen both the fairness and novelty claims.

**What is new and what is not.** Pilar and Wahlström already contain residual
EBM learning, warm-up, estimator-only initialization, and joint EBM-likelihood
training. We will not claim those components. The incremental distinction is a
trainable per-measurement inclusion gate plus rejection-cost regularization.
On one real RealPDEBench Cylinder PIV trajectory, a controlled three-seed
no-gate PINN-EBM implementation gives rMAE
\(0.24688\pm0.02633\) and momentum-residual RMS
\(0.02354\pm0.00394\); naPINN gives
\(0.16719\pm0.00105\) and \(0.00871\pm0.00070\), improvements of 32.3%
and 63.0%. However, LAD-PINN has the best held-out-PIV rMAE,
\(0.11138\pm0.00122\). Held-out PIV is not noise-free truth, Reynolds number
is fixed, and observation error is not separated from mismatch with the
nominal 2D Navier--Stokes model.

**B-PINN fairness.** The reviewer is correct: our evaluated model is
mean-field variational inference, not the HMC configuration of Yang et al. It
uses factorized Gaussian weight posteriors, one weight sample per training
step, KL weight \(10^{-6}\), and posterior-mean evaluation. We will rename it
B-PINN-VI throughout, describe these choices, and state explicitly that its
results neither reproduce HMC nor show Bayesian methods to be ineffective. We
did not run HMC and will not imply otherwise.

**Bayesian UQ versus gating.** We agree that the original contrast was
overstated. Bayesian inference can reduce sensitivity to noise with an
appropriate likelihood. naPINN presently point-estimates network and PDE
parameters and learns observation inclusion weights; it does not infer their
posterior. Although its gate has a latent-variable interpretation, the
relevant empirical distinction here is EBM likelihood without a gate versus
EBM-derived pointwise gating with rejection regularization. Posterior UQ and
measurement selection are complementary and could be combined.

**Statistical preprocessing baseline.** We implemented the full published
two-stage MAD-PINN procedure: 30,000 LAD updates, the
\(\operatorname{median}(|r|)/1.6777\), \(3\widehat{\sigma}\) screen, then
30,000 MSE updates on retained components. Across three real-PIV seeds it
gives rMAE \(0.14810\pm0.00103\), rMSE \(0.22537\pm0.00095\), momentum
RMS \(0.01399\pm0.00019\), and continuity RMS
\(0.01349\pm0.00125\). This is a stronger preprocessing baseline, but its
60,000 updates are not compute matched.

**Stability, hyperparameters, and corruptions.** A persistent-failure stress
test injects bias plus linear drift into 19 of 192 sensor identities across
all 200 real-PIV frames. It is synthetic corruption over real PIV, not a
naturally observed failure. The manifest-recorded primary naPINN condition
gives rMAE \(0.18234\pm0.00711\), failure AUROC
\(0.96234\pm0.00086\), 99.29% failed rejection, and 47.51% clean rejection;
LAD is better in held-out rMAE at \(0.13212\pm0.00251\). A separately
labeled rejection-cost sensitivity gives rMAE
\(0.16928\pm0.00131\), AUROC \(0.96673\pm0.00149\), 98.86% failed
rejection, and 33.68% clean rejection. It improves selectivity but does not
replace the primary condition. We did not complete a full EMA-coefficient
sweep. The correct convention is new-batch weight \(\beta=0.05\), old-state
decay \(\rho=0.95\), with an upper clamp at ten times the running scale.

The submission's additional Gaussian and Laplace cases remain controlled IID
tests; they do not answer the deployment concern. We will distinguish those
from the new natural-PIV study and the injected persistent-failure stress test,
and avoid claiming comprehensive hyperparameter optimization.

**Training cost and references.** MSE, LAD, PINN-EBM, and naPINN each use
30,000 PINN updates; the EBM methods add 5,000 estimator-only updates. Mean
end-to-end observations on shared A100 GPUs were 1373.8, 1451.3, 1717.2,
and 1714.4 s, respectively. They are not hardware-isolated wall-clock matches.
MAD uses 60,000 updates and averaged 2993.6 s. We will report these exact
budgets and limitations. The Baydin survey year will be corrected from 1989
to 2018; the cited keys were also checked for resolution and uniqueness.

In a revision, we will rename and delimit B-PINN-VI, rewrite Related Work to
present UQ and gating as complementary, narrow the novelty statement, add the
direct PINN-EBM and MAD comparisons and real-PIV tables, and make the
calibration, model-discrepancy, and timing limitations explicit.

# Character counts

- Meta Review / Area Chair response body: 3,755 characters
- Reviewer 6SDM response body: 4,107 characters
- Reviewer aoJS response body: 3,882 characters
- Reviewer 6XZg response body: 4,670 characters
