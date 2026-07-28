# Response to the Area Chair

<!-- Submission-ready text for Area Chair. HTML comments are evidence pointers
for internal audit and must be stripped before posting; run
scripts/rebuttal/strip_response_comments.py to emit the final text.
Character budget: 10,000 excluding comments. -->

We thank the AC for a precise diagnosis. We ran the requested real-data
evaluation, and it produced a result we did not anticipate: **our advantage
over the closest prior is small where the governing equation is exact and large
where it is misspecified**, in the direction the AC's own remark about model
discrepancy predicts. We report it in full below, including the conditions
where we lose.

## 1. A real measurement problem

We evaluated on real time-resolved PIV from the RealPDEBench Cylinder
dataset: 200 frames, 192 fixed irregular training sensors (38,400 velocity
vectors), and 1,549,400 held-out velocity vectors at 7,747 spatial locations
disjoint from the training sensors. We corrupted only the training sensors
and left the held-out PIV untouched, so evaluation is against an independent
real measurement.

We deliberately did not invent a new corruption for this experiment. We
applied the corruption generator of our submitted code at its **default,
predeclared intensity**, so the setting could not be tuned after seeing
results. Seeds 40-42, mean +- sample s.d.:

| Method | 10% rMAE | 10% rMSE | 15% rMAE | 15% rMSE |
| --- | ---: | ---: | ---: | ---: |
| PINN (MSE) | 0.40576 | 0.42918 | 0.57380 | 0.59140 |
| LAD-PINN | 0.24617 | 0.27780 | 0.26997 | 0.30028 |
| OrPINN q=2.9 | 0.21426 | 0.25026 | 0.23106 | 0.26514 |
| naPINN | **0.15517** | **0.21532** | **0.16023** | **0.22266** |

naPINN is lowest in both metrics at both severities, and lower than the
strongest baseline for each of the three seeds individually, not only on
average. Its gate excludes 98.99% / 99.23% of the injected gross components
while excluding 2.95% / 4.23% of the background-only components.

<!-- analysis/results/runs/rebuttal_realpde_legacy_4g/aggregation.json, 48/48 runs -->

To test whether this depends on one trajectory, we predeclared and ran a
144-run matrix over three RealPDEBench PIV scenarios (Controlled Cylinder,
FSI, Foil) x 10%/15% x eight methods x seeds 40-42. naPINN-MSE has the lowest
mean rMAE in all six dataset/severity conditions, but the lowest mean rMSE in
only four: on FSI, OrPINN q=2.9 is lowest in rMSE at both severities
(0.60933/0.61541 versus 0.64632/0.67973). We also retain a failure: on Foil
at 15%, the q=2.9 variant of naPINN collapses on one seed (rMAE 0.70040). We
report the full matrix rather than the favourable subset.

<!-- outputs/rebuttal/realpdebench_multidataset/aggregation.json, 144/144 -->

We also ran five structured corruption families on the real PIV -- persistent
sensor bias plus drift at 10/20/30%, AR(1) temporal correlation, and spatially
clustered burst failure. Here the outcome is genuinely mixed: LAD-PINN is best
on both field metrics under AR(1), 10% drift and spatial burst, while naPINN
is best on both under 20% and 30% drift. The persistent-failure calibration
does not transfer perfectly to temporally correlated corruption.

<!-- outputs/rebuttal/realpde_recovery_20260726/injected_piv_aggregation_strict.json, 102/102 -->

We did not include RealPDEBench Combustion. Its measured channel is OH*
intensity, not velocity, and we have no validated observation operator from
OH* to the velocity-pressure state our solver assumes. Producing a number
there would compare two different problems.

## 2. Observation error together with physics-model discrepancy

This is the part of the AC's comment we take most seriously, and our honest
answer is a negative result.

Applying a nominal 2-D incompressible Navier-Stokes residual with a latent
pressure to real PIV *is* the scenario the AC describes: the governing model
is wrong at some level, and any residual-based error model must absorb that
discrepancy. To test what this does to inference, we ran an inverse problem on
the 30%-drift real-PIV condition, learning the Reynolds number jointly with
the field (initialised at 8000; dataset metadata value 10031):

| Method | learned Re | mean rel. error | field rMAE | field rMSE |
| --- | ---: | ---: | ---: | ---: |
| MSE | 1331.81 +- 111.03 | 86.72% | 0.37221 | 0.49041 |
| LAD | 3634.24 +- 1650.42 | 63.77% | 0.23303 | 0.35739 |
| PINN-EBM | 7681.08 +- 5927.76 | 51.54% | 0.28567 | 0.36947 |
| naPINN | 2306.53 +- 550.97 | 77.01% | **0.22825** | **0.31499** |

naPINN recovers the field best and the coefficient worst among the robust
methods. **Every method exceeds 51% relative error**, and PINN-EBM's
per-seed values are 14260.66 / 2757.24 / 6025.34. We therefore report this as
failed coefficient identification under combined observation and model
discrepancy, not as a success. It also shows concretely that a low nominal PDE
residual does not imply a recovered physical constant: what is identified is
an effective coefficient of the observation-plus-model pair. We do not claim
our method separates the two discrepancies, and we now believe that separation
requires an explicit model-discrepancy term, which naPINN does not have.

The AC's remark that a mis-specified physics model forces the observation
error model to absorb the discrepancy turns out to be measurable in our
results, and Section 3 reports it: the method that relies most heavily on the
learned error density is precisely the one that collapses on real PIV.

<!-- outputs/rebuttal/realpde_recovery_20260726/injected_piv_aggregation_strict.json, 12/12 inverse-Re -->

## 3. Pilar and Wahlstrom [31], and the regime boundary this exposes

The AC is right and we accept the criticism without reservation: [31] is the
closest prior, not a motivating reference. It already studies inverse PINNs
under unknown non-Gaussian measurement noise and already learns the residual
noise distribution with a jointly trained EBM. Our introduction did not convey
that, and we will rewrite it.

We implemented it as a direct baseline -- the learned EBM negative
log-likelihood is the data loss, backpropagated to the PINN, the EBM and the
PDE parameter, with no gate and no rejection cost -- and ran it on every
condition in this response. The comparison is the most informative result we
obtained.

**Where the governing equation is exactly correct, the prior is highly
competitive.** At 15% corruption naPINN is lower on all three benchmarks, but
only by modest margins (the naPINN column is our submitted table, unchanged):

| PDE | PINN-EBM rMAE | naPINN rMAE |
| --- | ---: | ---: |
| Allen-Cahn | 0.153 | **0.134** |
| Burgers | 0.077 | **0.072** |
| lambda-omega | 0.082 | **0.076** |

Its raw EBM surprise also ranks corrupted measurements about as well as our
gate does (AUROC 0.98770 vs 0.98766), so the gate is not what makes corruption
identifiable.

**Where the governing equation is misspecified, the gap opens up.** Across the
13 real-PIV conditions in Section 1 -- Legacy-4G at two severities, three
RealPDEBench scenarios at two severities, and five structured corruption
families -- direct PINN-EBM is **never best, and never even third**. Its best
rank is 4th; on FSI at 10% it is 8th of 8, and under spatially clustered burst
failure it is last of 6, behind plain MSE. Over the same 13 conditions naPINN
is first in all eight gross-outlier conditions and in the two most severe
persistent-failure conditions.

We think this is the substantive answer to the AC's second point rather than a
separate topic. The AC wrote that if physics-model discrepancy is not modelled
explicitly, the observation error model must absorb it, creating very complex
error distributions. Our results are exactly what that predicts: the method
that *makes the residual density do that work* is the strongest when the model
is right and the least reliable when it is wrong, because the density is then
fitting model discrepancy and sensor error together. A mechanism that only
decides **whether to use a measurement**, and never lets that decision reshape
the data-fitting objective itself, degrades far more gracefully. We note that
a fixed robust loss (LAD) also beats direct likelihood optimisation on all
five structured real-PIV families, which supports the same reading.

We therefore do not claim general superiority over [31]. We claim a boundary:
learning and directly optimising the residual likelihood is preferable under a
correctly specified model, and explicit measurement selection is preferable
under the model discrepancy that real measurements bring. We will restructure
the paper around this, positioning [31] as the closest prior with the direct
comparison table above, and stating the mechanistic differences (detached
density objective, explicit per-measurement inclusion, gated base
reconstruction loss, rejection regularisation) as the reason for the different
failure behaviour rather than as a claim of general improvement.

We are explicit about what this does *not* establish: we cannot separate
observation error from model discrepancy, so the boundary is a robust
empirical association across 25 conditions, not an isolated causal mechanism.

## 4. What we cannot claim

The injected corruptions are controlled stress tests on a real measured field,
not observed sensor failures with ground-truth labels. The held-out PIV is an
independent measurement, not noise-free physical truth. The nominal PDE is
misspecified and we cannot decompose that from observation error. Our method's
advantage is regime-dependent, and we have documented the regimes where it
loses.

We hope the real-data evidence, and our willingness to report the adverse
comparisons alongside it, addresses the substance of the meta-review.
