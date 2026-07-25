# naPINN NeurIPS 2026 Rebuttal: Cross-Server Experiment Handoff

Last updated: 2026-07-25

Audited repository snapshot before this document was added:
`6e9d8e80656f8ead8d2852a3b82058a0005c3643`

## 1. Purpose and status

This document is the operational handoff for continuing the naPINN rebuttal
experiments on another GPU server. It has two purposes:

1. freeze the scientific and reporting decisions already made; and
2. define the next experiment: apply the submitted synthetic
   four-Gaussian-plus-point-outlier protocol to real Cylinder PIV, first at
   the exact legacy setting and then in a transparent scale search around
   that setting.

This document distinguishes three evidence states:

- **VERIFIED**: a complete non-smoke artifact exists and has been aggregated;
- **PLANNED**: the protocol is specified here but no citable result exists;
- **NOT RUN**: no defensible implementation or full artifact exists; and
- **RESPONSE HOLD**: evidence may be verified internally, but the author has
  not approved its numerical results or ranking for reviewer communication.

The proposed real-PIV four-Gaussian scale search is **PLANNED**, not verified
rebuttal evidence. Do not cite any result from it until all required seeds,
methods, integrity checks, and aggregation are complete.

All current direct naPINN-versus-PINN-EBM numerical comparisons are
**RESPONSE HOLD**. Preserve their artifacts and use them to plan further
experiments, but do not place their values, rankings, or resulting
contribution/novelty conclusions in a reviewer response yet. The response
direction will be chosen only after the planned real-PIV legacy-corruption,
scale, robust-loss, and closest-prior checks are as complete as feasible and
the author explicitly releases the comparison for response use.

## 2. Non-negotiable constraints

1. `paper/neurips_2026.tex` is frozen. Do not edit it, and do not say that the
   paper or supplement was revised during author response.
2. New experimental results may be communicated to reviewers, but the final
   OpenReview responses must contain no links and must stay below 10,000
   characters per reviewer.
3. The submitting author must confirm the exact response deadline and time
   zone.
4. Natural or unmodified PIV results are excluded from rebuttal evidence by
   author instruction. Only controlled corruption injected into real PIV is
   eligible.
5. Smoke tests demonstrate implementation health only. They are never
   scientific evidence.
6. Report positive, mixed, and adverse results. Do not retain only settings
   in which naPINN wins.
7. Every comparison must use identical corrupted observations within a
   `(corruption seed, ratio, scale)` block.
8. All full training runs must use an explicit device in `cuda:0` through
   `cuda:7`. The server is shared; inspect free memory before dispatch and do
   not call its wall times hardware-isolated benchmarks.
9. Do not overwrite existing verified artifacts. Use a new output root for
   the four-Gaussian PIV experiment.
10. The generated Burgers and lambda--omega data and current additional runs
    are traceable new experiments, not bitwise reproductions or replacements
    of the submitted tables.
11. Direct PINN-EBM results are internal evidence under `RESPONSE HOLD`.
    Additional experiments may refine the response strategy, but must not
    delete, overwrite, or selectively discard existing outcomes.

## 3. Main rebuttal position

### 3.1 What the rebuttal must accomplish

The Area Chair identified two score-moving issues:

- no compelling real example, especially under both observation corruption
  and physics-model discrepancy; and
- insufficient recognition of Pilar and Wahlström as the closest prior.

The response should lead with controlled corruption over real PIV. Because
the Area Chair and reviewer aoJS explicitly raised Pilar--Wahlström, their
eventual responses must address the relationship directly and technically.
The numerical comparison and final positioning are currently on
`RESPONSE HOLD` pending the additional experiment campaign and an explicit
author decision. This does not require volunteering a global reduction of
the paper's contribution claim in every response. Additional parameter,
compute, sensitivity, preprocessing, and structured-corruption results
should be routed only to the reviewer concerns they answer.

### 3.2 Real-data claim boundary

RealPDEBench contains five dataset families:

- Cylinder;
- Controlled Cylinder;
- Fluid--Structure Interaction (FSI);
- Foil; and
- Combustion.

The first four include real time-resolved PIV velocity measurements.
Combustion contains real OH* chemiluminescence intensity rather than a direct
PDE state. Cylinder is not the only possible dataset. It is the currently
implemented, lowest-assumption stationary-flow case compatible with the
pressure-latent 2D incompressible Navier--Stokes PINN.

Safe wording:

> We selected stationary Cylinder as the lowest-assumption real-PIV case
> compatible with our pressure-latent 2D Navier--Stokes implementation.

Controlled Cylinder and FSI require moving-boundary, control, or coupled
physics treatment. Foil is a planar slice of a three-dimensional flow and
therefore has stronger model-mismatch concerns. Combustion would require a
new observation operator from latent thermo-chemical states to OH*
intensity.

The current rebuttal experiment scope is deliberately limited to Cylinder.
The other RealPDEBench datasets remain valid candidates for future
validation after implementing their dataset-specific moving-boundary,
coupled-physics, three-dimensional-model-discrepancy, or observation-operator
requirements. They are deferred rather than ruled out. Do not present this
as a promise that those experiments will be completed during the rebuttal.

Do not claim that no public noisy PDE data exist. The defensible gap is
narrower:

> We did not identify a public inverse-PDE corruption benchmark that jointly
> supplies known severe-corruption labels, a paired uncorrupted physical
> reference, the governing PDE, and unknown coefficients.

The held-out Cylinder PIV is an independent real measurement reference, not
noise-free physical ground truth. Applying nominal 2D Navier--Stokes to it
creates a useful model-discrepancy stress, but naPINN does not separate
observation error from physics-model discrepancy.

### 3.3 Closest-prior and novelty position

#### Response-scope rule

Do not foreground PINN-EBM or proactively recast naPINN as only a shallow
increment in responses that do not raise this relationship. Address it where
the reviewer or Area Chair explicitly asks:

- the Area Chair says Pilar--Wahlström is "much closer ... than is
  acknowledged";
- reviewer aoJS identifies it as the closest prior and requests a direct
  comparison; and
- reviewer 6SDM requests a learned unknown-noise/EBM-likelihood baseline,
  which requires an objective-level comparison but not a separate global
  novelty concession.

The mechanism-level explanation below is approved for planning, but the final
response wording and all direct numerical results remain on `RESPONSE HOLD`.
Do not turn the compact wording into a submitted paragraph until the author
chooses the response direction after the additional experiments.

In those responses, acknowledge the actual shared components without
minimizing the full method. Pilar and Wahlström already use:

- learning an unknown residual distribution with an EBM;
- MSE warm-up;
- estimator-only initialization;
- using learned EBM negative log likelihood directly as the PINN data loss;
  and
- joint PINN, EBM, and PDE-parameter optimization.

Then state the methodological distinction firmly and concretely:

- **PINN-EBM:** the learned EBM negative log likelihood is itself the PINN
  measurement objective. Its gradient passes through the residual and jointly
  updates the PINN, PDE parameter, and EBM.
- **naPINN:** the density estimator is trained on detached residuals. Its
  density score is used as input to an explicit trainable per-measurement
  inclusion gate. The PINN receives a gate-weighted reconstruction objective,
  while rejection-cost regularization prevents indiscriminate rejection.
- This separation gives the density estimator, inclusion mechanism, and
  reconstruction loss different roles. It also permits the gate to be
  composed with MSE, L1, or q-Gaussian reconstruction rather than making EBM
  likelihood the reconstruction loss itself.

Thus the response should not merely say that naPINN "adds a gate." It should
explain the different objective and gradient paths and the explicit selective
learning mechanism. At the same time, do not claim that residual EBM
learning, warm-up, estimator initialization, or raw anomaly ranking was first
introduced by naPINN.

Recommended compact wording for an explicitly raised closest-prior question:

> We agree that PINN-EBM is closely related because both approaches learn a
> residual density with an EBM. Their training roles are different:
> PINN-EBM directly uses EBM NLL as the PINN measurement objective, whereas
> naPINN fits the estimator on detached residuals and uses its score to train
> an explicit per-measurement gate. The PINN is optimized with a gated base
> reconstruction loss plus rejection regularization, separating density
> estimation, inclusion, and reconstruction. We therefore include the
> direct no-gate PINN-EBM comparison to isolate this difference.

Do not argue that PINN-EBM failed. In the completed synthetic comparison,
direct PINN-EBM has the lowest field rMAE in all nine PDE/ratio conditions.
These values are retained internally but are not currently approved for the
reviewer response. After the experiment campaign, any released comparison
must include the complete relevant outcome rather than only settings
favorable to naPINN. The existing empirical result limits an
accuracy-superiority claim over PINN-EBM; it does not by itself erase the
methodological distinction above.

### 3.4 Bayesian baseline position

The submitted B-PINN implementation is mean-field variational inference, not
the HMC configuration of Yang et al. Call it **B-PINN-VI**. Its results
neither reproduce HMC nor establish that Bayesian methods are ineffective.

Bayesian posterior inference and adaptive measurement selection are
complementary. naPINN is a point-estimate method with learned inclusion
weights; it does not replace posterior uncertainty quantification. No
validated HMC comparison has been run, and a rushed sampler result must not
be implied.

## 4. Exact legacy synthetic corruption protocol

This is the protocol that must define the center of the new real-PIV
experiment.

### 4.1 Background distribution

Each scalar background error is drawn from the equal-weight mixture

\[
p(z)=\frac14\sum_{k=1}^{4}\mathcal N(\mu_k,\sigma_k^2),
\]

with raw component `(mean, standard deviation)` pairs

```text
(-9.0, 2.0)
(-0.3, 4.0)
( 2.7, 0.6)
( 8.5, 1.0)
```

The raw mixture has mean `0.475`, variance `45.271875`, and standard
deviation approximately `6.728438`. It is therefore non-zero-mean and
strongly multimodal.

The current code does not scale this mixture by the clean-data standard
deviation. It uses

\[
s_{\rm legacy}=0.1\,\operatorname{mean}(|y_{\rm reference}|),
\qquad
\epsilon_{\rm bg}=s_{\rm legacy}z.
\]

For two-output problems, `mean(abs(y_reference))` is pooled over both
components.

### 4.2 Gross point outliers

The code selects vector measurement rows without replacement. At every
selected row, each scalar component receives an independent **positive**
additive offset

\[
\epsilon_{\rm gross}=a\,\widehat\sigma_{\rm bg},
\qquad a\sim U(3,10),
\]

where `sigma_bg_hat` is the unbiased empirical standard deviation of the
entire realized background-error tensor before gross-outlier injection.

Important consequences:

- outliers are added to the background-corrupted value; they do not replace
  the observation;
- the gross offsets are positive, not randomly signed;
- Burgers and lambda--omega select vector rows and corrupt both components;
- the factor is relative to background-noise standard deviation, not
  clean-data standard deviation; and
- changing the background scale also changes the absolute gross-outlier
  amplitude.

The submitted paper describes replacement by uniformly sampled values, while
the code adds positive offsets. This paper/code discrepancy must not be
hidden. For a code-compatible legacy transfer, follow the implementation
above and state the actual semantics in any response.

### 4.3 Observation counts

| Benchmark | Vector/scalar rows | 5% count | 10% count | 15% count |
| --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | 22,500 scalar rows | 1,120 | 2,250 | 3,375 |
| Burgers | 33,525 vector rows | 1,690 | 3,360 | 5,030 |
| lambda--omega | 56,250 vector rows | 2,800 | 5,700 | 8,450 |

The realized fractions differ slightly from the nominal labels because the
runner uses fixed integer counts. Residual components are flattened before
EBM fitting, gating, and detection scoring.

### 4.4 Legacy scale expressed in clean/reference standard deviations

The following values are derived diagnostics, not new training results.
They use the clean/reference arrays in the current checkout and the
theoretical raw-mixture standard deviation. Realized empirical values vary
slightly by corruption seed.

| Dataset | pooled mean absolute value | pooled reference std. | expected background std. / reference std. | gross offset range / reference std. |
| --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | 0.220573 | 0.334266 | 0.444 | 1.33--4.44 |
| Burgers | 0.298357 | 0.356456 | 0.563 | 1.69--5.63 |
| lambda--omega | 0.407265 | 0.479105 | 0.572 | 1.72--5.72 |
| Cylinder PIV, nondimensional training velocities | 0.470548 | 0.454142 | 0.697 | 2.09--6.97 |

For Cylinder PIV, the exact legacy center is therefore already a severe
corruption setting when measured against the pooled pre-injection training
PIV standard deviation. On the current prepared artifact:

```text
velocity scale U                     = 0.3343667 m/s
pooled mean(abs(u/U, v/U))           = 0.4705483
pooled std(u/U, v/U)                 = 0.4541421
s_legacy                             = 0.0470548
expected background-noise std        = 0.3166055
expected background std / PIV std    = 0.6971508
expected gross offset / PIV std      = 2.0915 to 6.9715
```

Recompute and record these diagnostics on the destination server. In reviewer
text, call the original PIV values **pre-injection measurements** or a
**reference**, not clean physical truth.

## 5. Priority new experiment: legacy-matched corruption on real PIV

**Status: PLANNED.**

### 5.1 Scientific purpose

This experiment places the same corruption mechanism used in the submitted
synthetic benchmarks on a real measured field with irregular sensors and a
nominal PDE model. It is more directly comparable to the submitted tables
than the existing persistent-failure experiment while still retaining real
PIV geometry and model discrepancy.

It does not demonstrate naturally occurring labeled sensor failure. The
field is real; the added corruption and its labels are controlled.

This handoff requests full runs only for Cylinder. Controlled Cylinder, FSI,
Foil, and Combustion are future extension candidates under the model changes
described in Section 3.2, not part of the current experiment queue.

### 5.2 Data and split

Use the existing prepared RealPDEBench Cylinder artifact:

```text
trajectory                         10031.h5
frames                             1000--1199 inclusive (200)
fixed irregular training sensors  192
training velocity vectors          38,400
held-out spatial locations         7,747
held-out velocity vectors          1,549,400
training/held-out spatial overlap  0
prepared artifact SHA-256          0aea0fe1e85ed15ee381443106e2478cf61554c91c5969c2ab62e78d2b47c0c7
```

Inject corruption only at the 38,400 training vector rows. Leave every
held-out PIV value bitwise unchanged.

At the two requested ratios:

```text
10% gross-outlier rows = 3,840 velocity vectors = 7,680 scalar components
15% gross-outlier rows = 5,760 velocity vectors = 11,520 scalar components
```

Both `u` and `v` at a selected row receive independent gross-offset factors,
matching the Burgers/lambda--omega vector semantics. The remaining training
components still contain the four-Gaussian background injection.

### 5.3 Artifact-generation algorithm

Implement a dedicated generator, for example:

```text
scripts/rebuttal/inject_realpdebench_legacy_4g.py
```

Do not modify the base prepared NPZ in place. For each corruption seed:

1. load the unmodified prepared Cylinder artifact;
2. extract the 38,400 pre-injection `(u,v)` training vectors;
3. nondimensionalize by `U = Re * nu / D`, exactly as the PIV experiment;
4. compute the pooled `mean(abs(y_reference))` and pooled sample standard
   deviation;
5. sample one shared raw 4G realization for all 76,800 scalar components;
6. multiply it by `0.1 * mean(abs(y_reference))` at the legacy center;
7. compute the unbiased empirical standard deviation of that background
   tensor;
8. draw one random permutation of the 38,400 vector rows;
9. use its first 3,840 rows for 10% and first 5,760 rows for 15%, making the
   severity subsets nested;
10. draw independent uniform factors for `u` and `v`, and add
    `factor * empirical_background_std`;
11. convert back to physical units for storage if the NPZ schema requires it;
12. preserve the original full fields as `u_clean_mps` and `v_clean_mps`;
13. store a two-component `training_corruption_mask` marking only gross
    point outliers; and
14. verify that all held-out values and all non-training spatial locations
    are bitwise unchanged.

Because every training component receives background 4G noise, a false entry
in the gross-outlier mask means **background-only**, not physically clean.
Accordingly, report `gross-outlier rejection` and `background-only
rejection`, not `failed` and `clean` rejection.

Store the following metadata in every generated artifact:

- parent artifact path and SHA-256;
- generator source commit and dirty patch status;
- corruption seed;
- raw GMM component parameters;
- requested and realized row ratio;
- base-scale multiplier and resolved scale;
- gross-scale multiplier and resolved uniform range;
- pooled pre-injection mean absolute value and standard deviation;
- realized background mean and standard deviation;
- realized background standard deviation divided by reference standard
  deviation;
- realized gross-offset minimum, median, maximum, and quantiles in both
  background-standard-deviation and reference-standard-deviation units;
- gross row indices and a checksum of them;
- checksum of the background-noise tensor;
- confirmation that 10% indices are a subset of 15% indices;
- confirmation that held-out PIV is bitwise unchanged; and
- final artifact SHA-256.

Materialize or checksum the underlying mixture-component choices, Gaussian
draws, outlier permutation, and uniform factors. This makes the corruption
paired across methods and scale settings even if training is distributed
across heterogeneous servers.

### 5.4 Exact-center run comes first

Before any scale search, run the exact legacy center:

```text
raw GMM parameters       unchanged
base noise.scale         0.1
gross factor range       U(3,10)
gross row ratios         10%, 15%
naPINN rejection cost    0.5
EMA update weight        0.05
PDE/data weights         current PIV defaults unless method-specific prior
model/update/evaluation  current audited PIV protocol
```

The exact-center matrix is the primary result because it is not selected to
favor naPINN. Report it even if naPINN loses.

### 5.5 Methods at the exact center

The minimum matched matrix is:

| Method | Role |
| --- | --- |
| MSE-PINN | standard MSE reference |
| LAD-PINN | L1 robust-loss reference |
| OrPINN `q=2.9` | submitted q-Gaussian robust-loss reference |
| direct PINN-EBM | closest prior; no gate |
| naPINN-MSE | submitted gate with MSE reconstruction loss |
| naPINN-L1 | gate composed with L1 reconstruction loss |
| naPINN-q2.9 | gate composed with the same q-Gaussian loss |

If capacity permits, also include OrPINN `q=1.9`. Every naPINN loss variant
must be compared with its ungated matched loss. Do not tune a q value only
for naPINN.

For direct PINN-EBM, include both:

- the equal PDE/data-weight condition; and
- the literature-motivated PIV joint PDE weight 50 condition.

If additional weights are evaluated, preserve the complete frozen grid. While
`RESPONSE HOLD` is active, retain it internally; if the author later releases
it, report the complete relevant grid rather than only the favorable weight.

These experiments are required for internal decision-making. Their direct
naPINN-versus-PINN-EBM rankings remain on `RESPONSE HOLD` until the author
reviews the completed campaign.

The current real-PIV runner hard-codes squared reconstruction loss inside the
`napinn` branch and has no `napinn_lad` or `napinn_q29` method. These variants
therefore require an audited runner extension before full runs. The extension
must:

- preserve the detached EBM objective and existing gate;
- change only the reconstruction/base data loss;
- add the resolved loss and q value to config and metrics;
- give every method a collision-free output tag; and
- pass forward/backward smoke tests for all three naPINN losses.

### 5.6 Training budgets

- MSE, LAD, and OrPINN: 30,000 PINN updates.
- PINN-EBM and all naPINN variants:
  5,000 MSE warm-up PINN updates, 5,000 estimator-only updates, and
  25,000 joint PINN updates. This is 30,000 PINN updates plus 5,000
  estimator-only updates.
- MAD-PINN, if added later: 30,000 LAD updates, fixed MAD screening, then
  30,000 retained-data MSE updates. It is a 60,000-update,
  non-compute-matched reference.

The current PIV architecture remains five hidden layers, width 80, `tanh`.
Do not change architecture, batches, collocation procedure, normalization, or
evaluation while changing corruption scale.

## 6. Scale exploration around the exact center

**Status: PLANNED exploratory extension.**

### 6.1 Parameterization

Keep the raw GMM component parameters and gross-outlier ratios fixed. Define
two dimensionless multipliers:

\[
b=\frac{\text{background scale}}{\text{legacy background scale}},
\qquad
o=\frac{\text{gross factor range}}{U(3,10)}.
\]

Resolve them as

```text
noise.scale = 0.1 * b
gross factor range = U(3*o, 10*o)
```

Use the geometric grid

```text
b in {0.5, 1.0, 2.0}
o in {0.5, 1.0, 2.0}
ratios in {0.10, 0.15}
```

The exact legacy center is `(b,o)=(1,1)`.

Changing `b` scales both background noise and gross offsets because gross
offsets are defined from the realized background standard deviation.
Changing `o` varies gross severity relative to that background. The full
two-dimensional grid is therefore required to distinguish global corruption
amplitude from gross-to-background contrast.

For Cylinder PIV, theoretical expected scales relative to the pooled
pre-injection PIV standard deviation are:

```text
background std / reference std = 0.69715 * b
gross offset / reference std   = [2.0915, 6.9715] * b * o
```

Always report realized empirical values from each generated artifact.

### 6.2 Anti-cherry-picking design

The purpose is to identify the regime in which gating helps, but the
experiment must remain interpretable.

1. Freeze the complete 3-by-3 scale grid, both ratios, methods, metrics, and
   selection rule before inspecting new held-out results.
2. Use seed 39 only as an exploratory/development seed.
3. Evaluate every grid cell on seed 39; preserve the complete heatmap,
   including failures.
4. Select at most three non-center candidate cells using the rule below.
5. Freeze those candidates before running reporting seeds 40, 41, and 42.
6. Report the exact center regardless of outcome.
7. If a selected non-center result is cited, disclose that it was selected
   from the stated development grid and report all selected candidates, not
   only the successful one.
8. If time permits, the strongest design is to run all nine scale cells on
   seeds 40--42 and report a complete phase diagram. This removes the need
   for candidate selection.

Candidate ranking should use the best naPINN loss variant against the best
non-naPINN baseline, but a candidate is eligible only if, on seed 39:

- the naPINN variant is lower in both held-out rMAE and rMSE;
- it improves its matched ungated loss in both metrics;
- learned-parameter or physics-residual behavior is not catastrophically
  worse;
- gross-outlier rejection is at least 80%; and
- background-only rejection is at most 40%.

Rank eligible cells by the smaller of the relative rMAE and rMSE gains over
the strongest non-naPINN baseline. If no cell is eligible, do not relax the
rule after seeing results; report that no promising regime was found.

A descriptive "confirmed win" requires the seeds-40--42 mean of one naPINN
variant to be lower than every frozen baseline in both rMAE and rMSE. With
only three seeds, avoid formal significance claims. Also show per-seed paired
differences.

### 6.3 Execution priority

Use this order:

1. generate and validate paired corruption artifacts;
2. run exact-center 10% and 15% matrices on seeds 40--42;
3. run the complete seed-39 scale grid;
4. freeze candidate cells;
5. run candidates on seeds 40--42;
6. optionally confirm the complete 3-by-3 grid;
7. aggregate only after all required cells are complete.

Do not tune rejection cost, EMA momentum, PDE weight, q, or architecture
inside the scale sweep. Any such tuning is a separately labeled experiment
with its own calibration and complete reporting rule.

## 7. Destination-server implementation and launch checklist

### 7.1 Transfer

Transfer:

- the repository at the recorded source revision plus every intended patch;
- the prepared Cylinder NPZ and its checksum;
- `configs/experiment/realpdebench_cylinder_*.yaml`;
- `scripts/rebuttal/run_realpdebench.py`;
- the new corruption generator and scale-search queue after implementation;
- any completed reference aggregates needed for comparison; and
- this handoff document.

The repository `requirements.txt` is not a complete reproducible Python
environment specification. Capture the source server environment
(`python`, PyTorch, CUDA, NumPy, SciPy, scikit-learn, PyYAML, and other
imports) and record the destination versions in every run. Do not assume
that installing only `requirements.txt` reproduces the current environment.

The current audited environment reported:

```text
PyTorch       2.11.0a0+eb65b36914.nv26.02
NumPy         2.1.0
SciPy         1.17.0
scikit-learn  1.8.0
PyYAML        6.0.1
pandas        3.0.1
```

Exact package equality is preferable, but the essential requirement is that
the destination version, CUDA stack, and source snapshot are recorded and
smoke-tested.

### 7.2 Data integrity

Verify before launch:

```bash
sha256sum \
  analysis/results/runs/rebuttal_realpde/data/realpdebench_cylinder_10031_frames1000-1200_stride1.npz
```

Expected SHA-256:

```text
0aea0fe1e85ed15ee381443106e2478cf61554c91c5969c2ab62e78d2b47c0c7
```

If synthetic checks are also transferred, the current regenerated-data
checksums are:

```text
ba83755bfd03115838341d757784e85d7f91854e457d5be856de26fd773d0d27  Burgers2D_3-5/data.npz
d046ba0f2c0c46e54c301b2c2a1c241da956d09bde70c4c9b11ee3f416bf92f6  LambdaOmega_Spiral_2/data.npz
```

### 7.3 Static and smoke validation

After implementing the generator, runner variants, and queue:

```bash
python -m compileall -f -q pinnlab analysis scripts/rebuttal
python -m pinnlab.train --help
python -m analysis.evaluate_checkpoint --help
python -m scripts.rebuttal.run_realpdebench --help

python - <<'PY'
from pathlib import Path
import yaml
for path in Path("configs").rglob("*.yaml"):
    yaml.safe_load(path.read_text())
print("YAML: OK")
PY

for script in $(find scripts analysis/scripts -name '*.sh' -type f); do
  bash -n "$script"
done

git diff --check
git diff --exit-code -- paper/neurips_2026.tex
```

Run one short forward/backward smoke test per affected method and one data-only
generation check per ratio/scale family. Smoke outputs must be placed under a
clearly labeled smoke root and excluded from aggregation.

### 7.4 GPU dispatch

Before launching:

```bash
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv
```

Use only `cuda:0` through `cuda:7`. Avoid a device with less than 4096 MiB
free. A PINN run is small enough to share a high-memory GPU, but existing
processes must not be terminated or displaced.

Prefer assigning a complete
`(corruption seed, ratio, b, o, method-family)` block to one device. Use a
deterministic queue and save status after every child process. A command
template after variant implementation is:

```bash
python -m scripts.rebuttal.run_realpdebench \
  --exp-config configs/experiment/realpdebench_cylinder_napinn_lad.yaml \
  --variant-config configs/experiment/realpdebench_cylinder_4g_r10_b1_o1.yaml \
  --seed 40 \
  --device cuda:0 \
  --run-name seed_40_r10_b1_o1
```

The exact filenames above are planned names, not files that currently exist.
The queue must fail rather than overwrite an existing non-empty run
directory. Partial runs should be quarantined and rerun into a new directory.

### 7.5 Required per-run outputs

Every completed run must contain:

- exact command;
- source commit and dirty status;
- resolved config;
- input artifact path and SHA-256;
- corruption metadata and scale diagnostics;
- optimizer and corruption seeds;
- device and software metadata;
- `status: complete` or equivalent evidence flag;
- final checkpoint;
- field rMAE and rMSE;
- nominal momentum and continuity residual RMS;
- effective or learned PDE parameter, if enabled;
- raw-EBM AUROC where applicable;
- gate AUROC, gross-outlier rejection, background-only rejection, and
  retained fraction where applicable;
- warm-up, estimator-only, joint, evaluation, and end-to-end times;
- PINN and estimator-only update counts; and
- peak allocated and reserved GPU memory.

Aggregate mean and sample standard deviation over reporting seeds. Preserve
per-seed values and paired differences. Exclude calibration seed 39 whenever
reporting seeds 40--42 are requested.

## 8. Existing verified evidence retained for internal decision-making

**Response-use status for direct naPINN-versus-PINN-EBM comparisons: HOLD.**
The values below remain visible so that the next server does not repeat,
overwrite, or selectively ignore prior evidence. They are not approved text
for the reviewer response. Complete the planned comparisons first, then ask
the author to choose whether and how the direct results should be used.

### 8.1 Evidence-status summary

| Topic | Status | Conclusion |
| --- | --- | --- |
| 5/10/15% four-Gaussian synthetic core | VERIFIED | 3 PDEs, 6 methods, seeds 40--42 complete |
| PINN-EBM PDE-weight grid | VERIFIED / RESPONSE HOLD | weights 1/10/50 complete; outcome is PDE-dependent |
| EMA and rejection-cost sweeps | VERIFIED | EMA is modest over tested range; cost 1.0 is strongly adverse |
| Other noise families and naPINN losses | VERIFIED | composability is noise-dependent |
| MAD and selector baselines | VERIFIED | mixed; MAD is not compute matched |
| Injected Cylinder PIV | VERIFIED; direct comparison on RESPONSE HOLD | persistent, AR(1), and spatial-burst matrices complete |
| PIV inverse-Re extension | VERIFIED, adverse | does not establish successful coefficient recovery |
| HMC B-PINN | NOT RUN | submitted baseline is B-PINN-VI |
| PIV legacy 4G point-outlier transfer | PLANNED | defined by this handoff |

### 8.2 Complete synthetic four-Gaussian field result

Field rMAE, mean over seeds 40--42:

| PDE | Outlier ratio | MSE | LAD | Or q1.9 | Or q2.9 | PINN-EBM | naPINN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Allen--Cahn | 5% | 0.3372 | 0.2710 | 0.2218 | 0.1616 | **0.0724** | 0.0962 |
| Allen--Cahn | 10% | 0.5995 | 0.3071 | 0.4279 | 0.2389 | **0.0800** | 0.1016 |
| Allen--Cahn | 15% | 0.8396 | 0.3592 | 0.6636 | 0.2632 | **0.0833** | 0.1006 |
| Burgers | 5% | 0.2164 | 0.2005 | 0.1630 | 0.1347 | **0.0418** | 0.0802 |
| Burgers | 10% | 0.4033 | 0.2273 | 0.2646 | 0.1625 | **0.0602** | 0.0783 |
| Burgers | 15% | 0.5862 | 0.2598 | 0.3921 | 0.1941 | **0.0478** | 0.0818 |
| lambda--omega | 5% | 0.1528 | 0.1613 | 0.0899 | 0.1226 | **0.0453** | 0.0738 |
| lambda--omega | 10% | 0.2885 | 0.1777 | 0.1298 | 0.1394 | **0.0443** | 0.0755 |
| lambda--omega | 15% | 0.4214 | 0.1919 | 0.1754 | 0.1512 | **0.0398** | 0.0699 |

Direct PINN-EBM is first and naPINN second in all nine field conditions.
This supports breadth and improvement over fixed-loss baselines, not
superiority over the closest prior. Direct raw EBM surprise also has slightly
higher AUROC than the gate in the three severe synthetic cases.

The submitted main tables reported ten trials. This additional audited matrix
uses three reporting seeds and regenerated Burgers/lambda--omega datasets.
Do not pool these runs with the submitted ten-trial numbers or describe them
as a regeneration of the submitted table.

The complete PINN-EBM PDE-weight grid `(1,10,50)` has rMAE:

```text
Allen--Cahn   0.08334 / 0.08226 / 0.78955
Burgers       0.04783 / 0.04776 / 0.04122
lambda--omega 0.03976 / 0.03708 / 0.04551
```

The preferred weight is PDE-dependent, and weight 50 catastrophically
degrades Allen--Cahn. Do not select a single favorable weight post hoc.

### 8.3 PDE-parameter recovery

At 15% four-Gaussian corruption, absolute parameter errors for
`MSE / LAD / Or q2.9 / PINN-EBM / naPINN` are:

```text
Allen--Cahn epsilon  0.00367 / 0.00266 / 0.00313 / 0.00272 / 0.00273
Burgers viscosity    0.00558 / 0.00178 / 0.00123 / 0.00080 / 0.00271
lambda--omega beta   0.32778 / 0.08441 / 0.05231 / 0.00966 / 0.01340
```

All three unknown coefficients are now evaluated, but winners vary. Direct
PINN-EBM is stronger than naPINN for severe Burgers and lambda--omega
parameter recovery; Allen--Cahn is nearly tied.

The 30% injected-PIV inverse-Re extension is adverse. Against metadata
`Re=10031`, learned Re is approximately:

```text
MSE       1309 ± 155
LAD       4133 ± 506
PINN-EBM  4651 ± 2561
naPINN    2284 ± 533
```

naPINN has the lowest nominal physics residual but worse Re recovery than LAD
and PINN-EBM. Treat the learned value as a potentially effective coefficient
under model discrepancy, not successful metadata-parameter identification.

### 8.4 Robust-loss composition and noise families

Lambda--omega field rMAE at 15% extra outliers:

| Background | MSE | LAD | Or q2.9 | PINN-EBM | na-MSE | na-L1 | na-q2.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 0.21689 | 0.08798 | 0.07908 | 0.06834 | **0.05422** | 0.06011 | 0.05954 |
| Laplace | 0.31439 | 0.07414 | 0.07108 | 0.06930 | 0.05800 | **0.05506** | 0.05900 |
| Student-t | 0.09335 | 0.04694 | 0.06184 | **0.04648** | 0.09641 | 0.04686 | 0.06550 |
| Four-Gaussian | 0.42138 | 0.19194 | 0.15123 | **0.03976** | 0.06989 | 0.14179 | 0.13770 |

Safe conclusion:

> Alternate reconstruction losses are composable with the gate, but their
> interaction is noise-dependent.

Do not claim universal synergy, noise-family invariance, or cross-PDE
generality from this one-PDE matrix.

### 8.5 Existing controlled real-PIV results

The strongest current real-PIV evidence uses persistent sensor identity
failure. The exact 30% condition means:

- 58 of 192 sensor identities, or 30.2083%;
- both `u` and `v` at every one of 200 frames;
- 11,600 velocity vectors or 23,200 of 76,800 scalar measurements;
- a constant signed `3 * component reference std` bias plus an independently
  signed linear drift ending at `2 * component reference std`;
- corruption seed 20260726 shared by optimizer seeds 40--42; and
- unchanged held-out PIV and unchanged measurements at the 134 non-failed
  training sensors.

This is not point corruption, dropout, corruption of 30% of frames, or
naturally observed failure.

At calibration-selected rejection cost 0.10, 30% persistent-failure
rMAE/rMSE is:

| Method | rMAE | rMSE |
| --- | ---: | ---: |
| MSE | 0.37100 | 0.49065 |
| LAD | 0.23558 | 0.35840 |
| OrPINN q2.9 | 0.29745 | 0.41328 |
| direct PINN-EBM | **0.20329** | 0.31052 |
| naPINN | 0.21194 | **0.30034** |

naPINN beats MSE, LAD, and OrPINN in both field metrics. Against the closest
prior, the result is mixed: PINN-EBM has 4.1% lower rMAE, while naPINN has
3.3% lower rMSE and much lower nominal momentum/continuity residuals.
Direct raw-EBM AUROC `0.95046` is higher than gate AUROC `0.93445`.
naPINN rejects 93.22% of grossly corrupted and 28.64% of reference-clean
scalars.

At 20% persistent failure, naPINN rMAE/rMSE is `0.16889/0.24869`, about
1.8% below LAD on both. At 10%, calibrated naPINN gives
`0.15501/0.23226`, while LAD remains best at `0.13212/0.20881`.

For AR(1), rMAE for `MSE/LAD/Or/PINN-EBM/naPINN` is
`0.22047/0.12330/0.14541/0.20891/0.14997`. For spatial burst, it is
`0.15981/0.11358/0.12634/0.23979/0.15167`. LAD is best on both field
comparisons. naPINN has the lowest nominal physics residuals, but rejects
only 64.97% of AR(1)-corrupted scalars. Calibration transfer is incomplete.

The PIV cost-0.10 calibration used seed 39, with seeds 40--42 for reporting.
It must not be called fully held-out-blind because earlier 10% cost-0.01
held-out runs predated the refined 0.05/0.10 calibration grid.

### 8.6 Preprocessing and selector conclusions

MAD-PINN uses 30,000 LAD updates, the published scalar screen

\[
\widehat\sigma=\operatorname{median}(|r|)/1.6777,\qquad
|r|\leq3\widehat\sigma,
\]

then 30,000 retained-data MSE updates. At severe synthetic corruption its
rMAE is `0.32482/0.24696/0.13008` on Allen--Cahn/Burgers/lambda--omega.
It removes 98.4--98.6% of gross outliers and 10.8--17.8% of clean scalars.
It improves over LAD but remains worse than direct PINN-EBM and naPINN. It is
not compute matched.

The estimator-free fixed-quantile selector gives severe rMAE
`0.54862/0.37395/0.23718` and removes only about one third of the 15%
outliers because its fixed rejection fraction is 5%. The learnable-threshold
gate gives `0.22245/0.08493/0.06644`: worse than naPINN on Allen--Cahn,
close on Burgers, and slightly better on lambda--omega. It is trainable and
must not be described as fixed preprocessing.

### 8.7 Hyperparameter and implementation discrepancies

The implemented EMA convention is

\[
s_t=(1-m)s_{t-1}+m\widehat s_t.
\]

The configured update weight is `m=0.05`, so old-state decay is
`rho=0.95`, not 0.99. The completed severe Allen--Cahn sensitivity gives
rMAE `0.11397/0.10060/0.10898` for `m=0.01/0.05/0.10`. Variability
overlaps; do not claim universal optimality.

The submitted appendix states Allen--Cahn rejection cost 1.0, while the
initial rebuttal core used 0.5. The paper-aligned cost 1.0 gives rMAE
`0.25878/0.56180/0.87859` at 5%/10%/15% and rejects only 2--3% of known
outliers. Preserve and disclose both configurations; do not silently replace
one.

Additional known discrepancies:

- likelihood-gate initializer fields in current YAMLs are not passed to the
  gate constructor; defaults are used;
- the implementation is measurement driven with PDE plus data loss, despite
  the general paper equation containing boundary/initial terms;
- `data_loss_balancer` is the historical name for per-measurement gating, not
  auxiliary dynamic PDE/data-loss balancing;
- several default-named YAMLs are ablations, not canonical paper recipes;
- L1 and q-Gaussian naPINN are new rebuttal variants, whereas the submission
  describes MSE naPINN; and
- Baydin's automatic-differentiation article is from 2018, not 1989.

### 8.8 Compute accounting

- ordinary MSE/LAD/OrPINN: 30,000 PINN updates;
- PINN-EBM/naPINN: 5,000 warm-up plus 25,000 joint PINN updates and 5,000
  estimator-only updates;
- MAD-PINN: 60,000 PINN updates.

In the severe synthetic runs, estimator-only initialization took about
20--24 seconds, or 2.6--4.3% of total training. Joint time also contains
estimator work, so estimator-only time is not total overhead.

Observed total synthetic training time for direct PINN-EBM / naPINN was:

```text
Allen--Cahn    537.5 ± 69.6 / 568.2 ± 63.3 seconds
Burgers        658.5 ± 15.2 / 827.8 ± 154.6 seconds
lambda--omega  793.5 ± 227.1 / 798.2 ± 42.1 seconds
```

For the 30% persistent-failure PIV matrix, observed end-to-end seconds were
approximately `1705.9/1859.1/1764.0/2396.4/2425.6` for
`MSE/LAD/Or q2.9/PINN-EBM/naPINN`.

The conservative 35,000-full-PINN-update ordinary baselines remain worse than
PINN-EBM and naPINN on all three severe synthetic PDEs. This is an
update-count check, not a wall-clock match. All recorded timing comes from
shared A100 servers and is observational.

For reference, severe rMAE for 35k
`MSE/LAD/Or q2.9` versus the 30k-plus-estimator
`PINN-EBM/naPINN` comparison was:

```text
Allen--Cahn    0.88440 / 0.34645 / 0.30771  vs  0.08334 / 0.10060
Burgers        0.59215 / 0.24871 / 0.19923  vs  0.04783 / 0.08179
lambda--omega  0.44750 / 0.18358 / 0.14191  vs  0.03976 / 0.06989
```

## 9. Reviewer-specific response emphasis

### Area Chair

Lead with controlled real-PIV evidence and explain why it is more informative
than a pure synthetic field: real geometry, irregular sensors, real
measurement structure, and nominal model discrepancy remain. State clearly
that corruption labels are injected and that observation/model discrepancy
is not decomposed. Because the Area Chair explicitly raised
Pilar--Wahlström, reserve space for one focused paragraph that acknowledges
the shared EBM components and explains the different objective, gradient
path, explicit gate, and rejection regularization. Do not insert direct
comparison numbers or finalize the positioning while `RESPONSE HOLD` is
active. Do not frame the entire response as a voluntary global narrowing of
the contribution.

### Reviewer 6SDM

Address:

- real/semi-realistic PIV;
- persistent, AR(1), spatial-burst, and the planned legacy 4G point-outlier
  transfer;
- direct PINN-EBM, MAD, and matched robust-loss baselines;
- EMA convention and sensitivity;
- exact update accounting and observational wall time.

Reviewer 6SDM asks for the learned-noise baseline, so explain the objective
difference when the response direction is approved. Direct numerical results
remain on `RESPONSE HOLD`. Do not add a broader originality concession that
the reviewer did not request.

### Reviewer aoJS

Answer point by point:

- because this reviewer explicitly raises the closest prior, acknowledge the
  shared EBM and staged components, then state the objective and gradient-path
  differences in Section 3.3 rather than merely saying "we add a gate";
- reserve the requested direct no-gate comparison slot, but keep all numbers,
  rankings, and claims on `RESPONSE HOLD` until the additional experiments
  and explicit author decision are complete;
- distinguish learned likelihood, fixed screening, learnable threshold, and
  gated inclusion;
- report epsilon, viscosity, and beta recovery;
- state end-to-end schedule and cost; and
- acknowledge the Baydin 2018 typo.

### Reviewer 6XZg

Concede that the submitted B-PINN is VI rather than HMC. State that Bayesian
UQ and adaptive selection are complementary. Give the faithful MAD
preprocessing result and its 60,000-update cost. Do not claim an HMC
comparison or broad Bayesian inferiority.

## 10. Prohibited claims

Do not write any of the following:

- a voluntary global statement that naPINN's contribution should be reduced
  to "only a gate" when the closest-prior relationship was not raised;
- "There are no public noisy PDE datasets."
- "Cylinder is the only possible RealPDEBench case."
- "Held-out PIV is noise-free ground truth."
- "The injected PIV failures are naturally occurring."
- "naPINN consistently wins on real data."
- "naPINN outperforms the closest prior on synthetic reconstruction."
- "Direct PINN-EBM fails under severe outliers."
- "Residual EBM learning or staged initialization is unique to naPINN."
- "Only the reliability gate can rank corruption."
- "naPINN separates sensor error from PDE-model discrepancy."
- "Parameter recovery is uniformly superior."
- "The submitted B-PINN is faithful HMC."
- "All methods have equal total optimization cost."
- "The 35k extension is wall-clock matched."
- "Shared-server timing proves speed superiority."
- "The method is insensitive to rejection cost or EMA."
- "The tested corruptions cover real-world noise."
- "The scale search is verified" before complete aggregation.
- "The regenerated runs reproduce or replace the submitted tables."
- "The paper has been revised."

## 11. Final acceptance gate for new evidence

The new PIV legacy-corruption result may enter a reviewer response only after:

1. generator invariants and checksums pass;
2. every required method and seed has `status: complete`;
3. no smoke, seed-39 calibration, partial, or failed artifact enters the
   reporting aggregate;
4. mean, sample standard deviation, and per-seed paired differences are
   verified;
5. field, physics, detection, parameter, timing, and memory metrics agree
   with source JSON;
6. positive, mixed, and adverse cells are retained;
7. the exact-center result is reported independently of the scale search;
8. all cited non-center settings follow the predeclared selection rule or the
   complete grid is reported;
9. `git diff --check` passes;
10. `git diff --exit-code -- paper/neurips_2026.tex` is empty; and
11. `rebuttal/response_matrix.md`, the Korean report, and
    `docs/PROGRESS.md` are updated from verified artifacts only.
12. for any direct naPINN-versus-PINN-EBM comparison, the author explicitly
    removes `RESPONSE HOLD` after reviewing the completed experiment
    campaign.

The current response should remain conservative if this gate is not met. A
planned experiment is not evidence.
