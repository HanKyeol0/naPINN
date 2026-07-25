# Reviewer comments

## Submission constraints

- Venue: NeurIPS 2026 Main Track
- Initial author-response deadline: July 27, 2026 (official schedule; exact
  OpenReview deadline/time should be confirmed by the authors)
- Author + Reviewer + AC discussion period: July 27--August 3, 2026
- Global word or character limit: none identified
- Per-reviewer limit: 10,000 characters
- Whether manuscript/code changes are allowed: the paper and supplemental
  material cannot be revised during author response; responses may report new
  results; no links are permitted in rebuttals
- Available compute budget: not supplied by the authors

Official source checked July 24, 2026:
`https://neurips.cc/Conferences/2026/MainTrackHandbook`

## Meta Review by Area Chair

The paper proposes naPINN, which attempts to learn an error model for the observation process, and then uses this to more robustly develop a PINN (by down-weighting unreliable observations) to solves the underlying inverse problem.

The main strength of the paper is that it tackles a common well-motived and important problem, namely that standard PINNs use MSE, and thus implicitly assume the observation errors are Gaussian. If there are non-Gaussian errors, which there often are, this can quickly lead to biases. The other strength of the paper is that the method is relatively simple (and thus likely to find application), and is well demonstrated on a range of synthetic problems.

The main weakness of the paper is that there is no demonstration of the method on real problems. Given the motivation for the paper is that our assumed error models are often wrong, it seems an oversight that the paper doesn’t demonstrate the method making a valuable difference on a real problem. The problem with synthetic problems is that our imagination when dreaming up forms of observation model-error is limited - nature can be much more perverse. In addition, often there is physics model discrepancy (ie the underlying PDEs are wrong) as well as observation model discrepancy (ie the Gaussian error assumption is wrong). If we don’t account for physics model discrepancy explicitly, the observation error model has to do that work, and this creates very complex error distributions. It would be good to demonstrate the method in this kind of scenario. An additional weakness is that the paper doesn’t fairly account for existing prior work such as the paper by Pilar and Wahlstrom (ref 31 in the paper). Although this work is mentioned in the introduction, it is much closer to the proposed approach than is acknowledged in the paper.

The paper is currently headed towards rejection. If the authors are able to add a compelling real example, showing that their approach can deliver a significant benefit in a real problem, that may move the reviewers to reconsider their current scores.

## Review 1 by Reviewer 6SDM

**Summary:**
This paper proposes naPINN, a noise-adaptive physics-informed neural network for inverse PDE problems with corrupted measurement data. The main idea is to estimate the reliability of data residuals during training, and use a trainable gate to downweight unreliable measurement points. The method uses a staged training procedure: first warm up a standard PINN, then initialize a residual noise estimator, and finally jointly train the PINN, estimator, and reliability gate. Experiments are conducted on three synthetic 2D PDE benchmarks with non-Gaussian noise and injected outliers.

**Contribution Type:** General: Most submissions will fall into this type.
**Strengths And Weaknesses:**
The paper studies an important problem, and the proposed reliability gate is intuitive. However, I have several concerns.

First, all experiments are synthetic controlled stress tests. The paper is motivated by real sensor corruptions, but no real or semi-realistic data experiment is included.

Second, the PDE benchmarks are still standard and not very close to complex real systems. It is unclear whether the method works under model mismatch, irregular sensors, correlated noise, sensor-level failure, or drift.

Third, the baseline comparison is not strong enough. More relevant baselines such as MAD-PINN, PINN with learned unknown noise / EBM likelihood, and tuned robust losses should be included.

Fourth, some hyperparameters are not well justified. In particular, the EMA normalization coefficient is important for the residual estimator and gate, but no sensitivity study is provided. The notation of beta in the main text and rho in the appendix is also confusing.

**Quality:** 2: not good
**Clarity:** 3: good
**Significance:** 2: not good
**Originality:** 2: not good
**Questions:**
Can the authors add one real or semi-realistic experiment to test beyond synthetic iid outliers?

Can the authors include stronger baselines, such as MAD-PINN and PINN-EBM without the reliability gate?

Can the authors test correlated noise, sensor-level failure, drift, or model mismatch?

Can the authors clarify the EMA coefficient choice and add sensitivity study for key hyperparameters?

Can the authors provide compute-matched or wall-clock-matched comparison?

**Limitations:**
See Weaknesses and Questions

**Rating:** 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
**Confidence:** 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

## Review 2 by Reviewer aoJS

Summary:
The paper presents naPINN, a noise-adaptive Physics-Informed Neural Network framework for inverse PDE problems with corrupted measurement data. The motivation is that standard PINNs usually use an MSE data loss, which works well under clean or Gaussian noise but becomes sensitive to non-Gaussian noise and gross outliers. In such settings, a small number of bad measurements can dominate the gradients and bias both the recovered solution and the learned PDE parameters.

To address this, the paper introduces a residual-based reliability estimation mechanism. After a warm-up stage, the model uses the measurement residuals to estimate a noise distribution. In the main version, this estimator is a one-dimensional Energy-Based Model. The estimated residual reliability is then passed through a trainable gate that downweights unreliable measurements in the data loss. A rejection-cost regularizer is added to prevent the trivial solution where the model rejects too many data points.

The method is evaluated on three 2D time-dependent PDE benchmarks: Allen–Cahn, Burgers, and lambda–omega reaction–diffusion. The experiments use sparse measurements corrupted by multimodal non-Gaussian noise and varying outlier ratios. The paper compares against vanilla PINN, B-PINN, LAD-PINN, and OrPINN baselines. Results show that naPINN improves reconstruction accuracy under corruption, identifies many outliers through the learned gate, and gives better PDE parameter recovery on the Allen–Cahn benchmark. The appendix further studies alternative backbones, additional noise distributions, estimator choices, staged training, rejection-cost sensitivity, residual normalization, and training cost.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths
The problem is important and well motivated. Real measurement-driven inverse problems often involve corrupted sensors, non-Gaussian noise, outliers, and incomplete boundary or initial information. Standard PINNs are known to be sensitive to such corruptions because the data loss can be dominated by large residuals. So the paper addresses a practical weakness of PINNs.

The core idea is useful. Instead of only replacing MSE with a fixed robust loss, naPINN tries to estimate which measurements are reliable during training. The combination of residual density estimation, trainable reliability gating, and rejection-cost regularization is a sensible framework. The rejection-cost term is also important because without it the gate could simply reject most measurements and reduce the data loss artificially.

The empirical results are strong on the chosen benchmarks. Across Allen–Cahn, Burgers, and lambda–omega reaction–diffusion systems, naPINN consistently improves over vanilla PINN and robust baselines under 5%, 10%, and 15% outlier corruption. The results are averaged over 10 independent trials and include standard deviations, which makes the evaluation more credible than a single-run comparison. The visual results also support the quantitative table.

The paper has good ablations. I especially liked the staged-training ablation, because it shows that warm-up is important for stabilizing the residual estimator on some PDEs.

The method is also reasonably lightweight. The reported overhead over a vanilla PINN is modest for the tested problems, and the method appears easy to add on top of existing PINN training pipelines.

Weaknesses
My main concern is originality relative to the closest prior work. The paper cites Pilar and Wahlström, Physics-informed Neural Networks with Unknown Measurement Noise, as reference [31], but only in passing as motivation. Pilar and Wahlström already study inverse PINNs under unknown non-Gaussian measurement noise, and they already jointly train an energy-based model to learn the residual noise distribution, then use the resulting likelihood as the data loss. This is essentially the residual-based EBM noise-distribution estimation that naPINN describes. I understand that naPINN adds the reliability gate and rejection-cost regularization, which is a meaningful extension, but the paper should compare directly to this closest prior or include an ablation that removes the gate and uses only the learned EBM likelihood.

The evaluation is still limited to controlled synthetic benchmarks. The paper is upfront about this, but it remains a significant limitation. The corruptions are injected by the authors and the outlier labels are known for analysis. Real sensor corruption can be spatially correlated, temporally persistent, heteroscedastic, biased, drifting, or correlated across variables. The current experiments are useful stress tests, but they do not fully establish that naPINN will work in real deployment-scale inverse problems.

The parameter recovery evidence is incomplete. The paper motivates inverse physics recovery and treats unknown PDE parameters in all three benchmarks, but detailed parameter reconstruction is only reported for the Allen–Cahn parameter. It would be useful to report viscosity recovery for Burgers and beta recovery for the lambda–omega system as well. Otherwise, the evidence is stronger for field reconstruction than for general PDE parameter identification.

There is also a possible training-budget fairness issue. The main text says all methods use 30,000 training steps, but naPINN additionally uses 5,000 estimator-initialization iterations that are not counted toward this total. The reported per-epoch overhead is small, but the total wall-clock comparison should include the warm-up, estimator initialization, and joint-training stages. This may not change the conclusion, but it should be accounted for clearly.

There are also minor reference and formatting issues. The reference [3] in this paper lists the Baydin automatic differentiation survey as 1989 paper, but its the JMLR article of 2018.

Quality: 3: good
Clarity: 3: good
Significance: 2: not good
Originality: 2: not good
Questions:
Can the authors compare directly against the closest EBM-based unknown-measurement-noise PINN baseline? Since that prior work [31] also learns an unknown noise distribution using an EBM, this comparison is important to isolate the contribution of the proposed reliability gate and rejection-cost regularization.

Can the authors provide a simple ablation with warm-up plus residual-density estimation but without a trainable gate? For example, a fixed residual-thresholding or fixed likelihood-weighting baseline would help show whether the trainable gate and rejection-cost term are necessary, rather than the gains coming mainly from residual screening.

Can the authors report PDE parameter recovery for all benchmarks? The paper reports Allen–Cahn parameter recovery, but Burgers viscosity and the lambda–omega reaction parameter are also treated as unknown. Reporting these would better support the claim of recovering the underlying physics, not only reconstructing the solution field.

Can the authors clarify the total training cost? The paper says all methods use 30,000 steps, but naPINN uses an additional 5,000 estimator-initialization iterations. Please report end-to-end wall-clock time including warm-up, estimator initialization, joint training, and estimator updates, not only per-epoch overhead.

Limitations:
Partially. The paper clearly acknowledges that the evaluation is based on controlled simulated PDE benchmarks with injected corruptions and does not replace real sensor validation. This is good.

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

## Review 3 by Reviewer 6XZg

**Summary:**
The paper introduces naPINN, a PINN modification for dealing with noisy data for recovering physics from corrupt measurements. The method is benchmarked on Allen-Canh, Burgers and lambda-omega reaction diffusion PDEs. While in principle some arguments in the paper have certain validity, the evaluation of the proposed method against other methods poses certain questions in terms of clarity and fairness.

**Contribution Type:** General: Most submissions will fall into this type.
**Strengths And Weaknesses:**
Here are my comments:

Strength of the submission:

there is indeed often happening a mismatch between the sensor-based measurements and the assumption on the Gaussian noise which can cause challenges in deployment of the inverse problems to real case studies.
the methodology provides sufficient statistical grounding to believe the implementation
quantitative results and anomaly detection show strong favor to the algorithm, however, while in the comparison matrix naPIN is definitely favorable to other PINNs, the implementation and reasoning for making certain assumptions is where the main weaknesses of this submission lie.
Hence the weaknesses of the submission are:

BPINN used for comparison of the results seems to be different from the set-up proposed by Yang et al. in the original BPINN publication. Where in the core BPINN a more superior HMC posterior sampling is used to calibrate the model in the implementation proposed by authors of this submission we see a factorized Gaussian weighted posterior, which to say is in full fairness is a weaker model. Hence if there exists a more superior BPINN implementation it would be more fair to compare results to that one, instead of using a weaker version which would inevitably show worse performance compared to naPINN.
The claim regarding uncertainty quantification in "2 Related work" in the paragraph of "Robust learning..." needs to be reasoned better. As of now it is unclear to the reader what the authors are trying to justify. Yes measurement selection is not the same as using a statistical method to eliminate effects of the noise and outliers, but essentially they do serve the same purpose. Beyesian methods have been widely used for many years to reduce the effect of noise and anomalies from data (including experimental data) when making predictions. If you want to suggest that methods are ineffective, you need to provide more concrete reasoning and grounding for the claim.
at the same time when looking closely to naPINN it appears to be formed on the Bayesian latent-variable model itself, hence the fact that paper is claiming contrasting differences with the Bayesian UQ framework is somewhat over-exaggerated. Essentially naPINN and BPINN are mostly differ in the approximation method and likelihood choice.
when comparing to PINN it is unfair not to remove baseline outliers, normal practice in data science and PINN applications when working with noisy measurements would include basic staistical data pre-prosessing, which would remove some points that for example exceed z-score limit etc. a more fair baseline would be required here to show effectiveness of the method.
things like model stability and hyperparameter optimization need to be discussed here in better detail.
noise distribution seems to be the same everywhere, would be good to include different noise benchmarks.
not clear regarding the trianing costs - seems to be an important information to discuss in main body.
please check the references carefully, reference 3 has wrong publication year, likely there are other inconsistencies.
All in all the proposed benchmarking is somewhat weakly presented to justify the claims authors provide.

**Quality:** 2: not good
**Clarity:** 3: good
**Significance:** 3: good
**Originality:** 2: not good
**Questions:**
Main suggestion is to have a more detailed benchkmarking and design a fair comparison with other methods. Not sure if naPINN presents enough differences not to just be a subset of BPINN

**Limitations:**
no, authors need to carefully revise the experiment design and benchmarking strategy.

**Rating:** 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
