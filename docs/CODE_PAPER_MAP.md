# Code-to-paper map

Use this map to trace a paper statement to implementation and configuration
before answering a reviewer or changing a claim.

| Topic | Paper location | Implementation | Configuration/evidence | Notes |
| --- | --- | --- | --- | --- |
| Overall algorithm | Method, Algorithm 1 | `pinnlab/train.py` | `phase` and `ebm.init_train_epochs` | 5k warm-up + 25k joint PINN steps; estimator init is additional |
| Shared PINN backbone | Implementation details: Model implementation | `pinnlab/models/mlp.py` | `configs/model/mlp.yaml` | Five hidden layers, width 80, tanh |
| Bayesian baseline | Appendix: B-PINN | `pinnlab/models/bpinn.py` | `configs/model/bpinn.yaml` | KL weight is `1e-6`; posterior mean is used at evaluation |
| PDE and measurement objectives | PINNs for inverse problems; Training objective | `pinnlab/experiments/*.py`, `pinnlab/train.py` | common loss weights | Reported implementation is PDE + data; no auxiliary dynamic cross-loss balancing |
| Residual normalization | Staged warm-up; running-std ablation | each experiment's `running_std` update | `ebm.std_mode`, `ebm.momentum` | Paper notation is internally inconsistent (`beta` in the update, `rho=0.99` in hyperparameters). Current configs use update weight `0.05`, hence old-state decay `0.95`; this is not numerically equivalent to decay `0.99` |
| EBM estimator | Residual reliability; estimator implementation | `pinnlab/utils/ebm.py::EBM` | `ebm.density_estimator: ebm` | Scalar log unnormalized density; partition approximated on `[-10,10]` |
| GMM estimator | Estimator module comparison | `pinnlab/utils/ebm.py::TrainableGMM` | `density_estimator: gmm` | Default is three components |
| KDE estimator | Estimator module comparison | `pinnlab/utils/kde.py::KDE` | `density_estimator: kde` | Nonparametric buffer with Silverman bandwidth |
| Estimator factory | Estimator module comparison | `pinnlab/utils/density.py` | `ebm` YAML section | Common scalar interface |
| Likelihood gate | Joint optimization; gate implementation | `TrainableLikelihoodGate` in `pinnlab/utils/ebm.py` | `kind: gated_trainable` | Code uses standardized log density, an equivalent sign convention. Current experiment YAML gate initializers are not passed to this class; its constructor defaults are used |
| Quantile gate | Gate ablation | `QuantileThresholdGate` | `kind: quantile` | Estimator-free ablation |
| Learnable residual threshold | Gate ablation | `LearnableThresholdGate` | `kind: threshold` | Estimator-free ablation |
| Rejection cost | Method and sensitivity appendix | gate classes and experiment data loss | `rejection_cost` | Paper reports `1.0` for Allen-Cahn and `0.5` for the other PDEs; current tracked YAMLs use `0.5` where configured, so the reported Allen-Cahn main config is not presently identified |
| Robust data losses | Additional noise distributions; Loss functions | `pinnlab/utils/data_loss.py` | `data_loss.kind` | MSE, L1, and q-Gaussian |
| Noise families | Experimental setup | `pinnlab/data/noise.py` | `noise.kind` | `G`, `4G`, `Laplace`, `StudentT` |
| Gross outliers | Noise injection | each experiment's noisy-data initialization | `noise.extra_noise` | Outlier count encodes the intended ratio for that dataset |
| Allen-Cahn | Dataset details | `pinnlab/experiments/allencahn2d.py` | `allencahn2d*.yaml` | Analytical reference; learns `eps` when enabled |
| Burgers | Dataset details | `pinnlab/experiments/burgers2d.py` | `burgers2d*.yaml` | Requires generated `Burgers2D_3-5/data.npz`; learns `nu` |
| Lambda-omega RD | Dataset details | `pinnlab/experiments/lambdaomega2d.py` | `lambdaomega2d*.yaml` | Requires generated `LambdaOmega_Spiral_2/data.npz`; learns `beta` |
| Metrics | Baselines and metrics; Evaluation metrics | each experiment's `eval_on_grid` | common 120-grid evaluation | Vector-field metrics pool both output components |
| Checkpoints and reload | Implementation workflow | `pinnlab/train.py`, `analysis/evaluate_checkpoint.py` | run `config.yaml`, `best.pt`, `final.pt` | Experiment state includes estimator, gate, running scale, and PDE parameters |
| Paper plots | Main results and appendices | `analysis/plots/`, experiment video/evaluation hooks | `paper/figures/` | Analysis artifacts must be explicitly promoted to paper figures |

## Current YAML inventory

These are current working recipes, not a verified reproduction manifest.

| YAML | Current role selected by fields |
| --- | --- |
| `allencahn2d.yaml` | learnable residual-threshold ablation; EBM disabled |
| `allencahn2d_laplace.yaml` | EBM likelihood gate under Laplace noise |
| `allencahn2d_quantile.yaml` | quantile-gate ablation |
| `allencahn2d_studentt.yaml` | EBM likelihood gate under Student-t noise |
| `burgers2d_3.yaml` | Gaussian q-Gaussian baseline; phase and gate disabled |
| `burgers2d_3_laplace.yaml` | EBM likelihood gate under Laplace noise |
| `burgers2d_3_quantile.yaml` | quantile-gate ablation |
| `burgers2d_3_studentt.yaml` | EBM likelihood gate under Student-t noise |
| `burgers2d_3_thres.yaml` | learnable residual-threshold ablation |
| `lambdaomega2d.yaml` | learnable residual-threshold ablation; density disabled |
| `lambdaomega2d_laplace.yaml` | ungated Laplace run; phase disabled |
| `lambdaomega2d_quantile.yaml` | quantile-gate ablation |
| `lambdaomega2d_studentt.yaml` | EBM likelihood gate under Student-t noise |
| `ablation2_*.yaml` | learnable residual-threshold ablations |

Before claiming reproducibility, create or identify immutable canonical configs
for every reported main-table condition and record their run artifacts/seeds.
