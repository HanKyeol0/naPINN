# naPINN NeurIPS 2026 Rebuttal 한국어 종합 보고서

## 1. 문서 목적과 사용 원칙

이 문서는 NeurIPS 2026 review와 Meta Review를 한국어로 이해하고,
author response의 논리와 추가 실험 결과를 내부적으로 검토하기 위한
보고서다. 실제 OpenReview에 게시할 영문 답변은
`rebuttal/response_draft.md`에 별도로 유지한다.

중요한 전제는 다음과 같다.

- Author response 기간에는 제출한 paper와 supplementary material을
  수정하여 다시 올릴 수 없다. 따라서 이 보고서에서 “revision에서
  명확히 하겠다”는 표현은 reviewer와의 communication 및 향후
  camera-ready 수정 계획을 뜻한다.
- 현재 `paper/neurips_2026.tex`는 사용자가 rollback한 상태이며, 이
  보고서 작성 과정에서 변경하지 않는다.
- Reviewer가 제기한 문제 중 맞는 지적은 명확하게 인정한다. 특히
  Pilar--Wahlström의 prior contribution, B-PINN baseline의
  mean-field VI 성격, EMA notation 오류, 5,000 estimator-only update,
  Baydin 연도 오류는 방어적으로 부정하지 않는다.
- 추가 실험의 결과가 mixed outcome이면 그대로 보고한다. naPINN이
  LAD-PINN보다 held-out-PIV agreement에서 우수하다고 주장하지 않는다.
- 중요한 nuance를 유지하기 위해 다음 용어는 English를 병기하거나
  그대로 사용한다: `held-out-PIV agreement`, `nominal-physics
  residual`, `model discrepancy`, `observation error`, `closest prior`,
  `no-gate PINN-EBM`, `reliability gate`, `rejection-cost regularization`,
  `mean-field variational inference`, `HMC`, `compute matched`,
  `wall-clock matched`, `sensitivity`, `fixed Reynolds number`.

## 2. Executive summary

현재 review의 가장 큰 문제는 두 가지다.

1. **Real problem에 대한 검증 부재**
2. **Pilar--Wahlström PINN-EBM과의 novelty 및 comparison 부족**

이를 위해 RealPDEBench Cylinder의 실제 time-resolved PIV를 사용한
3-seed 실험과, Pilar--Wahlström의 핵심 objective를 구현한
`PINN-EBM (no gate)` comparison을 추가로 수행했다. 또한 persistent
sensor bias와 linear drift를 실제 PIV training measurements에 주입한
structured-failure stress test, 그리고 faithful MAD-PINN comparison을
수행했다.

핵심 결론은 “naPINN이 모든 baseline보다 우수하다”가 아니다.

- Natural real PIV에서는 **LAD-PINN이 held-out-PIV agreement에서 가장
  우수**하다.
- naPINN은 **nominal Navier--Stokes residual이 가장 낮고**,
  `PINN-EBM (no gate)`보다 rMAE를 32.3%, momentum-residual RMS를
  63.0% 낮춘다.
- 따라서 real-data evidence는 `universal win`이 아니라
  **measurement agreement와 nominal-physics consistency 사이의
  tradeoff**를 보여준다.
- Injected persistent sensor failure에서는 naPINN이 failed measurement를
  잘 ranking하지만, primary setting은 clean scalar의 47.5%도
  reject한다. 이 수치는 중요한 selectivity limitation이다.
- `rejection cost = 0.01` sensitivity는 clean rejection을 33.7%로
  낮추지만, 여전히 상당하며 primary setting을 대체하지 않는다.

Novelty도 좁혀서 설명해야 한다.

- Residual EBM learning, MSE warm-up, estimator-only initialization,
  EBM likelihood를 이용한 joint training은 Pilar--Wahlström의 prior
  contribution이다.
- naPINN의 방어 가능한 incremental contribution은 learned residual
  reliability를 **trainable per-measurement inclusion gate**로 변환하고,
  indiscriminate rejection을 막는 **rejection-cost regularization**을
  추가한 점이다.

현재 rebuttal에서 해결하지 못한 사항도 명확하다.

- Burgers viscosity와 lambda--omega reaction parameter의 새로운
  recovery 결과는 original dataset/run artifact 부재로 보고할 수 없다.
- Full EMA-coefficient sweep는 수행하지 않았다.
- Real-PIV experiment는 Reynolds number를 metadata 값으로 고정했기
  때문에 inverse-coefficient experiment가 아니다.
- Observation error와 physics-model discrepancy를 별도로 식별하지
  않는다.
- Wall-clock measurements는 shared A100에서 얻은 observational values로,
  hardware-isolated benchmark가 아니다.

## 3. Author response 제약

- Venue: NeurIPS 2026 Main Track
- Reviewer별 response limit: 10,000 characters
- New experimental results 보고 가능
- Paper 및 supplementary material 수정 업로드 불가
- Rebuttal response 내 link 금지
- Discussion 시작 일정은 2026년 7월 27일로 공지되어 있으나, 정확한
  OpenReview invitation deadline과 timezone은 저자가 직접 확인해야 한다.

현재 영문 response body의 기계적 character count는 다음과 같다.

| 대상 | Character count |
| --- | ---: |
| Meta Review / Area Chair | 3,755 |
| Reviewer 6SDM | 4,107 |
| Reviewer aoJS | 3,882 |
| Reviewer 6XZg | 4,670 |

모든 response는 10,000-character limit보다 작으며 URL과 Markdown link를
포함하지 않는다.

---

## 4. Meta Review by Area Chair

### 4.1 Review 번역

이 논문은 observation process의 error model을 학습하고, 신뢰할 수 없는
observation의 weight를 낮추어 underlying inverse problem을 더 robust하게
푸는 PINN인 naPINN을 제안한다.

논문의 주요 장점은 일반적이고 동기가 타당하며 중요한 문제를 다룬다는
점이다. Standard PINN은 MSE를 사용하므로 observation error가 Gaussian일
것이라고 암묵적으로 가정한다. 실제로는 non-Gaussian error가 자주
발생하고, 이 경우 bias가 빠르게 생길 수 있다. 또 다른 장점은 방법이
상대적으로 단순하여 실제 활용 가능성이 높고, 여러 synthetic problem에서
잘 검증되었다는 점이다.

가장 큰 약점은 real problem에서의 demonstration이 없다는 점이다. 논문의
동기가 assumed error model이 실제로는 자주 틀린다는 것인데, real
problem에서 방법이 가치 있는 차이를 만든다는 것을 보여주지 않은 것은
중요한 누락으로 보인다. Synthetic problem에서는 저자가 상상한 형태의
observation-model error만 만들 수 있지만, 실제 자연 현상은 훨씬 더
복잡하고 예측하기 어려울 수 있다.

또한 실제 문제에서는 observation model discrepancy, 즉 Gaussian error
assumption의 오류뿐 아니라 physics model discrepancy, 즉 underlying
PDE 자체가 부정확한 경우도 흔하다. Physics model discrepancy를
명시적으로 다루지 않으면 observation error model이 그 역할까지 떠맡게
되고, 결과적으로 매우 복잡한 error distribution이 만들어진다. 이런
scenario에서 방법을 검증하면 좋겠다.

추가적인 약점은 Pilar and Wahlström의 연구, 즉 논문의 reference [31]과
같은 prior work를 공정하게 다루지 않았다는 점이다. Introduction에서
언급하기는 했지만, 이 연구는 논문이 인정한 것보다 제안 방법과 훨씬
가깝다.

현재 논문은 reject 방향으로 가고 있다. 저자들이 compelling real
example을 추가하여 real problem에서 significant benefit을 보인다면,
reviewer들이 현재 score를 재고할 가능성이 있다.

### 4.2 AC 지적의 의미

AC의 메시지는 명확하다. 단순한 wording correction이나 소규모 ablation이
아니라, score를 움직일 수 있는 핵심 evidence는 real example이다. 동시에
Pilar--Wahlström과의 관계를 축소해서 표현하면 originality concern이
해결되지 않는다.

따라서 response의 우선순위는 다음과 같다.

1. Real PIV experiment를 먼저 제시한다.
2. Result가 mixed임을 숨기지 않는다.
3. Closest prior의 contribution을 인정하고 novelty를 좁힌다.
4. Gate가 learned likelihood alone에 비해 무엇을 추가하는지 direct
   no-gate comparison으로 보인다.
5. Model discrepancy를 “해결했다”고 주장하지 않고 stress test라고
   한정한다.

### 4.3 AC에 대한 우리 답변의 한국어 번역

RealPDEBench Cylinder trajectory 한 개에서 3-seed experiment를
수행했다. 200개의 real time-resolved PIV frame을 사용했고, 192개의
fixed irregular training sensor에서 38,400개의 velocity vector를
관측했다. Training spatial location과 겹치지 않는 7,747개 spatial
location의 1,549,400 velocity vector를 held-out evaluation에 사용했다.
모든 방법은 동일한 pressure-latent \(u,v,p\) network와 nominal 2D
incompressible Navier--Stokes residual을 사용했다. Reynolds number는
trajectory metadata 값으로 고정했다.

Unmodified real-PIV result는 informative하지만 mixed하다. LAD-PINN의
held-out-PIV rMAE는 \(0.11138\pm0.00122\)로 가장 낮고, naPINN은
\(0.16719\pm0.00105\)다. Held-out PIV는 independent real measurement이지
noise-free ground truth가 아니다.

반면 controlled `PINN-EBM (no gate)` comparison과 비교하면 naPINN은
rMAE를 \(0.24688\pm0.02633\)에서 \(0.16719\pm0.00105\)로 32.3%
낮추고, nominal momentum-residual RMS를
\(0.02354\pm0.00394\)에서 \(0.00871\pm0.00070\)로 63.0% 낮춘다.
Continuity-residual RMS도 82.6% 낮다. 즉, 이 실험은 learned residual
likelihood만 사용하는 것보다 gating이 유용하고 nominal-physics
fidelity를 개선한다는 evidence는 제공하지만, fixed robust loss에 대한
universal superiority를 보이지는 않는다.

추가로 real PIV 위에 structured sensor failure를 주입했다. 192개 sensor
중 19개의 identity에 대해 두 velocity component 모두에 persistent
bias와 linear drift를 200개 모든 frame에 적용했다. 이는 naturally
observed failure가 아니라 real PIV 위에 주입한 controlled corruption이다.

Manifest에 primary로 기록한 \(\lambda_{\mathrm{rej}}=0.005\)에서 naPINN은
MSE-PINN 대비 rMAE를 \(0.21182\pm0.00261\)에서
\(0.18234\pm0.00711\)로 낮추고, failure AUROC
\(0.96234\pm0.00086\), failed-scalar rejection 99.29%를 얻었다.
하지만 clean scalar도 47.51% reject했다. LAD-PINN은 rMAE
\(0.13212\pm0.00251\)로 여전히 가장 우수하다.

별도로 표시한 \(\lambda_{\mathrm{rej}}=0.01\) sensitivity는 rMAE를
\(0.16928\pm0.00131\)로 개선하고 clean rejection을 33.68%로 낮추면서
failed rejection 98.86%를 유지했다. 그러나 이 sensitivity가 primary
setting을 대체한다고 주장하지 않는다.

Pilar--Wahlström에 대한 original positioning이 불충분했다는 점에
동의한다. 그들의 PINN-EBM은 이미 residual EBM learning, MSE warm-up,
estimator-only initialization, EBM negative log likelihood를 이용한
joint training을 포함한다. 이들은 naPINN의 contribution이 아니다.
우리의 incremental contribution은 trainable per-measurement gate와
rejection regularization이다.

Real experiment는 inverse coefficient를 추정하지 않고, observation
error와 physics-model discrepancy를 별도로 식별하지 않으며, 하나의
trajectory와 하나의 nominal model만 다룬다. Gate rejection을 identified
observation error로 해석하지 않고 이 limitation을 명시한다.

---

## 5. Reviewer 6SDM

### 5.1 Review 번역

#### Summary

논문은 corrupted measurement data가 있는 inverse PDE problem을 위한
noise-adaptive PINN인 naPINN을 제안한다. 핵심 아이디어는 training 중
data residual의 reliability를 추정하고, trainable gate로 신뢰할 수 없는
measurement point의 weight를 낮추는 것이다.

Training은 staged procedure를 사용한다. 먼저 standard PINN을 warm-up한
뒤 residual noise estimator를 초기화하고, 마지막으로 PINN, estimator,
reliability gate를 joint training한다. 실험은 non-Gaussian noise와
injected outlier가 있는 세 개의 synthetic 2D PDE benchmark에서 수행된다.

#### Strengths and weaknesses

논문은 중요한 문제를 다루며 proposed reliability gate는 직관적이다.
하지만 다음과 같은 우려가 있다.

1. 모든 실험이 synthetic controlled stress test다. 논문의 동기는 real
   sensor corruption인데 real 또는 semi-realistic data experiment가 없다.
2. PDE benchmark가 standard problem에 머물고 complex real system과
   가깝지 않다. Model mismatch, irregular sensor, correlated noise,
   sensor-level failure, drift에서 작동하는지 불분명하다.
3. Baseline comparison이 충분히 강하지 않다. MAD-PINN, learned unknown
   noise 또는 EBM likelihood를 사용하는 PINN, tuned robust loss처럼 더
   관련성 높은 baseline을 포함해야 한다.
4. 일부 hyperparameter의 근거가 부족하다. 특히 residual estimator와
   gate에 중요한 EMA normalization coefficient에 sensitivity study가
   없다. Main text의 \(\beta\)와 appendix의 \(\rho\) notation도 혼란스럽다.

#### Questions

- Synthetic IID outlier를 넘어서는 real 또는 semi-realistic experiment를
  하나 추가할 수 있는가?
- MAD-PINN과 reliability gate가 없는 PINN-EBM 같은 stronger baseline을
  포함할 수 있는가?
- Correlated noise, sensor-level failure, drift, model mismatch를 시험할
  수 있는가?
- EMA coefficient 선택을 명확히 하고 key hyperparameter sensitivity를
  추가할 수 있는가?
- Compute-matched 또는 wall-clock-matched comparison을 제공할 수 있는가?

#### 평가

- Quality: 2 / not good
- Clarity: 3 / good
- Significance: 2 / not good
- Originality: 2 / not good
- Rating: 2 / Reject
- Confidence: 5 / 매우 높은 확신

### 5.2 Reviewer 6SDM에 대한 우리 답변의 한국어 번역

#### Real data, irregular sensors, and mismatch

RealPDEBench Cylinder의 실제 PIV 200 frame, 192 fixed irregular training
sensor, spatially disjoint dense held-out PIV set을 사용했다. Nominal 2D
incompressible Navier--Stokes와 fixed metadata Reynolds number를
사용했다.

3-seed 결과에서 LAD-PINN이 held-out-PIV rMAE
\(0.11138\pm0.00122\)로 가장 좋고, naPINN은
\(0.16719\pm0.00105\)다. Closest no-gate PINN-EBM과 비교하면 naPINN은
rMAE를 32.3%, nominal momentum-residual RMS를 63.0% 개선한다.

이 결과는 mixed outcome이다. Held-out PIV는 noise-free truth가 아니며,
setup은 natural observation error와 model discrepancy를 분리하지 않고
inverse coefficient도 추정하지 않는다. 또한 PIV error의 spatial 또는
temporal correlation structure를 정량화하지 않았으므로 controlled
correlated-noise experiment라고 표현하지 않는다.

Structured test에서는 192개 sensor 중 19개에 persistent bias와 linear
drift를 모든 frame과 두 component에 주입했다. Primary naPINN은 rMAE
\(0.18234\pm0.00711\)로 MSE-PINN의
\(0.21182\pm0.00261\)보다 낮고, AUROC
\(0.96234\pm0.00086\), failed-scalar rejection 99.29%를 얻었다.
그러나 clean scalar rejection은 47.51%이며, LAD-PINN rMAE
\(0.13212\pm0.00251\)가 가장 우수하다.

\(\lambda_{\mathrm{rej}}=0.01\) sensitivity는 clean rejection을 33.68%로
낮추고 rMAE \(0.16928\pm0.00131\), failed rejection 98.86%를 얻는다.
이것은 sensitivity이며 primary \(\lambda_{\mathrm{rej}}=0.005\)를 대체하지
않는다.

#### Stronger baselines

LAD, learned EBM likelihood without a gate, faithful two-stage MAD-PINN을
3-seed로 비교했다. Natural PIV에서:

- PINN-EBM (no gate): rMAE \(0.24688\pm0.02633\)
- naPINN: rMAE \(0.16719\pm0.00105\)
- MAD-PINN: rMAE \(0.14810\pm0.00103\), rMSE
  \(0.22537\pm0.00095\), momentum RMS
  \(0.01399\pm0.00019\), continuity RMS
  \(0.01349\pm0.00125\)

MAD-PINN은
\(\widehat{\sigma}=\operatorname{median}(|r|)/1.6777\),
\(|r|\leq3\widehat{\sigma}\) screen을 적용한 뒤 MSE retraining을 수행한다.
30,000 LAD update와 30,000 MSE update, 총 60,000 PINN update를 사용하므로
30,000-update method와 compute matched가 아니다.

Submitted OrPINN은 \(q\in\{1.9,2.9\}\)를 사용하지만 comprehensive
robust-loss hyperparameter sweep는 수행하지 않았다. 이 값을
exhaustively tuned setting이라고 주장하지 않는다.

#### EMA coefficient

실제 update는 다음과 같다.

\[
\sigma_{\mathrm{run}}
\leftarrow
(1-\beta)\sigma_{\mathrm{run}}+\beta s_{\mathcal B},
\qquad \beta=0.05.
\]

여기서 \(\beta=0.05\)는 new-batch weight이며 old-state decay로 쓰면
\(\rho=0.95\)다. Appendix에 적힌 0.99는 실제 구현과 일치하지 않는다.
Batch scale은 update 전에 \(10^{-6}\)으로 lower clamp되고, 현재 running
scale의 10배로 upper clamp된다.

Full EMA-coefficient sweep는 완료하지 않았으므로 coefficient
insensitivity를 주장할 수 없다. Submitted EMA-versus-per-batch
normalization ablation이 보여주는 범위만 유지한다.

#### Compute

MSE, LAD, PINN-EBM, naPINN은 각각 30,000 PINN update를 사용한다.
naPINN의 30,000 PINN update는 5,000 warm-up과 25,000 joint update이며,
naPINN과 PINN-EBM은 여기에 5,000 estimator-only update를 추가로
수행한다.

Shared A100에서 관찰한 mean end-to-end time은 다음과 같다.

| Method | Mean end-to-end time |
| --- | ---: |
| MSE-PINN | 1,373.8 s |
| LAD-PINN | 1,451.3 s |
| PINN-EBM | 1,717.2 s |
| naPINN | 1,714.4 s |
| MAD-PINN two-stage pipeline | 2,993.6 s |

이 값들은 모든 stage와 evaluation을 포함하지만 shared GPU에서 측정했기
때문에 hardware-isolated wall-clock match가 아니다.

---

## 6. Reviewer aoJS

### 6.1 Review 번역

#### Summary

논문은 corrupted measurement data가 있는 inverse PDE problem을 위한
noise-adaptive PINN framework인 naPINN을 제안한다. Standard PINN은
clean 또는 Gaussian noise에서는 잘 작동할 수 있지만, non-Gaussian noise와
gross outlier가 있으면 MSE data loss가 큰 residual에 민감하다. 소수의
bad measurement가 gradient를 지배하여 reconstructed solution과 learned
PDE parameter 모두를 bias할 수 있다.

이를 해결하기 위해 residual-based reliability estimation을 사용한다.
Warm-up 이후 measurement residual로 noise distribution을 추정한다.
Main version에서는 one-dimensional Energy-Based Model을 사용한다.
Estimated residual reliability를 trainable gate에 통과시켜 unreliable
measurement의 data-loss weight를 낮춘다. Gate가 모든 measurement를
reject하는 trivial solution을 막기 위해 rejection-cost regularizer를
추가한다.

Allen--Cahn, Burgers, lambda--omega reaction--diffusion의 세 개 2D
time-dependent PDE에서 평가한다. Sparse measurement에 multimodal
non-Gaussian noise와 여러 outlier ratio를 적용하고, vanilla PINN,
B-PINN, LAD-PINN, OrPINN과 비교한다. naPINN은 corruption 아래에서
reconstruction accuracy를 개선하고, gate를 통해 많은 outlier를 식별하며,
Allen--Cahn에서 더 나은 PDE parameter recovery를 보인다. Appendix에는
alternative backbone, additional noise distribution, estimator choice,
staged training, rejection-cost sensitivity, residual normalization,
training cost가 포함되어 있다.

#### Strengths

- 문제는 중요하고 동기가 충분하다. Real measurement-driven inverse
  problem에서는 corrupted sensor, non-Gaussian noise, outlier, incomplete
  boundary/initial information이 흔하다.
- Fixed robust loss만 사용하는 대신 training 중 measurement reliability를
  추정하는 아이디어는 유용하다.
- Residual density estimation, trainable reliability gating,
  rejection-cost regularization의 조합은 합리적이다.
- 세 benchmark에서 5%, 10%, 15% outlier corruption에 대해 vanilla PINN과
  robust baseline보다 좋은 결과를 보인다.
- 10 independent trial의 mean과 standard deviation을 보고하여 single-run
  comparison보다 신뢰도가 높다.
- Staged-training ablation은 warm-up이 residual estimator 안정화에
  중요하다는 점을 보여준다.
- Tested problem에서는 vanilla PINN 대비 overhead가 크지 않고 기존
  PINN training pipeline에 쉽게 추가할 수 있어 보인다.

#### Weaknesses

가장 큰 우려는 closest prior work 대비 originality다. Pilar and
Wahlström은 이미 unknown non-Gaussian measurement noise가 있는 inverse
PINN을 연구했고, EBM으로 residual noise distribution을 학습한 뒤 그
likelihood를 data loss로 사용했다. 이는 naPINN이 설명하는 residual-based
EBM noise estimation과 본질적으로 매우 가깝다.

naPINN의 reliability gate와 rejection-cost regularization은 의미 있는
extension이지만, closest prior와 직접 비교하거나 gate를 제거하고 learned
EBM likelihood만 사용하는 ablation을 포함해야 한다.

Evaluation은 여전히 controlled synthetic benchmark에 한정된다. Real
sensor corruption은 spatially correlated, temporally persistent,
heteroscedastic, biased, drifting, variable 간 correlated 형태일 수 있다.
현재 실험은 유용한 stress test지만 deployment-scale inverse problem에서의
작동을 충분히 증명하지 못한다.

Parameter recovery evidence도 불완전하다. 세 benchmark 모두에서 unknown
PDE parameter를 다룬다고 설명하지만 detailed parameter reconstruction은
Allen--Cahn parameter에 대해서만 보고한다. Burgers viscosity와
lambda--omega beta도 보고해야 general PDE parameter identification
claim을 뒷받침할 수 있다.

Training-budget fairness 문제도 있다. 모든 방법이 30,000 training step을
사용한다고 했지만 naPINN에는 이 숫자에 포함되지 않은 5,000
estimator-initialization iteration이 있다. Per-epoch overhead뿐 아니라
warm-up, estimator initialization, joint training 전체를 포함한
wall-clock comparison이 필요하다.

Reference [3]의 Baydin automatic differentiation survey는 1989가 아니라
2018 JMLR article이다.

#### Questions

- Closest EBM-based unknown-measurement-noise PINN baseline과 직접 비교할
  수 있는가?
- Warm-up과 residual-density estimation은 사용하지만 trainable gate는
  없는 ablation을 제공할 수 있는가?
- 모든 benchmark의 PDE parameter recovery를 보고할 수 있는가?
- Warm-up, estimator initialization, joint training, estimator update를
  모두 포함한 total wall-clock time을 제공할 수 있는가?

#### 평가

- Quality: 3 / good
- Clarity: 3 / good
- Significance: 2 / not good
- Originality: 2 / not good
- Rating: 3 / Borderline reject
- Confidence: 4 / 높은 확신

### 6.2 Reviewer aoJS에 대한 우리 답변의 한국어 번역

#### Closest prior and gate ablation

Pilar--Wahlström이 이미 MSE warm-up, estimator-only EBM initialization,
learned EBM negative log likelihood를 data loss로 사용하는 joint
optimization을 제안했다는 점에 동의한다. 이 component들을 novel하다고
주장하지 않는다. 우리의 좁은 contribution은 residual likelihood를
trainable per-measurement inclusion weight로 변환하고 rejection을
regularize하는 것이다.

동일한 RealPDEBench trajectory, backbone, irregular train/held-out split,
seed, 30,000 PINN update를 사용하여 gate와 rejection cost가 없는 closest
prior objective를 구현했다. Pilar--Wahlström의 Navier--Stokes setting에
보고된 joint-phase physics weight도 적용했다.

3-seed 결과는 다음과 같다.

| Method | rMAE | rMSE | Momentum RMS | Continuity RMS |
| --- | ---: | ---: | ---: | ---: |
| PINN-EBM (no gate) | \(0.24688\pm0.02633\) | \(0.33706\pm0.03480\) | \(0.02354\pm0.00394\) | \(0.04654\pm0.00338\) |
| naPINN | \(0.16719\pm0.00105\) | \(0.26616\pm0.00124\) | \(0.00871\pm0.00070\) | \(0.00812\pm0.00139\) |
| Relative reduction | 32.3% | 21.0% | 63.0% | 82.6% |

이 comparison은 learned residual density alone과
gating + rejection regularization의 차이를 직접 분리한다.

#### Real-data scope

실험은 200 frame의 real time-resolved PIV, 192 fixed irregular training
sensor, zero spatial overlap인 1,549,400 held-out velocity vector를
사용한다. LAD-PINN의 held-out-PIV rMAE
\(0.11138\pm0.00122\)가 가장 낮으므로 naPINN이 robust loss를
dominate한다고 주장하지 않는다.

Held-out PIV는 noise-free truth가 아니라 independent real measurement다.
Fixed Reynolds number의 nominal 2D incompressible Navier--Stokes constraint는
model-discrepancy stress test를 제공하지만 observation error와 model
discrepancy는 별도로 identified되지 않는다.

Fixed-screening comparison인 MAD-PINN은 rMAE
\(0.14810\pm0.00103\), rMSE \(0.22537\pm0.00095\), momentum RMS
\(0.01399\pm0.00019\), continuity RMS
\(0.01349\pm0.00125\)다. 60,000 PINN update를 사용하므로
30,000-update method와 compute matched가 아니다.

#### PDE parameter recovery

Submitted evidence가 detailed parameter recovery를 지원하는 것은
Allen--Cahn뿐이라는 점에 동의한다. Original Burgers와 lambda--omega
run artifact 및 generated dataset이 현재 없기 때문에 viscosity나 reaction
parameter 값을 책임 있게 새로 보고할 수 없다.

Real-PIV experiment도 Reynolds number를 고정했으므로 field reconstruction
experiment이지 inverse-coefficient evidence가 아니다. General
parameter-recovery claim을 좁히고 detailed coefficient evidence가
Allen--Cahn에 한정된다는 점을 명확히 한다.

#### Total cost and reference

naPINN은 5,000 warm-up + 25,000 joint PINN update와 별도의 5,000
estimator-only update를 사용한다. PINN-EBM도 같은 estimator-only count를
추가한다. 모든 stage와 evaluation을 포함한 observational mean time은
MSE 1,373.8 s, LAD 1,451.3 s, PINN-EBM 1,717.2 s, naPINN 1,714.4 s다.

Baydin automatic-differentiation survey의 올바른 연도는 2018이며,
1989는 bibliography error다. 이 오류는 인정하고 수정 계획을 알린다.

---

## 7. Reviewer 6XZg

### 7.1 Review 번역

#### Summary와 strength

논문은 noisy measurement에서 physics를 recover하기 위한 PINN modification인
naPINN을 소개하며 Allen--Cahn, Burgers, lambda--omega
reaction--diffusion PDE에서 평가한다. 논문의 일부 주장은 원칙적으로
타당하지만, 다른 방법과의 comparison clarity와 fairness에 문제가 있다.

Sensor-based measurement와 Gaussian noise assumption 사이의 mismatch가
real inverse problem deployment에서 문제가 될 수 있다는 동기는 타당하다.
Methodology는 implementation을 신뢰할 수 있을 정도의 statistical
grounding을 제공하며, quantitative result와 anomaly detection은 제안
algorithm에 유리하다. 그러나 implementation과 일부 assumption의
근거가 논문의 주요 약점이다.

#### Weaknesses

1. **B-PINN comparison fairness**

   논문에서 사용한 B-PINN은 Yang et al. original B-PINN의 setup과 다르게
   보인다. Original publication의 core B-PINN은 더 강한 HMC posterior
   sampling을 사용하지만, 논문의 implementation은 factorized Gaussian
   weighted posterior를 사용하며 더 약한 model이다. 더 강한 B-PINN
   implementation이 있다면 그것과 비교하는 것이 공정하다.

2. **Bayesian uncertainty quantification에 대한 과도한 주장**

   Related Work의 `Robust learning` paragraph에서 uncertainty quantification에
   대한 claim의 근거가 부족하다. Measurement selection과 statistical
   method를 이용한 noise/outlier effect reduction이 동일하지는 않지만
   비슷한 목적을 수행한다. Bayesian method는 experimental data를 포함한
   noisy/anomalous data의 영향을 줄이는 데 오래 사용되어 왔다. Ineffective
   하다고 주장하려면 더 구체적인 근거가 필요하다.

3. **naPINN과 Bayesian latent-variable model의 관계**

   naPINN 자체가 Bayesian latent-variable model에 기반한 것으로 보이므로
   Bayesian UQ framework와의 차이를 과장한 것으로 보인다. 실질적으로는
   naPINN과 B-PINN이 approximation method와 likelihood choice에서 주로
   다를 수 있다.

4. **Basic preprocessing baseline 부재**

   Noisy measurement를 다루는 일반적인 data science 또는 PINN
   application에서는 z-score threshold 같은 basic statistical
   preprocessing으로 일부 outlier를 제거한다. Baseline PINN에서 outlier를
   제거하지 않은 것은 불공정할 수 있으므로 더 공정한 baseline이 필요하다.

5. **Stability와 hyperparameter optimization**

   Model stability와 hyperparameter optimization을 더 자세히 논의해야 한다.

6. **Noise benchmark 다양성**

   Noise distribution이 모든 실험에서 같은 것으로 보인다. Different
   noise benchmark가 필요하다.

7. **Training cost**

   Training cost가 불명확하며 main body에서 논의할 중요한 정보다.

8. **Reference 오류**

   Reference [3]의 publication year가 틀렸으며 다른 inconsistency도
   확인해야 한다.

전체적으로 benchmarking presentation이 논문의 claim을 정당화하기에
약하다. naPINN이 B-PINN의 subset에 지나지 않는지 의문이며, 더 상세하고
공정한 benchmarking design이 필요하다.

#### 평가

- Quality: 2 / not good
- Clarity: 3 / good
- Significance: 3 / good
- Originality: 2 / not good
- Rating: 2 / Reject
- Confidence: 4 / 높은 확신

### 7.2 Reviewer 6XZg에 대한 우리 답변의 한국어 번역

#### What is new and what is not

Pilar--Wahlström은 이미 residual EBM learning, warm-up,
estimator-only initialization, joint EBM-likelihood training을 포함한다.
이 component들을 naPINN contribution으로 주장하지 않는다. Incremental
distinction은 trainable per-measurement inclusion gate와 rejection-cost
regularization이다.

RealPDEBench PIV trajectory의 controlled no-gate PINN-EBM은 rMAE
\(0.24688\pm0.02633\), momentum-residual RMS
\(0.02354\pm0.00394\)이고, naPINN은 각각
\(0.16719\pm0.00105\), \(0.00871\pm0.00070\)으로 32.3%와 63.0%
개선한다. 그러나 LAD-PINN이 held-out-PIV rMAE
\(0.11138\pm0.00122\)로 가장 좋다.

#### B-PINN fairness

Reviewer의 지적이 맞다. Evaluated model은 Yang et al.의 HMC
configuration이 아니라 mean-field variational inference다. Factorized
Gaussian weight posterior, training step당 one weight sample, KL weight
\(10^{-6}\), posterior-mean evaluation을 사용한다.

따라서 이를 `B-PINN-VI`로 명확히 부르고, 이 결과가 HMC를 reproduce하지
않으며 Bayesian method가 일반적으로 ineffective하다는 evidence도 아님을
밝힌다. HMC를 실행하지 않았으며 실행한 것처럼 암시하지 않는다.

#### Bayesian UQ versus gating

Original contrast가 과장되었다는 점에 동의한다. Appropriate likelihood를
사용하는 Bayesian inference는 noise sensitivity를 줄일 수 있다.
naPINN은 현재 network와 PDE parameter를 point estimate하고 observation
inclusion weight를 학습하며, parameter posterior를 inference하지 않는다.

Gate에 latent-variable interpretation이 있더라도 이 rebuttal에서
empirically 검증한 차이는 다음과 같다.

- EBM likelihood without gate
- EBM-derived pointwise gating with rejection regularization

Posterior UQ와 measurement selection은 complementary하며 결합될 수 있다.

#### Statistical preprocessing baseline

Published MAD-PINN의 full two-stage procedure를 구현했다.

1. LAD-PINN 30,000 update
2. \(\operatorname{median}(|r|)/1.6777\)로 scale estimation
3. \(3\widehat{\sigma}\) threshold로 component-level screening
4. Retained component에서 MSE-PINN 30,000 update

3-seed real-PIV 결과는 rMAE \(0.14810\pm0.00103\), rMSE
\(0.22537\pm0.00095\), momentum RMS
\(0.01399\pm0.00019\), continuity RMS
\(0.01349\pm0.00125\)다. Stronger preprocessing baseline이지만 총
60,000 update이므로 compute matched가 아니다.

#### Stability, hyperparameters, and corruptions

Persistent-failure stress test에서 192개 sensor 중 19개에 bias와 linear
drift를 모든 200 real-PIV frame에 주입했다. 이는 naturally observed
failure가 아니라 real PIV 위의 synthetic corruption이다.

Primary naPINN은 rMAE \(0.18234\pm0.00711\), AUROC
\(0.96234\pm0.00086\), failed rejection 99.29%, clean rejection 47.51%다.
LAD-PINN은 rMAE \(0.13212\pm0.00251\)로 더 좋다.

Rejection-cost sensitivity는 rMAE \(0.16928\pm0.00131\), AUROC
\(0.96673\pm0.00149\), failed rejection 98.86%, clean rejection 33.68%다.
Selectivity를 개선하지만 primary condition을 대체하지 않는다.

Full EMA coefficient sweep는 수행하지 않았다. Correct convention은
new-batch weight \(\beta=0.05\), old-state decay \(\rho=0.95\), upper
clamp는 running scale의 10배다.

Submitted Gaussian과 Laplace case도 controlled IID test이므로 deployment
concern을 해결하지 않는다. Natural-PIV study 및 injected persistent
failure와 구분하고 comprehensive hyperparameter optimization을
주장하지 않는다.

#### Training cost and references

MSE, LAD, PINN-EBM, naPINN은 30,000 PINN update를 사용하고, EBM method는
5,000 estimator-only update를 추가한다. Shared A100 mean end-to-end
observation은 각각 1,373.8, 1,451.3, 1,717.2, 1,714.4 s다. Hardware-isolated
wall-clock match가 아니다. MAD-PINN은 60,000 update, 평균 2,993.6 s다.

Baydin survey year는 1989에서 2018로 바로잡아야 한다.

---

## 8. 추가 실험 설계와 검증 결과

### 8.1 Natural real-PIV experiment

#### Dataset와 split

- Dataset: RealPDEBench Cylinder real time-resolved PIV
- Trajectory: `10031.h5`
- Frames: 1000--1199, 총 200 frame
- Reported Reynolds number: 10,031
- Training sensors: 192 fixed irregular spatial locations
- Training observations: 38,400 velocity vectors
- Held-out spatial locations: 7,747
- Held-out observations: 1,549,400 velocity vectors
- Training/held-out spatial overlap: 0
- Seeds: 40, 41, 42

#### Model

- Inputs: nondimensionalized \(x,y,t\)
- Outputs: pressure-latent \(u,v,p\)
- Constraint: nominal 2D incompressible Navier--Stokes
- Reynolds number: metadata 값으로 fixed
- Backbone: 5 hidden layers, width 80, `tanh`
- 공통 PINN update budget: 30,000

#### Result

아래 값은 3-seed mean \(\pm\) sample standard deviation이다.

| Method | Held-out rMAE | Held-out rMSE | Momentum RMS | Continuity RMS | Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| MSE-PINN | \(0.139813\pm0.002511\) | \(0.198120\pm0.001924\) | \(0.026651\pm0.000330\) | \(0.028818\pm0.001144\) | -- |
| LAD-PINN | **\(0.111378\pm0.001225\)** | **\(0.183773\pm0.000947\)** | \(0.045523\pm0.000430\) | \(0.075696\pm0.000527\) | -- |
| MAD-PINN | \(0.148101\pm0.001025\) | \(0.225367\pm0.000950\) | \(0.013987\pm0.000194\) | \(0.013493\pm0.001246\) | \(0.6421\pm0.0061\) |
| PINN-EBM (no gate) | \(0.246876\pm0.026331\) | \(0.337063\pm0.034804\) | \(0.023541\pm0.003935\) | \(0.046543\pm0.003377\) | -- |
| naPINN, \(\lambda_{\mathrm{rej}}=0.01\) | \(0.167193\pm0.001053\) | \(0.266159\pm0.001242\) | **\(0.008709\pm0.000704\)** | **\(0.008119\pm0.001387\)** | \(0.757\pm0.014\) |

#### 해석

- LAD-PINN이 independent held-out PIV와 가장 잘 맞는다.
- naPINN은 MSE, LAD, MAD보다 field discrepancy가 크다.
- naPINN은 nominal physics residual이 가장 작다.
- no-gate PINN-EBM과 비교하면 naPINN은 rMAE 32.3%, rMSE 21.0%,
  momentum RMS 63.0%, continuity RMS 82.6%를 낮춘다.
- 이 결과는 `gating over learned likelihood alone`의 효과를 지지한다.
- 그러나 held-out PIV가 noise-free ground truth가 아니므로 field
  discrepancy와 true solution error를 동일시하면 안 된다.
- Low nominal-physics residual이 더 정확한 real flow를 뜻한다고 단정하면
  안 된다. Nominal 2D PDE 자체가 real experiment와 mismatch될 수 있다.

### 8.2 Rejection-cost behavior on natural PIV

- Original synthetic-scale rejection cost 0.5는 completed seed에서 모든
  measurement를 accept했다. Mean gate weight는 거의 1이었다.
- \(\lambda_{\mathrm{rej}}=0.005\)의 3-seed retention은
  \(0.671\pm0.220\)으로 seed variability가 매우 컸다.
- \(\lambda_{\mathrm{rej}}=0.01\)은 retention
  \(0.757\pm0.014\)로 훨씬 안정적이어서 natural-PIV comparison에
  사용했다.
- 이 선택은 universal default가 아니라 real-PIV residual scale에 대한
  calibration이다.

### 8.3 Injected persistent sensor-failure experiment

#### Corruption design

- Parent data: 위 natural real-PIV split
- Failed sensor identities: 19 / 192, 약 9.896%
- Affected components: \(u,v\) 모두
- Affected frames: 200 frame 전체
- Failed scalar measurements: 7,600
- Clean scalar measurements: 69,200
- Persistent signed bias: clean component standard deviation의 3배
- Linear drift: final frame에서 standard deviation의 2배까지 증가
- Held-out PIV: parent artifact와 동일
- Clean 173 training sensor: parent artifact와 동일

이는 `real sensor failure dataset`이 아니라 **real PIV training data 위에
주입한 structured failure**다.

#### Result

| Method | rMAE | rMSE | Momentum RMS | Continuity RMS | AUROC | Failed rejection | Clean rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MSE-PINN | \(0.211818\pm0.002607\) | \(0.288363\pm0.008881\) | \(0.049153\pm0.001011\) | \(0.044038\pm0.000762\) | -- | -- | -- |
| LAD-PINN | **\(0.132119\pm0.002515\)** | **\(0.208813\pm0.001574\)** | \(0.051266\pm0.000639\) | \(0.073476\pm0.000562\) | -- | -- | -- |
| PINN-EBM (no gate) | \(0.265452\pm0.088381\) | \(0.355668\pm0.082253\) | \(0.048471\pm0.014515\) | \(0.088892\pm0.015267\) | -- | -- | -- |
| naPINN, primary 0.005 | \(0.182338\pm0.007108\) | \(0.283806\pm0.009555\) | **\(0.004321\pm0.001909\)** | **\(0.004144\pm0.002158\)** | \(0.962342\pm0.000864\) | \(99.29\%\pm0.65\%\) | \(47.51\%\pm10.63\%\) |
| naPINN, 0.01 sensitivity | \(0.169280\pm0.001310\) | \(0.263777\pm0.003353\) | \(0.008319\pm0.000821\) | \(0.007139\pm0.000782\) | \(0.966731\pm0.001488\) | \(98.86\%\pm0.38\%\) | \(33.68\%\pm7.62\%\) |

#### 해석

Primary 0.005는 MSE-PINN보다 rMAE를 13.9%, rMSE를 1.6% 낮추며,
momentum과 continuity residual을 약 91% 낮춘다. Failed sensor ranking은
강하지만 clean rejection 47.5%는 큰 문제다.

Sensitivity 0.01은 MSE-PINN보다 rMAE를 20.1%, rMSE를 8.5% 낮추고,
clean rejection을 33.7%로 줄인다. 대신 primary보다 nominal-physics
residual이 크다.

두 setting 모두 LAD-PINN보다 held-out-PIV agreement가 나쁘다. 따라서
response에서는 anomaly-detection capability와 physics consistency를
강조할 수 있지만, clean measurement 보존이나 field reconstruction의
우월성을 과장하면 안 된다.

### 8.4 Closest-prior comparison

Pilar--Wahlström audit로 확인한 prior method의 핵심은 다음과 같다.

- MSE PINN warm-up
- Current residual에 대한 estimator-only EBM initialization
- Learned residual likelihood를 직접 data loss로 사용
- PINN, EBM, PDE parameter의 joint optimization
- Navier--Stokes joint phase에서 physics weight \(\omega=50\)

따라서 rebuttal의 novelty statement는 다음처럼 좁혀야 한다.

> Residual EBM learning과 staged schedule 자체가 아니라, learned residual
> reliability를 per-measurement gate로 변환하고 rejection cost로 gate의
> trivial rejection을 방지하는 것이 incremental contribution이다.

Natural PIV와 structured-failure 모두에서 no-gate PINN-EBM은 naPINN보다
field discrepancy와 physics residual이 크고 seed variability도 높았다.
그러나 이 결과를 Pilar--Wahlström 전체 method에 대한 일반적 우월성으로
확대하면 안 된다. 동일 backbone과 split에서 gate contribution을
분리하기 위한 controlled implementation으로 설명한다.

### 8.5 Compute accounting

| Method | PINN updates | Estimator-only updates | Mean observed end-to-end time |
| --- | ---: | ---: | ---: |
| MSE-PINN | 30,000 | 0 | 1,373.8 s |
| LAD-PINN | 30,000 | 0 | 1,451.3 s |
| PINN-EBM | 30,000 | 5,000 | 1,717.2 s |
| naPINN | 30,000 | 5,000 | 1,714.4 s |
| MAD-PINN | 60,000 | 0 | 2,993.6 s |

naPINN의 30,000 PINN update는 5,000 warm-up + 25,000 joint update다.
5,000 estimator-only update는 PINN parameter를 update하지 않지만 total
work에는 포함되어야 한다.

시간은 shared A100 환경에서 관찰한 값이므로 다음 표현은 피한다.

- “strict wall-clock-matched”
- “hardware-controlled timing”
- “MAD-PINN과 compute matched”

사용 가능한 표현은 다음과 같다.

- “same 30,000 PINN-update budget”
- “EBM methods additionally use 5,000 estimator-only updates”
- “observed end-to-end time on shared A100 GPUs”

---

## 9. 통합 rebuttal 전략

### 9.1 Score-moving priorities

#### Priority 1: Real-data evidence

AC가 score reconsideration의 조건으로 직접 언급했다. 따라서 각 response는
가능하면 real-PIV experiment로 시작한다. 다만 “compelling win”이라고
과장하지 않고 다음의 mixed result를 명확히 한다.

- LAD: best held-out-PIV agreement
- naPINN: best nominal-physics residual
- naPINN: substantial improvement over closest no-gate PINN-EBM

#### Priority 2: Closest-prior attribution

Originality를 방어하려고 Pilar--Wahlström과의 유사성을 축소하면 오히려
신뢰를 잃는다. Prior contribution을 먼저 인정하고 incremental novelty를
정확하게 한정한다.

#### Priority 3: Fairer baselines

- Direct PINN-EBM no-gate comparison
- Faithful two-stage MAD-PINN
- Existing LAD-PINN
- Submitted OrPINN의 두 \(q\) setting

다만 tuned robust-loss sweep를 수행했다고 주장하지 않는다.

#### Priority 4: Corrections and transparency

- B-PINN을 `B-PINN-VI`로 정확히 설명
- HMC comparison이 아님을 인정
- EMA \(\beta=0.05\), \(\rho=0.95\) convention 설명
- 5,000 estimator-only update를 total cost에 포함
- Baydin 2018 correction
- Parameter recovery evidence가 Allen--Cahn에 한정됨을 인정

### 9.2 Reviewer별 communication strategy

| 대상 | 첫 메시지 | 가장 중요한 evidence | 반드시 인정할 limitation |
| --- | --- | --- | --- |
| AC | Real PIV를 추가했고 closest prior를 재평가했다 | naPINN vs no-gate PINN-EBM, structured failure | LAD가 field agreement에서 우수, one trajectory, fixed Re |
| 6SDM | 요청한 real/irregular/structured test와 stronger baseline을 수행했다 | Real PIV, MAD, no-gate, compute accounting | Correlation을 정량화하지 않음, EMA full sweep 없음 |
| aoJS | Pilar--Wahlström이 closest prior라는 데 동의한다 | Direct no-gate comparison | Burgers/lambda--omega parameter result 없음 |
| 6XZg | B-PINN fairness 지적이 맞다 | B-PINN-VI clarification, MAD baseline | HMC 미실행, Bayesian UQ 비효율 주장 철회 |

### 9.3 Response ordering

각 response는 가능한 한 다음 순서로 구성한다.

1. Reviewer의 핵심 지적에 감사하고 맞는 부분을 인정
2. Real-PIV protocol과 mixed outcome
3. Direct closest-prior comparison
4. Structured failure 또는 MAD-PINN evidence
5. Compute 및 notation correction
6. 해결하지 못한 limitation
7. Revision에서 명확히 할 내용

### 9.4 피해야 할 표현

다음 표현은 evidence보다 강하므로 사용하지 않는다.

- “naPINN is best on real data.”
- “naPINN solves physics-model discrepancy.”
- “The gate accurately identifies observation noise.”
- “The structured failure is a real sensor failure.”
- “We compare against HMC B-PINN.”
- “All methods are compute matched.”
- “The EMA coefficient is insensitive.”
- “We demonstrate parameter recovery on all PDEs.”
- “The revised paper already contains these results.”

대신 다음처럼 쓴다.

- “LAD-PINN gives the best held-out-PIV agreement, whereas naPINN gives the
  lowest residuals under the nominal physics model.”
- “The real-PIV setup is a model-discrepancy stress test but does not separate
  observation error from model discrepancy.”
- “The injected-failure test is controlled corruption over real PIV.”
- “The evaluated Bayesian baseline is mean-field B-PINN-VI, not HMC.”
- “Timing values are observational measurements on shared GPUs.”
- “The detailed parameter-recovery evidence is limited to Allen--Cahn.”

### 9.5 Discussion period 대응

Initial response 이후 reviewer가 질문할 가능성이 높은 지점은 다음과 같다.

1. **왜 LAD보다 field error가 큰데 naPINN이 유용한가?**

   답변은 nominal-physics consistency와 learned-density/no-gate 대비 개선을
   설명하되, application objective에 따라 LAD가 더 적합할 수 있음을
   인정해야 한다.

2. **Low PDE residual이 real accuracy를 뜻하는가?**

   아니다. Nominal model mismatch 가능성이 있으므로 low residual을
   physical truth의 증거로 해석하지 않는다.

3. **왜 clean rejection이 이렇게 높은가?**

   Gate calibration이 data scale에 민감하고 observation/model discrepancy가
   residual에 함께 나타나기 때문이다. Sensitivity가 개선하지만 one-third
   clean rejection이 남아 있으며 limitation으로 인정한다.

4. **왜 inverse parameter를 추정하지 않았는가?**

   Real-PIV experiment에서는 Reynolds number를 metadata 값으로 고정하여
   controlled reconstruction/model-discrepancy test를 우선했다. 따라서
   inverse-coefficient evidence로 제시하지 않는다.

5. **Pilar--Wahlström보다 정확히 무엇이 새로운가?**

   Per-measurement reliability gate와 rejection regularizer가 incremental
   contribution이다. Residual EBM과 staged training은 prior work다.

6. **왜 HMC B-PINN을 실행하지 않았는가?**

   Submitted baseline은 mean-field VI였고 이를 정확히 한정한다. 급하게
   HMC implementation을 추가해 불공정한 comparison을 만드는 대신,
   잘못된 일반화 주장을 철회한다.

---

## 10. 남은 limitation과 내부 결정 사항

### 10.1 해결되지 않은 reviewer request

- Full EMA coefficient sensitivity sweep
- Burgers viscosity recovery
- Lambda--omega reaction parameter recovery
- Hardware-isolated wall-clock matching
- Multiple real trajectories 또는 multiple real systems
- Naturally labeled sensor-failure dataset
- Explicitly quantified spatial/temporal correlated noise
- Explicit model-discrepancy model
- HMC B-PINN comparison

### 10.2 Rebuttal에서의 처리

이 항목들은 future work로만 넘기기보다 현재 evidence의 범위를 정확히
한정하는 방식으로 답한다.

- “We did not complete...”라고 명확히 쓴다.
- 없는 숫자를 paper claim이나 smoke test에서 추정하지 않는다.
- Original synthetic table을 현재 checkout에서 regenerate했다고 말하지
  않는다.
- Smoke result를 rebuttal evidence로 사용하지 않는다.

### 10.3 현재 가장 방어 가능한 claim

> naPINN은 controlled synthetic corruption에서는 강한 성능을 보였으며,
> real-PIV stress test에서는 fixed robust loss를 universally 이기지 못했다.
> 그러나 closest no-gate residual-EBM objective보다 held-out-PIV
> discrepancy와 nominal-physics residual을 개선했고, structured persistent
> failure를 높은 AUROC로 ranking했다. 동시에 clean rejection이 상당하여
> calibration과 selectivity가 중요한 limitation으로 남는다.

이 claim은 positive result, negative result, closest-prior positioning,
real-data scope를 모두 일관되게 반영한다.

## 11. 최종 제출 전 checklist

- [ ] OpenReview의 정확한 response deadline과 timezone 확인
- [ ] Reviewer별 올바른 response box에 해당 section만 붙여넣기
- [ ] Response 안에 URL 또는 Markdown link가 없는지 재확인
- [ ] 각 response가 10,000 characters 이하인지 OpenReview 입력창에서도 확인
- [ ] `paper를 수정했다/업로드했다`는 표현이 없는지 확인
- [ ] Natural PIV와 injected structured failure를 혼동하지 않았는지 확인
- [ ] LAD가 held-out-PIV agreement에서 best라는 사실을 유지했는지 확인
- [ ] Clean rejection 47.5%와 sensitivity 33.7%를 누락하지 않았는지 확인
- [ ] `B-PINN-VI`, `not HMC`를 명확히 했는지 확인
- [ ] EMA \(\beta=0.05\), \(\rho=0.95\)를 혼동하지 않았는지 확인
- [ ] MAD-PINN 60,000-update non-compute-matched caveat 확인
- [ ] Burgers/lambda--omega parameter 결과가 없다는 사실을 유지했는지 확인
- [ ] Discussion 동안 새 질문에 답할 때 frozen aggregate 이외의 숫자를
      즉석에서 만들지 않기
