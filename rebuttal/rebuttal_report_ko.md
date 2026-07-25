# naPINN NeurIPS 2026 Rebuttal 한국어 종합 보고서

## 1. 목적, 범위, 현재 상태

이 문서는 Meta Review와 세 reviewer의 지적을 한국어로 정리하고,
author response의 논리, 추가 experiment 설계, 검증된 결과, 남은
limitation을 함께 기록하는 내부 보고서다. 중요한 nuance가 있는
terminology는 English를 유지한다.

사용자 지시에 따라 `paper/neurips_2026.tex`는 rollback 상태를 유지하며
이 rebuttal 작업에서 수정하지 않는다. 이 문서에서 “향후 명확히
하겠다”는 표현은 reviewer와의 communication 또는 camera-ready 이후의
계획을 뜻하며, 현재 제출 paper를 고친다는 뜻이 아니다.

결과의 상태는 다음과 같이 구분한다.

- `확정`: full run, non-smoke, seeds 40--42가 모두 완료되고 aggregation된
  결과
- `예비`: full run이지만 아직 seed 하나만 완료된 결과
- `진행 중`: runner와 smoke validation은 끝났으나 full matrix가 끝나지
  않은 결과
- `limitation`: 이번 response에서 실험으로 해결하지 않고 인정할 항목

Natural/unmodified PIV result는 사용자 요청에 따라 evidence table에서
제외한다. 이후 PIV 표는 모두 real PIV의 training measurements에
controlled corruption을 주입한 setting만 포함한다.

### Author decision: direct comparison response hold

현재까지 완료된 direct naPINN-versus-PINN-EBM 수치, ranking, 그리고
그로부터 도출되는 contribution/novelty 결론은 모두 내부 evidence로
보존하되 reviewer response에는 아직 사용하지 않는다. Real-PIV
legacy-corruption, scale, robust-loss, closest-prior 관련 실험을 가능한
범위까지 수행한 뒤 저자가 response 방향을 명시적으로 결정한다. 아래
표의 direct-comparison 수치는 experiment planning을 위한 내부 기록이며,
그 전까지 reviewer-facing report로 복사하면 안 된다.
이 문서 뒤쪽에 남아 있는 “direct result를 response에 report한다”는 기존
표현이 있다면 이 author decision이 우선하며, 추가 실험과 저자 release
전에는 실행 지침으로 사용하지 않는다.

## 2. 먼저 내릴 결론

### 2.1 Real-data 전략: 선택지 (a)가 아니라 제한된 선택지 (b)

“공개된 noisy PDE sensor dataset이 없다”는 절대적 주장은 사용하지 않는
것이 안전하다. RealPDEBench 공식 설명 자체가 real measurements에는
camera와 particle tracking에서 비롯된 measurement noise가 있다고
명시하기 때문이다. 따라서 선택지 (a)를 그대로 주장하면 reviewer가
공개 benchmark 하나만 들어도 반박할 수 있다.

대신 다음의 더 좁고 검증 가능한 문장을 사용한다.

> We did not identify a public inverse-PDE corruption benchmark that jointly
> provides known corruption labels, a clean paired physical reference, the
> governing PDE, and unknown coefficients.

추가 real-data evidence는 선택지 (b)를 따른다. RealPDEBench Cylinder의
real time-resolved PIV를 기반으로 training sensors에만 persistent bias,
linear drift, temporally correlated AR(1) corruption, spatial burst failure를
주입한다. Unmodified held-out PIV는 independent reference로 사용하되
`noise-free ground truth`라고 부르지 않는다.

이 선택은 “real dataset이므로 자동으로 real-world noise를 증명한다”는
주장이 아니다. 정확한 주장은 다음과 같다.

- field와 coordinates는 실제 PIV에서 온다.
- corruption은 controlled injection이며 label을 안다.
- irregular sensor geometry와 nominal Navier--Stokes model discrepancy는
  실제 data에서 유지된다.
- 따라서 pure synthetic PDE field보다 한 단계 현실적이지만, naturally
  occurring sensor-failure validation을 대체하지는 않는다.

Closest related work 다수가 synthetic field 또는 real-derived field에
injected noise를 사용했다는 사실은 contextual defense로는 쓸 수 있다.
그러나 AC가 명시적으로 compelling real example을 요구한 상황에서
“prior work도 모두 synthetic이었다”만으로 evaluation gap이 해소되지는
않는다. 이 점 때문에 injected real PIV를 추가 evidence로 사용한다.

### 2.2 Pilar--Wahlström에 관한 결론

PINN-EBM 관계를 모든 response에서 자발적으로 foreground하거나
naPINN의 전체 contribution을 “gate 하나의 incremental extension”으로
축소하지 않는다. AC와 aoJS처럼 closest prior를 직접 지적한 경우, 그리고
6SDM처럼 learned-noise/EBM baseline을 요청한 경우에만 필요한 범위에서
대응한다.

이때 Pilar--Wahlström이 이미 사용한 다음 요소는 정확히 인정한다.

- unknown residual noise distribution을 EBM으로 학습
- MSE warm-up
- residual estimator initialization
- learned EBM negative log-likelihood를 PINN data loss로 사용
- PINN, PDE parameter, EBM을 joint training

따라서 “EBM module을 PINN과 함께 학습하는 것”이나 staged initialization
자체를 naPINN의 novelty로 주장하면 안 된다.

그러나 naPINN을 단순히 “gate를 추가한 방법”이라고 설명하지도 않는다.
Objective와 gradient path의 차이를 구체적으로 제시한다.

- PINN-EBM은 EBM negative log-likelihood 자체를 PINN measurement
  objective로 사용하며 residual을 통한 gradient가 PINN, PDE parameter,
  EBM을 함께 update한다.
- naPINN의 estimator는 detached residual에 대한 density objective를
  담당하고, density score는 explicit trainable per-measurement gate의
  입력이 된다.
- PINN은 gate-weighted base reconstruction loss를 받으며,
  `rejection-cost regularization`이 indiscriminate rejection을 억제한다.
- Density estimation, inclusion, reconstruction의 역할이 분리되므로
  MSE, L1, q-Gaussian base loss와 결합할 수 있다.

사용자가 과거에 direct PINN-EBM이 동작하지 않았다고 관찰했더라도,
rebuttal에서 그렇게 주장하면 안 된다. 기존 direct comparison 결과는
삭제하거나 선택적으로 무시하지 않고 내부 evidence로 보존한다. 다만
현재는 `RESPONSE HOLD`이므로 수치, 승패, 그리고 그에 따른 novelty
positioning을 reviewer에게 보고하지 않는다. 추가 실험 완료 후 저자가
release한 경우에만 관련된 complete outcome을 사용한다.

### 2.3 현재 acceptance 전략의 현실적 평가

가장 score-moving한 evidence는 severe injected real-PIV result다. 30%
persistent failure에서 naPINN은 MSE/LAD/OrPINN보다 field error가 낮고
nominal-physics residual도 훨씬 작다. 반면 10% persistent failure의
earlier cost-0.01 sensitivity에서는 LAD-PINN이 field agreement 기준으로
naPINN보다 좋다. Direct PINN-EBM 대비 결과는 현재 response 방향의
근거로 사용하지 않고 내부 hold 상태로 둔다.

현재 승인 가능한 real-data 요약은 다음 범위다.

> Under 30% persistent sensor failure, naPINN improves both field metrics
> over MSE, LAD, and OrPINN. At 10% failure, LAD remains the strongest field
> baseline.

이 honest framing은 AC의 significant real benefit 요구에는 실질적인
evidence를 제공한다. Closest-prior numerical paragraph와 최종
originality positioning은 추가 실험 후 별도로 결정한다.

### 2.4 Reviewer별 score-moving 가능성

- AC: 가장 명시적으로 real example을 score-changing condition으로
  제시했다. 따라서 selected-cost 30% PIV와 correlated-failure 결과를
  opening에 둔다. 다만 injected corruption이고 LAD가 field metric에서
  이길 수 있으므로 “compelling significant benefit” 요구를 완전히
  충족한다고 장담할 수 없다.
- 6SDM: Confidence 5의 reject이므로 real PIV, stronger baselines,
  sensitivity, compute를 모두 채워도 score reversal 가능성은 낮다.
  빠진 experiment를 채웠다는 사실보다 adverse result까지 포함한
  evaluation completeness를 강조한다.
- aoJS: 기술적 solidness를 인정한 borderline reject이며 질문이 가장
  구체적이다. Direct-comparison slot은 남겨 두되 현재 수치와 ranking은
  hold한다. All-parameter recovery와 exact cost는 독립적으로 답할 수
  있으며, closest-prior response는 추가 실험 후 결정한다.
- 6XZg: HMC B-PINN을 실행하지 않으므로 fairness concern을 완전히
  해소하지 못한다. 대신 B-PINN-VI limitation, Bayesian methods에 대한
  과장 철회, MAD preprocessing으로 방어 가능한 범위만 답한다.

따라서 response 목표는 모든 score를 뒤집는다는 약속이 아니라, AC가
요청한 evidence를 최대한 제공하면서 scientific record를 정확히
정리하는 것이다.

## 3. Meta Review 번역과 답변

### 3.1 Reviewer comment 번역

논문은 observation process의 error model을 학습한 뒤, 신뢰할 수 없는
observation을 down-weight하여 inverse problem을 더 robust하게 푸는
naPINN을 제안한다.

장점은 standard PINN의 MSE가 observation error를 Gaussian으로
가정한다는 중요한 문제를 다룬다는 점이다. 실제 error가 non-Gaussian인
경우 bias가 빠르게 생길 수 있다. 방법이 상대적으로 단순하고 여러
synthetic problem에서 잘 보였다는 점도 장점이다.

가장 큰 약점은 real problem demonstration이 없다는 것이다. 논문의
동기가 실제 assumed error model이 자주 틀린다는 데 있으므로, real
problem에서 가치 있는 차이를 보이지 않은 것은 oversight다. Synthetic
problem의 observation-model error는 연구자의 상상에 제한되지만 자연은
훨씬 더 복잡할 수 있다.

실제 문제에서는 observation-model discrepancy뿐 아니라 underlying PDE가
틀린 physics-model discrepancy도 있을 수 있다. 이를 따로 account하지
않으면 observation error model이 두 discrepancy를 함께 떠맡아 매우
복잡한 distribution이 될 수 있다. 이런 scenario에서 demonstration이
필요하다.

또한 Pilar and Wahlström [31]은 introduction에서 언급한 것보다 훨씬
가까운 prior다. 현재 논문은 reject 방향이며, compelling real example이
추가되어 significant benefit을 보이면 score를 재고할 수 있다.

### 3.2 우리의 답변

지적을 인정하고, controlled injected-real-PIV experiment를 가장 먼저
제시한다. 200 real PIV frames, 192 fixed irregular training sensors,
38,400 training velocity vectors, training 위치와 겹치지 않는 7,747개
spatial locations의 1,549,400 held-out vectors를 사용한다.

Training sensors에는 persistent bias와 linear drift, AR(1) drift,
spatially correlated burst를 주입한다. Held-out PIV와 corruption되지
않은 training sensors는 injection 전 data와 bitwise identical하다.

동시에 이 실험의 한계를 명시한다.

- held-out PIV도 measurement이므로 clean physical truth가 아니다.
- nominal incompressible Navier--Stokes residual을 쓰므로 model
  discrepancy가 존재할 수 있다.
- method는 observation error와 model discrepancy를 separately
  identify하지 않는다.
- 따라서 이는 model mismatch를 포함한 stress test이지 discrepancy
  decomposition의 증명이 아니다.

Pilar--Wahlström의 선행 contribution을 전부 인정하고 direct no-gate
PINN-EBM을 추가한다. Novelty는 gate와 rejection regularization으로
제한한다.

여기서 prior와의 차이는 “prior가 only mild Gaussian noise만 썼다”로
과장하지 않는다. Official paper는 Gaussian, uniform, nonzero-mean
three-Gaussian mixture와 zero-mean mixture를 사용했다. 더 정확한 차이는
그들이 `homogeneous noise distribution`을 학습했고 explicit sparse
point-outlier fraction이나 per-measurement corruption label을 평가하지
않았다는 점이다. 그 논문 역시 heteroscedastic extension과 suitable
real-world dataset 적용을 future work로 남겼다.

## 4. Reviewer 6SDM 번역과 답변

### 4.1 Reviewer comment 번역

Reviewer는 problem이 중요하고 reliability gate가 intuitive하다고
평가했다. 하지만 다음을 우려했다.

1. 모든 experiment가 controlled synthetic stress test이고 real 또는
   semi-realistic data가 없다.
2. standard PDE benchmark만 사용하여 model mismatch, irregular sensors,
   correlated noise, sensor-level failure, drift에서 동작하는지 모른다.
3. MAD-PINN, learned unknown-noise/EBM likelihood, tuned robust loss 같은
   더 강한 baseline이 없다.
4. EMA normalization coefficient가 중요한데 sensitivity가 없고 main
   text의 beta와 appendix의 rho notation이 혼란스럽다.
5. compute-matched 또는 wall-clock-matched comparison이 필요하다.

평점은 Quality 2, Clarity 3, Significance 2, Originality 2,
overall `2: Reject`, Confidence 5였다.

### 4.2 우리의 답변

RealPDEBench experiment가 1번과 2번에 직접 대응한다. Sensor positions는
irregular하고, persistent identity-level failure, temporal correlation,
spatial correlation을 별도로 시험한다.

Baseline matrix는 다음과 같다.

- MSE-PINN
- LAD-PINN
- OrPINN \(q=1.9\), \(q=2.9\)
- direct PINN-EBM, no gate
- naPINN
- MAD-PINN: 30k LAD training, MAD screening, 30k retained-data MSE retraining
- estimator-free fixed-quantile gate와 learnable residual-threshold gate

EMA는 implementation convention을 정확히 설명한다. Update가

\[
s_t=(1-m)s_{t-1}+m\hat{s}_t
\]

이면 code/config의 값은 update weight \(m\)이다. 이를 decay coefficient로
쓰면 \(\rho=1-m\)이다. 따라서 \(m=0.01,0.05,0.10\) sensitivity를
보고하고, beta/rho가 같은 수라고 말하지 않는다. Implementation은
numerical guard로 batch scale을 최소 \(10^{-6}\), 최대 이전 running
scale의 10배로 clamp한다는 점도 response에서 숨기지 않는다.

Compute는 update count와 wall time을 분리해 답한다.

- 일반 baseline: 30,000 PINN updates
- PINN-EBM/naPINN: 5,000 warm-up + 25,000 joint = 30,000 PINN updates,
  별도로 5,000 estimator-only updates
- MAD-PINN: 30,000 LAD + 30,000 MSE = 60,000 PINN updates이며
  compute-matched baseline이 아님

Estimator initialization이 “몇 초뿐”이라고 미리 단정하지 않는다. 첫
완료 synthetic run에서는 약 18--21초였고 total training의 약
2--4%였다. 모든 최종값은 exact phase timing으로 보고한다. Shared A100의
wall time은 observational value이며 hardware-isolated benchmark라고
부르지 않는다.

## 5. Reviewer aoJS 번역과 답변

### 5.1 Reviewer comment 번역

Reviewer는 problem motivation, residual-density estimation과 trainable
gate의 조합, rejection-cost term, 10-trial synthetic result, staged
training ablation, 낮은 reported overhead를 장점으로 보았다.

주요 약점은 다음과 같다.

1. Pilar--Wahlström은 unknown non-Gaussian measurement noise를 EBM으로
   학습하고 그 likelihood를 data loss로 이미 사용했다. Gate와 rejection
   cost가 의미 있는 extension일 수 있지만 direct comparison이 필요하다.
2. 실제 sensor corruption은 spatially/temporally correlated,
   heteroscedastic, biased, drifting, cross-variable correlated할 수 있다.
3. Allen--Cahn 외에 Burgers viscosity와 lambda--omega beta recovery가
   보고되지 않았다.
4. naPINN의 5,000 estimator-initialization iterations가 30,000-step
   comparison 밖에 있어 total cost가 불명확하다.
5. Baydin automatic differentiation reference의 1989 표기는 2018이
   맞다.

평점은 Quality 3, Clarity 3, Significance 2, Originality 2,
overall `3: Borderline reject`, Confidence 4였다.

### 5.2 우리의 답변

Direct PINN-EBM을 가장 중요한 baseline으로 추가한다. 이 baseline은
warm-up 이후 EBM negative log-likelihood를 PINN data loss로 직접
backpropagate하며 gate와 rejection cost가 없다. 새 결과가 naPINN에
불리해도 모두 보고한다.

Gate-free/fixed-screening question에는 세 증거를 구분한다.

- direct PINN-EBM: learned likelihood only
- fixed-quantile gate: non-trainable within-batch quantile selection
- estimator-free threshold gate: threshold와 steepness는 trainable
- MAD-PINN: published fixed statistical screen을 쓰는 two-stage preprocessing

Parameter recovery는 세 benchmark 모두에서 기록한다.

- Allen--Cahn: \(\epsilon^\star=0.3\)
- Burgers: \(\nu^\star=0.01\)
- lambda--omega: \(\beta^\star=1.0\)

Cost schedule은 위 4.2와 같이 정확히 밝힌다. Baydin reference year는
reviewer가 맞으며, response에서 typo를 인정한다. Frozen paper source는
이번 작업에서 수정하지 않는다.

## 6. Reviewer 6XZg 번역과 답변

### 6.1 Reviewer comment 번역

Reviewer는 real measurement와 Gaussian-noise assumption의 mismatch가
중요하고 methodology에 statistical grounding이 있다고 보았다. 하지만
benchmarking clarity와 fairness에 다음 문제가 있다고 지적했다.

1. Submitted B-PINN은 original work의 stronger HMC가 아니라 factorized
   Gaussian posterior를 사용하므로 weaker comparison이다.
2. Bayesian UQ가 noise와 anomaly의 영향을 줄이는 데 비효율적인 것처럼
   쓴 부분은 과장되어 있다.
3. naPINN도 Bayesian latent-variable model과 유사한 면이 있어 B-PINN과
   본질적으로 완전히 다른 것처럼 contrast하면 안 된다.
4. Basic z-score/MAD preprocessing으로 outlier를 제거한 baseline이
   필요하다.
5. Stability, hyperparameter optimization, different noise distributions,
   training cost를 더 자세히 설명해야 한다.
6. Reference [3] 연도가 잘못되었고 다른 reference도 확인해야 한다.

Main question은 더 공정하고 상세한 benchmark를 요구하며, naPINN이
B-PINN의 단순한 subset 이상으로 충분히 구별되는지 의문을 제기한다.

평점은 Quality 2, Clarity 3, Significance 3, Originality 2,
overall `2: Reject`, Confidence 4였다.

### 6.2 우리의 답변

B-PINN 지적은 인정한다. Submitted implementation은
`mean-field variational inference`를 쓴 B-PINN-VI이며 HMC와 같지 않다.
이번 rebuttal에서 급하게 full-network HMC를 구현해 “faithful HMC”라고
부르는 것은 더 위험하다. 대신 baseline 명칭과 limitation을 정확히
밝힌다. Faithful HMC comparison에는 prior, step size, trajectory length,
burn-in, posterior sample count와 convergence diagnostics가 함께 필요하지만
current repository에는 그 implementation과 artifact가 없다. 이 조건 없이
새 sampler를 붙인 수치를 stronger B-PINN이라고 부르지 않는다.

Bayesian inference와 adaptive selection을 대립시키지 않는다. Bayesian
methods도 robust likelihood와 posterior inference를 통해 noise와
outlier를 다룰 수 있다. naPINN의 차이는 posterior UQ를 대체한다는 것이
아니라, point-estimate PINN에서 explicit inclusion weight를 학습한다는
것이다. 두 접근은 complementary하다.

Preprocessing에는 faithful MAD-PINN과 fixed-quantile gate를 추가하고,
별도 estimator-free learnable residual-threshold gate도 비교한다. Noise
family는 Gaussian, Laplace, Student-t, four-Gaussian,
persistent drift, AR(1), spatial burst를 포함한다. 이는 real-world
corruption 전체를 대표한다는 증명이 아니라 broader stress coverage다.

## 7. Additional experiment protocol

### 7.1 Synthetic core

- PDE: Allen--Cahn, Burgers, lambda--omega
- Four-Gaussian corruption ratio: 5%, 10%, 15%
- Seeds: 40, 41, 42
- Methods: MSE, LAD, OrPINN \(q=1.9,2.9\), direct PINN-EBM, naPINN
- Common PINN budget: 30,000 updates
- Outputs: field rMAE/rMSE, learned PDE parameter, absolute parameter error,
  phase time, peak GPU memory

Provenance limitation이 있다. Current checkout에는 submitted Burgers와
lambda--omega table을 만든 original run artifacts와 canonical manifest가
없었고, 두 numerical dataset은 current generator에서 다시 생성했다.
따라서 새 표는 traceable additional experiment이지만 submitted table의
bitwise reproduction 또는 replacement라고 부르지 않는다. Paper의 기존
수치와 smoke/additional run을 조용히 교체해서도 안 된다.

### 7.2 Synthetic robustness and ablation

- lambda--omega: 같은 15% severe extra-outlier injection 아래 background
  Gaussian, Laplace, Student-t, four-Gaussian noise
- naPINN base data loss: MSE, L1, OrPINN \(q=2.9\)
- EMA update weight: 0.01, 0.05, 0.10
- Rejection cost: 0.10, 0.30, 0.50, 0.70, 1.00
- Fixed-quantile gate, learnable residual-threshold gate, MAD-PINN

Paper/config audit에서 submitted appendix는 Allen--Cahn의
\(\lambda_{\mathrm{rej}}=1.0\), 나머지 PDE의 0.5를 적고 있지만 initial
rebuttal core config는 세 PDE 모두 0.5였음을 확인했다. 따라서 0.5
Allen result를 버리거나 덮어쓰지 않고 sensitivity로 보존하고,
paper-aligned 1.0을 모든 ratio와 seed에서 별도로 실행한다. 또한 같은
appendix의 sensitivity text가 0.9/1.0에서 degradation을 말하므로, 이
internal inconsistency는 reviewer에게 숨길 수 없으며 결과와 함께
clarify해야 한다.

### 7.3 Injected real PIV

- Dataset: RealPDEBench Cylinder real PIV, trajectory `10031.h5`
- Frames: 1000--1199, 총 200
- Training: 192 fixed irregular sensors, 38,400 velocity vectors
- Evaluation: training 위치와 겹치지 않는 7,747 spatial locations,
  1,549,400 velocity vectors
- PDE: pressure-latent 2D incompressible Navier--Stokes
- Reynolds number: metadata value 10031로 고정
- Held-out reporting seeds: 40, 41, 42
- Calibration seed: 39

Rejection-cost calibration grid는 0.005, 0.01, 0.05, 0.10, 0.5다.
처음 세 값 0.005/0.01/0.5만 계획했으나, seed-39의 0.01이 failed
rejection 98.52%와 동시에 clean rejection 84.37%를 보여 eligibility를
실패한 뒤 0.05/0.10을 추가했다. Chronology상 10% condition의 cost-0.01
held-out runs는 이 grid refinement 전에 이미 완료되어 있었다. 따라서
이 calibration을 완전한 `held-out-blind tuning`이라고 부르면 안 된다.
추가 값은 10% 성능이 아니라 seed-39 grid의 0.01--0.5 공백만을 근거로
정했고, 최종 selection formula도 seed 39만 사용한다. Selected cost는
모든 held-out condition에서 다시 실행하며, 기존 cost-0.01 결과는
sensitivity evidence로 별도 보존한다.
Selection rule은 failed rejection ≥90%, clean rejection ≤40%인 값 중
seed-39 rMAE 최소이며, eligible setting이 없으면 balanced rejection
error를 먼저 최소화한다.

Corruption은 training sensors에만 적용한다.

- Persistent failure: 10%, 20%, 30% sensor identities; component별 clean
  standard deviation의 3배 signed bias와 마지막 frame에서 2배가 되는
  signed linear drift
- AR(1): 15.1% sensor identities, \(\rho=0.98\), stationary scale 3 standard
  deviations
- Spatial burst: 15.1% sensor identities, 두 개의 20-frame burst,
  5 standard deviations

Primary field/detection table은 Reynolds number를 고정하여 method의
corruption robustness를 분리한다. 별도 inverse extension에서는 30%
persistent failure에서 positive log-parameterization으로 Re를 8000에서
시작해 metadata value 10031을 추정한다. 단, real PIV와 nominal 2D PDE의
model discrepancy 때문에 learned value는 physical metadata Re와
동일해야 하는 “clean ground truth”가 아니라 effective coefficient일 수
있다. 이 결과는 seeds 40--42가 모두 끝난 뒤 별도 표로 보고한다.

## 8. 확정 additional results

### 8.1 PIV rejection-cost calibration

30% persistent sensor bias/drift의 seed 39에서 frozen rule을 적용했다.

| Rejection cost | rMAE | AUROC | failed rejection | clean rejection | Eligible |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.005 | 0.46303 | 0.90437 | 97.87% | 91.01% | No |
| 0.010 | 0.33177 | 0.92647 | 98.52% | 84.37% | No |
| 0.050 | 0.21088 | **0.94985** | 97.98% | 61.20% | No |
| 0.100 | **0.19547** | 0.94509 | 93.79% | 18.56% | **Yes** |
| 0.500 | 0.29799 | 0.68691 | 33.28% | **0.62%** | No |

Failed rejection ≥90%와 clean rejection ≤40%를 동시에 만족한 유일한 값은
0.10이므로 이를 held-out PIV matrix 전체에 고정한다. Cost 0.05의 AUROC가
조금 더 높지만 clean data의 61.2%를 reject하므로 선택하지 않는다.

앞서 밝힌 대로 이 grid refinement 전에 10% cost-0.01 held-out 결과가
이미 완료되어 있었으므로 `fully held-out-blind`라고 부르지 않는다.
Selection script 자체는 seed 39 artifact만 읽으며, 선택된 0.10을 모든
condition과 seeds 40--42에서 다시 실행한다.

### 8.2 Real PIV + 30% persistent sensor bias/drift: 확정 primary comparison

Calibration에서 고정한 rejection cost 0.10과 모든 baseline의 seeds
40--42가 완료됐다. Mean ± sample standard deviation은 다음과 같다.

| Method | held-out rMAE | held-out rMSE | momentum RMS | continuity RMS |
| --- | ---: | ---: | ---: | ---: |
| MSE-PINN | 0.37100 ± 0.00190 | 0.49065 ± 0.00544 | 0.07605 ± 0.00096 | 0.06977 ± 0.00149 |
| LAD-PINN | 0.23558 ± 0.00678 | 0.35840 ± 0.00590 | 0.07195 ± 0.00299 | 0.07857 ± 0.00140 |
| OrPINN \(q=2.9\) | 0.29745 ± 0.00073 | 0.41328 ± 0.00164 | 0.08974 ± 0.00048 | 0.08378 ± 0.00178 |
| PINN-EBM, no gate | **0.20329 ± 0.01529** | 0.31052 ± 0.00543 | 0.07841 ± 0.02578 | 0.14667 ± 0.03555 |
| naPINN, rejection cost 0.10 | 0.21194 ± 0.01783 | **0.30034 ± 0.02341** | **0.01426 ± 0.00305** | **0.01176 ± 0.00201** |

naPINN은 MSE 대비 rMAE/rMSE를 42.9%/38.8%, LAD 대비
10.0%/16.2%, OrPINN 대비 28.7%/27.3% 낮춘다. 따라서 severe
identity-level failure에서는 fixed robust-loss 및 practical LAD
baseline보다 분명한 benefit이 있다.

Closest prior와의 결과는 mixed하다. Direct PINN-EBM은 rMAE가 naPINN보다
4.1% 낮지만, naPINN은 rMSE가 3.3% 낮고 momentum/continuity RMS가 각각
81.8%/92.0% 낮다. 따라서 `naPINN wins the real example`이라고 단순화하지
않고, field metric에 따라 direct prior와 순위가 바뀌며 naPINN은 훨씬
작은 nominal-physics residual을 제공하는 trade-off라고 답한다.

Corruption detection은 다음과 같다.

| Method / score | AUROC | failed rejection | clean rejection | retained fraction |
| --- | ---: | ---: | ---: | ---: |
| PINN-EBM raw EBM surprise | **0.95046 ± 0.00695** | -- | -- | -- |
| naPINN learned gate | 0.93445 ± 0.01306 | 93.22% ± 3.26% | 28.64% ± 13.07% | 51.85% ± 10.08% |

Calibration의 eligibility 조건은 held-out seeds에서도 평균적으로
유지됐지만 clean rejection의 seed variability는 크다. 따라서 30%
identity-level sensor failure에서도 corruption ranking이 작동했다는
evidence는 되지만, clean observation을 안정적으로 모두 보존했다는
evidence는 아니다. 또한 raw EBM score의 AUROC가 더 높으므로 gate가
anomaly ranking을 최초로 가능하게 한다고 주장하지 않는다.

Measured phase time은 warm-up 427.7 ± 86.1초, estimator-only
initialization 72.6 ± 14.0초, joint training 1922.3 ± 265.4초였고,
evaluation을 포함한 end-to-end time은 2425.6 ± 299.1초였다. Shared GPU
contention이 포함된 observational timing이며 hardware-isolated speed
benchmark로 해석하지 않는다. 같은 end-to-end 값은 MSE
1705.9 ± 256.6초, LAD 1859.1 ± 144.5초, OrPINN
1764.0 ± 168.5초, direct PINN-EBM 2396.4 ± 185.6초다.

### 8.3 Real PIV + 10% persistent sensor bias/drift: cost-0.01 sensitivity

다음은 seeds 40--42의 mean ± sample standard deviation이다. 이 표는
natural PIV result가 아니라, real PIV training sensors의 9.90%에
controlled persistent corruption을 주입한 결과다.

| Method | held-out rMAE | held-out rMSE | momentum RMS | continuity RMS |
| --- | ---: | ---: | ---: | ---: |
| MSE-PINN | 0.21182 ± 0.00261 | 0.28836 ± 0.00888 | 0.04915 ± 0.00101 | 0.04404 ± 0.00076 |
| LAD-PINN | **0.13212 ± 0.00251** | **0.20881 ± 0.00157** | 0.05127 ± 0.00064 | 0.07348 ± 0.00056 |
| PINN-EBM, no gate | 0.26545 ± 0.08838 | 0.35567 ± 0.08225 | 0.04847 ± 0.01452 | 0.08889 ± 0.01527 |
| naPINN, rejection cost 0.01 | 0.16928 ± 0.00131 | 0.26378 ± 0.00335 | 0.00832 ± 0.00082 | 0.00714 ± 0.00078 |

해석은 다음과 같다.

- naPINN은 MSE-PINN보다 rMAE를 20.1%, rMSE를 8.5% 낮춘다.
- no-gate PINN-EBM보다 rMAE를 36.2%, rMSE를 25.8% 낮춘다.
- naPINN의 nominal momentum/continuity residual은 MSE보다 각각 약
  83.1%/83.8% 작다.
- 그러나 LAD-PINN이 held-out field agreement에서 가장 좋다. 이를
  숨기거나 naPINN이 best라고 쓰면 안 된다.
- PINN-EBM의 큰 standard deviation도 그대로 보고해야 한다.

naPINN의 failure-detection 결과는 다음과 같다.

| Metric | mean ± sample std |
| --- | ---: |
| Failure-detection AUROC | 0.96673 ± 0.00149 |
| Failed-scalar rejection rate | 98.86% ± 0.38% |
| Clean-scalar rejection rate | 33.68% ± 7.62% |

AUROC와 failure rejection은 강하지만 clean rejection 33.7%는 무시할 수
없는 limitation이다. 따라서 “정확히 failed sensor만 제거한다”는 claim은
하지 않는다.

### 8.4 Real PIV + 10% persistent failure: calibration-selected cost 0.10

Earlier cost-0.01 sensitivity를 버리지 않은 채, seed-39에서 선택한 0.10을
같은 seeds 40--42에 다시 실행했다.

| Method | held-out rMAE | held-out rMSE | momentum RMS | continuity RMS |
| --- | ---: | ---: | ---: | ---: |
| MSE-PINN | 0.21182 ± 0.00261 | 0.28836 ± 0.00888 | 0.04915 ± 0.00101 | 0.04404 ± 0.00076 |
| LAD-PINN | **0.13212 ± 0.00251** | **0.20881 ± 0.00157** | 0.05127 ± 0.00064 | 0.07348 ± 0.00056 |
| OrPINN \(q=2.9\) | 0.16655 ± 0.00441 | 0.24307 ± 0.01013 | 0.06550 ± 0.00135 | 0.07164 ± 0.00122 |
| PINN-EBM, no gate | 0.26545 ± 0.08838 | 0.35567 ± 0.08225 | 0.04847 ± 0.01452 | 0.08889 ± 0.01527 |
| naPINN, rejection cost 0.10 | 0.15501 ± 0.00119 | 0.23226 ± 0.00463 | **0.01818 ± 0.00072** | **0.01932 ± 0.00154** |

naPINN은 MSE, OrPINN, direct PINN-EBM보다 두 field metric이 낮지만 LAD가
여전히 best다. Cost 0.01보다 naPINN field agreement와 clean retention은
좋아졌지만 nominal-physics residual은 커졌다.

| naPINN detection metric | mean ± sample std |
| --- | ---: |
| Gate failure-detection AUROC | 0.92989 ± 0.00134 |
| Raw EBM failure-detection AUROC | **0.94513 ± 0.00082** |
| Failed-scalar rejection rate | 84.97% ± 1.90% |
| Clean-scalar rejection rate | 3.63% ± 0.89% |
| Retained fraction | 88.32% ± 0.99% |

30% seed-39 calibration의 failed rejection ≥90% criterion은 10%
held-out condition에서 유지되지 않았다. 대신 clean rejection은 cost
0.01의 33.68%에서 3.63%로 크게 줄었다. 이는 rejection cost가 단순한
cosmetic hyperparameter가 아니라 failure recall과 clean retention의
trade-off를 제어하며, 한 severity에서 정한 constraint가 다른 severity로
완전히 transfer되지 않는다는 limitation이다. Raw EBM AUROC가 gate보다
높다는 결과도 그대로 유지한다.

같은 condition의 MAD-PINN은 rMAE/rMSE
0.15850 ± 0.00133 / 0.24077 ± 0.00059로 naPINN보다 조금 높고
LAD보다 높다. 반면 momentum/continuity RMS는 0.01443/0.01354로
naPINN보다 낮다. MAD screen은 corrupted 91.38% ± 0.49%와 clean
30.30% ± 0.43%를 reject하여 63.65% ± 0.40%를 retained한다. naPINN은
field와 clean retention이 더 좋고, MAD는 physics residual과 failure
recall이 더 좋은 trade-off다. MAD는 60,000-update,
non-compute-matched reference다.

Selected-cost naPINN end-to-end time은 5313.8 ± 2275.0초였지만, run별
shared-server contention이 크게 달라 variance가 매우 크다. 이를
cost-0.01 또는 baseline과의 controlled speed comparison으로 사용하지
않는다.

### 8.5 Real PIV + 20% persistent failure: calibration-selected cost 0.10

중간 severity의 matched seeds 40--42 결과도 완료됐다.

| Method | held-out rMAE | held-out rMSE | momentum RMS | continuity RMS |
| --- | ---: | ---: | ---: | ---: |
| MSE-PINN | 0.29806 ± 0.00414 | 0.37994 ± 0.00526 | 0.06124 ± 0.00112 | 0.05989 ± 0.00236 |
| LAD-PINN | 0.17195 ± 0.00576 | 0.25329 ± 0.00745 | 0.06163 ± 0.00248 | 0.07726 ± 0.00291 |
| OrPINN \(q=2.9\) | 0.23768 ± 0.00330 | 0.32710 ± 0.00941 | 0.07669 ± 0.00201 | 0.08331 ± 0.00128 |
| PINN-EBM, no gate | 0.20534 ± 0.02983 | 0.29317 ± 0.02936 | 0.06210 ± 0.00869 | 0.11887 ± 0.00783 |
| naPINN, rejection cost 0.10 | **0.16889 ± 0.00337** | **0.24869 ± 0.00871** | **0.01444 ± 0.00072** | **0.01408 ± 0.00036** |

naPINN은 LAD 대비 rMAE/rMSE를 각각 1.8%/1.8% 낮춘다. Improvement
margin은 30% failure보다 작지만, 이 setting에서는 direct PINN-EBM보다
field error도 낮다. LAD 대비 nominal momentum/continuity residual은
각각 76.6%/81.8% 낮다.

| Detection score | AUROC | failed rejection | clean rejection | retained |
| --- | ---: | ---: | ---: | ---: |
| PINN-EBM raw EBM surprise | **0.95991 ± 0.00790** | -- | -- | -- |
| naPINN raw EBM surprise | 0.94612 ± 0.00642 | -- | -- | -- |
| naPINN learned gate | 0.94192 ± 0.01330 | 91.25% ± 1.72% | 9.16% ± 1.03% | 74.59% ± 1.16% |

Raw EBM ranking이 gate보다 좋은 pattern은 다시 나타난다. 따라서
20% condition의 positive field result도 gate가 anomaly score 자체를
새로 가능하게 한다는 claim으로 해석하지 않는다.

같은 condition의 MAD-PINN은 rMAE/rMSE
0.17087 ± 0.00580 / 0.25418 ± 0.00785로 LAD와 거의 같고 naPINN보다
조금 높다. Momentum/continuity RMS는 0.01767/0.01529로 LAD보다 크게
낮지만 naPINN의 0.01444/0.01408보다는 높다. Fixed MAD screen은 known
corrupted scalars 91.60% ± 0.79%와 clean scalars 25.42% ± 0.82%를
reject하여 61.48% ± 0.80%를 retained한다. 이는 60,000-update,
non-compute-matched reference다.

### 8.6 Compute evidence for the same 10% cost-0.01 sensitivity

| Method | End-to-end wall time, seconds |
| --- | ---: |
| MSE-PINN | 1346.3 ± 13.2 |
| LAD-PINN | 1337.7 ± 58.8 |
| PINN-EBM | 1881.5 ± 394.0 |
| naPINN | 1857.1 ± 166.6 |

PINN-EBM과 naPINN은 30,000 PINN updates 외에 5,000 estimator-only
updates가 있다. Shared GPU contention 때문에 이 wall time을 엄밀한
speed benchmark로 해석하면 안 된다. 다만 omitted initialization을
포함한 end-to-end accounting이라는 점은 reviewer 질문에 답한다.

### 8.7 Real PIV + 30% persistent failure의 inverse-Re extension

Fixed-Re primary table과 별도로, 모든 method가 positive
log-parameterization의 Re를 8000에서 시작하여 metadata value 10031을
추정하도록 했다. Seeds 40--42 mean ± sample standard deviation은 다음과
같다.

| Method | held-out rMAE | held-out rMSE | learned Re | Re relative error | momentum RMS | continuity RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSE-PINN | 0.37046 ± 0.00155 | 0.48957 ± 0.00516 | 1309 ± 155 | 86.95% ± 1.55% | 0.07674 ± 0.00114 | 0.07026 ± 0.00127 |
| LAD-PINN | 0.23385 ± 0.00515 | 0.36157 ± 0.00655 | 4133 ± 506 | 58.80% ± 5.04% | 0.07128 ± 0.00157 | 0.07888 ± 0.00090 |
| PINN-EBM, no gate | **0.20197 ± 0.01048** | **0.30589 ± 0.00218** | **4651 ± 2561** | **53.63% ± 25.53%** | 0.06069 ± 0.01140 | 0.13219 ± 0.02388 |
| naPINN | 0.23092 ± 0.02378 | 0.31708 ± 0.01356 | 2284 ± 533 | 77.23% ± 5.32% | **0.02891 ± 0.02106** | **0.02926 ± 0.02736** |

이 extension은 naPINN의 parameter-recovery success를 보여주지 않는다.
Direct PINN-EBM이 field와 mean Re error에서 가장 좋지만 seed variability가
매우 크고, 모든 method의 learned Re가 metadata value보다 크게 낮다.
naPINN은 nominal momentum/continuity residual은 가장 낮지만 Re recovery는
LAD와 direct prior보다 나쁘다.

따라서 이 결과는 AC가 지적한 observation error와 physics-model
discrepancy의 confounding을 실증한다. Real PIV에 nominal pressure-latent
2D Navier--Stokes를 맞추면 learned coefficient는 literal metadata
parameter가 아니라 method-dependent effective coefficient가 될 수 있다.
naPINN은 두 discrepancy를 separately identify하지 않으며, low physics
residual만으로 correct physical parameter를 보장하지 않는다. 이 adverse
result는 fixed-Re real-data benefit과 함께 보고해야 한다.

### 8.8 Closest prior: 15% four-Gaussian synthetic, 확정 3-seed 결과

다음 direct PINN-EBM과 naPINN 비교는 seeds 40--42가 모두 완료됐다.
두 method는 같은 backbone, corrupted observations, 30,000 PINN updates,
5,000 estimator-only updates, PDE/data weight를 사용한다. Allen 값은
현재 rejection cost 0.5 run이며 paper-aligned 1.0 extension이 별도로
진행될 예정이다.

| PDE | Method | field rMAE | field rMSE | parameter abs. error | detection AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | PINN-EBM | **0.08334 ± 0.01430** | **0.08343 ± 0.01374** | epsilon: 0.00272 ± 0.00113 | raw EBM: **0.99419 ± 0.00188** |
| Allen--Cahn | naPINN | 0.10060 ± 0.02587 | 0.10038 ± 0.01791 | epsilon: 0.00273 ± 0.00027 | gate: 0.99286 ± 0.00159 |
| Burgers | PINN-EBM | **0.04783 ± 0.01230** | **0.06504 ± 0.01881** | nu: **0.00080 ± 0.00028** | raw EBM: **0.99320 ± 0.00210** |
| Burgers | naPINN | 0.08179 ± 0.00335 | 0.09983 ± 0.01022 | nu: 0.00271 ± 0.00024 | gate: 0.99037 ± 0.00152 |
| lambda--omega | PINN-EBM | **0.03976 ± 0.00158** | **0.05542 ± 0.00257** | beta: **0.00966 ± 0.00366** | raw EBM: **0.99420 ± 0.00012** |
| lambda--omega | naPINN | 0.06989 ± 0.00132 | 0.08851 ± 0.00102 | beta: 0.01340 ± 0.00057 | gate: 0.99214 ± 0.00052 |

이 결과는 reviewer의 closest-prior concern을 약화시키지 않고 오히려
강화한다. Direct PINN-EBM은 세 PDE 모두 field error가 더 낮고, Burgers와
lambda--omega parameter recovery도 더 좋다. Raw EBM AUROC도 naPINN
gate보다 약간 높다. 따라서 다음 주장은 사용할 수 없다.

- direct EBM-NLL은 severe outlier에서 실패한다.
- gain의 대부분은 trainable gate에서 온다.
- gate만 corruption ranking을 가능하게 한다.

Rebuttal에서 남는 방어는 explicit inclusion decision, rejection-cost
regularization, separated objectives, 그리고 injected-real-PIV에서의
behavior다. 이 결과만 보면 novelty/significance score를 뒤집을
가능성은 낮으며, response는 이를 숨기기보다 claim을 좁혀야 한다.

### 8.9 Closest-prior PDE-loss-weight sensitivity

Equal-weight held-out 결과를 이미 본 뒤 seed-39 calibration을 시작했기
때문에, seed 39만으로 favorable weight를 골라 보고하지 않았다. 대신
direct PINN-EBM의 PDE-loss weight \(1,10,50\)을 seeds 40--42에도 모두
실행하여 full grid를 공개한다.

| PDE | PDE weight | field rMAE | field rMSE | parameter abs. error |
| --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | 1 | 0.08334 ± 0.01430 | 0.08343 ± 0.01374 | 0.00272 ± 0.00113 |
| Allen--Cahn | 10 | **0.08226 ± 0.03434** | **0.08111 ± 0.02832** | **0.00117 ± 0.00048** |
| Allen--Cahn | 50 | 0.78955 ± 0.00594 | 0.77896 ± 0.00924 | 0.50221 ± 0.05523 |
| Burgers | 1 | 0.04783 ± 0.01230 | 0.06504 ± 0.01881 | 0.00080 ± 0.00028 |
| Burgers | 10 | 0.04776 ± 0.00959 | 0.06676 ± 0.01248 | **0.00033 ± 0.00027** |
| Burgers | 50 | **0.04122 ± 0.00630** | **0.05679 ± 0.01124** | 0.00087 ± 0.00068 |
| lambda--omega | 1 | 0.03976 ± 0.00158 | 0.05542 ± 0.00257 | 0.00966 ± 0.00366 |
| lambda--omega | 10 | **0.03708 ± 0.00087** | **0.05422 ± 0.00180** | 0.00334 ± 0.00092 |
| lambda--omega | 50 | 0.04551 ± 0.00164 | 0.08013 ± 0.00873 | **0.00297 ± 0.00069** |

Weight sensitivity는 PDE-dependent하다. Weight 10은 Allen--Cahn과
lambda--omega의 field mean을 조금 개선하고, weight 50은 Burgers를
개선하지만 Allen--Cahn을 catastrophic하게 악화시킨다. 따라서 tuned
direct prior가 equal-weight result보다 강해질 수 있다는 fairness concern은
유효하며, 하나의 universal optimal weight는 없다. 중요한 점은 weight
1/10/50 중 favorable cell만 선택하지 않았다는 것이다. naPINN과의 비교도
equal-weight 하나로 제한하지 않는다.

### 8.10 Correlated real-PIV corruption

같은 real PIV split에서 sensor-level correlation을 두 방식으로 시험했다.

- `AR(1)`: 15.1% fixed sensor identities에 temporal correlation
  \(\rho=0.98\), scale \(3\sigma\)
- `spatial burst`: 15.1% spatial sensors에 \(5\sigma\) corruption을
  두 개의 20-frame windows 동안 주입

두 조건 모두 seeds 40--42의 5-method matrix가 완성됐다.

| Corruption | Method | held-out rMAE | held-out rMSE | momentum RMS | continuity RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| AR(1) | MSE | 0.22047 ± 0.00068 | 0.28117 ± 0.00238 | 0.05440 ± 0.00026 | 0.05116 ± 0.00226 |
| AR(1) | LAD | **0.12330 ± 0.00054** | **0.19799 ± 0.00123** | 0.04886 ± 0.00066 | 0.07191 ± 0.00120 |
| AR(1) | OrPINN q=2.9 | 0.14541 ± 0.00340 | 0.21048 ± 0.00232 | 0.05696 ± 0.00258 | 0.06515 ± 0.00161 |
| AR(1) | PINN-EBM | 0.20891 ± 0.01607 | 0.31499 ± 0.01447 | 0.04120 ± 0.00919 | 0.09298 ± 0.01323 |
| AR(1) | naPINN | 0.14997 ± 0.00039 | 0.21480 ± 0.00196 | **0.01921 ± 0.00029** | **0.02007 ± 0.00027** |
| Spatial burst | MSE | 0.15981 ± 0.00052 | 0.21766 ± 0.00025 | 0.03967 ± 0.00090 | 0.03829 ± 0.00095 |
| Spatial burst | LAD | **0.11358 ± 0.00083** | **0.18549 ± 0.00077** | 0.04636 ± 0.00056 | 0.07346 ± 0.00083 |
| Spatial burst | OrPINN q=2.9 | 0.12634 ± 0.00219 | 0.18973 ± 0.00160 | 0.04737 ± 0.00072 | 0.06324 ± 0.00069 |
| Spatial burst | PINN-EBM | 0.23979 ± 0.00654 | 0.33950 ± 0.01824 | 0.04372 ± 0.01817 | 0.07795 ± 0.01918 |
| Spatial burst | naPINN | 0.15167 ± 0.00316 | 0.22627 ± 0.00122 | **0.01674 ± 0.00051** | **0.01762 ± 0.00067** |

| Corruption | Score | AUROC | failed rejection | clean rejection | retained |
| --- | --- | ---: | ---: | ---: | ---: |
| AR(1) | PINN-EBM raw EBM | 0.87252 ± 0.01659 | -- | -- | -- |
| AR(1) | naPINN raw EBM | **0.91443 ± 0.00082** | -- | -- | -- |
| AR(1) | naPINN gate | 0.91391 ± 0.00081 | 64.97% ± 7.89% | 3.86% ± 1.87% | 86.91% ± 2.78% |
| Spatial burst | PINN-EBM raw EBM | **0.99271 ± 0.00093** | -- | -- | -- |
| Spatial burst | naPINN raw EBM | 0.98237 ± 0.00529 | -- | -- | -- |
| Spatial burst | naPINN gate | 0.98303 ± 0.00482 | 100.00% | 5.53% ± 0.25% | 91.62% ± 0.25% |

LAD가 두 structured conditions 모두 field metric에서 가장 좋고,
naPINN은 우승하지 못한다. AR(1)에서는 OrPINN도 naPINN보다 field
agreement가 조금 좋다. Spatial burst에서 naPINN은 MSE보다 rMAE는
낮지만 rMSE는 높다. 반면 naPINN의 nominal momentum/continuity residual은
두 조건 모두 가장 낮다.

Detection도 condition-dependent하다. Spatial burst는 gate가 injected
failed scalars를 모두 reject하지만, AR(1)에서는 64.97%만 reject한다.
30% persistent-failure calibration에서 고른 rejection cost가 temporal
correlation으로 완전히 transfer되지 않는다는 뜻이다. 따라서 이 matrix는
“correlated noise에서도 consistently best”라는 근거가 아니라,
structured corruption coverage와 field/physics/detection trade-off를
보여주는 evidence다.

AR(1)의 MAD-PINN result도 seeds 40--42가 완료됐다. rMAE/rMSE는
0.15266 ± 0.00119 / 0.23266 ± 0.00322로 naPINN보다 높고 LAD보다 높다.
반면 momentum/continuity RMS는 0.01524/0.01333으로 naPINN보다 낮다.
MAD screen은 corrupted 94.34% ± 0.05%와 clean 27.60% ± 0.26%를
reject하여 62.32% ± 0.23%를 retained한다. naPINN gate의 64.97%
failure rejection과 3.86% clean rejection에 비해, MAD는 훨씬 높은
failure recall을 얻는 대신 clean data를 더 버리고 field agreement도
악화시킨다. 이 결과 역시 60,000-update, non-compute-matched
trade-off다.

Spatial burst MAD-PINN도 seeds 40--42가 완료됐다. rMAE/rMSE는
0.14967 ± 0.00075 / 0.22770 ± 0.00030이다. naPINN과 비교하면 MAD가
rMAE는 약 1.3% 낮고 naPINN이 rMSE는 약 0.6% 낮으며, LAD가 두 field
metrics 모두 가장 좋다. MAD momentum/continuity RMS는
0.01448/0.01364로 naPINN의 0.01674/0.01762보다 낮다. 두 selector 모두
corrupted scalars를 100% reject하지만 MAD는 clean scalars 33.60% ±
0.73%를 reject하고 naPINN은 5.53% ± 0.25%만 reject한다. 따라서
spatial burst에서도 MAD는 physics fit과 rMAE를 조금 개선하는 대신
clean retention을 크게 희생한다.

## 9. 확정 severe synthetic core와 PDE parameter recovery

15% four-Gaussian corruption의 seeds 40--42 mean ± sample standard
deviation이다. 모든 method는 동일한 field/data realization과 30,000 PINN
updates를 사용한다. PINN-EBM과 naPINN에는 추가로 5,000 estimator-only
updates가 있다.

| PDE | Method | field rMAE | field rMSE | learned parameter | parameter abs. error |
| --- | --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | MSE | 0.83957 ± 0.11123 | 0.70033 ± 0.08835 | epsilon=0.29633 ± 0.00178 | 0.00367 ± 0.00178 |
| Allen--Cahn | LAD | 0.35920 ± 0.04052 | 0.30535 ± 0.02906 | epsilon=0.29734 ± 0.00059 | 0.00266 ± 0.00059 |
| Allen--Cahn | OrPINN q=1.9 | 0.66361 ± 0.06553 | 0.55480 ± 0.04962 | epsilon=0.30653 ± 0.00277 | 0.00653 ± 0.00277 |
| Allen--Cahn | OrPINN q=2.9 | 0.26319 ± 0.06330 | 0.23233 ± 0.04521 | epsilon=0.29687 ± 0.00036 | 0.00313 ± 0.00036 |
| Allen--Cahn | PINN-EBM | **0.08334 ± 0.01430** | **0.08343 ± 0.01374** | epsilon=0.29728 ± 0.00113 | 0.00272 ± 0.00113 |
| Allen--Cahn | naPINN | 0.10060 ± 0.02587 | 0.10038 ± 0.01791 | epsilon=0.29727 ± 0.00027 | 0.00273 ± 0.00027 |
| Burgers | MSE | 0.58621 ± 0.03931 | 0.56572 ± 0.03663 | nu=0.01558 ± 0.00090 | 0.00558 ± 0.00090 |
| Burgers | LAD | 0.25976 ± 0.00367 | 0.25014 ± 0.00249 | nu=0.01178 ± 0.00059 | 0.00178 ± 0.00059 |
| Burgers | OrPINN q=1.9 | 0.39205 ± 0.01346 | 0.38496 ± 0.01559 | nu=0.01516 ± 0.00081 | 0.00516 ± 0.00081 |
| Burgers | OrPINN q=2.9 | 0.19407 ± 0.02426 | 0.19455 ± 0.01921 | nu=0.01123 ± 0.00074 | 0.00123 ± 0.00074 |
| Burgers | PINN-EBM | **0.04783 ± 0.01230** | **0.06504 ± 0.01881** | nu=0.00920 ± 0.00028 | **0.00080 ± 0.00028** |
| Burgers | naPINN | 0.08179 ± 0.00335 | 0.09983 ± 0.01022 | nu=0.01271 ± 0.00024 | 0.00271 ± 0.00024 |
| lambda--omega | MSE | 0.42138 ± 0.01071 | 0.44034 ± 0.01019 | beta=0.67222 ± 0.00543 | 0.32778 ± 0.00543 |
| lambda--omega | LAD | 0.19194 ± 0.00642 | 0.19914 ± 0.00568 | beta=0.91559 ± 0.00421 | 0.08441 ± 0.00421 |
| lambda--omega | OrPINN q=1.9 | 0.17540 ± 0.01494 | 0.19868 ± 0.01611 | beta=0.93914 ± 0.00453 | 0.06086 ± 0.00453 |
| lambda--omega | OrPINN q=2.9 | 0.15123 ± 0.00353 | 0.16390 ± 0.00343 | beta=0.94769 ± 0.00382 | 0.05231 ± 0.00382 |
| lambda--omega | PINN-EBM | **0.03976 ± 0.00158** | **0.05542 ± 0.00257** | beta=0.99034 ± 0.00366 | **0.00966 ± 0.00366** |
| lambda--omega | naPINN | 0.06989 ± 0.00132 | 0.08851 ± 0.00102 | beta=0.98660 ± 0.00057 | 0.01340 ± 0.00057 |

이 결과는 field reconstruction뿐 아니라 reviewer가 요청한 Burgers
viscosity와 lambda--omega beta recovery까지 포함한다. Direct PINN-EBM이
세 PDE 모두 field error가 가장 낮고, Burgers와 lambda--omega parameter
error도 가장 낮다. Allen--Cahn parameter error는 PINN-EBM과 naPINN이
사실상 같다. 따라서 gate의 value를 synthetic accuracy superiority로
설명할 수 없다.

동시에 naPINN은 세 PDE 모두 second-best field method이며 OrPINN \(q=2.9\)
대비 rMAE를 각각 61.8%, 57.9%, 53.8% 낮춘다. 따라서 fixed robust-loss
baselines 대비 효과는 유지되지만, 이 positive result가 closest-prior
comparison을 대신할 수는 없다.

Raw EBM likelihood alone도 corruption ranking을 잘한다. Direct PINN-EBM의
raw-score AUROC가 세 PDE 모두 naPINN gate보다 약간 높다. Gate의 더 좁은
역할은 density score를 explicit inclusion decision으로 바꾸고 rejection
regularization을 함께 학습하는 것이며, anomaly ranking 자체를 최초로
가능하게 한다는 것이 아니다.

### 9.1 Synthetic phase time

같은 severe runs에서 측정한 seconds의 mean ± sample standard
deviation이다. Shared GPU의 concurrent workload가 달랐으므로
hardware-isolated speed benchmark가 아니라 phase accounting evidence로만
사용한다.

| PDE | Method | warm-up | estimator-only initialization | joint | total training |
| --- | --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | PINN-EBM | 66.5 ± 5.6 | 22.9 ± 6.3 | 448.1 ± 58.9 | 537.5 ± 69.6 |
| Allen--Cahn | naPINN | 60.3 ± 2.7 | 20.2 ± 2.5 | 487.6 ± 59.6 | 568.2 ± 63.3 |
| Burgers | PINN-EBM | 88.0 ± 2.1 | 20.1 ± 1.7 | 550.4 ± 14.4 | 658.5 ± 15.2 |
| Burgers | naPINN | 95.8 ± 13.9 | 24.2 ± 5.9 | 707.8 ± 134.8 | 827.8 ± 154.6 |
| lambda--omega | PINN-EBM | 104.9 ± 33.0 | 23.7 ± 6.6 | 664.9 ± 187.8 | 793.5 ± 227.1 |
| lambda--omega | naPINN | 92.4 ± 3.8 | 21.1 ± 1.0 | 684.7 ± 39.6 | 798.2 ± 42.1 |

Estimator-only 5,000 updates는 이 runs에서 약 20--24초였고 total training의
약 2.6--4.3%다. 따라서 cost가 작다는 사용자의 intuition은 방향상
맞지만, “몇 초”라고 낮춰 말하지 않고 측정값을 제시한다. Joint phase
안에도 estimator update가 들어 있으므로 estimator-only time만으로
naPINN의 전체 overhead를 대표하지 않는다.

### 9.2 EMA coefficient sensitivity

Reviewer가 직접 요청한 Allen--Cahn 15% four-Gaussian condition에서 code
convention의 EMA update weight \(m\)을 바꿨다. Old-state decay로 쓰면
\(\rho=1-m\)이다.

| Update weight \(m\) | Decay \(\rho\) | field rMAE | field rMSE | epsilon abs. error |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.99 | 0.11397 ± 0.04817 | 0.11187 ± 0.03993 | 0.00271 |
| 0.05 | 0.95 | **0.10060 ± 0.02587** | **0.10038 ± 0.01791** | 0.00273 |
| 0.10 | 0.90 | 0.10898 ± 0.04136 | 0.10613 ± 0.03030 | 0.00282 |

세 값의 variance가 겹치므로 \(m=0.05\)가 universally optimal이라고
주장하지 않는다. 다만 tested range에서 mean field error 변화가 작고
default 0.05가 가장 낮았다. Main text의 beta와 appendix의 rho를 같은
숫자로 취급하지 않고, update-weight와 decay convention을 명시하는 것이
핵심 response다.

### 9.3 Rejection-cost sensitivity와 paper/config discrepancy

같은 Allen--Cahn 15% condition에서 \(\lambda_{\mathrm{rej}}\)를
0.10--1.00으로 바꾼 확정 결과다.

| Rejection cost | field rMAE | field rMSE | outlier rejection | clean rejection | gate AUROC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.09814 ± 0.01185 | 0.09987 ± 0.01075 | 99.27% ± 0.25% | 0.87% ± 0.21% | **0.99346 ± 0.00104** |
| 0.30 | 0.11383 ± 0.02902 | 0.11191 ± 0.02532 | 99.18% ± 0.27% | 0.88% ± 0.25% | 0.99097 ± 0.00314 |
| 0.50 | 0.10060 ± 0.02587 | 0.10038 ± 0.01791 | 99.19% ± 0.24% | 0.68% ± 0.13% | 0.99286 ± 0.00159 |
| 0.70 | **0.09564 ± 0.00269** | **0.09783 ± 0.00307** | 99.27% ± 0.21% | 0.61% ± 0.13% | 0.99324 ± 0.00169 |
| 1.00 | 0.87859 ± 0.05709 | 0.73081 ± 0.04713 | 2.21% ± 0.34% | **0.13% ± 0.07%** | 0.54961 ± 0.00952 |

0.10--0.70의 field mean은 서로 가깝고 0.70이 가장 낮지만, 1.00에서는
gate가 거의 모든 observation을 accept하여 outlier rejection과 field
reconstruction이 함께 붕괴한다. 이는 submitted appendix에 적힌
Allen--Cahn cost 1.0과 실제 initial core cost 0.5 사이의 discrepancy가
결과에 실질적인 영향을 준다는 뜻이다. 따라서 1.0 결과를 숨기거나
0.5로 조용히 대체하지 않는다. Reviewer에게는 exact configuration을
clarify하고, submitted value 1.0의 degradation을 인정해야 한다.

이 sweep 역시 post-hoc optimal-cost selection 근거로 사용하지 않는다.
Core result는 frozen 0.5를 유지하고, 표 전체를 sensitivity로 보고한다.
Paper-aligned 1.0의 5%/10% cells까지 seeds 40--42를 모두 실행한 결과는
다음과 같다.

| Corruption ratio | cost 0.5 field rMAE | cost 1.0 field rMAE | cost 1.0 outlier rejection | cost 1.0 gate AUROC |
| ---: | ---: | ---: | ---: | ---: |
| 5% | **0.09621 ± 0.01677** | 0.25878 ± 0.06094 | 3.27% | 0.57107 |
| 10% | **0.10159 ± 0.00811** | 0.56180 ± 0.08248 | 2.18% | 0.55669 |
| 15% | **0.10060 ± 0.02587** | 0.87859 ± 0.05709 | 2.21% | 0.54961 |

Cost 1.0의 degradation은 severe 15% 하나에 국한되지 않고 corruption
ratio가 높아질수록 커진다. Gate가 거의 모든 observation을 accept하며
AUROC도 random ranking에 가까워진다. 따라서 rebuttal에서 submitted
appendix value 1.0을 default가 잘 동작한 것처럼 방어하면 안 된다.
정직한 대응은 initial rebuttal core의 0.5와 paper-aligned 1.0을 모두
보여주고, exact configuration discrepancy와 sensitivity를 인정하는
것이다.

### 9.4 MAD-PINN preprocessing baseline

Published two-stage rule인 LAD 30,000 updates, scalar
\(\operatorname{median}(|r|)/1.6777\)와 \(k=3\) screen, retained-data MSE
30,000 updates를 사용했다.

| PDE | field rMAE | field rMSE | parameter abs. error | outlier rejection | clean rejection | retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Allen--Cahn | 0.32482 ± 0.03642 | 0.27377 ± 0.02822 | 0.01040 ± 0.01072 | 98.52% ± 0.26% | 17.85% ± 2.85% | 70.05% ± 2.46% |
| Burgers | 0.24696 ± 0.01307 | 0.24091 ± 0.01395 | 0.00167 ± 0.00076 | 98.59% ± 0.16% | 17.08% ± 0.46% | 70.69% ± 0.39% |
| lambda--omega | 0.13008 ± 0.00765 | 0.14535 ± 0.00497 | 0.03574 ± 0.00486 | 98.42% ± 0.05% | 10.76% ± 0.88% | 76.07% ± 0.74% |

MAD-PINN은 세 PDE 모두 LAD field error를 개선하지만 direct PINN-EBM과
naPINN보다 높다. Screening 자체는 severe outlier를 거의 모두 제거하나
clean scalar도 10.8--17.8% 제거한다. Total training time은 shared GPU
관측에서 PDE별 약 1025/1349/1498초였으며, 이는 stage-one과 stage-two를
합친 값이다. 이 baseline은 reviewer의 practical preprocessing 질문에
답하지만 60,000 PINN updates이므로 30,000-update method와 compute
matched라고 부르지 않는다.

### 9.5 Real PIV의 MAD-PINN preprocessing

30% persistent sensor failure에서도 같은 published screen을 적용했다.
Seeds 40--42 aggregate는 다음과 같다.

| Method | held-out rMAE | held-out rMSE | momentum RMS | continuity RMS |
| --- | ---: | ---: | ---: | ---: |
| LAD-PINN, stage-one reference | **0.23558 ± 0.00678** | **0.35840 ± 0.00590** | 0.07195 ± 0.00299 | 0.07857 ± 0.00140 |
| MAD-PINN, 60k updates | 0.25127 ± 0.00817 | 0.37205 ± 0.00738 | **0.02676 ± 0.00159** | **0.02146 ± 0.00054** |

MAD screen은 known failed scalars의 80.45% ± 1.66%와 clean scalars의
20.30% ± 0.54%를 제거하고, 전체의 61.53% ± 0.14%를 retained했다.
Field agreement는 stage-one LAD보다 오히려 나빠졌고 naPINN
(rMAE 0.21194, rMSE 0.30034)보다도 나쁘다. Nominal-physics residual은
LAD보다 작지만 naPINN보다 크다. 따라서 basic preprocessing만으로
naPINN result를 설명할 수 없다는 evidence는 되지만, MAD-PINN이
compute-matched라는 주장은 하지 않는다.

Shared-GPU에서 two-stage pipeline wall time은
6600.5 ± 1500.4초였고 정확히 60,000 PINN updates다. 이 큰 wall time은
동시 workload의 영향을 포함하므로 standalone runtime estimate로
일반화하지 않는다.

### 9.6 전체 four-Gaussian corruption-ratio core

5%, 10%, 15%의 54개 method/PDE group, 총 162개 held-out runs가 모두
완료됐다. 아래는 field rMAE의 mean ± sample standard deviation이다.
각 group은 동일한 seeds 40--42를 사용한다.

| PDE | Outlier ratio | MSE | LAD | OrPINN q=1.9 | OrPINN q=2.9 | PINN-EBM | naPINN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Allen--Cahn | 5% | 0.3372 ± 0.1219 | 0.2710 ± 0.0581 | 0.2218 ± 0.0525 | 0.1616 ± 0.0840 | **0.0724 ± 0.0158** | 0.0962 ± 0.0276 |
| Allen--Cahn | 10% | 0.5995 ± 0.1430 | 0.3071 ± 0.0262 | 0.4279 ± 0.1081 | 0.2389 ± 0.0662 | **0.0800 ± 0.0059** | 0.1016 ± 0.0218 |
| Allen--Cahn | 15% | 0.8396 ± 0.1112 | 0.3592 ± 0.0405 | 0.6636 ± 0.0655 | 0.2632 ± 0.0633 | **0.0833 ± 0.0143** | 0.1006 ± 0.0259 |
| Burgers | 5% | 0.2164 ± 0.0150 | 0.2005 ± 0.0147 | 0.1630 ± 0.0085 | 0.1347 ± 0.0135 | **0.0418 ± 0.0015** | 0.0802 ± 0.0076 |
| Burgers | 10% | 0.4033 ± 0.0036 | 0.2273 ± 0.0066 | 0.2646 ± 0.0071 | 0.1625 ± 0.0205 | **0.0602 ± 0.0322** | 0.0783 ± 0.0062 |
| Burgers | 15% | 0.5862 ± 0.0393 | 0.2598 ± 0.0037 | 0.3921 ± 0.0135 | 0.1941 ± 0.0243 | **0.0478 ± 0.0123** | 0.0818 ± 0.0033 |
| lambda--omega | 5% | 0.1528 ± 0.0099 | 0.1613 ± 0.0111 | 0.0899 ± 0.0025 | 0.1226 ± 0.0061 | **0.0453 ± 0.0061** | 0.0738 ± 0.0072 |
| lambda--omega | 10% | 0.2885 ± 0.0089 | 0.1777 ± 0.0062 | 0.1298 ± 0.0105 | 0.1394 ± 0.0049 | **0.0443 ± 0.0064** | 0.0755 ± 0.0154 |
| lambda--omega | 15% | 0.4214 ± 0.0107 | 0.1919 ± 0.0064 | 0.1754 ± 0.0149 | 0.1512 ± 0.0035 | **0.0398 ± 0.0016** | 0.0699 ± 0.0013 |

Direct PINN-EBM이 9개 field condition 모두 가장 낮고 naPINN이 모두
second-best다. 따라서 outlier ratio를 넓혀도 closest-prior에 대한
accuracy-superiority claim은 성립하지 않는다. 반면 fixed-loss 및
OrPINN baseline에 대한 naPINN의 benefit은 모든 condition에서 유지된다.

같은 runs의 PDE-parameter absolute error mean은 다음과 같다. Standard
deviation과 learned value까지 포함한 원자료는
`rebuttal_synthetic_aggregation.json`에 보존했다.

| PDE | Outlier ratio | MSE | LAD | OrPINN q=1.9 | OrPINN q=2.9 | PINN-EBM | naPINN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Allen--Cahn epsilon | 5% | 0.00408 | 0.00234 | **0.00091** | 0.00240 | 0.00183 | 0.00246 |
| Allen--Cahn epsilon | 10% | 0.00613 | 0.00307 | 0.00298 | 0.00330 | **0.00194** | 0.00325 |
| Allen--Cahn epsilon | 15% | 0.00367 | **0.00266** | 0.00653 | 0.00313 | 0.00272 | 0.00273 |
| Burgers viscosity | 5% | 0.00205 | 0.00107 | 0.00287 | **0.00081** | 0.00093 | 0.00200 |
| Burgers viscosity | 10% | 0.00267 | 0.00134 | 0.00342 | 0.00083 | **0.00082** | 0.00219 |
| Burgers viscosity | 15% | 0.00558 | 0.00178 | 0.00516 | 0.00123 | **0.00080** | 0.00271 |
| lambda--omega beta | 5% | 0.05703 | 0.05453 | 0.01977 | 0.03790 | 0.01191 | **0.01171** |
| lambda--omega beta | 10% | 0.15239 | 0.06707 | 0.03524 | 0.04424 | 0.01386 | **0.01128** |
| lambda--omega beta | 15% | 0.32778 | 0.08441 | 0.06086 | 0.05231 | **0.00966** | 0.01340 |

Parameter recovery의 winner는 PDE와 corruption ratio에 따라 달라진다.
따라서 이 table은 “모든 benchmark의 unknown coefficient를 평가했다”는
Reviewer aoJS의 completeness 요청에는 답하지만, naPINN의 universal
parameter-recovery superiority를 뒷받침하지는 않는다.

### 9.7 Additional noise families와 naPINN backbone loss

Lambda--omega의 15% extra-outlier condition에서 background noise를
Gaussian, Laplace, Student-t, four-Gaussian으로 바꾸고, naPINN의 base
reconstruction loss도 MSE/LAD/OrPINN-\(q=2.9\)로 바꿨다. 아래는 seeds
40--42 field rMAE mean이다.

| Noise family | MSE | LAD | OrPINN q=2.9 | PINN-EBM | naPINN-MSE | naPINN-LAD | naPINN-q2.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 0.21689 | 0.08798 | 0.07908 | 0.06834 | **0.05422** | 0.06011 | 0.05954 |
| Laplace | 0.31439 | 0.07414 | 0.07108 | 0.06930 | 0.05800 | **0.05506** | 0.05900 |
| Student-t | 0.09335 | 0.04694 | 0.06184 | **0.04648** | 0.09641 | 0.04686 | 0.06550 |
| Four-Gaussian | 0.42138 | 0.19194 | 0.15123 | **0.03976** | 0.06989 | 0.14179 | 0.13770 |

한 method가 모든 family에서 best가 아니다.

- Gaussian에서는 base naPINN-MSE가 best다.
- Laplace에서는 naPINN-LAD가 best다.
- Student-t에서는 direct PINN-EBM이 best이고 LAD 및 naPINN-LAD가 거의
  같으며, base naPINN-MSE는 MSE보다도 나쁘다.
- Four-Gaussian에서는 direct PINN-EBM이 best이고 base naPINN이
  second-best다. naPINN-LAD/q2.9는 base naPINN보다 크게 악화된다.

따라서 “complicated synthetic noise를 여러 개 시험했다”는 evaluation
coverage claim은 가능하지만, 이것이 모든 real-world corruption을
represent한다거나 naPINN이 noise family에 invariant하다는 claim은
불가능하다. 또한 gate gain이 reconstruction backbone과 독립적이지
않다는 limitation도 명시해야 한다. Full rMSE, parameter error, sample
standard deviation은 synthetic aggregation artifact에 보존했다.

### 9.8 Conservative 35k update-count comparison

PINN-EBM과 naPINN은 30,000 PINN updates 외에 5,000 estimator-only
updates를 사용한다. Reviewer의 compute-fairness 질문에 보수적으로
답하기 위해, estimator가 없는 MSE/LAD/OrPINN-\(q=2.9\)에는 35,000
`full PINN updates`를 주었다. 이는 estimator-only step보다 계산량이 큰
update를 5,000회 추가한 `conservative update-count match`이며,
wall-clock match는 아니다.

| PDE | 35k MSE | 35k LAD | 35k OrPINN q=2.9 | 30k PINN-EBM | 30k naPINN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Allen--Cahn | 0.88440 ± 0.05581 | 0.34645 ± 0.03387 | 0.30771 ± 0.07454 | **0.08334 ± 0.01430** | 0.10060 ± 0.02587 |
| Burgers | 0.59215 ± 0.02302 | 0.24871 ± 0.02657 | 0.19923 ± 0.01764 | **0.04783 ± 0.01230** | 0.08179 ± 0.00335 |
| lambda--omega | 0.44750 ± 0.02580 | 0.18358 ± 0.01124 | 0.14191 ± 0.02118 | **0.03976 ± 0.00158** | 0.06989 ± 0.00132 |

표의 값은 field rMAE, seeds 40--42의 mean ± sample standard
deviation이다. 35k ordinary baseline은 세 PDE 모두 direct PINN-EBM과
naPINN보다 높다. 그러나 ordinary baseline 자체의 30k→35k 변화는
일관되지 않다. 예를 들어 Allen--Cahn LAD는 0.35920에서 0.34645로
조금 낮아지지만 MSE와 OrPINN은 오히려 높아진다. 따라서 이 결과는
“추가 step이 항상 성능을 높인다”가 아니라, extra estimator-only update가
ordinary baseline에 불리한 update-count accounting 때문에 생긴
ranking은 아니라는 제한된 evidence로 사용한다.

35k total training time은 Allen--Cahn에서 MSE/LAD/OrPINN이 각각
436.5/483.2/424.3초, Burgers에서 689.5/617.9/615.6초,
lambda--omega에서 640.8/835.7/665.9초였다. Concurrent shared-server
load가 달랐으므로 이 값으로 speed superiority를 주장하지 않는다.

### 9.9 Estimator-free selector ablations

Reviewer의 “gate가 아니라 residual screening 자체가 gain의 원인인가?”
질문에 두 estimator-free selector를 추가했다.

- `fixed-quantile`: batch 안에서 residual 상위 5%에 weight 0을 주는
  non-trainable rule
- `learnable-threshold`: EBM 없이 residual threshold와 steepness를
  학습하는 gate. Fixed threshold라고 부르면 안 된다.

| PDE | Method | field rMAE | field rMSE | parameter abs. error | outlier rejection | clean rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Allen--Cahn | fixed-quantile | 0.54862 ± 0.15512 | 0.46304 ± 0.11639 | 0.00418 ± 0.00091 | 33.33% | 0.00% |
| Allen--Cahn | learnable-threshold | 0.22245 ± 0.02472 | 0.20101 ± 0.02024 | 0.00394 ± 0.00095 | 72.46% | 0.00% |
| Burgers | fixed-quantile | 0.37395 ± 0.03197 | 0.36432 ± 0.02790 | 0.00320 ± 0.00101 | 33.33% | 0.00% |
| Burgers | learnable-threshold | 0.08493 ± 0.01394 | 0.09745 ± 0.01380 | 0.00171 ± 0.00038 | 89.76% | 0.00% |
| lambda--omega | fixed-quantile | 0.23718 ± 0.00908 | 0.25674 ± 0.01029 | 0.11604 ± 0.00598 | 33.28% | 0.00% |
| lambda--omega | learnable-threshold | 0.06644 ± 0.00949 | 0.08171 ± 0.00859 | 0.01296 ± 0.00266 | 95.90% | 0.03% |

같은 severe setting의 naPINN rMAE는
0.10060/0.08179/0.06989이고 direct PINN-EBM은
0.08334/0.04783/0.03976이다. Fixed-quantile은 15% injected outlier 중
약 1/3만 제거하기 때문에 세 PDE 모두 약하다. Learnable-threshold는
훨씬 강하고 Burgers에서 naPINN과 비슷하며 lambda--omega에서는 naPINN보다
약 4.9% 낮은 rMAE를 보이지만, Allen--Cahn에서는 크게 나쁘고 direct
PINN-EBM보다 세 PDE 모두 높다.

따라서 fixed residual screening만으로 충분하다는 결론도, EBM gate가
항상 estimator-free threshold보다 우월하다는 결론도 둘 다 evidence와
맞지 않는다. 나오는 결론은 selector choice가 PDE-dependent하며,
naPINN의 gate contribution을 universal accuracy gain으로 분리해 주장하기
어렵다는 것이다.

## 10. 답변에서 피해야 할 표현

다음 표현은 evidence보다 강하므로 사용하지 않는다.

- “There are no public noisy PDE datasets.”
- “Real PIV is noise-free ground truth.”
- “The Pilar--Wahlström objective does not work.”
- “Staged EBM training is unique to naPINN.”
- “naPINN is consistently best on real PIV.”
- “The observation model separates sensor noise from PDE model discrepancy.”
- “Our B-PINN baseline is faithful HMC.”
- “All methods have exactly the same total optimization cost.”
- “Estimator initialization costs only a few seconds” without measured value.
- “The synthetic corruptions cover all real-world noise.”

## 11. 권장 response 순서

1. AC의 real-example 요청에 감사하고 injected-real-PIV protocol과 가장
   severe verified result를 먼저 제시한다.
2. naPINN이 MSE/LAD/OrPINN보다 좋은 severe result를 수치로 제시한다.
3. naPINN의 rMSE와 physics residual, failure AUROC, clean rejection을
   정확한 수치로 제시하고, 10% sensitivity에서는 LAD가 best임을 밝힌다.
4. AC/aoJS가 직접 제기한 Pilar--Wahlström paragraph 자리는 남겨 두되,
   shared component와 objective/gradient-path 차이를 설명하는 방향만
   준비한다.
5. Direct PINN-EBM 수치, ranking, novelty 결론은 추가 실험과 저자
   release 전까지 삽입하지 않는다.
6. Burgers/lambda--omega parameter recovery table을 보인다.
7. EMA, rejection cost, correlated noise, MAD/fixed preprocessing 결과를
   짧게 요약한다.
8. Exact update budget과 end-to-end time을 제시한다.
9. B-PINN-VI/HMC, clean-reference 부재, discrepancy non-identifiability,
   shared-GPU timing을 limitation으로 끝낸다.

## 12. Evidence 및 출처

- Frozen experiment matrix: `rebuttal/experiment_plan.yaml`
- Reviewer-to-evidence tracker: `rebuttal/response_matrix.md`
- Synthetic aggregation:
  `analysis/results/runs/rebuttal_synthetic_aggregation.json`
- Conservative 35k aggregation:
  `analysis/results/runs/rebuttal_synthetic_compute35_aggregation.json`
- Injected-PIV aggregation:
  `analysis/results/runs/rebuttal_realpde/injected_piv_aggregation.json`
- PIV calibration selection:
  `analysis/results/runs/rebuttal_realpde/piv_rejection_calibration_selection.json`
- Pilar--Wahlström official paper:
  `https://proceedings.mlr.press/v242/pilar24a.html`
- RealPDEBench official site:
  `https://realpdebench.github.io/`
- Original B-PINN paper: `https://arxiv.org/abs/2003.06097`
- OrPINN paper:
  `https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.L023302`
- MAD-PINN paper: `https://arxiv.org/abs/2210.10646`

Frozen additional-experiment matrix는 모두 완료됐으며, 이 문서는 final
aggregate와 positive, mixed, adverse result를 모두 반영한다.
