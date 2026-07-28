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
보존하되 reviewer response에는 아직 사용하지 않는다. 계획된 Real-PIV
legacy-corruption, scale, robust-loss, closest-prior campaign은 완료되어
strict validation까지 통과했다. 이제 저자가 direct comparison의 사용
여부와 response 방향을 명시적으로 결정해야 한다. 아래 표의
direct-comparison 수치는 내부 기록이며, author release 전에는
reviewer-facing report로 복사하면 안 된다.
이 문서 뒤쪽에 남아 있는 “direct result를 response에 report한다”는 기존
표현이 있다면 이 author decision이 우선하며, 저자 release 전에는 실행
지침으로 사용하지 않는다.

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

### 7.1 세 합성 PDE의 기본 비교 행렬

여기서 “기본 비교 행렬”은 Allen--Cahn, Burgers, lambda--omega 각각에
5%, 10%, 15%의 큰 이상치를 넣고 아래 여섯 방법을 같은 예산으로 비교한
실험 묶음이다. 내부 기록에서 사용했던 `synthetic core`라는 짧은 이름은
신경망의 core 구성요소를 뜻하지 않는다.

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
| MSE-PINN | 0.37077 ± 0.00552 | 0.49075 ± 0.00855 | 0.07650 ± 0.00028 | 0.07055 ± 0.00030 |
| LAD-PINN | 0.23411 ± 0.00458 | 0.35852 ± 0.00694 | 0.07400 ± 0.00094 | 0.07963 ± 0.00138 |
| OrPINN \(q=2.9\) | 0.29880 ± 0.00431 | 0.41375 ± 0.00656 | 0.08878 ± 0.00096 | 0.08250 ± 0.00250 |
| PINN-EBM, no gate | 0.29558 ± 0.08992 | 0.39053 ± 0.09202 | 0.10077 ± 0.03362 | 0.15588 ± 0.04155 |
| naPINN, rejection cost 0.10 | **0.21797 ± 0.01092** | **0.30886 ± 0.00296** | **0.01567 ± 0.00072** | **0.01379 ± 0.00089** |

새 strict aggregate에서는 naPINN이 MSE, LAD, OrPINN, direct PINN-EBM보다
rMAE와 rMSE가 모두 낮다. 다만 clean observation을 많이 거부하고 seed별
gate 동작이 흔들리므로, 이 결과를 모든 구조화 오염에서의 보편적 우위로
확대하지 않는다. Direct PINN-EBM은 표본 표준편차도 매우 크다.

Corruption detection은 다음과 같다.

| Method / score | AUROC | failed rejection | clean rejection | retained fraction |
| --- | ---: | ---: | ---: | ---: |
| PINN-EBM raw EBM surprise | 0.91928 ± 0.03547 | -- | -- | -- |
| naPINN raw EBM surprise | **0.93060 ± 0.00348** | -- | -- | -- |
| naPINN learned gate | 0.93002 ± 0.00395 | 92.90% ± 4.12% | 32.67% ± 24.25% | 49.13% ± 18.16% |

Calibration의 eligibility 조건은 held-out seeds에서도 평균적으로
유지됐지만 clean rejection의 seed variability는 매우 크다. 따라서 30%
identity-level sensor failure에서도 corruption ranking이 작동했다는
evidence는 되지만, clean observation을 안정적으로 보존했다는 evidence는
아니다. Raw estimator와 gate AUROC가 거의 같으므로 gate가 anomaly ranking
자체를 새로 가능하게 한다고 주장하지 않는다.

새 실행의 naPINN phase time은 warm-up 206.2 ± 37.5초,
estimator-only initialization 46.0 ± 1.5초, joint training
1572.6 ± 21.6초이며 end-to-end는 1825.4 ± 51.7초다. Direct PINN-EBM의
end-to-end는 1698.0 ± 35.1초다. Shared GPU에서 측정한 observational
timing이므로 hardware-isolated speed benchmark로 해석하지 않는다.

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
| MSE-PINN | 0.21017 ± 0.00548 | 0.28448 ± 0.01167 | 0.04866 ± 0.00097 | 0.04413 ± 0.00126 |
| LAD-PINN | **0.13199 ± 0.00017** | **0.20828 ± 0.00208** | 0.05159 ± 0.00052 | 0.07469 ± 0.00072 |
| OrPINN \(q=2.9\) | 0.16528 ± 0.00548 | 0.24054 ± 0.00826 | 0.06583 ± 0.00201 | 0.07090 ± 0.00107 |
| PINN-EBM, no gate | 0.20739 ± 0.01055 | 0.31176 ± 0.01785 | 0.03623 ± 0.00615 | 0.08531 ± 0.01766 |
| naPINN, rejection cost 0.10 | 0.15538 ± 0.00052 | 0.23260 ± 0.00328 | **0.01789 ± 0.00040** | **0.01891 ± 0.00085** |

naPINN은 MSE, OrPINN, direct PINN-EBM보다 두 field metric이 낮지만 LAD가
여전히 best다. Cost 0.01보다 naPINN field agreement와 clean retention은
좋아졌지만 nominal-physics residual은 커졌다.

| naPINN detection metric | mean ± sample std |
| --- | ---: |
| Gate failure-detection AUROC | 0.92914 ± 0.00080 |
| Raw EBM failure-detection AUROC | **0.94339 ± 0.00016** |
| Failed-scalar rejection rate | 86.15% ± 0.33% |
| Clean-scalar rejection rate | 4.03% ± 0.33% |
| Retained fraction | 87.85% ± 0.33% |

30% seed-39 calibration의 failed rejection ≥90% criterion은 10%
held-out condition에서 유지되지 않았다. 대신 clean rejection은 cost
0.01의 33.68%에서 4.03%로 크게 줄었다. 이는 rejection cost가 단순한
cosmetic hyperparameter가 아니라 failure recall과 clean retention의
trade-off를 제어하며, 한 severity에서 정한 constraint가 다른 severity로
완전히 transfer되지 않는다는 limitation이다. Raw EBM AUROC가 gate보다
높다는 결과도 그대로 유지한다.

같은 condition의 MAD-PINN은 rMAE/rMSE
0.15738 ± 0.00134 / 0.23950 ± 0.00144로 naPINN보다 조금 높고
LAD보다 높다. 반면 momentum/continuity RMS는 0.01445/0.01363으로
naPINN보다 낮다. MAD screen은 corrupted 91.57% ± 0.92%와 clean
30.42% ± 0.73%를 reject하여 63.53%를 retained한다. naPINN은
field와 clean retention이 더 좋고, MAD는 physics residual과 failure
recall이 더 좋은 trade-off다. MAD는 60,000-update,
non-compute-matched reference다.

Selected-cost naPINN end-to-end time은 1835.7 ± 11.5초다. Shared-server
측정이므로 cost-0.01 또는 baseline과의 hardware-isolated speed
comparison으로 사용하지 않는다.

### 8.5 Real PIV + 20% persistent failure: calibration-selected cost 0.10

중간 severity의 matched seeds 40--42 결과도 완료됐다.

| Method | held-out rMAE | held-out rMSE | momentum RMS | continuity RMS |
| --- | ---: | ---: | ---: | ---: |
| MSE-PINN | 0.30046 ± 0.00418 | 0.38291 ± 0.00385 | 0.06220 ± 0.00128 | 0.06070 ± 0.00103 |
| LAD-PINN | 0.17190 ± 0.00333 | 0.25310 ± 0.00208 | 0.06023 ± 0.00237 | 0.07765 ± 0.00126 |
| OrPINN \(q=2.9\) | 0.23666 ± 0.00390 | 0.32521 ± 0.00482 | 0.07747 ± 0.00019 | 0.08340 ± 0.00185 |
| PINN-EBM, no gate | 0.19577 ± 0.02248 | 0.29842 ± 0.00907 | 0.06206 ± 0.00646 | 0.11471 ± 0.01221 |
| naPINN, rejection cost 0.10 | **0.17113 ± 0.00251** | **0.25266 ± 0.00747** | **0.01432 ± 0.00037** | **0.01378 ± 0.00003** |

naPINN은 LAD보다 rMAE와 rMSE가 각각 약 0.4%와 0.2% 낮은 매우 작은
평균 차이를 보인다. 반복이 세 번뿐이므로 강한 우위로 해석하지 않는다.
이 setting에서는 direct PINN-EBM보다 field error도 낮고 nominal
momentum/continuity residual도 작다.

| Detection score | AUROC | failed rejection | clean rejection | retained |
| --- | ---: | ---: | ---: | ---: |
| PINN-EBM raw EBM surprise | **0.95706 ± 0.00724** | -- | -- | -- |
| naPINN raw EBM surprise | 0.94439 ± 0.00533 | -- | -- | -- |
| naPINN learned gate | 0.93683 ± 0.00957 | 90.66% ± 0.70% | 8.38% ± 0.32% | 75.33% ± 0.34% |

Raw EBM ranking이 gate보다 좋은 pattern은 다시 나타난다. 따라서
20% condition의 positive field result도 gate가 anomaly score 자체를
새로 가능하게 한다는 claim으로 해석하지 않는다.

같은 condition의 MAD-PINN은 rMAE/rMSE
0.17346 ± 0.00401 / 0.25574 ± 0.00195로 LAD와 naPINN보다 조금 높다.
Momentum/continuity RMS는 0.01808/0.01484다. Fixed MAD screen은 known
corrupted scalars 91.43% ± 0.85%와 clean scalars 25.46% ± 0.29%를
reject하여 61.49%를 retained한다. 이는 60,000-update,
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
| MSE-PINN | 0.37221 ± 0.00662 | 0.49041 ± 0.00917 | 1331.81 ± 111.03 | 86.72% ± 1.11% | 0.07606 ± 0.00215 | 0.07098 ± 0.00054 |
| LAD-PINN | 0.23303 ± 0.00332 | 0.35739 ± 0.00643 | 3634.24 ± 1650.42 | 63.77% ± 16.45% | 0.07269 ± 0.00072 | 0.07913 ± 0.00125 |
| PINN-EBM, no gate | 0.28567 ± 0.15308 | 0.36947 ± 0.11387 | **7681.08 ± 5927.76** | **51.54% ± 18.20%** | 0.07782 ± 0.01289 | 0.14246 ± 0.01231 |
| naPINN | **0.22825 ± 0.02198** | **0.31499 ± 0.00886** | 2306.53 ± 550.97 | 77.01% ± 5.49% | **0.02488 ± 0.01538** | **0.02777 ± 0.02501** |

이 extension은 naPINN의 parameter-recovery success를 보여주지 않는다.
naPINN은 field rMAE/rMSE가 가장 낮지만 Re 상대오차는 77.0%다. Direct
PINN-EBM은 mean Re error가 가장 낮아도 51.5%이고 seed별 learned Re가
14,260.66, 2,757.24, 6,025.34로 매우 크게 흔들린다. 모든 방법이 metadata
Re를 안정적으로 복원하지 못했다.

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
5,000 estimator-only updates, PDE/data weight를 사용한다. 현재 서버에서
rejection cost 0.5의 9/9 기본 비교와 paper-aligned 1.0의 9/9 extension이
모두 엄격 집계됐다. 이전 서버 요약은 별도 계보로 남기되, 아래 새
수치는 같은 서버와 같은 seeds의 비교로 해석한다.

| PDE | Method | field rMAE | field rMSE | parameter abs. error | detection AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | PINN-EBM | 0.10288 ± 0.02365 | 0.10169 ± 0.02132 | epsilon: **0.00215 ± 0.00036** | raw EBM: 0.99386 ± 0.00233 |
| Allen--Cahn | naPINN | **0.09202 ± 0.00352** | **0.09483 ± 0.00232** | epsilon: 0.00275 ± 0.00048 | gate: **0.99476 ± 0.00118** |
| Burgers | PINN-EBM | **0.04738 ± 0.01326** | **0.06743 ± 0.01227** | nu: **0.00065 ± 0.00018** | raw EBM: **0.99307 ± 0.00067** |
| Burgers | naPINN | 0.08186 ± 0.00240 | 0.09858 ± 0.00700 | nu: 0.00272 ± 0.00029 | gate: 0.99132 ± 0.00080 |
| lambda--omega | PINN-EBM | **0.04023 ± 0.00218** | **0.05564 ± 0.00313** | beta: **0.01016 ± 0.00409** | raw EBM: **0.99406 ± 0.00030** |
| lambda--omega | naPINN | 0.07056 ± 0.00121 | 0.08890 ± 0.00075 | beta: 0.01359 ± 0.00068 | gate: 0.99302 ± 0.00040 |

이 결과는 reviewer의 closest-prior concern을 없애지 않는다. Direct
PINN-EBM은 Burgers와 lambda--omega의 field error, 그리고 세 PDE의
parameter error가 더 낮다. Allen--Cahn field error와 detection AUROC는
naPINN이 더 낮거나 높다. Burgers와 lambda--omega의 raw EBM AUROC는
naPINN gate보다 약간 높다. 따라서 다음 주장은 사용할 수 없다.

- direct EBM-NLL은 severe outlier에서 실패한다.
- gain의 대부분은 trainable gate에서 온다.
- gate만 corruption ranking을 가능하게 한다.

Rebuttal에서 남는 방어는 명시적인 관측 포함/제외 결정, 관측을 버릴 때의
비용, density 학습과 field reconstruction의 분리, 그리고 실제 PIV 큰
이상치 실험에서의 결과다. 이 합성 결과를 숨기지 않고 claim을 조건부로
좁혀야 한다.

### 8.9 Closest-prior PDE-loss-weight sensitivity

Equal-weight held-out 결과를 이미 본 뒤 seed-39 calibration을 시작했기
때문에, seed 39만으로 favorable weight를 골라 보고하지 않았다. 대신
direct PINN-EBM의 PDE-loss weight \(1,10,50\)을 seeds 40--42에도 모두
실행하여 full grid를 공개한다.

| PDE | PDE weight | field rMAE | field rMSE | parameter abs. error |
| --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | 1 | 0.10288 ± 0.02365 | 0.10169 ± 0.02132 | 0.00215 ± 0.00036 |
| Allen--Cahn | 10 | **0.07629 ± 0.03903** | **0.07533 ± 0.03738** | **0.00141 ± 0.00042** |
| Allen--Cahn | 50 | 0.75508 ± 0.01691 | 0.74897 ± 0.01852 | 0.54693 ± 0.10998 |
| Burgers | 1 | 0.04738 ± 0.01326 | 0.06743 ± 0.01227 | 0.00065 ± 0.00018 |
| Burgers | 10 | 0.04239 ± 0.00763 | 0.06060 ± 0.01086 | **0.00025 ± 0.00011** |
| Burgers | 50 | **0.03496 ± 0.00199** | **0.04926 ± 0.00112** | 0.00054 ± 0.00020 |
| lambda--omega | 1 | 0.04023 ± 0.00218 | **0.05564 ± 0.00313** | 0.01016 ± 0.00409 |
| lambda--omega | 10 | **0.03995 ± 0.00345** | 0.05728 ± 0.00092 | 0.00398 ± 0.00143 |
| lambda--omega | 50 | 0.04384 ± 0.00282 | 0.07635 ± 0.00408 | **0.00289 ± 0.00117** |

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
| AR(1) | MSE | 0.22005 ± 0.00048 | 0.27981 ± 0.00029 | 0.05440 ± 0.00070 | 0.05108 ± 0.00136 |
| AR(1) | LAD | **0.12291 ± 0.00063** | **0.19727 ± 0.00107** | 0.04862 ± 0.00097 | 0.07134 ± 0.00133 |
| AR(1) | OrPINN q=2.9 | 0.14582 ± 0.00106 | 0.21065 ± 0.00051 | 0.05616 ± 0.00081 | 0.06556 ± 0.00170 |
| AR(1) | PINN-EBM | 0.20843 ± 0.01835 | 0.31694 ± 0.02182 | 0.04211 ± 0.00293 | 0.09907 ± 0.01109 |
| AR(1) | naPINN | 0.15036 ± 0.00064 | 0.21544 ± 0.00238 | **0.01925 ± 0.00018** | **0.02022 ± 0.00018** |
| Spatial burst | MSE | 0.15975 ± 0.00020 | 0.21735 ± 0.00037 | 0.04019 ± 0.00041 | 0.03901 ± 0.00017 |
| Spatial burst | LAD | **0.11321 ± 0.00009** | **0.18545 ± 0.00028** | 0.04602 ± 0.00062 | 0.07418 ± 0.00127 |
| Spatial burst | OrPINN q=2.9 | 0.12539 ± 0.00077 | 0.18828 ± 0.00012 | 0.04795 ± 0.00032 | 0.06300 ± 0.00021 |
| Spatial burst | PINN-EBM | 0.25594 ± 0.03873 | 0.35479 ± 0.02211 | 0.03473 ± 0.00954 | 0.08566 ± 0.01815 |
| Spatial burst | naPINN | 0.15318 ± 0.00354 | 0.22605 ± 0.00151 | **0.01730 ± 0.00105** | **0.01719 ± 0.00026** |

| Corruption | Score | AUROC | failed rejection | clean rejection | retained |
| --- | --- | ---: | ---: | ---: | ---: |
| AR(1) | PINN-EBM raw EBM | 0.88157 ± 0.01359 | -- | -- | -- |
| AR(1) | naPINN raw EBM | **0.91378 ± 0.00048** | -- | -- | -- |
| AR(1) | naPINN gate | 0.91307 ± 0.00085 | 61.41% ± 3.12% | 2.96% ± 0.50% | 88.21% ± 0.90% |
| Spatial burst | PINN-EBM raw EBM | **0.98745 ± 0.00436** | -- | -- | -- |
| Spatial burst | naPINN raw EBM | 0.98544 ± 0.00895 | -- | -- | -- |
| Spatial burst | naPINN gate | 0.98459 ± 0.01042 | 100.00% | 4.62% ± 0.85% | 92.50% ± 0.82% |

LAD가 두 structured conditions 모두 field metric에서 가장 좋고,
naPINN은 우승하지 못한다. AR(1)에서는 OrPINN도 naPINN보다 field
agreement가 조금 좋다. Spatial burst에서 naPINN은 MSE보다 rMAE는
낮지만 rMSE는 높다. 반면 naPINN의 nominal momentum/continuity residual은
두 조건 모두 가장 낮다.

Detection도 condition-dependent하다. Spatial burst는 gate가 injected
failed scalars를 모두 reject하지만, AR(1)에서는 61.41%만 reject한다.
30% persistent-failure calibration에서 고른 rejection cost가 temporal
correlation으로 완전히 transfer되지 않는다는 뜻이다. 따라서 이 matrix는
“correlated noise에서도 consistently best”라는 근거가 아니라,
structured corruption coverage와 field/physics/detection trade-off를
보여주는 evidence다.

AR(1)의 MAD-PINN result도 seeds 40--42가 완료됐다. rMAE/rMSE는
0.15158 ± 0.00088 / 0.23113 ± 0.00188로 naPINN보다 높고 LAD보다 높다.
반면 momentum/continuity RMS는 0.01529/0.01377으로 naPINN보다 낮다.
MAD screen은 corrupted 94.30% ± 0.12%와 clean 27.51% ± 0.24%를
reject하여 62.40%를 retained한다. naPINN gate의 61.41%
failure rejection과 2.96% clean rejection에 비해, MAD는 훨씬 높은
failure recall을 얻는 대신 clean data를 더 버리고 field agreement도
악화시킨다. 이 결과 역시 60,000-update, non-compute-matched
trade-off다.

Spatial burst MAD-PINN도 seeds 40--42가 완료됐다. rMAE/rMSE는
0.14851 ± 0.00072 / 0.22623 ± 0.00123이다. naPINN과 비교하면 MAD가
rMAE는 낮고 naPINN이 rMSE는 아주 조금 낮으며, LAD가 두 field
metrics 모두 가장 좋다. MAD momentum/continuity RMS는
0.01448/0.01354로 naPINN의 0.01730/0.01719보다 낮다. 두 selector 모두
corrupted scalars를 100% reject하지만 MAD는 clean scalars 34.04% ±
0.10%를 reject하고 naPINN은 4.62% ± 0.85%만 reject한다. 따라서
spatial burst에서도 MAD는 physics fit과 rMAE를 조금 개선하는 대신
clean retention을 크게 희생한다.

## 9. 세 합성 PDE의 15% 큰 이상치 엄격 집계와 PDE parameter 복원

현재 서버에서 다시 생성하고 162/162 실행을 모두 확인한 자료 중 15%
four-Gaussian 큰 이상치 조건의 seeds 40--42 평균 ± 표본 표준편차다.
모든 방법은 같은 좌표, 같은 오염 관측값, 같은 이상치 위치와 30,000
PINN update를 사용한다. PINN-EBM과 naPINN에는 추가로 5,000회
estimator-only update가 있다. 이전 서버에서 전달된 숫자와 하나의 표로
섞지 않고, 아래에는 현재 서버의 엄격 집계만 쓴다.

| PDE | Method | field rMAE | field rMSE | learned parameter | parameter abs. error |
| --- | --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | MSE | 0.85525 ± 0.10760 | 0.71225 ± 0.08689 | epsilon=0.29645 ± 0.00163 | 0.00355 ± 0.00163 |
| Allen--Cahn | LAD | 0.38339 ± 0.03061 | 0.32211 ± 0.02022 | epsilon=0.29688 ± 0.00088 | 0.00312 ± 0.00088 |
| Allen--Cahn | OrPINN q=1.9 | 0.67317 ± 0.08160 | 0.56108 ± 0.06205 | epsilon=0.30606 ± 0.00274 | 0.00606 ± 0.00274 |
| Allen--Cahn | OrPINN q=2.9 | 0.26754 ± 0.07363 | 0.23081 ± 0.05536 | epsilon=0.29679 ± 0.00060 | 0.00321 ± 0.00060 |
| Allen--Cahn | PINN-EBM | 0.10288 ± 0.02365 | 0.10169 ± 0.02132 | epsilon=0.29785 ± 0.00036 | **0.00215 ± 0.00036** |
| Allen--Cahn | naPINN | **0.09202 ± 0.00352** | **0.09483 ± 0.00232** | epsilon=0.29725 ± 0.00048 | 0.00275 ± 0.00048 |
| Burgers | MSE | 0.58642 ± 0.03949 | 0.56590 ± 0.03655 | nu=0.01566 ± 0.00085 | 0.00566 ± 0.00085 |
| Burgers | LAD | 0.25627 ± 0.00195 | 0.24634 ± 0.00099 | nu=0.01181 ± 0.00073 | 0.00181 ± 0.00073 |
| Burgers | OrPINN q=1.9 | 0.39356 ± 0.01173 | 0.38632 ± 0.01363 | nu=0.01515 ± 0.00080 | 0.00515 ± 0.00080 |
| Burgers | OrPINN q=2.9 | 0.19082 ± 0.02191 | 0.19223 ± 0.01710 | nu=0.01123 ± 0.00074 | 0.00123 ± 0.00074 |
| Burgers | PINN-EBM | **0.04738 ± 0.01326** | **0.06743 ± 0.01227** | nu=0.00935 ± 0.00018 | **0.00065 ± 0.00018** |
| Burgers | naPINN | 0.08186 ± 0.00240 | 0.09858 ± 0.00700 | nu=0.01272 ± 0.00029 | 0.00272 ± 0.00029 |
| lambda--omega | MSE | 0.42187 ± 0.01052 | 0.44110 ± 0.01033 | beta=0.67215 ± 0.00530 | 0.32785 ± 0.00530 |
| lambda--omega | LAD | 0.19097 ± 0.00784 | 0.19909 ± 0.00623 | beta=0.91630 ± 0.00410 | 0.08370 ± 0.00410 |
| lambda--omega | OrPINN q=1.9 | 0.17492 ± 0.01256 | 0.19818 ± 0.01311 | beta=0.93913 ± 0.00528 | 0.06087 ± 0.00528 |
| lambda--omega | OrPINN q=2.9 | 0.15263 ± 0.00309 | 0.16514 ± 0.00325 | beta=0.94766 ± 0.00403 | 0.05234 ± 0.00403 |
| lambda--omega | PINN-EBM | **0.04023 ± 0.00218** | **0.05564 ± 0.00313** | beta=0.98984 ± 0.00409 | **0.01016 ± 0.00409** |
| lambda--omega | naPINN | 0.07056 ± 0.00121 | 0.08890 ± 0.00075 | beta=0.98641 ± 0.00068 | 0.01359 ± 0.00068 |

이 결과는 field reconstruction뿐 아니라 reviewer가 요청한 Burgers
viscosity와 lambda--omega beta recovery까지 포함한다. Direct PINN-EBM은
Burgers와 lambda--omega의 field error와 세 PDE의 parameter error가 가장
낮다. 반면 Allen--Cahn field reconstruction은 naPINN이 가장 낮다.
따라서 gate의 가치를 모든 합성 조건의 정확도 우위로 설명할 수 없고,
field 복원과 parameter 복원의 승자도 다를 수 있다.

naPINN은 Allen--Cahn에서 1위, Burgers와 lambda--omega에서 2위다.
OrPINN \(q=2.9\) 대비 rMAE를 각각 약 65.6%, 57.1%, 53.8% 낮춘다.
따라서 고정 robust-loss baseline 대비 효과는 유지되지만, 이 결과가
closest-prior 비교를 대신할 수는 없다.

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
| Allen--Cahn | PINN-EBM | 132.7 ± 28.9 | 46.9 ± 1.6 | 896.8 ± 84.2 | 1076.4 ± 112.2 |
| Allen--Cahn | naPINN | 142.7 ± 7.7 | 47.6 ± 0.7 | 1150.4 ± 33.1 | 1340.7 ± 34.4 |
| Burgers | PINN-EBM | 162.1 ± 12.8 | 46.1 ± 1.0 | 1027.8 ± 25.2 | 1236.0 ± 32.0 |
| Burgers | naPINN | 163.0 ± 18.6 | 48.5 ± 3.1 | 1306.7 ± 90.3 | 1518.1 ± 111.8 |
| lambda--omega | PINN-EBM | 176.6 ± 14.9 | 47.3 ± 0.6 | 1048.6 ± 90.5 | 1272.5 ± 103.6 |
| lambda--omega | naPINN | 179.4 ± 18.9 | 47.5 ± 0.8 | 1329.8 ± 85.4 | 1556.7 ± 94.2 |

Estimator-only 5,000 updates는 이 runs에서 약 46--48초였고 total training의
약 3.0--4.4%다. 따라서 cost가 작다는 사용자의 intuition은 방향상
맞지만, “몇 초”라고 낮춰 말하지 않고 측정값을 제시한다. Joint phase
안에도 estimator update가 들어 있으므로 estimator-only time만으로
naPINN의 전체 overhead를 대표하지 않는다.

### 9.2 EMA coefficient sensitivity

Reviewer가 직접 요청한 Allen--Cahn 15% four-Gaussian condition에서 code
convention의 EMA update weight \(m\)을 바꿨다. Old-state decay로 쓰면
\(\rho=1-m\)이다.

| Update weight \(m\) | Decay \(\rho\) | field rMAE | field rMSE | epsilon abs. error |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.99 | 0.10778 ± 0.03716 | 0.10931 ± 0.03179 | 0.00264 |
| 0.05 | 0.95 | 0.09202 ± 0.00352 | 0.09483 ± 0.00232 | 0.00275 |
| 0.10 | 0.90 | **0.08551 ± 0.00874** | **0.09235 ± 0.00494** | 0.00262 |

세 값의 표본 표준편차가 겹치므로 \(m=0.10\)이 보편적으로 최적이라고
주장하지 않는다. 이 세 값에서는 0.10의 평균 field error가 가장 낮지만,
이는 Allen--Cahn 15% 한 조건의 국소 민감도 결과다. Main text의 beta와
appendix의 rho를 같은 숫자로 취급하지 않고, update-weight와 decay
convention을 명시하는 것이 핵심 response다.

### 9.3 Rejection-cost sensitivity와 paper/config discrepancy

다음 표는 현재 서버에서 Allen--Cahn 15% 조건의
\(\lambda_{\mathrm{rej}}\)를 0.10--1.00으로 바꾸어 각 값마다 seeds
40--42를 모두 실행한 엄격 집계다. 총 15회이며 일부 값이나 seed를
결과를 본 뒤 제외하지 않았다.

| Rejection cost | field rMAE | field rMSE | outlier rejection | clean rejection | gate AUROC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | **0.08847 ± 0.00549** | **0.09283 ± 0.00668** | 99.30% | 0.76% | **0.99403** |
| 0.30 | 0.09133 ± 0.00282 | 0.09631 ± 0.00330 | 99.25% | 0.72% | 0.99316 |
| 0.50 | 0.09202 ± 0.00352 | 0.09483 ± 0.00232 | 99.29% | 0.66% | 0.99377 |
| 0.70 | 0.09886 ± 0.01590 | 0.09869 ± 0.01361 | 99.28% | 0.72% | 0.99258 |
| 1.00 | 0.86890 ± 0.02391 | 0.72211 ± 0.02451 | 2.12% | **0.15%** | 0.55067 |

0.10--0.70의 field mean은 서로 가깝고 0.10이 가장 낮지만, 1.00에서는
gate가 거의 모든 observation을 accept하여 outlier rejection과 field
reconstruction이 함께 붕괴한다. 이는 submitted appendix에 적힌
Allen--Cahn cost 1.0과 실제 initial core cost 0.5 사이의 discrepancy가
결과에 실질적인 영향을 준다는 뜻이다. 따라서 1.0 결과를 숨기거나
0.5로 조용히 대체하지 않는다. Reviewer에게는 exact configuration을
clarify하고, submitted value 1.0의 degradation을 인정해야 한다.

이 sweep 역시 post-hoc optimal-cost selection 근거로 사용하지 않는다.
Core result는 frozen 0.5를 유지하고, 표 전체를 sensitivity로 보고한다.

추가로 paper-aligned cost 1.0의 5%·10%·15%, seeds 40--42는 현재 서버에서
새로 실행해 9/9 full run과 strict aggregate를 완료했다.

| Corruption ratio | cost 1.0 field rMAE | cost 1.0 field rMSE | epsilon 절대오차 | cost 1.0 outlier rejection | cost 1.0 gate AUROC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 0.26088 ± 0.07158 | 0.23520 ± 0.06119 | 0.00419 ± 0.00395 | 3.07% | 0.56380 |
| 10% | 0.56828 ± 0.08379 | 0.47785 ± 0.06770 | 0.00530 ± 0.00440 | 2.25% | 0.55831 |
| 15% | 0.86890 ± 0.02391 | 0.72211 ± 0.02451 | 0.00408 ± 0.00285 | 2.12% | 0.55067 |

Fresh aggregate:
`outputs/rebuttal/allen_cost1_recovery_20260726/aggregation_strict.json`

이 값은 이전 서버에서 전달된 cost-1.0 rMAE
0.25878/0.56180/0.87859와 조금 다르므로 두 실행을 하나로 합치거나
새 값으로 조용히 덮어쓰지 않는다. 그러나 gate가 이상치의 약 2--3%만
거부하고 오염률이 높아질수록 field reconstruction이 크게 악화된다는
결론은 일치한다. 같은 서버의 cost-0.5 기본 비교도 완료됐으므로 15%
조건에서는 동일 seed의 새 결과끼리 비교할 수 있다. 다만 이 새 비교가
제출 당시 실제 설정을 증명하지는 않는다.

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
| Allen--Cahn | 0.34354 ± 0.04332 | 0.28913 ± 0.03100 | 0.00735 ± 0.00583 | 98.52% ± 0.21% | 18.95% ± 2.25% | 69.11% ± 1.94% |
| Burgers | 0.24198 ± 0.01160 | 0.23555 ± 0.00988 | 0.00166 ± 0.00104 | 98.58% ± 0.14% | 16.67% ± 0.36% | 71.04% ± 0.32% |
| lambda--omega | 0.12735 ± 0.00622 | 0.14349 ± 0.00323 | 0.03766 ± 0.00322 | 98.42% ± 0.08% | 10.57% ± 1.09% | 76.24% ± 0.92% |

MAD-PINN은 세 PDE 모두 LAD field error를 개선하지만 direct PINN-EBM과
naPINN보다 높다. Screening 자체는 severe outlier를 거의 모두 제거하나
clean scalar도 10.6--19.0% 제거한다. 이 baseline은 reviewer의 practical preprocessing 질문에
답하지만 60,000 PINN updates이므로 30,000-update method와 compute
matched라고 부르지 않는다.

### 9.5 Real PIV의 MAD-PINN preprocessing

30% persistent sensor failure에서도 같은 published screen을 적용했다.
Seeds 40--42 aggregate는 다음과 같다.

| Method | held-out rMAE | held-out rMSE | momentum RMS | continuity RMS |
| --- | ---: | ---: | ---: | ---: |
| LAD-PINN, stage-one reference | **0.23411 ± 0.00458** | **0.35852 ± 0.00694** | 0.07400 ± 0.00094 | 0.07963 ± 0.00138 |
| MAD-PINN, 60k updates | 0.24848 ± 0.00831 | 0.36792 ± 0.00974 | **0.02655 ± 0.00022** | **0.02211 ± 0.00100** |

MAD screen은 known failed scalars의 80.63% ± 0.25%와 clean scalars의
20.16% ± 0.62%를 제거하고, 전체의 61.57% ± 0.38%를 retained했다.
Field agreement는 stage-one LAD보다 오히려 나빠졌고 naPINN
(rMAE 0.21797, rMSE 0.30886)보다도 나쁘다. Nominal-physics residual은
LAD보다 작지만 naPINN보다 크다. 따라서 basic preprocessing만으로
naPINN result를 설명할 수 없다는 evidence는 되지만, MAD-PINN이
compute-matched라는 주장은 하지 않는다.

두 단계 pipeline은 정확히 60,000 PINN updates다. Strict aggregate에는
두 단계 전체 wall time이 공통 metric으로 저장되지 않았으므로, 이전
서버의 wall time을 새 실행 시간으로 재사용하지 않는다.

### 9.6 전체 four-Gaussian corruption-ratio core

5%, 10%, 15%의 54개 method/PDE group, 총 162개 held-out runs가 모두
완료됐다. 아래는 field rMAE의 mean ± sample standard deviation이다.
각 group은 동일한 seeds 40--42를 사용한다.

| PDE | Outlier ratio | MSE | LAD | OrPINN q=1.9 | OrPINN q=2.9 | PINN-EBM | naPINN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Allen--Cahn | 5% | 0.3417 ± 0.0949 | 0.2412 ± 0.0416 | 0.2220 ± 0.0283 | 0.1624 ± 0.0823 | **0.0916 ± 0.0096** | 0.1389 ± 0.0701 |
| Allen--Cahn | 10% | 0.6263 ± 0.1423 | 0.2803 ± 0.0163 | 0.4060 ± 0.0482 | 0.2390 ± 0.0561 | **0.0785 ± 0.0126** | 0.1043 ± 0.0226 |
| Allen--Cahn | 15% | 0.8553 ± 0.1076 | 0.3834 ± 0.0306 | 0.6732 ± 0.0816 | 0.2675 ± 0.0736 | 0.1029 ± 0.0237 | **0.0920 ± 0.0035** |
| Burgers | 5% | 0.2148 ± 0.0135 | 0.1975 ± 0.0077 | 0.1650 ± 0.0112 | 0.1354 ± 0.0130 | **0.0393 ± 0.0023** | 0.0799 ± 0.0081 |
| Burgers | 10% | 0.4037 ± 0.0036 | 0.2311 ± 0.0108 | 0.2662 ± 0.0063 | 0.1624 ± 0.0202 | **0.0498 ± 0.0058** | 0.0802 ± 0.0109 |
| Burgers | 15% | 0.5864 ± 0.0395 | 0.2563 ± 0.0020 | 0.3936 ± 0.0117 | 0.1908 ± 0.0219 | **0.0474 ± 0.0133** | 0.0819 ± 0.0024 |
| lambda--omega | 5% | 0.1529 ± 0.0098 | 0.1585 ± 0.0110 | 0.0895 ± 0.0027 | 0.1229 ± 0.0060 | **0.0444 ± 0.0059** | 0.0735 ± 0.0073 |
| lambda--omega | 10% | 0.2886 ± 0.0087 | 0.1731 ± 0.0071 | 0.1301 ± 0.0141 | 0.1391 ± 0.0054 | **0.0497 ± 0.0142** | 0.0753 ± 0.0140 |
| lambda--omega | 15% | 0.4219 ± 0.0105 | 0.1910 ± 0.0078 | 0.1749 ± 0.0126 | 0.1526 ± 0.0031 | **0.0402 ± 0.0022** | 0.0706 ± 0.0012 |

Direct PINN-EBM이 9개 field 조건 중 8개에서 가장 낮고, Allen--Cahn
15%에서는 naPINN이 가장 낮다. 나머지 8개 조건에서는 naPINN이 2위다.
따라서 closest-prior에 대한 보편적인 정확도 우위 주장은 성립하지 않는다.
반면 fixed-loss 및 OrPINN baseline에 대한 naPINN의 장점은 모든 조건에서
유지된다.

같은 runs의 PDE-parameter absolute error mean은 다음과 같다. Standard
deviation과 learned value까지 포함한 원자료는
`rebuttal_synthetic_aggregation.json`에 보존했다.

| PDE | Outlier ratio | MSE | LAD | OrPINN q=1.9 | OrPINN q=2.9 | PINN-EBM | naPINN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Allen--Cahn epsilon | 5% | 0.00411 | 0.00254 | **0.00139** | 0.00260 | 0.00187 | 0.00268 |
| Allen--Cahn epsilon | 10% | 0.00592 | 0.00302 | 0.00262 | 0.00349 | **0.00242** | 0.00319 |
| Allen--Cahn epsilon | 15% | 0.00355 | 0.00312 | 0.00606 | 0.00321 | **0.00215** | 0.00275 |
| Burgers viscosity | 5% | 0.00208 | 0.00106 | 0.00290 | **0.00079** | 0.00080 | 0.00203 |
| Burgers viscosity | 10% | 0.00269 | 0.00148 | 0.00342 | 0.00080 | **0.00068** | 0.00215 |
| Burgers viscosity | 15% | 0.00566 | 0.00181 | 0.00515 | 0.00123 | **0.00065** | 0.00272 |
| lambda--omega beta | 5% | 0.05700 | 0.05471 | 0.01954 | 0.03765 | 0.01266 | **0.01171** |
| lambda--omega beta | 10% | 0.15227 | 0.06667 | 0.03525 | 0.04414 | 0.01222 | **0.01136** |
| lambda--omega beta | 15% | 0.32785 | 0.08370 | 0.06087 | 0.05234 | **0.01016** | 0.01359 |

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
| Gaussian | 0.21636 | 0.08686 | 0.07858 | 0.06853 | **0.05545** | 0.06058 | 0.06004 |
| Laplace | 0.31517 | 0.07447 | 0.07028 | 0.06921 | 0.05826 | **0.05631** | 0.05954 |
| Student-t | 0.09437 | **0.04516** | 0.06048 | 0.04545 | 0.09773 | 0.04578 | 0.06355 |
| Four-Gaussian | 0.42187 | 0.19097 | 0.15263 | **0.04023** | 0.07056 | 0.13995 | 0.13801 |

한 method가 모든 family에서 best가 아니다.

- Gaussian에서는 base naPINN-MSE가 best다.
- Laplace에서는 naPINN-LAD가 best다.
- Student-t에서는 LAD가 가장 낮고 direct PINN-EBM과 naPINN-LAD가 매우
  가깝다. base naPINN-MSE는 MSE보다도 나쁘다.
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
| Allen--Cahn | 0.87534 ± 0.06598 | 0.34754 ± 0.02568 | 0.32539 ± 0.07577 | 0.10288 ± 0.02365 | **0.09202 ± 0.00352** |
| Burgers | 0.59199 ± 0.02152 | 0.25301 ± 0.02045 | 0.20131 ± 0.01710 | **0.04738 ± 0.01326** | 0.08186 ± 0.00240 |
| lambda--omega | 0.44803 ± 0.02466 | 0.18677 ± 0.00963 | 0.14221 ± 0.02097 | **0.04023 ± 0.00218** | 0.07056 ± 0.00121 |

표의 값은 field rMAE, seeds 40--42의 mean ± sample standard
deviation이다. 35k ordinary baseline은 세 PDE 모두 direct PINN-EBM과
naPINN보다 높다. 그러나 ordinary baseline 자체의 30k→35k 변화는
일관되지 않다. 예를 들어 Allen--Cahn LAD는 30k에서 0.38339, 35k에서
0.34754로 낮아지지만 MSE와 OrPINN은 여전히 훨씬 높다. 따라서 이 결과는
“추가 step이 항상 성능을 높인다”가 아니라, extra estimator-only update가
ordinary baseline에 불리한 update-count accounting 때문에 생긴
ranking은 아니라는 제한된 evidence로 사용한다.

35k total training time은 Allen--Cahn에서 MSE/LAD/OrPINN이 각각
996.7/1077.8/884.0초, Burgers에서 1084.8/1022.8/1072.7초,
lambda--omega에서 1088.1/1230.1/1129.7초였다. Concurrent shared-server
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
| Allen--Cahn | fixed-quantile | 0.50940 ± 0.15872 | 0.43452 ± 0.11742 | 0.00420 | 33.33% | 0.00% |
| Allen--Cahn | learnable-threshold | 0.24415 ± 0.03941 | 0.21852 ± 0.02432 | 0.00387 | 71.94% | 0.00% |
| Burgers | fixed-quantile | 0.37315 ± 0.03110 | 0.36354 ± 0.02706 | 0.00321 | 33.33% | 0.00% |
| Burgers | learnable-threshold | 0.08557 ± 0.01422 | 0.09796 ± 0.01427 | 0.00172 | 89.75% | 0.00% |
| lambda--omega | fixed-quantile | 0.23726 ± 0.00883 | 0.25678 ± 0.00989 | 0.11599 | 33.28% | 0.00% |
| lambda--omega | learnable-threshold | 0.06730 ± 0.01120 | 0.08243 ± 0.00991 | 0.01308 | 95.90% | 0.03% |

같은 15% 설정의 naPINN rMAE는
0.09202/0.08186/0.07056이고 direct PINN-EBM은
0.10288/0.04738/0.04023이다. Fixed-quantile은 15% injected outlier 중
약 1/3만 제거하기 때문에 세 PDE 모두 약하다. Learnable-threshold는
훨씬 강하고 Burgers에서 naPINN과 비슷하며 lambda--omega에서는 naPINN보다
약 4.6% 낮은 rMAE를 보이지만, Allen--Cahn에서는 크게 나쁘다. Direct
PINN-EBM과 비교해도 세 PDE 모두 높다.

따라서 fixed residual screening만으로 충분하다는 결론도, EBM gate가
항상 estimator-free threshold보다 우월하다는 결론도 둘 다 evidence와
맞지 않는다. 나오는 결론은 selector choice가 PDE-dependent하며,
naPINN의 gate contribution을 universal accuracy gain으로 분리해 주장하기
어렵다는 것이다.

### 9.10 실제 Cylinder PIV에 제출 코드의 기본 오염 규칙을 적용한 실험 (`RESPONSE HOLD`)

Controlled corruption만 사용한 campaign은 완료되었다.
Natural/unmodified PIV는 제외했으며, 변형하지 않은 held-out PIV는
physical ground truth가 아니라 독립적인 real measurement로 해석한다.

`Legacy-4G`는 제출 코드가 합성 데이터에서 사용하던 four-Gaussian 배경
잡음과 큰 점 이상치 생성 규칙을 뜻한다. `(b,o)=(1,1)`은 수학적 참값을
안다는 뜻이 아니다. 배경 잡음의 크기 `b`와 큰 이상치의 추가 크기 `o`를
제출 코드의 기본값에서 바꾸지 않았다는 뜻이며, 결과를 보기 전에 반드시
보고하기로 정한 조건이다. 모델 parameter나 신경망 weight를 실제 PIV로
옮긴 것도 아니다. 합성 데이터에 쓰던 **오염 생성 규칙만** 실제 PIV의
학습 측정에 적용했다. 또한 이 문서의 “세 합성 PDE 기본 비교”는
Allen--Cahn, Burgers, lambda--omega와 5%/10%/15% 이상치 비율을 조합한
아홉 조건을 뜻한다. 신경망의 내부 core나 kernel을 뜻하지 않는다.

- 제출 코드의 기본 오염 강도 `(b,o)=(1,1)`: 48 verified runs, 16 complete three-seed groups,
  6 paired input blocks
- Predeclared seed-39 scale grid: 18 cells × 8 conditions, 총 144 verified
  runs; frozen selection rule은 threshold relaxation 없이 적용
- Fresh-seed confirmation stage: 72 verified runs, 24 complete three-seed
  groups, 9 paired input blocks

가장 방어력 높은 결과는 강도 탐색 결과로 선택하지 않은, 제출 코드의
기본 오염 강도 `(b,o)=(1,1)`이다. naPINN-MSE의 held-out-PIV rMAE/rMSE는 10%에서
0.15517 ± 0.00237 / 0.21532 ± 0.00338, 15%에서
0.16023 ± 0.00060 / 0.22266 ± 0.00304였다. 두 ratio에서 가장 강한
non-naPINN baseline은 OrPINN q=2.9였고, naPINN-MSE는 각각
rMAE 27.6%/30.7%, rMSE 14.0%/16.0% 낮았다. Seeds 40--42의 모든
paired comparison에서 두 field metric이 같은 방향이었다. Exact
10%/15%의 gross-outlier rejection은 98.99%/99.23%이고
background-only rejection은 2.95%/4.23%였다. 다만 gate AUROC와 raw
EBM AUROC가 사실상 같으므로 gate-only anomaly-ranking superiority는
주장하지 않는다.

Scale confirmation은 더 mixed하다. Frozen mean criterion으로 세 후보
중 두 후보가 confirmed였지만, 3/3 fresh seeds에서 두 field metric을
모두 개선한 후보는 rank 1 `(b,o)=(2,2)` 하나뿐이다. Rank 2
`(2,1)`은 평균상 개선됐지만 2/3 seeds에 그쳤고, rank 3
`(1,0.5)`는 rMAE 열세/rMSE 우세 trade-off로 confirmation에 실패했다.
따라서 reviewer-facing release가 승인되면 이 사전 고정 기본 강도를 primary
evidence로 사용하고 scale search는 selection 사실과 세 후보 전체를
공개할 수 있을 때만 secondary evidence로 쓰는 것이 안전하다. 기존
세 합성 PDE 기본 비교에서 direct PINN-EBM이 9개 중 8개 field 조건에서 우세했던
adverse result도 함께 유지해 결론을 regime-dependent benefit으로
한정해야 한다.

Complete-only evidence는
`analysis/results/runs/rebuttal_realpde_legacy_4g/aggregation.json`,
`analysis/results/runs/rebuttal_realpde_legacy_4g_scale/seed39_scale_selection.json`,
`analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/aggregation.json`,
`analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/candidate_validation.json`
에 보존했다. Positive, mixed, adverse outcome을 모두 유지한다. Direct
수치, ranking, selected-candidate identity, response positioning은 author
release 전까지 `RESPONSE HOLD`이다. 수치표, paired-seed 결과,
오염 강도, reviewer별 전략을 포함한 내부 한국어 보고서는
`rebuttal/reports/legacy4g_experiment_ko/report.md`에 있다.

### 9.11 Controlled Cylinder·FSI·Foil 144-run 일반화 실험

Cylinder 한 trajectory에만 유리한 결과인지 확인하기 위해 RealPDEBench의
속도 `u,v` 관측을 제공하는 Controlled Cylinder, Fluid--Structure
Interaction(FSI), Foil 세 시나리오를 같은 사전 고정 절차로 실행했다.
각 데이터셋에 10%와 15% 큰 이상치를 넣고, seeds 40--42와 여덟 방법을
모두 실행했다. 데이터셋 3개 × 이상치 비율 2개 × seed 3개 × 방법 8개로
총 144회이며 144/144 실행, 48/48 세-seed group, 18/18 paired input block이
모두 엄격 검사를 통과했다. 이 가운데 사용자가 요청한 새 FSI·Foil
실행은 96회다.

| 데이터셋·이상치 | naPINN-MSE rMAE | 가장 낮은 비-naPINN rMAE | naPINN-MSE rMSE | 전체 rMSE 1위 |
| --- | ---: | ---: | ---: | --- |
| Cylinder 10% | **0.14937** | OrPINN 0.22579 | **0.19981** | naPINN-MSE |
| Cylinder 15% | **0.15220** | OrPINN 0.23876 | **0.20209** | naPINN-MSE |
| Foil 10% | **0.09968** | OrPINN 0.19554 | **0.15158** | naPINN-MSE |
| Foil 15% | **0.11783** | OrPINN 0.21482 | **0.16361** | naPINN-MSE |
| FSI 10% | **0.39948** | OrPINN 0.42476 | 0.64632 | OrPINN 0.60933 |
| FSI 15% | **0.41764** | OrPINN 0.43626 | 0.67973 | OrPINN 0.61541 |

naPINN-MSE는 평균 rMAE 6/6에서 1위였지만 평균 rMSE는 4/6에서만
1위였다. FSI의 rMSE 두 조건에서는 OrPINN이 1위이고 naPINN-MSE는
3위다. Foil 15%의 naPINN-q2.9는 seed 40에서 정상 관측까지 과도하게
버려 평균 rMAE 0.70040, rMSE 0.71768로 실패했다. 이 seed도 제외하지
않았다. 따라서 이 실험은 큰 이상치가 있는 실제 PIV에서 naPINN-MSE의
rMAE가 일관되게 낮았다는 주장은 지지하지만, 모든 실제 데이터와 모든
지표에서 항상 1위라는 주장은 지지하지 않는다. 여덟 방법의 전체 순위는
`rebuttal/reports/method_rankings_and_pinn_ebm_audit_ko.md`에 있다.

Combustion은 속도나 압력이 아니라 OH* chemiluminescence intensity 한
채널을 관측한다. 현재의 2-D incompressible Navier--Stokes PINN에 이
영상을 바로 넣는 것은 같은 benchmark가 아니다. 검증된 reacting-flow
PDE와 OH* intensity observation operator가 없으므로 성능 숫자를 만들지
않고 적용 범위의 한계로 기록했다.

근거는
`outputs/rebuttal/realpdebench_multidataset/aggregation.json`
(SHA-256
`88acccad036ff36fb1bf23b95099a233d068b48eec58f6c1b3e562418e97e7a8`)과
`outputs/rebuttal/realpdebench_multidataset/combustion_applicability.json`이다.

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
  `outputs/rebuttal/realpde_recovery_20260726/injected_piv_aggregation_strict.json`
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

합성·민감도·추가 RealPDEBench·구조화 PIV 102회 행렬은 엄격 집계까지
완료됐다. 공식 PINN-EBM active-code A와 paper-architecture B도 각각
5회 완료되어 strict aggregate를 통과했다. B의 평가 MAE 평균은 A보다
0.00245 낮지만 반복 표준편차보다 작은 차이이므로, 구조 차이의 결정적
성능 우위로 해석하지 않는다. 완료된 결과의 positive, mixed, adverse
outcome은 모두 유지한다.
