# 반박 추가 실험 통합 결과 보고서

작성일: 2026-07-28  
형식: Markdown  
상태: **최종 집계 반영** — 완료된 897개 full training run과, 별도 저자-code
protocol의 공식 PINN-EBM A/B 각 5개 순차 반복을 반영했다. A/B는 naPINN과
계산량을 맞춘 비교가 아니라 원저자 구현의 재현 감사다.

## 먼저 답하는 결론

현재 결과를 “이상치가 있으면 naPINN, 없으면 PINN-EBM”이라는 한 문장으로
일반화하면 정확하지 않다.

1. 실제 Cylinder PIV에 제출 코드의 기본 four-Gaussian 배경 잡음과
   10%·15% 큰 이상치를 함께 적용한 조건에서는 naPINN-MSE가 rMAE와
   rMSE 모두 1위였다.
2. 세 합성 PDE의 같은 종류 오염에서는 direct PINN-EBM이 아홉 조건 중
   여덟 조건의 평균 rMAE·rMSE에서 1위였다. 예외인 Allen--Cahn 15%에서는
   naPINN이 두 field 지표 모두 1위였다.
3. 큰 이상치를 넣지 않고 four-Gaussian 배경 잡음만 넣은 합성 세 조건은
   direct PINN-EBM이 모두 1위였다.
4. 인공 오염을 추가하지 않은 natural PIV에서는 LAD-PINN이 두 field
   지표 모두 1위였고 PINN-EBM은 하위권이었다. 이 자료도 측정 PIV이므로
   잡음 없는 물리 참값이라는 뜻은 아니다.
5. Controlled Cylinder·FSI·Foil의 144회 큰-이상치 실험에서는
   naPINN-MSE가 평균 rMAE 6/6에서 1위였지만 평균 rMSE는 4/6에서만
   1위였다. FSI rMSE 두 조건에서는 OrPINN이 1위였다.
6. PDE parameter 복원은 한 방법의 보편적 우위를 보이지 않았다. 아홉
   PDE·이상치비율 조건에서 PINN-EBM이 5회, naPINN이 2회,
   OrPINN-q1.9와 q2.9가 각각 1회 가장 낮은 parameter error를 보였다.
7. Allen--Cahn rejection cost 1.0은 5%·10%·15%에서 알려진 이상치의
   약 2--3%만 거부했고 field error가 크게 악화됐다. 설정 민감도는
   확인됐지만, 새 실험은 제출 표를 만든 실제 설정의 계보를 증명하지
   않는다.
8. Structured Cylinder PIV의 고정 Reynolds 수 비교에서는 LAD가 AR(1),
   10% sensor drift, spatial burst의 rMAE·rMSE에서 1위였고, naPINN이
   더 심한 20%·30% sensor drift의 두 지표에서 1위였다. 따라서 이
   실험도 naPINN의 보편적 우위보다 오염 강도·구조 의존성을 보여 준다.
9. 30% sensor drift에서 Reynolds 수까지 함께 학습하면 naPINN이 field
   rMAE·rMSE는 가장 낮았지만, Reynolds 수 상대오차는 PINN-EBM이 가장
   낮았다. 그러나 그 최저값도 51.5%이고 seed 변동도 커서 어느 방법도
   coefficient를 신뢰성 있게 복원했다고 볼 수 없다.
10. Structured MAD-PINN은 알려진 실패 관측을 많이 제거했지만 정상 관측도
    20.2--34.0% 제거했고, 두 단계 합계 60,000 PINN update를 사용했다.
    다섯 고정-Re 조건 중 어느 field 지표에서도 단독 1위가 아니었다.

따라서 rebuttal의 과학적 주장은 다음 수준으로 좁히는 것이 안전하다.

> naPINN은 명시적인 관측 선택과 rejection regularization을 제공하며,
> 사전 고정한 여러 실제 PIV 큰-이상치 stress test에서 특히 rMAE를
> 일관되게 낮췄다. 그러나 direct PINN-EBM, robust loss, OrPINN과의 순위는
> PDE, 관측 분포, 실제 시나리오, 평가 지표에 따라 달라지며 naPINN의
> 보편적 우위를 주장할 수는 없다.

## 용어를 풀어 쓴 설명

- **제출 코드의 기본 오염 강도 `(b,o)=(1,1)`**: `b`는 모든 학습 관측에
  넣는 배경 잡음의 크기, `o`는 일부 관측에 추가하는 큰 이상치의 크기다.
  둘 다 제출 코드의 기본값에서 바꾸지 않았다는 뜻이다. 수학적 참값이나
  결과를 본 뒤 고른 최적점을 뜻하지 않는다.
- **합성 오염 규칙을 실제 PIV에 적용**: 모델 weight나 parameter를
  옮기는 transfer learning이 아니다. 합성 데이터에 쓰던 오염 생성법만
  실제 PIV의 학습 측정에 적용했다. 평가용 held-out PIV는 바꾸지 않았다.
- **세 합성 PDE 기본 비교**: Allen--Cahn, Burgers, lambda--omega와
  5%·10%·15% 이상치 비율을 조합한 아홉 조건이다. 신경망의 내부 core나
  코드 kernel을 뜻하지 않는다.
- **natural PIV**: 인공 잡음과 이상치를 추가하지 않은 원래 PIV 측정이다.
  실제 측정 오차가 없는 참값이라는 뜻은 아니다.
- **rMAE·rMSE**: 평가 field의 절대오차 또는 제곱오차를 기준 field의
  크기로 나눈 상대오차다. 둘 다 낮을수록 좋고, rMSE가 큰 오차에 더
  민감하다.
- **PDE parameter error**: 학습한 물리 계수와 합성 데이터 생성에 사용한
  계수의 절댓값 차이다. 낮을수록 좋다.
- **direct PINN-EBM**: residual의 EBM likelihood를 data loss로 직접
  쓰며 naPINN의 관측 선택 gate와 rejection cost는 쓰지 않는 방법이다.

## 확정 결과로 계산한 실행 범위

현재 이 문서의 확정 수치에는 897개 full training run이 들어 있다.

| 실험 묶음 | 완료 run |
| --- | ---: |
| Cylinder PIV, 제출 코드 기본 오염 강도 | 48 |
| Cylinder PIV, seed-39 오염 강도 탐색 | 144 |
| Cylinder PIV, 새 seed 후보 재확인 | 72 |
| Natural PIV robust-loss 및 no-gate/PINN-EBM | 27 |
| 합성 background-noise-only | 18 |
| Allen--Cahn cost 1.0 | 9 |
| 세 합성 PDE 5%·10%·15% 기본 비교 | 162 |
| 합성 EMA·cost·noise-family·selector supplement | 108 |
| PINN-EBM weight calibration 및 held-out | 27 |
| Controlled Cylinder·FSI·Foil | 144 |
| 35k PINN-update baseline | 27 |
| Synthetic MAD-PINN | 9 |
| Structured PIV fixed-Re·inverse-Re·MAD | 102 |
| **합계** | **897** |

Smoke test와 공식 PINN-EBM A/B는 서로 다른 저자-code protocol이므로 이
합계에서 제외했다. 공식 A와 B는 각각 순차 5회를 완료했고, 두 결과는
별도의 strict aggregate로 검증됐다. RealPDEBench 144회 중 새 non-Cylinder
FSI·Foil은 96회다.

## 조건별 모든 방법 순위

왼쪽 방법일수록 세 seed 평균 field error가 낮다. 이 표의 자연 PIV는
현재 저자 방침상 reviewer-facing 근거에서는 제외하지만, 내부 결론을
왜곡하지 않기 위해 함께 남긴다. 숫자까지 포함한 상세 표는
`rebuttal/reports/method_rankings_and_pinn_ebm_audit_ko.md`가 단일
authoritative 순위표다.

| 자료·오염 조건 | 평균 rMAE 순위 | 평균 rMSE 순위 |
| --- | --- | --- |
| Cylinder PIV 기본 강도, 10% | naMSE → naQ → OrPINN → naL1 → EBM50 → LAD → EBM1 → MSE | naMSE → OrPINN → naQ → naL1 → LAD → EBM50 → EBM1 → MSE |
| Cylinder PIV 기본 강도, 15% | naMSE → naQ → naL1 → OrPINN → LAD → EBM50 → EBM1 → MSE | naMSE → naQ → OrPINN → naL1 → LAD → EBM50 → EBM1 → MSE |
| 합성 Allen--Cahn, 5% | EBM → naPINN → Or-q2.9 → Or-q1.9 → LAD → MSE | EBM → naPINN → Or-q2.9 → Or-q1.9 → LAD → MSE |
| 합성 Allen--Cahn, 10% | EBM → naPINN → Or-q2.9 → LAD → Or-q1.9 → MSE | EBM → naPINN → Or-q2.9 → LAD → Or-q1.9 → MSE |
| 합성 Allen--Cahn, 15% | naPINN → EBM → Or-q2.9 → LAD → Or-q1.9 → MSE | naPINN → EBM → Or-q2.9 → LAD → Or-q1.9 → MSE |
| 합성 Burgers, 5% | EBM → naPINN → Or-q2.9 → Or-q1.9 → LAD → MSE | EBM → naPINN → Or-q2.9 → Or-q1.9 → LAD → MSE |
| 합성 Burgers, 10% | EBM → naPINN → Or-q2.9 → LAD → Or-q1.9 → MSE | EBM → naPINN → Or-q2.9 → LAD → Or-q1.9 → MSE |
| 합성 Burgers, 15% | EBM → naPINN → Or-q2.9 → LAD → Or-q1.9 → MSE | EBM → naPINN → Or-q2.9 → LAD → Or-q1.9 → MSE |
| 합성 lambda--omega, 5% | EBM → naPINN → Or-q1.9 → Or-q2.9 → MSE → LAD | EBM → naPINN → Or-q1.9 → Or-q2.9 → LAD → MSE |
| 합성 lambda--omega, 10% | EBM → naPINN → Or-q1.9 → Or-q2.9 → LAD → MSE | EBM → naPINN → Or-q1.9 → Or-q2.9 → LAD → MSE |
| 합성 lambda--omega, 15% | EBM → naPINN → Or-q2.9 → Or-q1.9 → LAD → MSE | EBM → naPINN → Or-q2.9 → Or-q1.9 → LAD → MSE |
| 합성 Allen--Cahn, 배경 잡음만 | EBM → naPINN | EBM → naPINN |
| 합성 Burgers, 배경 잡음만 | EBM → naPINN | EBM → naPINN |
| 합성 lambda--omega, 배경 잡음만 | EBM → naPINN | EBM → naPINN |
| Natural PIV | LAD → naL1 → OrPINN → naQ → naMSE → MSE → EBM50 → EBM10 → EBM1 | LAD → OrPINN → naL1 → naQ → naMSE → MSE → EBM50 → EBM10 → EBM1 |
| Controlled Cylinder, 10% | naMSE → naL1 → OrPINN → LAD → naQ → EBM50 → EBM1 → MSE | naMSE → naL1 → OrPINN → LAD → naQ → EBM50 → MSE → EBM1 |
| Controlled Cylinder, 15% | naMSE → naL1 → OrPINN → EBM50 → EBM1 → naQ → LAD → MSE | naMSE → naL1 → OrPINN → naQ → LAD → EBM50 → EBM1 → MSE |
| Foil, 10% | naMSE → naQ → naL1 → OrPINN → LAD → EBM1 → EBM50 → MSE | naMSE → naQ → naL1 → OrPINN → LAD → EBM1 → EBM50 → MSE |
| Foil, 15% | naMSE → naL1 → OrPINN → EBM1 → LAD → EBM50 → MSE → naQ | naMSE → naL1 → OrPINN → EBM1 → LAD → EBM50 → MSE → naQ |
| FSI, 10% | naMSE → OrPINN → LAD → naL1 → naQ → MSE → EBM50 → EBM1 | OrPINN → LAD → naMSE → naL1 → MSE → naQ → EBM1 → EBM50 |
| FSI, 15% | naMSE → OrPINN → naL1 → LAD → naQ → EBM50 → MSE → EBM1 | OrPINN → LAD → naMSE → naL1 → naQ → EBM50 → EBM1 → MSE |
| Structured PIV, AR(1) | LAD → OrPINN → naPINN → MAD → EBM → MSE | LAD → OrPINN → naPINN → MAD → MSE → EBM |
| Structured PIV, sensor drift 10% | LAD → naPINN → MAD → OrPINN → EBM → MSE | LAD → naPINN → MAD → OrPINN → MSE → EBM |
| Structured PIV, sensor drift 20% | naPINN → LAD → MAD → EBM → OrPINN → MSE | naPINN → LAD → MAD → EBM → OrPINN → MSE |
| Structured PIV, sensor drift 30% | naPINN → LAD → MAD → EBM → OrPINN → MSE | naPINN → LAD → MAD → EBM → OrPINN → MSE |
| Structured PIV, spatial burst | LAD → OrPINN → MAD → naPINN → MSE → EBM | LAD → OrPINN → MSE → naPINN → MAD → EBM |
| Structured PIV, sensor drift 30%, Re도 학습 | naPINN → LAD → EBM → MSE | naPINN → LAD → EBM → MSE |

`naMSE`, `naL1`, `naQ`는 각각 MSE, L1, q=2.9 reconstruction loss를
쓰는 naPINN이다. `EBM1`, `EBM10`, `EBM50`은 direct PINN-EBM의 PDE-loss
weight가 1, 10, 50인 설정이다. 각 캠페인이 실제로 실행한 방법만 순위에
넣었다.

## PDE parameter 복원

아래는 세 합성 PDE × 세 이상치 비율에서 parameter 절대오차가 낮은
순서다. Field 순위와 parameter 순위는 서로 다를 수 있다.

| PDE·이상치 | Parameter-error 순위 |
| --- | --- |
| Allen--Cahn 5% | Or-q1.9 0.001391 → EBM 0.001865 → LAD 0.002543 → Or-q2.9 0.002598 → naPINN 0.002682 → MSE 0.004113 |
| Allen--Cahn 10% | EBM 0.002418 → Or-q1.9 0.002620 → LAD 0.003022 → naPINN 0.003195 → Or-q2.9 0.003490 → MSE 0.005918 |
| Allen--Cahn 15% | EBM 0.002149 → naPINN 0.002753 → LAD 0.003121 → Or-q2.9 0.003209 → MSE 0.003553 → Or-q1.9 0.006063 |
| Burgers 5% | Or-q2.9 0.000791 → EBM 0.000799 → LAD 0.001064 → naPINN 0.002031 → MSE 0.002082 → Or-q1.9 0.002902 |
| Burgers 10% | EBM 0.000675 → Or-q2.9 0.000801 → LAD 0.001480 → naPINN 0.002152 → MSE 0.002691 → Or-q1.9 0.003417 |
| Burgers 15% | EBM 0.000646 → Or-q2.9 0.001226 → LAD 0.001813 → naPINN 0.002718 → Or-q1.9 0.005150 → MSE 0.005664 |
| lambda--omega 5% | naPINN 0.011708 → EBM 0.012657 → Or-q1.9 0.019542 → Or-q2.9 0.037652 → LAD 0.054712 → MSE 0.056996 |
| lambda--omega 10% | naPINN 0.011360 → EBM 0.012217 → Or-q1.9 0.035254 → Or-q2.9 0.044144 → LAD 0.066667 → MSE 0.152275 |
| lambda--omega 15% | EBM 0.010165 → naPINN 0.013586 → Or-q2.9 0.052341 → Or-q1.9 0.060873 → LAD 0.083705 → MSE 0.327853 |

이 표는 reviewer가 요구한 Burgers viscosity와 lambda--omega beta까지
모두 포함한다. 다만 새 current-code, three-seed rebuttal evidence이며 제출
당시 ten-trial 표의 bitwise 재현이라고 부르면 안 된다.

### 실제 PIV에서 Reynolds 수까지 함께 학습한 결과

아래 실험은 30%의 지속적인 sensor drift를 넣은 Cylinder PIV에서 field와
Reynolds 수를 동시에 학습했다. Metadata Reynolds 수는 10,031이다.
표의 상대오차는 각 seed의 절대 상대오차를 먼저 계산한 뒤 평균한 값이므로,
평균 Reynolds 수 하나로 다시 계산한 값과 다를 수 있다.

| 방법 | 학습한 Re, 평균 ± 표본 표준편차 | Re 상대오차 | field rMAE | field rMSE |
| --- | ---: | ---: | ---: | ---: |
| PINN-EBM | 7,681.08 ± 5,927.76 | **0.51537** | 0.28567 | 0.36947 |
| LAD-PINN | 3,634.24 ± 1,650.42 | 0.63770 | 0.23303 | 0.35739 |
| naPINN | 2,306.53 ± 550.97 | 0.77006 | **0.22825** | **0.31499** |
| MSE-PINN | 1,331.81 ± 111.03 | 0.86723 | 0.37221 | 0.49041 |

PINN-EBM의 seed별 Re는 14,260.66, 2,757.24, 6,025.34로 매우 크게
흔들렸다. 따라서 “PINN-EBM이 parameter reconstruction을 해결했다”고도,
“naPINN이 parameter를 가장 잘 복원했다”고도 말할 수 없다. 이 stress
test의 정직한 결론은 naPINN이 field를 가장 잘 복원했지만 모든 방법의
coefficient 식별은 실패했다는 것이다. 실제 PIV와 nominal 2-D
Navier--Stokes 사이의 model discrepancy가 섞이므로, 이 값은 깨끗한
물리 상수보다 현재 관측·모형 조합의 effective coefficient에 가깝다.

## Allen--Cahn rejection cost

15% 조건의 다섯 cost는 동일한 EMA update weight 0.05와 seeds 40--42를
사용했다.

| Rejection cost | rMAE | rMSE | 알려진 이상치 거부율 |
| ---: | ---: | ---: | ---: |
| 0.10 | **0.08847** | **0.09283** | 99.30% |
| 0.30 | 0.09133 | 0.09631 | 99.25% |
| 0.50 | 0.09202 | 0.09483 | 99.29% |
| 0.70 | 0.09886 | 0.09869 | 99.28% |
| 1.00 | 0.86890 | 0.72211 | 2.12% |

Cost 1.0을 5%·10%·15%에 별도로 적용한 9회에서도 rMAE가
0.26088·0.56828·0.86890으로 악화됐다. 이 결과는 cost 1.0이 현재
설정에서 gate를 사실상 모두-accept 쪽으로 보낸다는 것을 보여 준다.
그러나 제출 논문의 Allen--Cahn 표가 실제로 어떤 config에서 생성됐는지는
원 run lineage가 없어 아직 해결되지 않았다.

## 다른 reviewer 요청의 결과와 상태

| Reviewer 지적 | 상태 | 확인된 내용 |
| --- | --- | --- |
| 실제 또는 반실제 데이터 | 완료 | Cylinder 기본 강도와 Controlled Cylinder·FSI·Foil 144회 완료 |
| Cylinder 외 RealPDEBench | 완료 | FSI·Foil 96회 새 실행 완료; Combustion은 관측변수/PDE 불일치 한계 감사 완료 |
| Closest-prior PINN-EBM | 완료 | 동일-input current comparator와 weight 1·10·50, 공식 active-code A와 paper-architecture B 각 5회 strict aggregate 완료 |
| 모든 PDE parameter | 완료 | Allen epsilon, Burgers viscosity, lambda--omega beta 전 조건 보고 |
| Allen만 다른 rejection cost | 민감도 완료, 계보 미해결 | Cost 1.0의 부정적 결과 확인; 제출 run의 실제 설정은 모름 |
| EMA 민감도 | 완료 | update weight 0.01·0.05·0.10 완료; 한 값의 보편적 최적 주장 불가 |
| 다른 noise 분포 | 완료 | Gaussian·Laplace·Student-t·four-Gaussian에서 승자가 달라짐 |
| 기본 통계 전처리 | 완료 | fixed quantile·learnable threshold·synthetic MAD와 structured MAD 15회 완료 |
| Compute fairness | update-count 완료, wall-clock 한계 | 35k baseline 27회 완료; 격리 hardware wall-clock은 없음 |
| Correlated noise·drift·sensor failure | 완료 | Structured PIV fixed-Re 75회, inverse-Re 12회, MAD 15회 strict 집계 완료 |
| Real inverse parameter | 완료, 부정적 결과 | naPINN field 1위이나 Re 오차는 77.0%; 모든 방법의 Re 오차가 51.5% 이상 |
| HMC B-PINN | 한계 | 충실한 HMC 구현과 계산 protocol이 없어 새 성능을 주장하지 않음 |
| Heteroscedastic scale | 미실행 한계 | 사전 고정 scale function이 없음 |
| u/v 상관 잡음 | 미실행 한계 | 사전 고정 covariance matrix가 없음 |
| Gate initializer | 코드·논문 불일치 | YAML 필드가 likelihood gate에 전달되지 않아 저자 판단 필요 |
| Replacement outlier | 코드·논문 불일치 | 논문 문구는 replace, 현재 실행 코드는 positive additive offset |

따라서 “reviewer가 지적한 모든 경우가 실험으로 해결됐다”고 답하면 안
된다. 완료 실험, 현재 실행, 새로운 정의 없이는 정당하게 실행할 수 없는
한계를 분리해야 한다.

## Compute와 MAD-PINN 결과

MSE·LAD·OrPINN에 35,000 full PINN update를 준 27회 비교에서도 세 PDE
15% 조건의 field error는 30,000 PINN update와 5,000 estimator-only
update를 쓴 direct PINN-EBM·naPINN보다 높았다. 이는 update 수에 대한
보수적 비교이지 shared server wall-clock의 공정한 순위가 아니다.

Synthetic MAD-PINN은 30,000 LAD update 후 MAD screen을 만들고, 남은
관측으로 30,000 MSE update를 추가한다. 총 60,000 PINN update다.

| PDE | MAD rMAE | MAD rMSE | 이상치 거부 | 정상 관측 거부 |
| --- | ---: | ---: | ---: | ---: |
| Allen--Cahn | 0.34354 | 0.28913 | 98.52% | 18.95% |
| Burgers | 0.24198 | 0.23555 | 98.58% | 16.67% |
| lambda--omega | 0.12735 | 0.14349 | 98.42% | 10.57% |

큰 이상치를 잘 제거하지만 정상 관측도 적지 않게 버리고 계산량이 두
배이므로 compute-matched 우위로 표현하지 않는다.

Structured PIV의 MAD-PINN도 seed-matched 30,000-update LAD checkpoint에서
시작해 30,000-update masked-MSE 단계를 추가하므로 총 60,000 PINN
update다. 아래 비율은 알려진 실패 관측과 알려진 정상 관측 중 각각
제거한 비율이다.

| Structured 조건 | naPINN rMAE/rMSE | MAD rMAE/rMSE | naPINN 실패/정상 제거 | MAD 실패/정상 제거 |
| --- | ---: | ---: | ---: | ---: |
| AR(1) | 0.15036 / 0.21544 | 0.15158 / 0.23113 | 61.41% / 2.96% | 94.30% / 27.51% |
| Sensor drift 10% | 0.15538 / 0.23260 | 0.15738 / 0.23950 | 86.15% / 4.03% | 91.57% / 30.42% |
| Sensor drift 20% | **0.17113 / 0.25266** | 0.17346 / 0.25574 | 90.66% / 8.38% | 91.43% / 25.46% |
| Sensor drift 30% | **0.21797 / 0.30886** | 0.24848 / 0.36792 | 92.90% / 32.67% | 80.63% / 20.16% |
| Spatial burst | 0.15318 / **0.22605** | **0.14851** / 0.22623 | 100.00% / 4.62% | 100.00% / 34.04% |

MAD는 실패 관측을 많이 잡지만 정상 데이터도 훨씬 많이 버린다. 20%와
30% persistent drift에서는 naPINN이 두 field 지표 모두 낮았고,
spatial burst에서는 MAD의 rMAE만 근소하게 낮았다. 그러나 전체 여섯
방법을 포함하면 spatial burst의 1위는 LAD다. 따라서 이 결과를 MAD나
naPINN의 보편적 우위로 읽으면 안 된다.

## 결과 저장 위치와 검증 가능한 파일

각 current runner의 개별 실행 폴더에는 가능한 경우 `config.yaml`,
`run_metadata.json`, `train_history.jsonl`, 중간 `step_*.pt`, `final.pt`,
`metrics.json`을 저장한다. 공식 PINN-EBM 저자 코드는 원래 최종
`state_dict`를 저장하지 않으므로 A/B는 소스를 바꾸지 않고 공식 result
pickle과 추출 `metrics.json`을 보존한다.

| 근거 | 위치 | 상태 또는 SHA-256 |
| --- | --- | --- |
| 합성 162회 | `outputs/rebuttal/synthetic_recovery_20260726/aggregation_strict.json` | 162/162; `473b07059b568eea24657cf79dbe42be530700415d0a14be4e4488bd62bdc7cc` |
| Allen cost 1.0 | `outputs/rebuttal/allen_cost1_recovery_20260726/aggregation_strict.json` | 9/9; `59aab31b781270afb0a6c4e87cca9759daf84035c4d87bce9141fb3d32f186d5` |
| 합성 supplement | `outputs/rebuttal/synthetic_supplement_recovery_20260726/aggregation_strict.json` | 108/108; `4c0994090f5cf7fb359e8176c59f524c78f23cef4d8b50cb96c4bd832c79300b` |
| 35k baseline | `outputs/rebuttal/synthetic_compute35_recovery_20260726/aggregation_strict.json` | 27/27; `8569d5808b73595b26c1ba2a22fd3d6c1b9f1e3c3295fef391f2bfd83bee7738` |
| Synthetic MAD | `outputs/rebuttal/synthetic_mad_recovery_20260726/aggregation_strict.json` | 9/9; `c89b56ae9966b07465aa2e1dff67b12bb4ef1f008650f3311e33c924bae57836` |
| Multi-RealPDE | `outputs/rebuttal/realpdebench_multidataset/aggregation.json` | 144/144; `88acccad036ff36fb1bf23b95099a233d068b48eec58f6c1b3e562418e97e7a8` |
| Structured PIV | `outputs/rebuttal/realpde_recovery_20260726/injected_piv_aggregation_strict.json` | 102/102, 34/34 groups; `ca8d97d8951ec7ea8664ff3d58f6c0e04e814194063cce60d0e8414854c6de42` |
| 공식 PINN-EBM A | `outputs/rebuttal/pinn_ebm_upstream/runs/A_upstream_active_0b74f6f/seed000_nrun5_official_session_20260726/metrics.json` | 5/5 순차 반복 완료; metrics `94725942835f9006ee24db679a6edf93a81306388127911bb9ae6047c6f006ee` |
| 공식 PINN-EBM B | `outputs/rebuttal/pinn_ebm_upstream/runs/B_paper_spec_8x20/seed000_nrun5_official_session_20260726/metrics.json` | 5/5 순차 반복 완료; strict A/B aggregate에 포함 |
| 공식 PINN-EBM A/B aggregate | `outputs/rebuttal/pinn_ebm_upstream/aggregation_strict.json` | strict complete; `40a8f9f871dbbeb02fef4bf65b2639b66a8afa9832293e436ba1614825501878` |
| 전체 상태 | `outputs/status/reviewer_recovery_summary.json` | complete-only monitor |

기존 cross-server Legacy-4G 실행은 이미 checksum과 경로가 연결되어 있어
evidence chain을 깨지 않기 위해 `analysis/results/runs/` 아래에 그대로
보존했다. 새 실행은 모두 `outputs/` 아래에 저장한다.

## 권장 rebuttal 전략

1. 가장 먼저 사전 고정 실제 PIV 기본 강도와 multi-RealPDE 결과를
   제시한다. 실제 PIV 위에 통제된 이상치를 추가한 stress test이며
   held-out PIV도 noise-free truth가 아니라고 명시한다.
2. naPINN의 주장을 “모든 조건에서 최고”가 아니라 “명시적 관측 선택을
   제공하고, tested real-PIV gross-outlier 조건에서 rMAE가 일관되게
   낮았다”로 좁힌다.
3. 합성 아홉 조건 중 direct PINN-EBM 8회 우세, FSI rMSE의 OrPINN 우세,
   Foil naQ seed 실패, cost 1.0 실패를 숨기지 않는다. 불리한 결과를 먼저
   통제하면 closest-prior와 cherry-picking 지적에 더 정직하게 답할 수
   있다.
4. PINN-EBM은 current comparator와 공식 저자-code A/B를 분리한다.
   Current comparator는 같은 자료·backbone·PINN update를 맞춘 목적함수
   비교이고, 공식 A/B는 원 논문 과제의 구현 충실도 검사다. 서로 다른
   데이터의 숫자로 naPINN 승패를 직접 매기지 않는다.
5. PDE parameter 표는 completeness 답변으로 사용하되 보편적 parameter
   우위를 주장하지 않는다. 실제 PIV inverse-Re에서는 naPINN이 field
   오차는 가장 낮지만 Re 오차는 77.0%이고, PINN-EBM의 가장 낮은 Re
   오차도 51.5%다. 이를 model discrepancy와 effective coefficient의
   한계로 그대로 보고한다.
6. Cost 1.0 결과는 제출 설정을 정당화하는 자료가 아니라 paper/config
   불일치가 중요하다는 민감도 증거로 사용한다. 제출 run lineage가 없다는
   사실을 밝힌다.
7. 35k와 MAD 결과에는 update budget을 붙인다. Shared GPU 시간으로 speed
   superiority를 주장하지 않는다.
8. Structured PIV에서는 약한·상관 오염의 LAD 우세와 20%·30% persistent
   drift의 naPINN 우세를 함께 제시한다. MAD는 60k update와 정상 관측
   제거율을 반드시 붙인다.
9. 실험 protocol이 동결되지 않은 heteroscedastic·u/v covariance·HMC는
   급하게 숫자를 만들지 않고 limitation과 future protocol로 답한다.

## 공식 PINN-EBM A/B 재현의 최종 기록

공식 active code A(4 hidden layers × width 30)의 평가 MAE는
0.07467 ± 0.01141이고, 논문에 적힌 구조만 적용한 B(8 × 20)는
0.07222 ± 0.00873이다. B의 평균이 0.00245 낮지만 각 반복의 표준편차보다
작으므로, 이 5회 결과만으로 구조가 결정적으로 더 낫다고 주장하지 않는다.
B는 학습 시간이 10,057.83 ± 732.56초로 A의 6,817.53 ± 214.29초보다
길었다. 두 variant는 같은 공식 Cylinder wake 자료를 쓰지만 naPINN과의
직접 성능 순위나 compute-matched 비교에는 사용하지 않는다.
