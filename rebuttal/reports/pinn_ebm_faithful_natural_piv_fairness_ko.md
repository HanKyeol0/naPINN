# PINN-EBM 원본 재현, natural PIV robust loss, 합성 이상치 비교 감사

작성일: 2026-07-28  
용도: rebuttal 내부 의사결정 및 후속 실험 추적  
문서 형식: Markdown  
상태: 완료된 897개 full training run의 엄격 집계와 공식 PINN-EBM A/B 각 5회 strict aggregate 반영

## 기술 요약

현재 확인된 핵심 결론은 다음과 같다.

1. **현재 naPINN 저장소의 direct PINN-EBM 비교 코드는 PINN-EBM의 핵심 학습 원리와 gradient 흐름을 올바르게 구현하고 있지만, 저자 GitHub의 Navier–Stokes 실험을 그대로 복제한 코드는 아니었다.** 데이터셋, PINN 출력 방식, 네트워크 크기, EBM 크기, 학습 횟수, batch 크기, residual 정규화와 적분 방식이 모두 달랐다.
2. **합성 PDE 실험에서 PINN-EBM이 이상치가 있는데도 1위였던 이유를 “PINN-EBM만 더 쉬운 데이터를 받았다”로 설명할 근거는 없다.** seed 40의 3개 PDE와 5%·10%·15% 이상치 조건을 다시 생성해 비교한 결과, MSE-PINN, direct PINN-EBM, naPINN이 받은 좌표, 잡음이 들어간 관측값, 큰 이상치 위치가 조건별로 모두 같았다.
3. **다만 현재 합성 이상치 생성법은 PINN-EBM이 다루기 좋은 형태다.** 잡음 분포가 공간과 시간에 따라 변하지 않는 하나의 전역 분포이고, 큰 이상치도 좌표와 무관하게 무작위로 선택되며, 선택된 위치에는 항상 양의 방향으로 큰 값을 더한다. 따라서 residual 전체가 “좁은 주 분포와 한쪽의 넓은 꼬리 또는 보조 분포가 섞인 형태”가 된다. residual의 전체 확률분포를 직접 학습하는 EBM이 이 혼합 구조를 표현하기에 유리하다.
4. **natural PIV에서 naPINN에 robust loss를 결합하면 naPINN-MSE보다 field 오차가 낮아졌지만, 같은 robust loss를 gate 없이 쓴 방법보다 좋아지지는 않았다.** 전체 rMAE 1위는 gate 없는 LAD-PINN 0.112108이고 naPINN-L1은 0.112416으로 2위다. rMSE도 LAD-PINN과 OrPINN q2.9가 naPINN-L1보다 낮다. gate가 99% 이상 관측을 유지했으므로, natural PIV에서의 개선은 gate보다 robust reconstruction loss의 효과로 해석해야 한다.
5. **큰 이상치를 전혀 넣지 않고 four-Gaussian 배경 잡음만 사용한 새 합성 실험에서도 direct PINN-EBM이 세 PDE 모두 1위였다.** 따라서 기존 합성 실험의 PINN-EBM 1위를 “큰 이상치에 특이하게 잘 대응했기 때문”이라고만 설명할 수 없다. 현재 synthetic protocol에서는 복잡한 전역 배경 잡음의 likelihood를 직접 최적화하는 것 자체가 이미 큰 이점이다.
6. **Allen–Cahn에서 제출 부록에 적힌 rejection cost 1.0은 좋은 설정이 아니었다.** 새 서버의 9회 복구에서 5%·10%·15% rMAE가 0.26088·0.56828·0.86890으로 악화됐고, 알려진 이상치의 약 2–3%만 거부했다. 이 결과는 설정 민감도를 보여 주지만 제출 당시 표를 생성한 실제 설정이 무엇이었는지는 증명하지 못한다.
7. **새 destination-server 합성 core와 supplement는 분포와 PDE에 따라 1위가 바뀐다는 결론을 더 분명하게 만들었다.** Four-Gaussian에 큰 이상치를 더한 9조건 중 direct PINN-EBM이 8조건에서 평균 rMAE·rMSE 1위였고, Allen–Cahn 15%에서는 naPINN이 두 지표 모두 1위였다. Gaussian은 naPINN-MSE, Laplace는 naPINN-LAD, Student-t rMAE는 LAD, four-Gaussian은 direct PINN-EBM이 가장 낮았다. 어느 한 방법의 잡음 분포 무관 우위를 지지하지 않는다.
8. **Controlled Cylinder·FSI·Foil의 144회 실제 PIV stress test에서는 naPINN-MSE가 평균 rMAE 6/6 조건에서 1위였지만, 평균 rMSE는 4/6에서만 1위였다.** FSI의 rMSE 두 조건에서는 OrPINN이 1위였다. 따라서 추가 실제 PIV 결과도 보편적 우위가 아니라 지표와 시나리오에 따른 조건부 장점을 보여 준다.
9. **Structured Cylinder PIV 102회 엄격 집계에서는 LAD가 다섯 조건 중 세 조건의 두 field 지표에서 1위이고, naPINN은 20%와 30% persistent drift에서 1위였다.** 30% inverse-Re에서 naPINN의 field 오차는 가장 낮았지만 Re 상대오차는 77.0%였고, 모든 방법의 Re 상대오차가 51.5% 이상이었다. 이 결과는 field reconstruction과 PDE coefficient identification을 분리해서 해석해야 함을 보여 준다.

이 결과만으로 기존 비교가 부정하다고 볼 수는 없다. 각 방법이 같은 자료와 같은 계산량을 받은 **목적함수 비교**로서는 정당하다. 그러나 이 결과를 “PINN-EBM보다 naPINN이 모든 종류의 큰 이상치에 강하다”는 보편적 주장에 사용할 수는 없다. rebuttal에서는 실제 PIV의 강한 오염 조건에서 나타난 naPINN의 장점과, 합성 PDE에서 나타난 direct PINN-EBM의 장점을 함께 보고하고 주장 범위를 좁히는 편이 안전하다.

## 완료된 실험의 최종 결과

이 절에서 **최종 결과**라고 부르는 수치는 필요한 seed가 모두 끝났고, 실패한 실행이 없으며, 완결된 실행만 받아들이는 집계 검사를 통과한 값이다. 공식 PINN-EBM A/B도 각각 다섯 번의 순차 실행과 result-pickle hash 검사를 통과한 별도 최종 증거다.

현재 이 문서가 엄격 집계로 다루는 현재 저장소 full training run은 총 897개다. 기존 795개에 structured-PIV fixed-Re 75개, inverse-Re 12개, MAD-PINN 15개를 더한 수다. 짧은 smoke test와 서로 다른 공식 저자-code protocol인 A/B는 이 합계에서 제외한다. RealPDEBench 144개 중 사용자가 요청한 새 non-Cylinder FSI·Foil 실행은 96개다.

| 실험 묶음 | 완료 상태 | 최종 결론 | rebuttal에서의 취급 |
| --- | --- | --- | --- |
| 실제 PIV, 사전 고정 Legacy-4G 기본 강도 | 48/48 완료, 실패 0 | 10%와 15% 큰 이상치 조건 모두 naPINN-MSE가 rMAE와 rMSE 1위 | 가장 강한 실제 PIV 근거. 다만 실제 PIV 위에 통제된 인공 오염을 추가한 실험이라고 밝혀야 한다. |
| 실제 PIV, 오염 강도 탐색 | 144/144 완료, 실패 0 | 오염 강도에 따라 순위가 바뀌었다. 모든 설정에서 naPINN이 이긴 것은 아니다. | 결과를 본 뒤 고른 설정을 주 근거로 쓰지 않고, 민감도와 경계조건을 설명하는 데 사용한다. |
| 실제 PIV, 새 seed 후보 재확인 | 72/72 완료, 실패 0 | 강한 `(b,o)=(2,2)` 조건에서는 naPINN-MSE가 3/3 seed에서 두 field 지표 모두 우세했다. 약한 이상치 `(1,0.5)` 조건은 rMAE와 rMSE의 1위가 달랐다. | 긍정적 결과와 혼합 결과를 함께 보존한다. |
| 추가 인공 오염 없는 natural PIV | 27/27 완료, 실패 0 | LAD-PINN이 rMAE와 rMSE 1위다. naPINN-L1은 rMAE 2위·rMSE 3위이고, direct PINN-EBM weight 50은 두 지표 모두 7위다. | 중요한 내부 진단이지만, 현재 author 지침상 reviewer-facing 주 근거에서는 제외한다. |
| 합성 PDE, four-Gaussian 배경 잡음만 사용하고 큰 이상치 0% | 18/18 완료, 실패 0 | Allen–Cahn, Burgers, lambda–omega 모두 direct PINN-EBM이 naPINN보다 낮은 rMAE와 rMSE를 보였다. | PINN-EBM의 합성 실험 장점이 큰 이상치 처리에서만 생긴 것이 아님을 보여 주는 경계조건이다. |
| 합성 PDE, four-Gaussian 배경 잡음과 5%·10%·15% 큰 이상치 | 162/162 완료, 54/54 groups | direct PINN-EBM이 8/9 조건의 평균 rMAE·rMSE 1위이고, Allen–Cahn 15%에서는 naPINN이 두 지표 모두 1위다. | 새 생성자료의 auditable rebuttal 증거이며, 제출 당시 ten-trial 표의 bitwise 재현이라고 부르지 않는다. |
| 합성 EMA·rejection cost·noise family·selector supplement | 108/108 완료, 36/36 groups | EMA와 cost 결과는 국소 민감도를 보이고, noise family와 selector 결과는 PDE·분포에 따른 혼합 순위를 보였다. | 유리한 값만 고르지 않고 전 범위와 부정적 cost-1.0 결과를 함께 보고한다. |
| 합성 direct PINN-EBM PDE weight | calibration 9/9, held-out 18/18 완료 | Allen–Cahn field는 weight 10, Burgers field는 50이 가장 낮고, lambda–omega는 지표별 순서가 다르다. Allen–Cahn weight 50은 크게 실패했다. | Weight 1·10·50 전체를 보고하고 하나의 tuned weight만 고르지 않는다. |
| Allen–Cahn, rejection cost 1.0 | 9/9 완료, 실패 0 | 5%·10%·15%에서 이상치의 약 2–3%만 거부했고 오염률 증가에 따라 rMAE가 0.26088·0.56828·0.86890으로 악화됐다. | 부정적 민감도 결과로 공개한다. 제출 당시 설정 lineage와는 분리한다. |
| Controlled Cylinder·FSI·Foil, 10%·15% 큰 이상치 | 144/144 완료, 48/48 groups | naPINN-MSE가 평균 rMAE 6/6에서 1위, 평균 rMSE 4/6에서 1위다. FSI rMSE는 OrPINN이 1위이고 Foil 15% naPINN-q2.9의 seed 실패도 포함했다. | 유리한 데이터셋이나 seed만 고르지 않고 여섯 조건의 전체 순위를 공개한다. Combustion은 관측변수와 PDE가 맞지 않아 적용 불가 한계로 보고한다. |
| 35,000 PINN-update baseline | 27/27 완료, 9/9 groups | MSE·LAD·OrPINN에 35k PINN update를 줘도 세 PDE 15%에서 direct PINN-EBM과 naPINN보다 field 오차가 컸다. | update 수에 대한 보수적 비교로만 사용한다. hardware-isolated wall-clock match라고 부르지 않는다. |
| Synthetic MAD-PINN | 9/9 완료, 3/3 groups | 알려진 이상치 약 98.4–98.6%를 제거했지만 정상 scalar도 약 10.6–19.0% 제거했다. | 60k PINN update의 비동일 계산량임을 함께 밝힌다. |
| Structured PIV fixed-Re·inverse-Re·MAD | 102/102 완료, 34/34 groups | LAD는 AR(1), 10% drift, spatial burst에서 두 field 지표 1위이고, naPINN은 20%·30% drift에서 1위다. inverse-Re는 모든 방법이 metadata Re를 크게 놓쳤다. | structured noise coverage와 coefficient 식별 실패를 함께 보고하며, MAD의 60k update를 명시한다. |
| 합성 입력 공정성 감사 | 3개 PDE × 3개 큰 이상치 비율, 총 9조건 통과 | 세 방법이 같은 좌표, 원래 값, 최종 오염값, 큰 이상치 위치를 받았다. 특정 방법만 더 쉬운 입력을 받은 흔적은 없었다. | 성능 실험이 아니라 비교 입력의 동일성을 확인한 감사 결과다. |
| 공식 PINN-EBM 재현 A/B | A/B 각 5/5 완료, strict aggregate 완료 | A/B 평가 MAE는 각각 0.07467 ± 0.01141, 0.07222 ± 0.00873이다. | 평균 차이는 반복 표준편차보다 작다. 구조의 결정적 성능 우위나 naPINN과의 직접 우열로 해석하지 않는다. |

사전 고정한 실제 PIV 기본 오염 강도에서 가장 중요한 최종 수치는 다음과 같다. 이 설정은 강도 탐색 결과를 본 뒤 고른 점이 아니라, 제출 코드의 오염 크기를 그대로 적용하기로 미리 고정한 기준점이다.

| 큰 이상치 행 비율 | naPINN-MSE rMAE | 가장 강한 비-naPINN rMAE | naPINN-MSE rMSE | 가장 강한 비-naPINN rMSE |
| ---: | ---: | ---: | ---: | ---: |
| 10% | **0.15517 ± 0.00237** | OrPINN q=2.9: 0.21426 ± 0.00415 | **0.21532 ± 0.00338** | OrPINN q=2.9: 0.25026 ± 0.00363 |
| 15% | **0.16023 ± 0.00060** | OrPINN q=2.9: 0.23106 ± 0.00508 | **0.22266 ± 0.00304** | OrPINN q=2.9: 0.26514 ± 0.00374 |

숫자는 seeds 40–42의 평균 ± 표본 표준편차이며 낮을수록 좋다. 10%와 15% 조건 모두에서 naPINN-MSE는 세 seed 각각에서 OrPINN q=2.9보다 rMAE와 rMSE가 낮았다. 그러나 이 결론의 범위는 **RealPDEBench Cylinder PIV, 제출 코드와 같은 four-Gaussian 배경 잡음, 양의 큰 오차 추가, 사전 고정 기본 강도**에 한정된다.

완료된 실험 전체를 한 문장으로 정리하면 다음과 같다.

> 실제 PIV에 사전 고정한 강한 Legacy-4G 오염을 넣은 조건에서는 naPINN-MSE가 가장 좋았지만, 추가 인공 오염이 없는 natural PIV에서는 LAD-PINN이 가장 좋았다. 합성 PDE의 전역 four-Gaussian 배경 잡음에서는 큰 이상치를 넣지 않아도 direct PINN-EBM이 가장 좋았다. 따라서 어느 한 방법의 보편적 우위가 아니라 자료와 오염 분포에 따른 조건부 우위로 해석해야 한다.

최종 집계의 원본 근거는 다음 파일에 있다.

- 실제 PIV Legacy-4G 기본 강도: `analysis/results/runs/rebuttal_realpde_legacy_4g/aggregation.json`
- 실제 PIV 오염 강도 탐색: `analysis/results/runs/rebuttal_realpde_legacy_4g_scale/seed39_scale_selection.json`
- 실제 PIV 새 seed 재확인: `analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/aggregation.json`
- natural PIV naPINN robust loss: `outputs/rebuttal/natural_piv/aggregation.json`
- natural PIV baseline과 direct PINN-EBM: `outputs/rebuttal/natural_piv_baselines/aggregation.json`
- 합성 배경 잡음 전용 실험: `outputs/rebuttal/synthetic_background_only/aggregation.json`
- Allen–Cahn rejection-cost-1.0 복구: `outputs/rebuttal/allen_cost1_recovery_20260726/aggregation_strict.json`
- 합성 PDE 5%·10%·15% core:
  `outputs/rebuttal/synthetic_recovery_20260726/aggregation_strict.json`
- 합성 EMA·rejection·noise·selector supplement:
  `outputs/rebuttal/synthetic_supplement_recovery_20260726/aggregation_strict.json`
- 합성 PINN-EBM seed-39 weight calibration:
  `outputs/rebuttal/synthetic_pinn_ebm_weight_calibration_20260726/aggregation_strict.json`
- 합성 PINN-EBM held-out weight 10·50:
  `outputs/rebuttal/synthetic_pinn_ebm_weight_heldout_20260726/aggregation_strict.json`
- Controlled Cylinder·FSI·Foil 144-run:
  `outputs/rebuttal/realpdebench_multidataset/aggregation.json`
- RealPDEBench Combustion 적용 가능성 감사:
  `outputs/rebuttal/realpdebench_multidataset/combustion_applicability.json`
- 35,000 PINN-update baseline:
  `outputs/rebuttal/synthetic_compute35_recovery_20260726/aggregation_strict.json`
- Synthetic MAD-PINN:
  `outputs/rebuttal/synthetic_mad_recovery_20260726/aggregation_strict.json`
- Structured PIV fixed-Re·inverse-Re·MAD:
  `outputs/rebuttal/realpde_recovery_20260726/injected_piv_aggregation_strict.json`
- 합성 입력 공정성 감사: `outputs/audits/synthetic_pinn_ebm_fairness/seed_40.json`

## 이 문서에서 사용하는 용어

- **natural PIV**: 인위적인 잡음이나 큰 이상치를 새로 추가하지 않은 실제 PIV 측정자료를 뜻한다. “잡음이 전혀 없는 참값”이라는 뜻은 아니다. 실제 측정과정에서 생긴 오차는 원래 자료에 남아 있을 수 있다.
- **배경 잡음**: 큰 이상치를 선택하기 전에 모든 학습 관측에 적용하는 비교적 일상적인 측정오차다. 현재 합성 실험에서는 네 Gaussian 분포가 섞인 분포를 사용한다.
- **큰 이상치 또는 gross outlier**: 배경 잡음보다 훨씬 큰 오차를 일부 측정행에 추가한 것이다. 현재 합성 설정에서는 선택된 측정행에 배경 잡음 표준편차의 3–10배인 양의 값을 더한다.
- **residual**: 관측값에서 PINN 예측값을 뺀 차이다. residual이 0에 가까우면 예측과 관측이 가깝다.
- **scalar residual**: 속도처럼 여러 성분이 있는 residual을 성분별 숫자로 펼쳤을 때의 각 숫자를 뜻한다. 예를 들어 한 측정행의 `u,v` 오차는 두 개의 scalar residual이 된다.
- **EBM**: residual이 어떤 값에서 자주 나타나고 어떤 값에서 드문지를 유연한 신경망으로 학습하는 확률모형이다. Gaussian 하나로 모양을 고정하지 않으므로 비대칭 분포나 여러 봉우리가 있는 분포도 표현할 수 있다.
- **direct PINN-EBM**: EBM이 계산한 residual의 음의 로그우도를 PINN의 data loss로 직접 사용하는 방법이다. naPINN의 신뢰도 gate는 사용하지 않는다.
- **gate**: 각 관측을 얼마나 신뢰할지 0과 1 사이의 가중치로 정하는 장치다. 낮은 가중치는 그 관측이 PINN 학습에 미치는 영향을 줄인다.
- **robust loss**: 큰 residual이 생겨도 손실과 gradient가 MSE만큼 급격히 커지지 않도록 설계한 관측 재구성 손실이다. 이 문서에서는 L1과 q-Gaussian 기반 손실을 뜻한다.
- **hash 또는 checksum**: 파일이나 배열의 내용을 짧은 문자열로 요약한 지문이다. 두 hash가 같으면 비교한 내용이 같았는지 매우 강하게 확인할 수 있다.
- **smoke test**: 전체 학습을 하기 전에 입력과 출력 shape, forward 계산, backward gradient가 정상인지 아주 짧게 확인하는 실행이다. smoke test의 성능 숫자는 논문 결과로 사용하지 않는다.

## 서로 다른 세 종류의 PINN-EBM 실험을 분리한다

“원본 PINN-EBM과 완벽하게 일치”라는 표현은 논문과 실행 중인 공식 코드가 서로 다른 네트워크 구조를 적고 있기 때문에 하나의 설정으로 충족할 수 없다. 그래서 다음 세 실험을 명시적으로 분리했다.

| 구분 | 무엇을 확인하는 실험인가 | 주요 설정 | 결과를 비교할 수 있는 범위 |
| --- | --- | --- | --- |
| A. 공식 코드 그대로 | GitHub commit `0b74f6f`에서 실제 실행되는 코드를 그대로 재현 | 공식 Cylinder wake 수치해석 자료, PINN 4 hidden layers × width 30, EBM 3 hidden layers × width 5와 dropout 0.5, PINN update 100,000회, 공식 반복 5회 | 저자 GitHub 코드가 이 서버에서 내는 결과를 확인할 수 있다. 현재 natural PIV 또는 naPINN 수치와 직접 순위를 매기면 안 된다. |
| B. 논문 네트워크 구조 | 논문 본문에 적힌 PINN 구조를 공식 코드에 적용 | A와 같지만 PINN만 8 hidden layers × width 20으로 변경 | 논문에 적힌 구조와 공식 실행 코드의 구조 차이가 결과에 영향을 주는지 확인할 수 있다. A와 별개로 보고해야 한다. |
| C. 현재 과제에서 계산량을 맞춘 비교 | 같은 natural PIV 자료와 같은 PINN 계산량에서 direct EBM likelihood와 naPINN gate를 비교 | 현재 5 hidden layers × width 80 PINN, data batch 2,048, PDE collocation batch 8,192, PINN update 30,000회, EBM 단독 update 5,000회 | 현재 논문의 방법 비교와 rebuttal에 사용할 수 있다. “공식 저자 코드 재현”이라고 부르면 안 된다. |

A와 B는 같은 공식 자료를 사용하고, 하나의 seed 0 난수 흐름 안에서 공식 코드의 세 방법 순서 `[0, 3, 2]`를 유지하며 각 방법을 다섯 번 학습한다. 각 실행은 총 150만 PINN update를 포함하므로 여러 시간이 아니라 하루 이상이 걸릴 수 있는 장기 실험이다.

공식 commit에는 실행에 필요한 `prop_noise` 값이 정의되어 있지 않다. 소스 코드는 수정하지 않고, 이전 공식 commit과 논문의 additive-noise 식에 맞춰 `prop_noise=0`을 입력 설정으로 공급했다. 이 한 가지 실행 가능성 보완은 결과 문서에 명시한다.

이 문서에서 A를 “공식 코드 그대로”라고 부르는 범위는 **소스 commit, 실행 설정, 데이터, seed 흐름**이다. 공식 저장소에는 원 저자 실행환경의 Python, PyTorch, CUDA 버전을 고정하는 requirements 또는 environment 파일이 없다. 따라서 현재 서버의 package와 RTX A6000에서 실행한 결과를 원 저자의 하드웨어와 dependency까지 bit 단위로 같은 환경이라고 주장할 수는 없다. 모든 현재 package와 hardware 정보는 각 `run_metadata.json`에 저장한다.

현재 상태는 다음과 같다.

| 실험 | 상태 | 저장 위치 |
| --- | --- | --- |
| A. 공식 active code 4×30 | 공식 5회 완료; metrics와 result pickle 보존 | `outputs/rebuttal/pinn_ebm_upstream/runs/A_upstream_active_0b74f6f/seed000_nrun5_official_session_20260726/` |
| B. 논문 구조 8×20 | 공식 5회 완료; strict aggregate 포함 | `outputs/rebuttal/pinn_ebm_upstream/runs/B_paper_spec_8x20/seed000_nrun5_official_session_20260726/` |
| A/B 구성요소 forward·backward 검사 | 통과; 성능 근거로 사용하지 않는 smoke test | `outputs/rebuttal/pinn_ebm_upstream/smoke/` |
| A/B 코드·자료·설정 감사 | 통과 | `outputs/rebuttal/pinn_ebm_upstream/audits/` |
| A/B strict aggregate | strict complete | `outputs/rebuttal/pinn_ebm_upstream/aggregation_strict.json` |

2026-07-27 16:44 UTC snapshot에서 A는 1,500,000/1,500,000
update-equivalent와 공식 5회를 모두 완료했다. A의 5회 평균 ± 표본
표준편차는 lambda1 0.98861 ± 0.00646, lambda2 0.01100 ± 0.00057,
평가 MAE 0.07467 ± 0.01141, validation NLL 0.06020 ± 0.10658,
PDE mean-squared residual 0.000488 ± 0.000035다. A의 `metrics.json` SHA-256은
`94725942835f9006ee24db679a6edf93a81306388127911bb9ae6047c6f006ee`다.
공식 코드의 metric key 이름에는 `rmse`가 들어가지만 실제 계산은 평균
절댓값이므로 여기서는 평가 MAE라고 부른다.

최종적으로 B도 1,500,000/1,500,000 update-equivalent와 공식 5회를
완료했다. B의 5회 평균 ± 표본 표준편차는 lambda1 0.99522 ± 0.00229,
lambda2 0.00940 ± 0.00060, 평가 MAE 0.07222 ± 0.00873, validation NLL
0.04712 ± 0.03738, PDE mean-squared residual 0.000364 ± 0.000036이다.
A의 해당 값은 0.98861 ± 0.00646, 0.01100 ± 0.00057, 0.07467 ± 0.01141,
0.06020 ± 0.10658, 0.000488 ± 0.000035다. B의 평균 평가 MAE는 0.00245
낮지만 반복 변동보다 작고 학습 시간은 A보다 길다(10,057.83초 대
6,817.53초). 따라서 A/B는 구조 차이가 결과를 크게 뒤집는다는 증거가
아니며, 공식 코드와 논문 명시 구조를 모두 실행했다는 재현 근거다.

공식 코드는 학습 이력과 평가 배열은 저장하지만 최종 PyTorch `state_dict`는 저장하지 않는다. A/B의 소스 실행을 수정하지 않는 것이 우선이므로, 이 두 실행은 공식 result pickle과 추출한 `metrics.json`을 보존하고 모델 checkpoint를 새로 끼워 넣지 않는다. 반대로 현재 저장소 runner로 실행하는 C, natural PIV, 합성 실험은 `final.pt`, 중간 checkpoint, `metrics.json`, 실행 설정과 학습 이력을 모두 `outputs/` 아래에 저장한다.

## 현재 direct PINN-EBM과 원본 구현의 일치·불일치

현재 direct PINN-EBM에서 residual은 `관측값 - PINN 예측값`이다. 먼저 MSE로 PINN을 준비하고, 준비된 PINN의 residual을 사용해 EBM만 별도로 초기화한다. 그 다음 공동학습에서는 EBM이 계산한 residual의 음의 로그우도와 PDE residual을 더해 최소화한다. 이때 residual을 PINN에서 분리하지 않기 때문에 EBM likelihood의 gradient가 PINN까지 전달된다. 이 핵심 경로는 원 방법과 일치한다.

그러나 실험 전체가 원본과 같은 것은 아니다.

| 항목 | 원 논문 또는 공식 코드 | 현재 C 비교 코드 | 해석 |
| --- | --- | --- | --- |
| 자료 | `cylinder_nektar_wake.mat` 수치해석 wake | RealPDEBench Cylinder 실제 PIV | 서로 다른 과제다. |
| PINN 출력 | streamfunction `psi`와 pressure를 출력하고 미분하여 속도 `u,v`를 계산 | `u,v,p`를 직접 출력하고 연속방정식 residual도 사용 | 물리 제약을 표현하는 방식이 다르다. |
| PINN 크기 | 논문 8×20, active code 4×30 | 5×80 | 어느 원본 구조와도 같지 않다. |
| EBM 크기 | 3×5, dropout 0.5 | width 32, dropout 없음 | 현재 EBM의 표현력이 더 크고 regularization은 더 약하다. |
| EBM 정규화 적분 | batch residual 범위에 맞춘 1,001점 동적 격자 | 표준화한 residual의 고정 구간에서 수치 적분 | likelihood의 수치적 정의가 다르다. |
| 학습량 | PINN 100,000 update; 10,000회 뒤 EBM 2,000회 초기화 | PINN 30,000 update; 5,000회 뒤 EBM 5,000회 초기화 | 현재 rebuttal의 계산량 통일 규칙에 맞춘 별도 protocol이다. |
| batch | data 200, PDE 100 | data 2,048, PDE 8,192 | 최적화 중 gradient 변동성과 PDE 기여가 달라진다. |
| PDE 가중치 | Navier–Stokes PINN-EBM에서 50 | 1, 10, 50을 모두 실행 | 결과를 보고 좋은 값만 고르지 않고 전체 민감도를 보고한다. |

따라서 C를 설명할 때는 “PINN-EBM의 direct likelihood 목적함수를 현재 과제와 계산량에 맞춰 비교한 구현”이라고 써야 한다. “저자 코드를 완벽하게 재현한 구현”이라고 쓰면 안 된다.

## 합성 실험에서 PINN-EBM만 유리한 데이터를 받지는 않았다

seed 40에 대해 Allen–Cahn, Burgers, lambda–omega의 5%·10%·15% 큰 이상치 조건을 다시 생성했다. 각 조건에서 MSE-PINN, direct PINN-EBM, naPINN이 받은 다음 네 배열의 hash를 비교했다.

- 학습 좌표
- 큰 이상치를 넣기 전의 깨끗한 값
- 배경 잡음과 큰 이상치를 모두 넣은 최종 관측값
- 큰 이상치를 넣은 측정행의 위치

총 9개 조건 모두에서 네 hash의 고유값 개수는 각각 1이었다. 즉, 같은 조건 안에서 세 방법이 받은 자료는 같았다. 또한 5% 큰 이상치 위치는 10% 위치의 부분집합이고, 10% 위치는 15% 위치의 부분집합이었다. 이상치 비율이 커질 때 기존 이상치 위치를 유지한 채 새 위치를 추가하는 구조도 의도대로 작동했다.

감사 결과는 `outputs/audits/synthetic_pinn_ebm_fairness/seed_40.json`에 저장했다.

이 검사는 **방법 간 입력 자료가 같은지**를 확인한다. seeds 41–42까지 모든 부동소수점 배열을 다시 감사한 것은 아니므로, seed 40의 완전한 pairing 검사를 세 seed 전체의 결과로 과장하지 않는다. 다만 runner가 seed만 바꿔 같은 생성 경로를 사용하므로, 특정 방법에만 쉬운 표본을 제공하는 코드 분기는 발견되지 않았다.

학습 중 EBM 갱신도 확인했다. direct PINN-EBM은 PINN과 EBM parameter를 하나의 새 Adam optimizer에 넣어 공동학습한다. naPINN은 EBM을 그 optimizer에 넣지 않는 대신, 각 공동학습 batch에서 EBM의 내부 optimizer를 별도로 한 번 실행한 뒤 갱신된 density score로 gate를 계산한다. 따라서 naPINN의 EBM만 초기화 뒤 고정되는 비대칭은 없다. 오히려 direct PINN-EBM은 EBM 초기화가 끝난 뒤 새 Adam optimizer를 만들기 때문에 초기화 단계의 Adam momentum을 이어받지 않고, naPINN의 내부 EBM optimizer는 그 상태를 이어받는다. 이 차이는 direct PINN-EBM에 명백히 유리한 설정으로 보기 어렵다.

## 현재 합성 이상치 분포는 direct EBM likelihood에 유리하다

이 절에서 말하는 “유리하다”는 부정한 특혜가 있다는 뜻이 아니다. 방법의 가정과 시험 분포의 구조가 잘 맞는다는 뜻이다.

### 배경 잡음이 하나의 전역 분포다

배경 잡음의 분포는 좌표나 시간에 따라 달라지지 않는다. 예를 들어 소용돌이의 중심과 경계, 초기 시간과 후기 시간에서 서로 다른 잡음 분포를 사용하지 않는다. 모든 위치의 scalar residual을 한 줄로 모아 하나의 밀도를 학습하는 EBM의 가정과 잘 맞는다.

### 큰 이상치가 모두 양의 방향이다

큰 이상치가 선택된 측정행에는 배경 잡음 표준편차의 3배에서 10배 사이인 **양의 값**을 더한다. seed 40 감사에서 세 PDE의 5%·10%·15% 조건 모두 큰 이상치의 전체 오차가 양수였다.

여기에는 논문 문장과 실행 코드의 차이도 있다. 현재 논문은 선택된 관측값을 높은 크기의 uniform 표본으로 “대체한다”고 설명하지만, 실제 실행 코드는 먼저 four-Gaussian 배경 잡음을 만든 뒤 선택된 위치의 잡음에 양의 값을 **추가한다**. Burgers와 lambda–omega에서는 같은 측정행의 `u,v` 두 성분 모두에 양의 추가량을 넣되, 두 성분의 크기는 각각 따로 뽑는다. 기존 보고 수치를 코드 실행의 결과로 사용할 때는 “positive additive gross offsets”라고 설명하는 것이 정확하다.

15% 조건의 대표 수치는 다음과 같다.

| PDE | 배경 잡음만 받은 값의 오차 표준편차 | 큰 이상치 행의 평균 전체 오차 | 큰 이상치 오차가 양수인 비율 |
| --- | ---: | ---: | ---: |
| Allen–Cahn | 0.1486 | 0.9722 | 100% |
| Burgers | 0.2016 | 1.3331 | 100% |
| lambda–omega | 0.2737 | 1.8029 | 100% |

큰 이상치의 평균 오차는 세 PDE 모두 배경 잡음 표준편차의 약 6.5배다. 이것은 residual 분포에 한쪽으로 치우친 두 번째 넓은 덩어리 또는 긴 꼬리를 만든다. EBM은 residual이 Gaussian이라고 고정하지 않고 이 비대칭 혼합 모양 자체를 학습할 수 있다.

### PINN-EBM의 큰 residual 영향은 MSE와 다르다

MSE에서는 residual이 두 배가 되면 gradient의 크기도 대체로 두 배가 된다. 그래서 큰 이상치가 예측기를 강하게 끌어당긴다.

PINN-EBM에서는 data loss가 residual 자체의 제곱이 아니라, EBM이 학습한 확률밀도의 음의 로그값이다. 예측기에 전달되는 힘은 residual의 절댓값이 아니라 **학습된 log-density 곡선의 기울기**로 결정된다. EBM이 큰 양의 residual 구간을 실제 잡음 분포의 보조 꼬리로 학습하면, residual 값이 크더라도 그 구간에서 예측기를 끌어당기는 힘이 작아질 수 있다.

그러므로 “PINN-EBM은 큰 이상치를 명시적으로 제거하지 않으니 반드시 이상치에 약해야 한다”는 전제는 맞지 않는다. 명시적인 제거는 없지만, 학습한 비정규 residual 분포가 큰 residual의 영향력을 자동으로 줄일 수 있다.

### naPINN gate에는 추가적인 선택 문제가 있다

naPINN은 EBM 점수를 보고 각 측정을 얼마나 신뢰할지 정하는 gate를 추가한다. 이 gate가 이상치를 정확히 낮은 가중치로 보내면 도움이 된다. 그러나 다음 두 오류 가능성도 생긴다.

- 실제로 유용한 측정을 잘못 거부한다.
- 이상치를 충분히 낮은 가중치로 보내지 못한다.

또한 측정을 거부하는 데에는 rejection cost가 있으므로, naPINN은 “관측값을 이용할지”와 “거부 비용을 낼지”를 동시에 최적화한다. direct PINN-EBM에는 이 이산적인 선택에 가까운 추가 부담이 없다. 이전 15% 합성 결과에서 direct EBM의 이상치 구분 AUROC가 약 0.994였고 naPINN gate보다 소폭 높았다는 점도, 이 분포에서는 direct EBM이 큰 이상치 구간을 이미 잘 구분했음을 보여 준다.

## 공정한 비교를 위해 바꾸는 것은 알고리즘이 아니라 실험 분리와 보고 방식이다

현재 관측자료 pairing이 맞고, direct likelihood의 gradient에도 심각한 오류가 발견되지 않았다. 이런 상황에서 PINN-EBM이 1위라는 이유만으로 PINN-EBM 구현을 약하게 바꾸는 것은 정당하지 않다. 필요한 수정은 다음과 같다.

1. **공식 재현과 현재 과제 비교를 별도 표로 보고한다.** A/B는 공식 Cylinder wake 재현이고, C는 natural PIV의 계산량 통일 비교다.
2. **PDE 가중치 1, 10, 50을 모두 보고한다.** 결과를 본 뒤 가장 좋은 값 하나만 PINN-EBM 대표값으로 선택하지 않는다.
3. **큰 이상치가 없는 배경 잡음 전용 합성 실험을 추가한다.** 이 실험은 PINN-EBM의 장점이 four-Gaussian 배경 분포 학습에서 이미 생기는지, 큰 이상치가 추가될 때 새로 생기는지를 분리한다.
4. **추가 오염이 없는 natural PIV에서 같은 계산량의 기본 방법과 비교한다.** MSE, LAD, OrPINN q=2.9와 direct PINN-EBM의 PDE 가중치 전체를 seeds 40–42에서 실행한다.
5. **naPINN과 robust reconstruction loss의 결합을 같은 natural PIV split에서 실행한다.** naPINN-MSE, naPINN-L1, naPINN-q2.9를 seeds 40–42에서 비교한다.
6. **논문의 이상치 생성 설명과 코드를 일치시킨다.** 현재 결과를 유지한다면 논문과 rebuttal에는 “관측값 대체”가 아니라 “배경 잡음에 양의 큰 오차를 추가”했다고 써야 한다. 정말 대체형 이상치를 주장하려면 별도 생성법으로 다시 실험해야 한다.
7. **보편적 gross-outlier 강건성 주장은 피한다.** 후속 본문 또는 보충자료에서는 양·음 방향 이상치, 좌표에 따라 발생확률이 달라지는 이상치, 연속된 공간·시간 구간이 통째로 손상되는 이상치를 별도 stress test로 추가하는 것이 더 설득력 있다.

3–5번은 모두 `outputs/` 아래에서 완료됐으며, 세 seed 전체와 사전 고정한 PINN-EBM 가중치 1·10·50을 빠짐없이 집계했다.

## natural PIV의 naPINN과 robust loss 실험

여기서 natural PIV는 인위적인 four-Gaussian 잡음이나 큰 이상치를 새로 넣지 않은 실제 PIV 학습자료다. 실제 측정자료이므로 참 이상치 label은 없다. 따라서 rMAE와 rMSE는 계산할 수 있지만, “어느 관측이 이상치인지 정확히 맞혔다”는 AUROC나 이상치 제거율은 계산하면 안 된다.

세 조합은 다음과 같다.

| 방법 | naPINN gate 뒤에 사용하는 관측 재구성 손실 | 기대하는 성질 |
| --- | --- | --- |
| naPINN-MSE | residual 제곱 | 작은 오차를 강하게 줄이지만 남은 큰 residual의 영향을 크게 받을 수 있다. |
| naPINN-L1 | residual 절댓값 | 큰 residual의 영향이 MSE보다 천천히 증가한다. |
| naPINN-q2.9 | q-Gaussian 기반 손실 | 중앙부를 맞추면서 두꺼운 꼬리 residual의 영향을 완화하는 절충을 시도한다. |

세 방법 모두 같은 PIV 파일, 같은 공간 분리, fixed Reynolds number, 5×80 PINN, data batch 2,048, PDE batch 8,192, PINN update 30,000회와 EBM 단독 update 5,000회를 사용한다. seeds 40–42의 9개 실행이 모두 끝나야 최종 순위를 기록한다.

9개 실행과 strict aggregation이 모두 완료됐다. 낮을수록 좋은 두 field 오차에서 순위는 동일했다.

| 순위 | 조합 | rMAE, seeds 40–42 평균 ± 표본 표준편차 | rMSE, seeds 40–42 평균 ± 표본 표준편차 |
| ---: | --- | ---: | ---: |
| 1 | naPINN-L1 | **0.112416 ± 0.001610** | **0.186754 ± 0.001744** |
| 2 | naPINN-q2.9 | 0.122562 ± 0.001513 | 0.189499 ± 0.002136 |
| 3 | naPINN-MSE | 0.139353 ± 0.001868 | 0.197751 ± 0.001248 |

L1과 q2.9의 개선 방향은 세 seed에서 모두 같았다. 같은 seed의 naPINN-MSE에서 뺀 paired difference는 다음과 같다. 음수이면 robust loss 조합의 field 오차가 더 낮다는 뜻이다.

| 비교 | rMAE 차이 평균 ± 표본 표준편차 | rMSE 차이 평균 ± 표본 표준편차 |
| --- | ---: | ---: |
| naPINN-L1 − naPINN-MSE | **−0.026937 ± 0.001667** | **−0.010997 ± 0.000656** |
| naPINN-q2.9 − naPINN-MSE | **−0.016791 ± 0.000536** | **−0.008252 ± 0.000889** |

그러나 이 결과를 “gate가 natural PIV 이상치를 잘 제거했다”로 해석하면 안 된다. L1 조합은 최종 scalar 관측의 평균 99.38%를 유지했고, q2.9 조합은 99.15%, MSE 조합은 사실상 100%를 유지했다. 실제 PIV에 참 이상치 label도 없으므로 gate의 이상치 판별 성능을 계산할 수 없다.

또한 field 오차 개선과 물리 residual 사이에 trade-off가 나타났다.

| 조합 | 최종 유지 비율 | Navier–Stokes 운동량 residual RMS | 연속방정식 residual RMS |
| --- | ---: | ---: | ---: |
| naPINN-MSE | 1.0000 | **0.02629** | **0.02822** |
| naPINN-q2.9 | 0.9915 | 0.04118 | 0.06123 |
| naPINN-L1 | 0.9938 | 0.04415 | 0.07443 |

즉, naPINN-L1은 공간적으로 분리한 PIV 평가값에는 가장 가깝지만, 학습이 끝난 모델의 PDE residual은 naPINN-MSE보다 크다. 평가자료도 잡음 없는 수치해석 참값이 아니라 실제 PIV 측정이므로, 이 결과는 “측정 field 예측이 개선됐다”는 근거이지 “물리적으로 가장 정확한 해가 확정됐다”는 근거는 아니다.

gate 없는 matched baseline까지 합치면 전체 순위는 다음과 같다.

| rMAE 순위 | 방법 | 평균 ± 표본 표준편차 |
| ---: | --- | ---: |
| 1 | LAD-PINN | **0.112108 ± 0.001450** |
| 2 | naPINN-L1 | 0.112416 ± 0.001610 |
| 3 | OrPINN q2.9 | 0.121429 ± 0.001211 |
| 4 | naPINN-q2.9 | 0.122562 ± 0.001513 |
| 5 | naPINN-MSE | 0.139353 ± 0.001868 |
| 6 | MSE-PINN | 0.139760 ± 0.002817 |
| 7 | direct PINN-EBM, PDE weight 50 | 0.228312 ± 0.006576 |
| 8 | direct PINN-EBM, PDE weight 10 | 0.237386 ± 0.088476 |
| 9 | direct PINN-EBM, PDE weight 1 | 0.303340 ± 0.037082 |

| rMSE 순위 | 방법 | 평균 ± 표본 표준편차 |
| ---: | --- | ---: |
| 1 | LAD-PINN | **0.183531 ± 0.002101** |
| 2 | OrPINN q2.9 | 0.185683 ± 0.001520 |
| 3 | naPINN-L1 | 0.186754 ± 0.001744 |
| 4 | naPINN-q2.9 | 0.189499 ± 0.002136 |
| 5 | naPINN-MSE | 0.197751 ± 0.001248 |
| 6 | MSE-PINN | 0.198223 ± 0.002260 |
| 7 | direct PINN-EBM, PDE weight 50 | 0.314401 ± 0.020966 |
| 8 | direct PINN-EBM, PDE weight 10 | 0.469826 ± 0.265598 |
| 9 | direct PINN-EBM, PDE weight 1 | 0.750336 ± 0.091382 |

같은 reconstruction loss끼리 gate의 추가 효과를 비교하면 다음과 같다. 양수는 naPINN gate 조합이 더 나빴다는 뜻이다.

| paired 비교 | rMAE 평균 차이 | rMSE 평균 차이 | 해석 |
| --- | ---: | ---: | --- |
| naPINN-L1 − LAD-PINN | +0.000308 | +0.003223 | 평균적으로 gate 없는 LAD가 두 지표 모두 더 낮다. |
| naPINN-q2.9 − OrPINN q2.9 | +0.001133 | +0.003816 | gate 없는 OrPINN이 두 지표 모두 더 낮다. |
| naPINN-MSE − MSE-PINN | −0.000406 | −0.000472 | naPINN 평균이 아주 조금 낮지만 seed별 방향이 섞이고 차이가 매우 작다. |

따라서 natural PIV에서 field 오차를 줄인 주된 요인은 **L1 또는 q-Gaussian reconstruction loss**다. gate가 추가적인 성능 향상을 만들었다는 근거는 없다. 그렇다고 gate가 항상 해롭다고 일반화할 수도 없다. 이 natural PIV에는 통제된 gross outlier가 없고 gate도 거의 모든 관측을 유지했기 때문이다.

direct PINN-EBM은 natural PIV에서 1위가 아니었다. 세 PDE 가중치 중 50이 가장 안정적이었지만 rMAE 7위, rMSE 7위였다. 가중치 1에서는 likelihood를 낮추는 동안 PDE residual이 크게 증가했고, 가중치 10은 seed에 따른 변동이 매우 컸다. 가중치 50의 운동량 residual RMS는 0.02558로 낮았지만 field rMAE는 0.22831이었다. 즉, 강한 PDE 가중치가 물리식 residual을 억제했어도 held-out PIV field 예측이 가장 좋아지지는 않았다.

| 항목 | 현재 상태 또는 위치 |
| --- | --- |
| 전체 실행 | 9/9 완료, 실패 0 |
| 개별 모델, 중간 checkpoint, 평가값 | `outputs/rebuttal/natural_piv/runs/` |
| 개별 log | `outputs/rebuttal/natural_piv/logs/` |
| job 상태 | `outputs/rebuttal/natural_piv/status/` |
| 검증된 최종 집계 | `outputs/rebuttal/natural_piv/aggregation.json`, `aggregation.csv` |
| gate 없는 baseline과 PINN-EBM 전체 가중치 실행 | `outputs/rebuttal/natural_piv_baselines/runs/` |
| baseline 검증 집계 | `outputs/rebuttal/natural_piv_baselines/aggregation.json`, `aggregation.csv` |

## 큰 이상치 없는 합성 배경 잡음 전용 실험

이 실험은 기존 합성 PDE에서 four-Gaussian 배경 잡음은 그대로 두고, 큰 이상치를 넣는 측정행 수만 0으로 만든다. direct PINN-EBM과 naPINN을 Allen–Cahn, Burgers, lambda–omega, seeds 40–42에서 비교한다. 총 18개 실행이다.

이 실험의 목적은 “PINN-EBM이 큰 이상치를 잘 처리해서 이기는가?”만 묻는 것이 아니다. 오히려 “큰 이상치가 없어도 복잡한 four-Gaussian 배경분포를 직접 학습하는 것만으로 이미 유리한가?”를 분리해 확인한다.

18개 실행과 세 seed 집계가 모두 완료됐다. rMAE와 rMSE의 순위는 세 PDE에서 모두 같았다.

| PDE | rMAE 순위, 평균 ± 표본 표준편차 | rMSE 순위, 평균 ± 표본 표준편차 |
| --- | --- | --- |
| Allen–Cahn | 1. **direct PINN-EBM 0.09165 ± 0.01163** → 2. naPINN 0.09769 ± 0.00915 | 1. **direct PINN-EBM 0.09011 ± 0.01160** → 2. naPINN 0.09651 ± 0.00639 |
| Burgers | 1. **direct PINN-EBM 0.03737 ± 0.00180** → 2. naPINN 0.07537 ± 0.00755 | 1. **direct PINN-EBM 0.05055 ± 0.00045** → 2. naPINN 0.08935 ± 0.01026 |
| lambda–omega | 1. **direct PINN-EBM 0.04473 ± 0.00461** → 2. naPINN 0.07287 ± 0.00631 | 1. **direct PINN-EBM 0.05886 ± 0.00542** → 2. naPINN 0.08955 ± 0.00761 |

이 결과는 질문 3에 직접 답한다. direct PINN-EBM의 synthetic 1위는 gross outlier가 있을 때만 생기는 현상이 아니다. gross outlier를 0으로 만들어도 세 PDE 모두 direct PINN-EBM이 더 낮은 평균 field 오차를 보였다. Burgers에서는 rMAE가 naPINN보다 약 50% 낮았고, lambda–omega에서는 약 39% 낮았다. Allen–Cahn의 차이는 약 6%로 더 작았고 seed 변동성도 두 방법의 평균 차이보다 크므로, Allen–Cahn의 작은 차이는 강한 우위 증거로 과장하지 않는다.

naPINN gate가 최종적으로 거부한 scalar 측정의 평균 비율은 Allen–Cahn 0.019%, Burgers 0.030%, lambda–omega 0.030%에 불과했다. 큰 이상치가 없으므로 거의 모든 관측을 유지하는 방향은 자연스럽지만, 이 경우 naPINN은 사실상 MSE 재구성에 가까워지고 direct EBM likelihood의 이점을 얻지 못한다.

따라서 기존 synthetic 결과를 설명할 때는 다음처럼 말하는 것이 정확하다.

> 현재 합성 protocol은 좌표와 무관한 전역 four-Gaussian residual 분포를 사용한다. direct PINN-EBM은 이 비정규 likelihood를 data objective로 직접 사용하므로, gross outlier가 없어도 naPINN-MSE보다 유리하다. gross outlier가 추가된 뒤에도 이 기본 장점이 유지된 것이다.

이 결론은 현재 서버에서 새로 생성한 Burgers와 lambda–omega 자료를 사용한 **이번 0% gross 캠페인 내부 비교**에 근거한다. 이전 서버의 5%·10%·15% 수치와 차이의 크기를 직접 빼서 “gross outlier의 추가 효과”로 계산하지 않는다.

| 항목 | 현재 상태 또는 위치 |
| --- | --- |
| 전체 실행 | 18/18 완료, 실패 0 |
| 개별 모델과 평가값 | `outputs/rebuttal/synthetic_background_only/` |
| queue 상태 | `outputs/status/rebuttal_synthetic_background_only/` |
| 검증된 세 seed 집계 | `outputs/rebuttal/synthetic_background_only/aggregation.json`, `aggregation.csv` |

## 한계와 불확실성

- 공식 재현 A/B는 모두 완료됐지만 source·data·setting-faithful 재현이다. 원 저자 환경의 package와 하드웨어까지 bitwise-identical 재현이라는 뜻은 아니다.
- 공식 저장소가 원 실행환경의 package 버전을 고정하지 않았기 때문에 A는 source·data·setting-faithful reproduction이지, 원 저자 환경과 bitwise-identical reproduction은 아니다.
- 합성 pairing 감사는 seed 40에서 모든 조건을 재구성해 확인했다. 코드 경로는 seeds 41–42에서도 같지만, 세 seed의 모든 배열을 별도로 hash 감사한 결과로 표현하지 않는다.
- 논문은 큰 이상치를 높은 값으로 대체한다고 적지만 실행 코드는 배경 잡음에 양의 큰 값을 추가한다. 이 차이를 수정하지 않으면 실험 설명과 실제 증거가 어긋난다.
- Burgers와 lambda–omega 자료는 이 서버에 원래 없어서 현재 설정으로 다시 생성했다. 현재 NPZ 컨테이너 hash는 이전 서버 handoff의 컨테이너 hash와 다르다. 따라서 새 background-only 결과를 이전 서버 실험의 byte-identical 재현이라고 부르지 않는다.
- natural PIV의 평가는 잡음 없는 참 유동장이 아니라 공간적으로 분리한 실제 PIV 측정값과 비교한다. 평가 오차에는 평가측정 자체의 오차도 포함될 수 있다.
- 세 seed 평균은 방향을 확인하는 반복 실험이지만, 아주 작은 차이를 강한 통계적 우위로 해석하기에는 반복 수가 적다.

## rebuttal 전략

가장 방어 가능한 전략은 naPINN의 보편적 우위를 주장하는 것이 아니라, **어떤 오염 구조에서 gate가 추가 가치를 주는지**를 분명히 하는 것이다.

1. 실제 PIV에 사전 고정한 강한 Legacy-4G 오염과 10%·15% 큰 이상치를 넣은 조건에서는 naPINN-MSE가 가장 낮은 평균 오차를 보였다고 말한다.
2. 합성 PDE의 전역 four-Gaussian 분포와 한쪽 방향 큰 이상치 조건에서는 direct PINN-EBM이 더 좋았다는 불리한 결과도 함께 공개한다.
3. 이 차이를 실패로만 해석하지 않고, direct EBM likelihood가 전역 residual 혼합분포를 충분히 표현할 때는 gate가 불필요하거나 오히려 선택 오차를 추가할 수 있다는 경계조건으로 설명한다.
4. natural PIV에서 naPINN-L1과 naPINN-q2.9는 naPINN-MSE보다 좋아졌지만, 같은 loss의 gate 없는 LAD와 OrPINN보다 좋아지지 않았다. 따라서 이 결과는 “robust loss를 naPINN에 결합할 수 있다”는 모듈성 근거로만 사용하고, gate의 추가 성능 향상 근거로 사용하지 않는다.
5. natural PIV에서 direct PINN-EBM은 모든 가중치에서 field 오차 하위권이었고, 합성 background-only에서는 모든 PDE에서 1위였다. 이 대비를 분포 적합성의 근거로 사용한다. 전역 four-Gaussian 합성 residual은 direct likelihood에 잘 맞지만, 실제 PIV에서는 같은 장점이 재현되지 않았다.
6. official A/B 결과는 현재 naPINN과의 승패 표가 아니라, comparator 구현의 충실성을 입증하는 별도 재현 근거로 사용한다.

reviewer 답변에서 사용할 수 있는 현재의 안전한 문장은 다음과 같다.

> We audited the direct PINN-EBM comparator against the paper and the authors' code, and separately launched source-faithful and paper-architecture reproductions. The existing compute-matched comparator preserves the direct residual-likelihood objective and its gradient path, but differs from the original Navier–Stokes experiment in data, parameterization, architecture, batching, and training budget. We therefore label it as an objective-faithful comparator rather than an exact reproduction. A paired-input audit found no method-specific data advantage in the synthetic benchmark. Direct PINN-EBM also remained best when gross offsets were removed and only the global four-Gaussian background was retained, indicating that direct optimization of this homogeneous residual likelihood is already advantageous. In contrast, it ranked below the MSE and robust-loss baselines on unmodified real PIV for all predeclared PDE weights. We therefore interpret the comparison as distribution-dependent rather than claiming universal superiority of either method.

공식 A/B 집계는 `outputs/rebuttal/pinn_ebm_upstream/aggregation_strict.json`
(SHA-256 `40a8f9f871dbbeb02fef4bf65b2639b66a8afa9832293e436ba1614825501878`)에
보존했다. 이는 comparator 구현의 충실성 증거이며, naPINN과의 직접 성능
비교 숫자는 아니다.

## 답이 남은 질문

다음 세 질문에는 이번 실험으로 답했다.

- 배경 잡음 전용 합성 실험에서도 direct PINN-EBM이 세 PDE 모두 1위였다.
- natural PIV에서 naPINN-L1과 naPINN-q2.9는 naPINN-MSE보다 좋아졌지만, gate 없는 LAD와 OrPINN보다 좋아지지는 않았다.
- natural PIV PINN-EBM은 PDE 가중치에 민감했다. 가중치 50이 가장 안정적이고 물리 residual도 낮았지만 field 오차는 전체 7위였으며, 가중치 1과 10은 더 나빴다.

아직 답이 남은 질문은 다음과 같다.

- 양·음 방향을 모두 갖는 이상치 또는 공간적으로 뭉친 이상치에서도 direct PINN-EBM의 장점이 유지되는가?
두 번째 질문에는 답했다. 8×20의 평균 평가 MAE는 더 낮지만 차이
0.00245가 각 반복의 표준편차보다 작으므로, 이 다섯 반복만으로 구조적
성능 차이가 반복 변동보다 크다고 말할 수 없다.

첫 질문은 별도의 stress-test protocol이 필요하다.
