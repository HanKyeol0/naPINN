# Reviewer 지적별 실험 결과와 증거 보존 상태

작성일: 2026-07-28  
용도: rebuttal 내부 점검 및 추가 실험 우선순위 결정  
문서 형식: Markdown  
상태: synthetic·민감도·추가 RealPDEBench·structured PIV 및 공식 PINN-EBM
A/B 모두 엄격 집계 완료

완료 수치, 전체 방법 순위, 용어 설명과 rebuttal 전략을 한곳에 모은
통합 문서는 `rebuttal/reports/final_experiment_results_ko.md`다. 현재는
완료 artifact만 확정 수치로 사용한다.

## 먼저 답해야 할 질문

Reviewer가 요청한 실험의 숫자가 기존 rebuttal 문서에 많이 정리되어 있는
것은 맞다. 그러나 **모든 결과가 이 서버에서 원시 실행 결과까지 다시
검증 가능한 상태로 정리되어 있는 것은 아니다.**

현재 결과는 다음 세 종류로 나뉜다.

1. **원시 결과까지 확인 가능**: 각 run의 설정, metric, checkpoint 또는
   공식 결과 파일과 complete-only 집계가 현재 서버에 모두 있다.
2. **문서에 숫자는 있지만 원시 결과가 없음**: 이전 서버에서 얻은 평균과
   표준편차가 rebuttal 문서에는 남아 있지만, 그 숫자를 만든 개별 run과
   집계 JSON이 현재 서버에는 없다. 숫자를 참고할 수는 있어도 완전한
   재검증 증거라고 부르면 안 된다.
3. **실험하지 않음 또는 실험으로 답할 수 없음**: 아직 실행하지 않았거나,
   reviewer의 지적이 용어·비교 범위·식별 가능성에 관한 것이어서 새
   숫자보다 정확한 설명이 필요한 경우다.

특히 사용자가 질문한 두 항목은 최초 감사 시점에는 모두 두 번째
종류에 해당했다. 이후 합성 PDE parameter reconstruction 162회,
Allen–Cahn cost-1.0 extension 9회, 실제 PIV inverse-Re 12회를 이 서버에서
새로 완료해 첫 번째 종류로 승격했다.

- Allen–Cahn, Burgers, lambda–omega의 제출 당시 PDE parameter recovery
  원시 aggregate는 이 서버에 없다. 대신 5%·10%·15% 전체를 새로 실행한
  destination-server 162-run aggregate가 엄격 검증됐다. 이것은 제출 표의
  bitwise 재현이 아니라 별도 rebuttal 증거 계보다.
- Allen–Cahn의 rejection cost 0.5와 1.0 비교 및 다섯 개 cost 민감도
  숫자는 문서에 남아 있지만, 원래 sweep aggregate는 이 서버에 없다.
  다만 cost 1.0의 5%·10%·15%, seeds 40–42 총 9회는 새로 복구하고
  strict aggregate까지 완료했다. cost 0.5 core도 162회 synthetic
  recovery의 일부로 엄격 집계됐으며, 15% 다섯 비용 sweep도 108-run
  supplement의 strict aggregate에서 완료됐다.

따라서 “reviewer가 지적한 모든 항목에 새 실험 결과가 있느냐”에 대한
정확한 답은 **아니다**. 다만 이번에 실행 대상으로 고정한 합성 parameter,
Allen–Cahn cost, structured PIV, inverse-Re, MAD, 추가 RealPDEBench 범위는
모두 원시 산출물과 엄격 집계까지 완료됐다. HMC, heteroscedastic noise,
상관된 `u,v` 오차처럼 protocol을 고정하지 않았거나 rebuttal 기간에
무리하게 구현하지 않기로 한 항목은 여전히 미실험 또는 limitation이다.

현재 서버에서 다시 실행할 전체 범위와 저장 위치는
`configs/rebuttal/reviewer_recovery_manifest.yaml`에 고정했다. 이 파일은
단순한 할 일 목록이 아니다. 각 캠페인마다 다음 정보를 함께 기록한다.

- 몇 개의 개별 학습 run과 몇 개의 세 seed 평균 그룹이 정확히 있어야
  완료인지
- seed 39 calibration과 seeds 40–42 최종 평가를 어느 서로 다른 폴더에
  저장하는지
- 어떤 실험이 다른 실험의 체크포인트를 먼저 필요로 하는지
- strict aggregate가 확인할 학습 update 수와 출력 파일 위치
- 아직 실험 protocol을 과학적으로 고정하지 못해 실행하지 않는 reviewer
  요청은 무엇인지

이 manifest에 적힌 수와 맞지 않는 부분 실행은 최종 표에 들어가지 않는다.
전체 진행률은 `outputs/status/reviewer_recovery_summary.json`에도 기록한다.
이 JSON은 완료 개수와 strict aggregate 존재 여부만 담으며, 중간 성능값은
의도적으로 넣지 않는다.

## Destination-server recovery 진행 스냅샷

아래 표는 2026-07-27 16:44 UTC까지 확인한 상태다. 진행 중인 정확한
개수는 `outputs/status/reviewer_recovery_summary.json`을 따른다. `완료`는 단순히
프로세스가 끝났다는 뜻이 아니라, 현재까지 생성된 각 run이 설정,
checkpoint, 입력 해시와 학습 예산 검사를 통과했다는 뜻이다. 다만
캠페인 전체가 끝나기 전에는 개별 run의 성능값으로 순위를 만들지 않는다.

| 캠페인 | 상태 | 정확한 완료 조건 | 결과 저장 위치 |
| --- | --- | ---: | --- |
| 세 합성 PDE의 5%·10%·15% 기본 비교 | 162/162 엄격 집계 완료, 증거 사용 가능 | 162 runs, 54 groups | `outputs/rebuttal/synthetic_recovery_20260726/` |
| Controlled Cylinder·FSI·Foil | 144/144 엄격 집계 완료, 증거 사용 가능 | 144 runs, 48 groups | `outputs/rebuttal/realpdebench_multidataset/` |
| PINN-EBM 저자 코드 A / 논문 구조 B | A/B 각 5/5 완료, strict aggregate 완료 | variant별 공식 5회 결과와 result pickle | `outputs/rebuttal/pinn_ebm_upstream/aggregation_strict.json` |
| Allen–Cahn rejection cost 1.0 | 9/9 엄격 집계 완료 | 9 runs, 3 groups | `outputs/rebuttal/allen_cost1_recovery_20260726/` |
| EMA·rejection·noise family·selector | 108/108 엄격 집계 완료, 증거 사용 가능 | 중복 기준점을 한 번만 학습한 108 runs, 36 groups | `outputs/rebuttal/synthetic_supplement_recovery_20260726/` |
| PINN-EBM PDE weight | calibration 9/9와 held-out 18/18 모두 엄격 완료 | seed 39의 9 runs와 seeds 40–42의 18 runs | `outputs/rebuttal/synthetic_pinn_ebm_weight_*_20260726/` |
| 보수적 35k update 비교 | 27/27 엄격 집계 완료, 증거 사용 가능 | 27 runs, 9 groups | `outputs/rebuttal/synthetic_compute35_recovery_20260726/` |
| Synthetic MAD-PINN | 9/9 엄격 집계 완료, 증거 사용 가능 | 9 runs, 3 groups, run당 총 60k PINN updates | `outputs/rebuttal/synthetic_mad_recovery_20260726/` |
| Structured PIV와 inverse-Re, MAD-PINN | **102/102 엄격 집계 완료, 증거 사용 가능** | 75 + 12 + 15 = 102 runs, 34 groups | `outputs/rebuttal/realpde_recovery_20260726/` |

Structured-PIV의 naPINN은 과거 cost-0.01 민감도 설정을 재사용하지 않는다.
Seed-39 calibration에서 고정한 rejection cost 0.10 config를 fixed-Re 75회와
inverse-Re 12회 모두에 명시적으로 전달한다. Earlier cost-0.01 결과는
별도의 시간순 민감도 증거로 보존한다. 이 구분이 없으면 “calibration에서
선택한 설정으로 전체 held-out matrix를 재실행한다”는 사전 계획과 실제
queue가 서로 달라진다.

반면 완료된 추가 RealPDEBench 144-run 행렬은 persistent-drift
calibration을 다른 데이터셋으로 옮기는 실험이 아니라, 세 데이터셋에
동일한 Legacy-4G 오염과 canonical synthetic gate cost 0.5를 적용하는
별도의 사전 고정 일반화 stress test다. 생성된 모든 naPINN resolved
config가 cost 0.5로 일치함을 확인했다. 이 행렬은 144/144로 이미
완료됐으며, 결과를 보고 cost 0.10으로 바꾸지 않았다. 최종 보고서에서는
`non-Cylinder Legacy-4G, cost 0.5`와
`Cylinder structured drift, seed-39-selected cost 0.10`을 같은 설정인
것처럼 합치지 않는다. 이 행렬은 Controlled Cylinder 48회에 FSI 48회와
Foil 48회를 더한 것이므로, 사용자가 요청한 Cylinder 이외의 새 실행은
FSI와 Foil 합계 96회다.

## 이 문서의 판정 기준

| 상태 | 의미 | rebuttal에서 사용할 때의 원칙 |
| --- | --- | --- |
| `원시 증거 완료` | 필요한 full run과 complete-only 집계가 현재 서버에 있다. Smoke test와 중간 로그는 제외한다. | 결과의 범위와 불리한 결과까지 함께 공개하는 조건으로 수치 인용 가능 |
| `문서 요약만 존재` | 평균값은 남아 있지만 그 값을 만든 원시 metric 또는 집계가 현재 서버에 없다. | 원래 서버의 결과를 회수하거나 재실험하기 전에는 검증 완료로 표시하지 않음 |
| `진행 중` | full run이 실행 중이며 아직 최종 metric이 없다. | 최종 결과 파일이 생기기 전에는 성능 근거로 사용하지 않음 |
| `미실험` | reviewer가 요청한 비교를 아직 실행하지 않았다. | 명시적인 limitation으로 남기거나 사전 고정한 실험을 수행 |
| `설명으로 대응` | 새 실험보다 잘못된 용어, 비교 범위 또는 식별 불가능성을 바로잡는 것이 핵심이다. | 실험한 것처럼 보이게 만들지 않고 정확히 인정하고 설명 |

## 현재 서버에서 원시 증거까지 완료된 실험

| 실험 | 완료 범위 | 현재 확인 가능한 근거 | 해석 |
| --- | ---: | --- | --- |
| RealPDEBench Cylinder, 제출 코드와 같은 Legacy-4G 오염 기본 강도 | 48/48 full runs | `analysis/results/runs/rebuttal_realpde_legacy_4g/aggregation.json` | 10%와 15% 큰 이상치 조건에서 naPINN-MSE가 가장 낮은 field error를 보였다. |
| Cylinder Legacy-4G 오염 강도 탐색 | 144/144 full runs | `analysis/results/runs/rebuttal_realpde_legacy_4g_scale/seed39_scale_selection.json` | 오염 강도에 따라 방법 순위가 달라졌다. 결과를 본 뒤 유리한 강도만 주 결과로 고르면 안 된다. |
| Cylinder Legacy-4G 새 seed 재확인 | 72/72 full runs | `analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/aggregation.json` | 강한 오염에서는 naPINN 우위, 약한 오염에서는 지표별 우승 방법이 달라지는 혼합 결과가 확인됐다. |
| 추가 인공 오염이 없는 natural PIV | 27/27 full runs | `outputs/rebuttal/natural_piv/aggregation.json`, `outputs/rebuttal/natural_piv_baselines/aggregation.json` | LAD-PINN이 가장 좋았다. 현재 author 지침상 reviewer-facing 주 근거에서는 제외한다. |
| 합성 PDE, four-Gaussian 배경 잡음만 있고 큰 이상치 0% | 18/18 full runs | `outputs/rebuttal/synthetic_background_only/aggregation.json` | 세 PDE 모두 direct PINN-EBM이 naPINN보다 좋았다. |
| 합성 PDE, four-Gaussian 배경 잡음과 5%·10%·15% 큰 이상치 | 162/162 full runs, 54/54 groups | `outputs/rebuttal/synthetic_recovery_20260726/aggregation_strict.json` | direct PINN-EBM이 8/9 조건의 평균 rMAE·rMSE에서 1위였고, Allen–Cahn 15%에서는 naPINN이 두 지표 모두 1위였다. 제출 표의 bitwise 재현이 아닌 새 증거 계보다. |
| 합성 PDE 민감도·잡음 분포·selector 보충 행렬 | 108/108 full runs, 36/36 groups | `outputs/rebuttal/synthetic_supplement_recovery_20260726/aggregation_strict.json` | EMA, rejection cost, Gaussian/Laplace/Student-t, naPINN robust backbone, fixed-quantile와 learnable-threshold를 현재 서버에서 다시 검증했다. 결과는 조건 의존적이며 보편적 우위를 지지하지 않는다. |
| Controlled Cylinder·FSI·Foil Legacy-4G 일반화 행렬 | 144/144 full runs, 48/48 groups | `outputs/rebuttal/realpdebench_multidataset/aggregation.json` | naPINN-MSE가 평균 rMAE 6/6 조건에서 1위였고, 평균 rMSE는 4/6에서 1위였다. FSI의 rMSE는 OrPINN이 1위였고, Foil 15%의 naPINN-q2.9 한 seed는 크게 실패했다. |
| 35k PINN-update 보수적 비교 | 27/27 full runs, 9/9 groups | `outputs/rebuttal/synthetic_compute35_recovery_20260726/aggregation_strict.json` | MSE, LAD, OrPINN에 35,000 PINN update를 줘도 세 PDE 15% 조건에서 direct PINN-EBM과 naPINN의 30,000-PINN-update 결과보다 field 오차가 컸다. Wall-clock matching 증거는 아니다. |
| Synthetic MAD-PINN | 9/9 full runs, 3/3 groups | `outputs/rebuttal/synthetic_mad_recovery_20260726/aggregation_strict.json` | 15%에서 알려진 이상치 약 98.4–98.6%를 제거했지만 clean scalar도 약 10.6–19.0% 제거했다. 두 단계 합계 60,000 PINN update이므로 compute-matched reference가 아니다. |
| 합성 입력 공정성 감사 | 3 PDE × 3 이상치 비율 | `outputs/audits/synthetic_pinn_ebm_fairness/seed_40.json` | 같은 조건의 방법들이 같은 좌표, 관측값과 이상치 위치를 받았음을 확인했다. 성능 실험은 아니다. |
| Allen–Cahn rejection cost 1.0 복구 | 9/9 full runs, 실패 0 | `outputs/rebuttal/allen_cost1_recovery_20260726/aggregation_strict.json` | 5%·10%·15%에서 gate가 알려진 이상치의 약 2–3%만 거부했고, 오염률 증가에 따라 field error가 크게 악화됐다. |
| Structured PIV fixed-Re·inverse-Re·MAD | 102/102 full runs, 34/34 groups | `outputs/rebuttal/realpde_recovery_20260726/injected_piv_aggregation_strict.json` | LAD가 AR(1), 10% drift, spatial burst에서 두 field 지표 1위였고, naPINN은 20%와 30% drift에서 두 지표 1위였다. inverse-Re에서는 모든 방법의 Re 상대오차가 51.5% 이상이었다. |

## Reviewer 지적별 전체 상태

### 실제 자료, 센서 오염, 물리모형 불일치

| Reviewer 지적 | 현재 상태 | 현재 남아 있는 내용 | 필요한 후속 조치 |
| --- | --- | --- | --- |
| 합성 자료 외의 실제 또는 반실제 사례 | `원시 증거 완료` | Cylinder PIV Legacy-4G와 Controlled Cylinder·FSI·Foil 144-run strict aggregate | 실제 PIV held-out 측정은 잡음 없는 물리 참값이 아니며 controlled injection stress test라는 범위를 함께 밝힌다. |
| 지속적인 센서 고장과 drift | `원시 증거 완료` | 10%·20%·30%, 여섯 방법, seeds 40–42가 strict aggregate에 있다. naPINN은 20%·30%에서 두 field 지표 1위이나 10%에서는 LAD가 1위다. | 유리한 20%·30%만 떼지 않고 세 강도를 함께 보고한다. |
| 시간상관 AR(1) 오염 | `원시 증거 완료` | 여섯 방법 × seeds 40–42가 완료됐다. LAD가 rMAE/rMSE 1위이고 naPINN은 3위다. | persistent-drift calibration이 시간상관 오염에 그대로 최적이라고 주장하지 않는다. |
| 공간적으로 뭉친 burst 오염 | `원시 증거 완료` | 여섯 방법 × seeds 40–42가 완료됐다. LAD가 두 field 지표 1위다. | naPINN의 100% failure rejection과 4.62% clean rejection을 field 1위로 과장하지 않는다. |
| 위치·시간·상태에 따라 분산이 바뀌는 heteroscedastic noise | `미실험` | 해당 조건 없음 | 여력이 있으면 위치별 noise scale을 사전 고정해 추가하고, 아니면 미검증 범위로 명시한다. |
| 속도 `u,v` 성분 사이의 상관된 측정오차 | `미실험` | 기존 큰 이상치 크기는 성분별 독립 추출이다. | 상관계수를 고정한 `u,v` innovation 실험을 추가하거나 미검증 범위로 명시한다. |
| 관측오차와 PDE model discrepancy가 동시에 있을 때의 성능 | `일부 원시 증거 완료` | 실제 PIV에 nominal 2D Navier–Stokes를 적용한 Legacy-4G 실험 | Stress test로는 사용할 수 있지만 관측오차와 model discrepancy 각각의 영향을 분해했다고 주장하면 안 된다. |
| 실제 사례에서 PDE coefficient 복원 | `12/12 원시 증거 완료, 부정적 결과` | naPINN이 field rMAE/rMSE는 가장 낮았지만 Re 상대오차는 77.0%였다. 가장 낮은 Re 상대오차도 PINN-EBM의 51.5%이며 seed별 Re가 크게 흔들렸다. | 어떤 방법의 coefficient 복원 성공으로도 제시하지 않고 model discrepancy 아래 식별 실패로 보고한다. |
| 한 개 Cylinder trajectory를 넘는 일반성 | `144/144 원시 증거 완료` | Controlled Cylinder, FSI, Foil의 사전 고정 행렬이 48 groups와 18 paired input blocks를 모두 통과했다. | FSI rMSE의 불리한 결과와 Foil naPINN-q2.9의 seed 실패를 포함해 전체 결과를 보고한다. |
| Combustion 실측 자료 | `현재 방법 적용 불가` | 실측 채널은 속도 `u,v`가 아니라 OH* intensity다. | 검증된 관측 연산자와 연소 PDE 없이 Navier–Stokes PINN을 임의 적용하지 않는다. 적용 불가 근거와 필요한 모델을 기록한다. |

### 가장 가까운 prior, baseline, robust loss

| Reviewer 지적 | 현재 상태 | 현재 남아 있는 내용 | 필요한 후속 조치 |
| --- | --- | --- | --- |
| Gate가 없는 direct PINN-EBM과 비교 | `Cylinder 원시 증거 완료`, `세 합성 PDE 기본 비교 원시 증거 완료` | Cylinder Legacy-4G에는 equal-weight와 PDE-weight-50 direct PINN-EBM이 모두 있다. 세 합성 PDE × 세 이상치 비율의 162/162 strict aggregate도 통과했다. | 두 증거 모두 범위를 제한해 사용하고, 이전 서버의 “아홉 조건 모두” 순위와 새 rerun의 8/9 순위를 서로 다른 계보로 공개한다. |
| PINN-EBM 저자 GitHub 코드와 논문 구조 재현 | `A/B 모두 완료` | 공식 A/B 각 5회와 result pickle이 strict aggregate를 통과했다. 평가 MAE는 A 0.07467 ± 0.01141, B 0.07222 ± 0.00873이다. | 평균 차이는 반복 표준편차보다 작으므로 구조의 결정적 성능 우위로 주장하지 않는다. A/B는 naPINN과 별도 protocol이다. |
| PINN-EBM PDE-loss weight 공정성 | `합성 calibration 9/9·held-out 18/18 원시 증거 완료` | Allen–Cahn field는 weight 10, Burgers field는 50이 가장 낮았고, lambda–omega는 rMAE와 rMSE의 순서가 달랐다. Allen–Cahn weight 50은 크게 실패했다. | 기존 weight 1 core와 새 10·50을 모두 제시하고 하나의 보편적 최적 weight를 주장하지 않는다. |
| MAD-PINN 또는 통계적 전처리 baseline | `synthetic 9/9·structured 15/15 원시 증거 완료` | 두 MAD stage 2 캠페인은 seed-matched LAD checkpoint를 사용했고 strict aggregate를 통과했다. Structured MAD는 5개 오염 조건을 포함한다. | MAD의 총 60k PINN update와 정상 관측의 20.2–34.0%까지 제거한 비용을 함께 밝힌다. |
| 고정 residual screening | `selector 및 synthetic·structured MAD 원시 증거 완료` | fixed-quantile, learnable-threshold, synthetic MAD 9회, structured MAD 15회가 모두 엄격 집계됐다. | Learnable-threshold는 학습되는 규칙이므로 고정 전처리라고 부르지 않는다. |
| Robust loss를 충분히 비교했는가 | `Cylinder·합성 core·추가 backbone 원시 증거 완료` | Cylinder에는 MSE, LAD, OrPINN q=2.9 및 naPINN MSE/L1/q2.9가 있다. 합성 core에는 MSE, LAD, OrPINN q=1.9/2.9, PINN-EBM, naPINN이 있고, supplement에는 lambda–omega의 naPINN-LAD/q2.9가 있다. | 사용한 q값을 사전 선택값으로 설명하고 “완전한 hyperparameter 최적화”라고 부르지 않는다. |
| HMC B-PINN과 비교 | `미실험, 설명으로 대응` | 제출 baseline은 mean-field variational B-PINN-VI다. | 이를 HMC라고 부르지 않고 한계로 인정한다. 검증 없는 급한 HMC 실행은 하지 않는다. |
| Bayesian UQ와 naPINN의 차이 | `설명으로 대응` | naPINN은 posterior UQ가 아니라 관측별 포함 가중치를 학습하는 point-estimate 방법이다. | 두 접근은 경쟁 관계만이 아니라 결합 가능한 방법이라고 범위를 바로잡는다. |

### PDE parameter reconstruction과 민감도

| Reviewer 지적 | 현재 상태 | 현재 남아 있는 내용 | 필요한 후속 조치 |
| --- | --- | --- | --- |
| 세 benchmark의 모든 PDE parameter 복원 | `162/162 strict aggregate 완료` | 새 destination-server aggregate가 5%·10%·15%에서 Allen–Cahn \(\epsilon\), Burgers viscosity, lambda–omega \(\beta\)의 학습값과 절대오차를 세 PDE × 여섯 방법 × 세 seed 전체에 저장한다. | Field error와 parameter error를 분리해 모두 보고하고, 제출 당시 ten-trial 표의 재현이라고 부르지 않는다. |
| 제출 논문의 Allen–Cahn parameter table 재현 | `paper-reported only` | TeX 표는 있지만 원래 ten-trial config와 artifact 연결이 없다. | 제출 당시 config와 run lineage를 찾는다. 새 rebuttal run을 제출 표의 재현이라고 부르면 안 된다. |
| EMA coefficient와 beta/rho 표기 | `세 값 원시 증거 완료 + 설명 필요` | 코드의 update weight \(m=0.05\)는 old-state decay로 쓰면 \(\rho=0.95\)다. \(m=0.01,0.05,0.10\)의 Allen–Cahn 15% rMAE는 0.10778/0.09202/0.08551이다. | 한 조건의 민감도이므로 \(m=0.10\)이 보편적으로 최적이라고 주장하지 않고, 표기 불일치는 실험과 별개로 명확히 정정한다. |
| Rejection-cost 민감도 | `cost 1.0·cost 0.5·다섯 비용 sweep 원시 증거 완료` | 15%에서 cost 0.10/0.30/0.50/0.70/1.00의 rMAE는 0.08847/0.09133/0.09202/0.09886/0.86890이다. 앞의 네 값은 알려진 이상치 약 99.3%를 거부하지만, 1.00은 2.12%만 거부한다. | 성능을 본 뒤 cost 0.10을 새 기본값으로 고르지 않는다. 이 sweep은 민감도 증거이고 제출 당시 실제 config의 계보 문제는 별도로 남는다. |
| Allen–Cahn만 rejection cost가 1.0인 이유 | `새 민감도 증거 완료`, `제출 설정 lineage 미해결` | Paper는 Allen–Cahn 1.0, 다른 PDE 0.5라고 적지만 초기 rebuttal core는 세 PDE 모두 0.5였다. 새 cost-1.0 extension은 9/9 완료됐으나 제출 당시 run은 아니다. | 제출 당시 config를 찾아 실제 사용값을 확인한다. 새 실험은 과학적 민감도로만 보고하고 제출 표의 lineage 증명으로 사용하지 않는다. |
| Gate initializer 민감도 | `미실험 + 설명 필요` | YAML initializer field가 실제 gate constructor에 전달되지 않아 기본값이 사용됐다. | 실제 기본값과 inert field를 인정한다. 새 controlled study 없이 initializer를 최적화했다고 주장하지 않는다. |
| Optimization stability | `paper-reported only` | 제출 논문의 staged-training ablation 설명만 있고 현재 서버의 원시 trace는 없다. | 원래 artifact 회수 또는 제한된 severe three-seed 재실행 |
| Gaussian 외의 noise family | `69-run 원시 증거 완료` | Gaussian, Laplace, Student-t의 63회와 four-Gaussian naPINN robust-loss 6회가 108-run supplement에 포함되어 strict aggregate를 통과했다. | Gaussian은 naPINN-MSE, Laplace는 naPINN-LAD, Student-t rMAE는 LAD, four-Gaussian은 direct PINN-EBM이 가장 낮았다. 잡음 분포에 무관한 우위라고 주장하지 않는다. |

### 계산량, 재현성, 논문·코드 일치

| Reviewer 지적 | 현재 상태 | 현재 남아 있는 내용 | 필요한 후속 조치 |
| --- | --- | --- | --- |
| 전체 학습 계산량 | `schedule은 코드로 확인`, `시간 수치는 문서 요약만 존재` | naPINN은 PINN 30k updates와 estimator-only 5k updates를 쓴다. 30k PINN은 5k warm-up + 25k joint다. | 원시 phase-timing artifact를 회수한다. |
| 35k update-count 비교 | `27/27 원시 증거 완료` | run당 정확히 35,000 PINN update를 사용한 MSE/LAD/OrPINN의 별도 strict aggregate가 있다. | PINN update-count 보수적 control로만 사용하고 wall-clock match라고 부르지 않는다. |
| 같은 wall-clock의 비교 | `미실험` | 기존 시간은 공유 GPU에서 측정됐다. | 필요하면 동일한 예약 GPU에서 equal-elapsed-time protocol을 실행한다. |
| 논문이 말한 “이상치로 대체”와 코드의 “양의 큰 오차 추가” | `불일치 확인` | 현재 원시 evidence는 positive additive gross offsets에 해당한다. | 실제 구현을 정확히 설명한다. Replacement 주장을 유지하려면 별도 paired replacement 실험이 필요하다. |
| Baydin 인용 연도 | `설명으로 대응` | 2018이 맞고 기존 표기는 오기다. | Rebuttal에서 인정하고 camera-ready에서 수정 |
| 시간 overhead 백분율 | `설명으로 대응` | 표에 적힌 시간으로 다시 계산하면 일부 본문 백분율과 맞지 않는다. | 산술을 정정한다. 새 실험은 필요하지 않다. |

## 사용자가 특별히 질문한 두 결과의 현재 해석

### PDE parameter reconstruction

기존 문서에는 세 PDE의 5%·10%·15% 결과가 모두 표로 정리되어 있다.
그러나 parameter error의 우승 방법은 PDE와 이상치 비율에 따라 달라진다.
따라서 이 결과로 “naPINN이 모든 coefficient를 가장 정확히 복원한다”고
주장할 수 없다. 특히 실제 PIV의 inverse-Re 결과는 nominal PDE residual이
작아도 metadata coefficient를 정확히 복원한다는 보장이 없음을 보여 주는
반대 사례다.

현재 destination-server strict aggregate는 세 PDE, 세 오염 비율, 여섯
방법과 seed 40--42를 모두 연결한다. 따라서 새 rebuttal 실험의 parameter
수치는 원시 run까지 추적할 수 있다. 다만 제출 당시 ten-trial 표와 그
원래 config를 연결하는 lineage는 여전히 없으므로, 새 결과를 제출 표의
bitwise 재현이라고 부르면 안 된다.

실제 PIV의 30% persistent drift에서 Reynolds 수를 8,000으로 초기화하고
metadata 값 10,031과 비교한 12회 strict 결과는 다음과 같다. Field 오차와
parameter 오차를 같은 성공 지표로 취급하면 안 되므로 별도 열로 나눈다.

| 방법 | 학습 Re, 평균 ± 표본 표준편차 | Re 상대오차 | field rMAE | field rMSE |
| --- | ---: | ---: | ---: | ---: |
| MSE | 1,331.81 ± 111.03 | 86.72% | 0.37221 | 0.49041 |
| LAD | 3,634.24 ± 1,650.42 | 63.77% | 0.23303 | 0.35739 |
| PINN-EBM | 7,681.08 ± 5,927.76 | **51.54%** | 0.28567 | 0.36947 |
| naPINN | 2,306.53 ± 550.97 | 77.01% | **0.22825** | **0.31499** |

naPINN은 field reconstruction은 가장 좋지만 Re 복원에는 실패했다.
PINN-EBM도 평균 Re 오차가 가장 낮다는 이유만으로 성공이라고 볼 수 없다.
상대오차가 51.5%이고 seed별 Re가 14,260.66, 2,757.24, 6,025.34로 크게
흔들렸기 때문이다. 이 표는 reviewer에게 “실제 자료에서도 PDE parameter를
정확히 복원했다”고 답하는 근거가 아니라, observation error와 model
discrepancy가 함께 있을 때 coefficient identification이 실패할 수 있다는
부정적 근거다.

근거:
`outputs/rebuttal/realpde_recovery_20260726/injected_piv_aggregation_strict.json`

### Allen–Cahn rejection cost

새 서버에서 cost 1.0의 5%·10%·15% 조건을 seeds 40–42로 다시 실행했다.
9개 run 모두 `final.pt`, resolved config, 30,000 PINN update,
5,000 estimator-only update, finite metric 검사를 통과했다. strict
aggregate는 다음과 같다.

| 큰 이상치 비율 | field rMAE, 평균 ± 표본 표준편차 | field rMSE, 평균 ± 표본 표준편차 | \(\epsilon\) 절대오차 | 알려진 이상치 rejection | 알려진 clean rejection |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 0.26088 ± 0.07158 | 0.23520 ± 0.06119 | 0.00419 ± 0.00395 | 3.07% ± 0.67%p | 0.012% ± 0.003%p |
| 10% | 0.56828 ± 0.08379 | 0.47785 ± 0.06770 | 0.00530 ± 0.00440 | 2.25% ± 0.66%p | 0.049% ± 0.035%p |
| 15% | 0.86890 ± 0.02391 | 0.72211 ± 0.02451 | 0.00408 ± 0.00285 | 2.12% ± 0.35%p | 0.150% ± 0.046%p |

근거:
`outputs/rebuttal/allen_cost1_recovery_20260726/aggregation_strict.json`

이 값은 이전 서버에서 문서로 전달된
rMAE 0.25878/0.56180/0.87859와 완전히 같지는 않다. 따라서 이전 숫자를
새 로컬 결과로 바꾸어 부르지 않고, 위 표를 **destination-server fresh
recovery**로 따로 기록한다. 두 결과는 모두 cost 1.0에서 gate가 거의
모든 이상치를 받아들이고 오염률이 높아질수록 field error가 크게
악화된다는 같은 결론을 지지한다.

이 결과는 “Allen–Cahn에는 1.0이 더 적절하다”는 근거가 아니다. 오히려
현재 기록만 보면 1.0이 좋지 않았다는 민감도 증거다. 더 중요한 문제는
제출 논문의 표기와 실제 제출 run의 설정을 연결하는 원시 config가 없다는
점이다. 새로 0.5와 1.0을 비교해도 어느 값이 제출 당시 실제로
사용됐는지는 증명할 수 없다. 그래서 다음 두 질문을 분리해야 한다.

1. **제출 당시 실제 설정은 무엇이었는가?** 원래 config와 run artifact를
   찾아야 답할 수 있다.
2. **현재 protocol에서 어느 cost가 더 안정적인가?** cost
   0.10/0.30/0.50/0.70/1.00의 같은 destination-server 비교가 완료됐다.
   0.10--0.70은 서로 비슷하지만 1.00은 gate가 거의 모든 이상치를
   받아들이면서 크게 악화됐다. 이는 민감도 결론이지 제출 config의
   계보 증명은 아니다.

## 108-run 보충 실험에서 새로 확인된 결과

이 절의 값은 모두
`outputs/rebuttal/synthetic_supplement_recovery_20260726/aggregation_strict.json`
에서 읽은 seeds 40--42의 평균과 표본 표준편차다. 총 108개 full run과
36개 세-seed group이 모두 완료됐고, 빠진 seed·중복 run·불완전
checkpoint·잘못된 update 수·무한대 또는 NaN metric이 없음을 검사했다.

### EMA 값은 한 조건에서 비교했으며, 0.10이 보편적 최적값이라는 뜻은 아니다

EMA는 최근 residual 통계가 갑자기 흔들리지 않도록 이전 통계와 새
통계를 섞는 방법이다. 구현의 \(m\)은 **새 mini-batch 통계에 주는
가중치**다. 따라서 이전 통계에 주는 가중치, 즉 decay를 \(\rho\)라고
쓰면 \(\rho=1-m\)이다.

| 새 통계 가중치 \(m\) | 이전 통계 decay \(1-m\) | field rMAE | field rMSE |
| ---: | ---: | ---: | ---: |
| 0.01 | 0.99 | 0.10778 ± 0.03716 | 0.10931 ± 0.03179 |
| 0.05 | 0.95 | 0.09202 ± 0.00352 | 0.09483 ± 0.00232 |
| 0.10 | 0.90 | **0.08551 ± 0.00874** | **0.09235 ± 0.00494** |

이 표는 Allen–Cahn, 15% four-Gaussian 큰 이상치라는 한 조건만 비교한다.
이 조건에서는 \(m=0.10\)의 평균 field error가 가장 낮고 \(m=0.01\)의
seed 변동이 가장 크다. 그러나 세 PDE와 여러 오염률 전체에서 0.10을
검증한 것이 아니므로, rebuttal에서는 “0.10이 최적”이 아니라 “0.01,
0.05, 0.10 범위에서 방법이 작동하며 이 한 severe cell에서는 0.10이
가장 낮았다”고 제한해 말해야 한다.

### Rejection cost 1.0은 gate가 이상치를 거의 거부하지 않게 만들었다

Rejection cost는 관측을 버릴 때 지불하는 벌점이다. 값이 너무 크면
gate는 이상치까지 받아들이는 편을 택할 수 있다. 같은 Allen–Cahn 15%
조건에서 이를 직접 확인했다.

| Rejection cost | field rMAE | 알려진 이상치 중 거부한 비율 | 알려진 정상값 중 거부한 비율 |
| ---: | ---: | ---: | ---: |
| 0.10 | **0.08847 ± 0.00549** | 99.30% | 0.76% |
| 0.30 | 0.09133 ± 0.00282 | 99.25% | 0.72% |
| 0.50 | 0.09202 ± 0.00352 | 99.29% | 0.66% |
| 0.70 | 0.09886 ± 0.01590 | 99.28% | 0.72% |
| 1.00 | 0.86890 ± 0.02391 | 2.12% | 0.15% |

0.10--0.70은 평균 오차가 비교적 가깝고 알려진 이상치의 약 99.3%를
거부했다. 1.00에서는 이상치 거부율이 2.12%로 급락하면서 field error도
크게 증가했다. 이 결과는 부록의 1.0 설정에 대한 강한 민감도 경고다.
다만 새 실험으로 제출 당시 어떤 config가 실제 사용됐는지는 알 수 없다.

### 잡음 분포가 바뀌면 가장 좋은 reconstruction loss도 바뀌었다

아래 표는 lambda–omega에 15% 큰 이상치를 넣고, 그 아래에 깔리는 배경
잡음의 분포를 바꾼 결과다. 각 행은 평균 rMAE가 낮은 순서다.

| 배경 잡음 | rMAE 순위 |
| --- | --- |
| Gaussian | 1. **naPINN-MSE 0.05545** → 2. naPINN-q2.9 0.06004 → 3. naPINN-LAD 0.06058 → 4. direct PINN-EBM 0.06853 → 5. OrPINN q2.9 0.07858 → 6. LAD 0.08686 → 7. MSE 0.21636 |
| Laplace | 1. **naPINN-LAD 0.05631** → 2. naPINN-MSE 0.05826 → 3. naPINN-q2.9 0.05954 → 4. direct PINN-EBM 0.06921 → 5. OrPINN q2.9 0.07028 → 6. LAD 0.07447 → 7. MSE 0.31517 |
| Student-t | 1. **LAD 0.04516** → 2. direct PINN-EBM 0.04545 → 3. naPINN-LAD 0.04578 → 4. OrPINN q2.9 0.06048 → 5. naPINN-q2.9 0.06355 → 6. MSE 0.09437 → 7. naPINN-MSE 0.09773 |
| Four-Gaussian | 1. **direct PINN-EBM 0.04023** → 2. naPINN-MSE 0.07056 → 3. naPINN-q2.9 0.13801 → 4. naPINN-LAD 0.13995 → 5. OrPINN q2.9 0.15263 → 6. LAD 0.19097 → 7. MSE 0.42187 |

Gaussian에서는 naPINN-MSE, Laplace에서는 naPINN-LAD, Student-t
rMAE에서는 gate 없는 LAD, four-Gaussian에서는 direct PINN-EBM이
가장 낮다. 특히 Student-t에서 naPINN-MSE는 MSE보다도 약간 나빴다.
따라서 이 실험은 naPINN의 잡음 분포 무관 우위를 보여 주지 않는다.
대신 base reconstruction loss와 residual 분포의 조합이 중요하다는
근거다.

### 간단한 residual screening은 PDE에 따라 성패가 달랐다

Fixed-quantile은 residual이 큰 순서대로 전체 scalar의 고정 5%를
제외한다. 실제 큰 이상치는 15%이므로 원리상 알려진 이상치의 약 3분의
1만 제거한다. Learnable-threshold는 residual cutoff를 학습하지만 EBM
밀도 추정기는 쓰지 않는다. 따라서 후자는 “고정 전처리”가 아니라
단순한 학습형 gate ablation이다.

| PDE | Fixed-quantile rMAE | Learnable-threshold rMAE | naPINN rMAE | direct PINN-EBM rMAE |
| --- | ---: | ---: | ---: | ---: |
| Allen–Cahn | 0.50940 | 0.24415 | **0.09202** | 0.10288 |
| Burgers | 0.37315 | 0.08557 | 0.08186 | **0.04738** |
| lambda–omega | 0.23726 | 0.06730 | 0.07056 | **0.04023** |

고정 5% 제거는 세 PDE 모두 약했다. Learnable-threshold는
Allen–Cahn에서는 크게 나빴고, Burgers에서는 naPINN과 가까웠으며,
lambda–omega에서는 naPINN보다 조금 낮았다. 그러나 direct PINN-EBM은
Burgers와 lambda–omega에서 둘보다 더 낮았다. 이 결과는 복잡한 EBM
gate만이 유일한 해법이라는 주장을 지지하지 않으며, PDE마다 선택 규칙의
효과가 달라진다는 혼합 결과로 보고해야 한다.

### PINN-EBM의 PDE weight는 하나의 값이 모든 PDE와 지표를 지배하지 않았다

PINN-EBM의 data loss는 residual EBM의 음의 로그우도이고, PDE weight는
여기에 더하는 PDE residual loss의 상대적 크기다. Weight가 커지면
물리식을 더 강하게 맞추지만, field reconstruction이나 PDE parameter
복원이 반드시 함께 좋아지는 것은 아니다.

아래는 15% four-Gaussian 큰 이상치에서 seeds 40--42의 평균이다.
Weight 1은 세 합성 PDE와 세 이상치 비율을 모두 포함한 기본 비교 실험에서,
weight 10·50은 별도의 엄격한 held-out 집계에서
aggregate에서 읽었다. Equal-weight 결과를 이미 본 뒤였기 때문에
seed 39에서 하나를 고르지 않고 10과 50을 held-out에서 모두 실행했다.

| PDE | 지표 | Weight 1 | Weight 10 | Weight 50 | 가장 낮은 값 |
| --- | --- | ---: | ---: | ---: | --- |
| Allen–Cahn | field rMAE | 0.10288 | **0.07629** | 0.75508 | Weight 10 |
| Allen–Cahn | field rMSE | 0.10169 | **0.07533** | 0.74897 | Weight 10 |
| Allen–Cahn | \(\epsilon\) 절대오차 | 0.002149 | **0.001408** | 0.546933 | Weight 10 |
| Burgers | field rMAE | 0.04738 | 0.04239 | **0.03496** | Weight 50 |
| Burgers | field rMSE | 0.06743 | 0.06060 | **0.04926** | Weight 50 |
| Burgers | viscosity 절대오차 | 0.000646 | **0.000252** | 0.000536 | Weight 10 |
| lambda–omega | field rMAE | 0.04023 | **0.03995** | 0.04384 | Weight 10 |
| lambda–omega | field rMSE | **0.05564** | 0.05728 | 0.07635 | Weight 1 |
| lambda–omega | \(\beta\) 절대오차 | 0.010165 | 0.003981 | **0.002894** | Weight 50 |

Allen–Cahn의 weight 50은 field와 parameter 모두 크게 실패했다. Burgers는
field에는 50, parameter에는 10이 가장 낮다. Lambda–omega는 rMAE,
rMSE, parameter의 최적 weight가 각각 10, 1, 50으로 모두 다르다.
따라서 rebuttal에서는 tuned PINN-EBM의 가장 좋은 cell 하나만 naPINN과
비교하지 않고, 세 weight 전체와 이 trade-off를 함께 보여 주는 것이
정당하다.

근거:

- Weight 1:
  `outputs/rebuttal/synthetic_recovery_20260726/aggregation_strict.json`
- Seed 39 calibration:
  `outputs/rebuttal/synthetic_pinn_ebm_weight_calibration_20260726/aggregation_strict.json`
- Held-out weight 10·50:
  `outputs/rebuttal/synthetic_pinn_ebm_weight_heldout_20260726/aggregation_strict.json`

## 새 RealPDEBench 실험의 고정 범위

RealPDEBench 공식 저장소와 dataset card를 확인해 Cylinder 외 시나리오를
다음처럼 분류했다.

| 데이터셋 | 실측 입력 | 현재 실험 |
| --- | --- | --- |
| Controlled Cylinder | PIV 속도 `u,v` | 48/48 완료 |
| Fluid–Structure Interaction (FSI) | PIV 속도 `u,v` | 48/48 완료 |
| Foil | PIV 속도 `u,v` | 48/48 완료 |
| Combustion | OH* intensity 한 채널 | 적용 가능성 감사 완료; 성능 실험은 부적합 |

세 유체 데이터셋에는 동일하게 다음 frozen protocol을 적용한다.

- 첫 1,000–1,199 frame
- 고정된 192개 불규칙 공간 sensor
- 학습 sensor와 겹치지 않는 held-out 공간
- corruption seeds 40–42
- Legacy-4G 배경 잡음과 10%·15% positive additive gross offsets
- MSE, LAD, OrPINN q=2.9, direct PINN-EBM PDE weight 1과 50,
  naPINN-MSE, naPINN-L1, naPINN-q2.9
- 데이터셋 3개 × 이상치 비율 2개 × seed 3개 × 방법 8개 =
  총 144 full runs, 144/144 완료

아래 순위는 세 seed의 평균 오차가 작은 순서다. `naMSE`는 MSE로 학습하는
naPINN, `naL1`은 L1 data loss를 쓰는 naPINN, `naQ`는 q=2.9 robust
data loss를 쓰는 naPINN을 뜻한다. `EBM1`과 `EBM50`은 PINN-EBM의 PDE
loss weight가 각각 1과 50인 두 고정 설정이다.

| 데이터셋·이상치 | 평균 field rMAE 순위 | 평균 field rMSE 순위 |
| --- | --- | --- |
| Cylinder 10% | naMSE < naL1 < OrPINN < LAD < naQ < EBM50 < EBM1 < MSE | naMSE < naL1 < OrPINN < LAD < naQ < EBM50 < MSE < EBM1 |
| Cylinder 15% | naMSE < naL1 < OrPINN < EBM50 < EBM1 < naQ < LAD < MSE | naMSE < naL1 < OrPINN < naQ < LAD < EBM50 < EBM1 < MSE |
| Foil 10% | naMSE < naQ < naL1 < OrPINN < LAD < EBM1 < EBM50 < MSE | naMSE < naQ < naL1 < OrPINN < LAD < EBM1 < EBM50 < MSE |
| Foil 15% | naMSE < naL1 < OrPINN < EBM1 < LAD < EBM50 < MSE < naQ | naMSE < naL1 < OrPINN < EBM1 < LAD < EBM50 < MSE < naQ |
| FSI 10% | naMSE < OrPINN < LAD < naL1 < naQ < MSE < EBM50 < EBM1 | OrPINN < LAD < naMSE < naL1 < MSE < naQ < EBM1 < EBM50 |
| FSI 15% | naMSE < OrPINN < naL1 < LAD < naQ < EBM50 < MSE < EBM1 | OrPINN < LAD < naMSE < naL1 < naQ < EBM50 < EBM1 < MSE |

따라서 naPINN-MSE는 평균 rMAE에서는 6/6 조건에서 1위였고, 평균
rMSE에서는 Cylinder와 Foil의 4/6 조건에서 1위였다. FSI의 rMSE 두
조건에서는 OrPINN이 1위였으며, naPINN-MSE는 3위였다. 이 결과는
naPINN이 모든 데이터셋과 모든 지표에서 항상 최고라는 주장을 지지하지
않는다. 대신 gross outlier가 들어간 세 실제 PIV 시나리오에서 rMAE가
일관되게 낮았다는 더 좁은 주장을 지지한다.

Foil 15%의 `naQ`는 seed 40에서 과도하게 많은 정상 관측까지 버려 평균
rMAE 0.70040, rMSE 0.71768로 실패했다. 이 실패도 평균에서 제외하지
않았다. 즉 robust loss를 naPINN에 결합한다고 언제나 개선되는 것은
아니다.

엄격 집계 근거:

- `outputs/rebuttal/realpdebench_multidataset/aggregation.json`
  (144/144, smoke run 0, 사후 성능 선택 없음,
  SHA-256 `88acccad036ff36fb1bf23b95099a233d068b48eec58f6c1b3e562418e97e7a8`)
- `outputs/rebuttal/realpdebench_multidataset/manifest_frozen.json`
  (실험 범위와 설정을 실행 전에 고정)
- `outputs/rebuttal/realpdebench_multidataset/combustion_applicability.json`
  (Combustion 적용 가능성 감사)

실제 실험에서는 각 시나리오 파일명에 기록된 Reynolds number를 metadata
값으로 보존하지만, 이를 깨끗한 물리 coefficient 정답이라고 과장하지
않는다. 실제 PIV에 nominal 2D incompressible Navier–Stokes를 적용하기
때문에, 학습된 coefficient는 model discrepancy가 섞인 effective
coefficient가 될 수 있다.

Combustion을 제외하는 것은 불리한 데이터셋을 숨기는 결정이 아니다.
현재 실측 combustion 자료는 속도장이 아니라 OH* intensity 영상이다.
OH* intensity에서 속도와 압력으로 가는 검증된 observation operator와
연소 반응 PDE를 정의하지 않고 현재 Navier–Stokes PINN을 적용하면 서로
다른 문제를 같은 benchmark처럼 보이게 만드는 오류가 된다.

공식 자료 출처:

- [RealPDEBench 공식 GitHub 저장소](https://github.com/AI4Science-WestlakeU/RealPDEBench)
- [RealPDEBench 공식 Hugging Face dataset card](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench)

## Rebuttal 전략

현재 가장 안전한 전략은 실험 개수를 많아 보이게 하는 것이 아니라,
각 수치의 증거 수준을 정확히 구분하는 것이다.

1. **현재 서버에 full artifact가 있는 결과만 “재검증 완료”라고 쓴다.**
   문서에만 남은 숫자는 원래 artifact 회수 또는 재실험 전까지
   `evidence needed`로 표시한다.
2. **PDE parameter recovery는 보편적 우위 주장이 아니라 혼합 결과로
   제시한다.** 모든 PDE·비율의 parameter error를 함께 보여 주고,
   실제 PIV inverse-Re 실패도 숨기지 않는다.
3. **Allen–Cahn cost 문제는 제출 설정의 계보와 새 민감도 실험을
   분리한다.** 새 실험이 제출 당시 실제 설정을 증명한다고 주장하지
   않는다.
4. **non-Cylinder RealPDE 결과는 하나의 데이터셋에서 고른 사례가 아니라
   미리 고정한 세 시나리오 전체 결과로 보고한다.** 유리한 데이터셋만
   골라 쓰지 않고 세 데이터셋과 두 이상치 비율의 전체 순위를 공개한다.
5. **Combustion은 억지 성능표 대신 적용 불가 이유를 투명하게 쓴다.**
   이는 빠진 실험이 아니라 관측변수와 PDE가 맞지 않는 현재 방법의
   범위를 밝히는 것이다.
6. **closest-prior에 불리한 결과도 함께 남긴다.** 이상치를 넣지 않은
   합성 실험과 세 합성 PDE × 세 이상치 비율의 기본 비교에서 direct
   PINN-EBM이 강했던 사실은 숨기지
   않고, naPINN의 장점을 명시적인 관측 선택과 rejection regularization,
   그리고 특정 실제 오염 조건에서의 결과로 좁혀 설명한다.
