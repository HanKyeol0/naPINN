# 실험별 방법 순위, PINN-EBM 구현 감사, 결과 저장 위치

작성일: 2026-07-27  
용도: 저자 내부 검토  
공개 상태: 직접적인 naPINN–PINN-EBM 수치와 순위는 현재 `RESPONSE HOLD`

## 먼저 결론

질문의 두 문장을 그대로 받아들이면 안 된다.

1. **실제 PIV 학습자료에 four-Gaussian 배경 잡음과 큰 이상치를 함께 넣은 기본 강도 실험**에서는, 큰 이상치를 학습 측정행의 10% 또는 15%에 넣었을 때 `naPINN-MSE`가 rMAE와 rMSE 모두 평균 1위였다.
2. 새로 엄격 집계한 **인공적인 세 PDE 데이터**에서는 같은 종류의 four-Gaussian 배경 잡음과 큰 이상치를 5%, 10%, 15%로 넣은 아홉 조건 중 여덟 조건에서 direct PINN-EBM이 rMAE와 rMSE 모두 1위였다. 예외는 Allen–Cahn 15%로, 이 조건에서는 naPINN이 두 지표 모두 평균 1위였다.
3. 이번 Legacy-4G 실제 PIV 캠페인에는 “배경 잡음만 넣고 큰 이상치는 전혀 넣지 않은 별도 학습 실험”이 없다. 그러나 그 질문을 분리하기 위해 **합성 세 PDE의 four-Gaussian 배경 잡음 전용 실험**을 새로 수행했고, 세 PDE 모두 direct PINN-EBM이 naPINN보다 좋았다.
4. 새로 완료한 **추가 인공 오염 없는 natural PIV의 27개 엄격 비교**에서는 PINN-EBM이 1위가 아니었다. rMAE와 rMSE 모두 LAD-PINN이 1위였고, PINN-EBM은 PDE weight 50도 전체 7위였다.
5. 새로 완료한 **Controlled Cylinder·FSI·Foil 144회 실험**에서는 naPINN-MSE가 평균 rMAE 6/6 조건에서 1위였고, 평균 rMSE는 Cylinder와 Foil의 4/6 조건에서 1위였다. FSI의 rMSE 두 조건에서는 OrPINN이 1위였다.
6. 새로 엄격 집계한 **structured Cylinder PIV 102회 실험**에서는 LAD-PINN이 AR(1), 10% drift, spatial burst의 두 field 지표에서 1위였고, naPINN은 20%와 30% drift에서 두 지표 1위였다. 즉, structured corruption 전체에서 naPINN이 항상 가장 좋은 것은 아니다.

따라서 가장 정확한 한 문장 요약은 다음과 같다.

> 실제 PIV의 사전 고정 Legacy-4G 오염 조건에서는 naPINN-MSE가 가장 좋았다. 새 합성 PDE 엄격 재실행에서는 direct PINN-EBM이 9개 큰-이상치 조건 중 8개와 3개 배경-잡음-전용 조건에서 가장 좋았고, Allen–Cahn 15%에서는 naPINN이 가장 좋았다. 추가 인공 오염 없는 natural PIV에서는 LAD-PINN이 가장 좋았고 PINN-EBM은 하위권이었다. 세 RealPDEBench PIV 시나리오의 큰 이상치 실험에서는 naPINN-MSE가 rMAE 6/6에서 가장 낮았지만, FSI rMSE에서는 OrPINN보다 나빴다.

이 새 162-run 결과는 현재 서버에서 감사 가능한 rebuttal 증거지만 제출 표의
bitwise 재현은 아니다. 이전 서버에서 옮겨온 요약은 PINN-EBM이 9/9
조건에서 1위라고 기록한다. 두 계보를 합치거나 더 유리한 쪽만 고르지
않고, 새 strict aggregate와 이전 요약의 차이를 함께 보존한다.

여기서 `background-only`라는 기록은 별도 실험 이름이 아니다. 하나의 오염된 학습자료 안에서 **큰 이상치가 추가되지 않고 four-Gaussian 배경 잡음만 받은 개별 측정 성분**을 뜻한다. 이 이름을 “잡음만 넣은 별도 학습 조건”으로 해석하면 안 된다.

## 순위표를 읽는 방법

- 왼쪽에 있는 방법일수록 평가 오차가 낮다.
- 숫자는 seeds 40–42 평균이다. 후보 탐색 실험도 최종 재확인 값은 seeds 40–42 평균이다.
- `PDE weight 1`과 `PDE weight 50`은 PINN-EBM 공동학습 단계에서 PDE 잔차에 곱한 가중치가 각각 1과 50이라는 뜻이다.
- 평균 순위는 통계적 유의성을 뜻하지 않는다. 반복이 세 번뿐이므로, 작은 평균 차이는 강한 우열 증거로 해석하지 않는다.
- 실제 PIV의 평가값은 공간적으로 분리한 별도 PIV 측정이다. 잡음이 전혀 없는 물리적 참값은 아니다.

## 모든 조건의 순위를 한 표로 정리

아래 표는 현재 확인 가능한 순위를 한곳에 모은 것이다. 숫자와 주의사항은 뒤의 상세 표에서 확인할 수 있다.

| 자료와 오염 조건 | rMAE 순위 | rMSE 순위 | 증거 상태 |
| --- | --- | --- | --- |
| 실제 PIV, 기본 four-Gaussian 배경 잡음과 10% 큰 이상치 | naPINN-MSE → naPINN-q2.9 → OrPINN q=2.9 → naPINN-L1 → PINN-EBM, PDE weight 50 → LAD-PINN → PINN-EBM, PDE weight 1 → MSE-PINN | naPINN-MSE → OrPINN q=2.9 → naPINN-q2.9 → naPINN-L1 → LAD-PINN → PINN-EBM, PDE weight 50 → PINN-EBM, PDE weight 1 → MSE-PINN | 완료된 원 집계 파일 있음 |
| 실제 PIV, 기본 four-Gaussian 배경 잡음과 15% 큰 이상치 | naPINN-MSE → naPINN-q2.9 → naPINN-L1 → OrPINN q=2.9 → LAD-PINN → PINN-EBM, PDE weight 50 → PINN-EBM, PDE weight 1 → MSE-PINN | naPINN-MSE → naPINN-q2.9 → OrPINN q=2.9 → naPINN-L1 → LAD-PINN → PINN-EBM, PDE weight 50 → PINN-EBM, PDE weight 1 → MSE-PINN | 완료된 원 집계 파일 있음 |
| 실제 PIV, 15% 큰 이상치, 배경 배율 2, 이상치 추가 배율 2 | naPINN-MSE → naPINN-L1 → OrPINN q=2.9 → LAD-PINN → naPINN-q2.9 → PINN-EBM, PDE weight 1 → PINN-EBM, PDE weight 50 → MSE-PINN | naPINN-MSE → naPINN-L1 → OrPINN q=2.9 → LAD-PINN → naPINN-q2.9 → PINN-EBM, PDE weight 1 → PINN-EBM, PDE weight 50 → MSE-PINN | 완료된 원 집계 파일 있음; 3/3 seed에서 방향 재현 |
| 실제 PIV, 15% 큰 이상치, 배경 배율 2, 이상치 추가 배율 1 | naPINN-MSE → OrPINN q=2.9 → LAD-PINN → naPINN-L1 → naPINN-q2.9 → PINN-EBM, PDE weight 50 → PINN-EBM, PDE weight 1 → MSE-PINN | naPINN-MSE → OrPINN q=2.9 → LAD-PINN → naPINN-L1 → naPINN-q2.9 → PINN-EBM, PDE weight 50 → PINN-EBM, PDE weight 1 → MSE-PINN | 완료된 원 집계 파일 있음; 2/3 seed에서 방향 재현 |
| 실제 PIV, 15% 큰 이상치, 배경 배율 1, 이상치 추가 배율 0.5 | PINN-EBM, PDE weight 50 → naPINN-MSE → naPINN-q2.9 → PINN-EBM, PDE weight 1 → naPINN-L1 → OrPINN q=2.9 → LAD-PINN → MSE-PINN | naPINN-MSE → PINN-EBM, PDE weight 50 → naPINN-q2.9 → naPINN-L1 → OrPINN q=2.9 → LAD-PINN → PINN-EBM, PDE weight 1 → MSE-PINN | 완료된 원 집계 파일 있음; 두 지표의 1위가 다름 |
| 합성 Allen–Cahn, four-Gaussian 배경 잡음과 5% 큰 이상치 | direct PINN-EBM → naPINN → OrPINN q=2.9 → OrPINN q=1.9 → LAD-PINN → MSE-PINN | direct PINN-EBM → naPINN → OrPINN q=2.9 → OrPINN q=1.9 → LAD-PINN → MSE-PINN | 새 162-run strict aggregate |
| 합성 Allen–Cahn, four-Gaussian 배경 잡음과 10% 큰 이상치 | direct PINN-EBM → naPINN → OrPINN q=2.9 → LAD-PINN → OrPINN q=1.9 → MSE-PINN | direct PINN-EBM → naPINN → OrPINN q=2.9 → LAD-PINN → OrPINN q=1.9 → MSE-PINN | 새 162-run strict aggregate |
| 합성 Allen–Cahn, four-Gaussian 배경 잡음과 15% 큰 이상치 | naPINN → direct PINN-EBM → OrPINN q=2.9 → LAD-PINN → OrPINN q=1.9 → MSE-PINN | naPINN → direct PINN-EBM → OrPINN q=2.9 → LAD-PINN → OrPINN q=1.9 → MSE-PINN | 새 162-run strict aggregate; naPINN은 2/3 seed에서 우세 |
| 합성 Burgers, four-Gaussian 배경 잡음과 5% 큰 이상치 | direct PINN-EBM → naPINN → OrPINN q=2.9 → OrPINN q=1.9 → LAD-PINN → MSE-PINN | direct PINN-EBM → naPINN → OrPINN q=2.9 → OrPINN q=1.9 → LAD-PINN → MSE-PINN | 새 162-run strict aggregate |
| 합성 Burgers, four-Gaussian 배경 잡음과 10% 큰 이상치 | direct PINN-EBM → naPINN → OrPINN q=2.9 → LAD-PINN → OrPINN q=1.9 → MSE-PINN | direct PINN-EBM → naPINN → OrPINN q=2.9 → LAD-PINN → OrPINN q=1.9 → MSE-PINN | 새 162-run strict aggregate |
| 합성 Burgers, four-Gaussian 배경 잡음과 15% 큰 이상치 | direct PINN-EBM → naPINN → OrPINN q=2.9 → LAD-PINN → OrPINN q=1.9 → MSE-PINN | direct PINN-EBM → naPINN → OrPINN q=2.9 → LAD-PINN → OrPINN q=1.9 → MSE-PINN | 새 162-run strict aggregate |
| 합성 lambda–omega, four-Gaussian 배경 잡음과 5% 큰 이상치 | direct PINN-EBM → naPINN → OrPINN q=1.9 → OrPINN q=2.9 → MSE-PINN → LAD-PINN | direct PINN-EBM → naPINN → OrPINN q=1.9 → OrPINN q=2.9 → LAD-PINN → MSE-PINN | 새 162-run strict aggregate |
| 합성 lambda–omega, four-Gaussian 배경 잡음과 10% 큰 이상치 | direct PINN-EBM → naPINN → OrPINN q=1.9 → OrPINN q=2.9 → LAD-PINN → MSE-PINN | direct PINN-EBM → naPINN → OrPINN q=1.9 → OrPINN q=2.9 → LAD-PINN → MSE-PINN | 새 162-run strict aggregate |
| 합성 lambda–omega, four-Gaussian 배경 잡음과 15% 큰 이상치 | direct PINN-EBM → naPINN → OrPINN q=2.9 → OrPINN q=1.9 → LAD-PINN → MSE-PINN | direct PINN-EBM → naPINN → OrPINN q=2.9 → OrPINN q=1.9 → LAD-PINN → MSE-PINN | 새 162-run strict aggregate |
| 합성 Allen–Cahn, four-Gaussian 배경 잡음만, 큰 이상치 0% | direct PINN-EBM → naPINN | direct PINN-EBM → naPINN | 새 3-seed 집계 완료 |
| 합성 Burgers, four-Gaussian 배경 잡음만, 큰 이상치 0% | direct PINN-EBM → naPINN | direct PINN-EBM → naPINN | 새 3-seed 집계 완료 |
| 합성 lambda–omega, four-Gaussian 배경 잡음만, 큰 이상치 0% | direct PINN-EBM → naPINN | direct PINN-EBM → naPINN | 새 3-seed 집계 완료 |
| 실제 PIV Legacy-4G 배경 잡음만, 큰 이상치 0% | 실행하지 않음 | 실행하지 않음 | 합성 background-only와 혼동하면 안 됨 |
| 추가 인공 오염 없는 natural PIV, 새 엄격 비교 | LAD-PINN → naPINN-L1 → OrPINN q=2.9 → naPINN-q2.9 → naPINN-MSE → MSE-PINN → PINN-EBM weight 50 → PINN-EBM weight 10 → PINN-EBM weight 1 | LAD-PINN → OrPINN q=2.9 → naPINN-L1 → naPINN-q2.9 → naPINN-MSE → MSE-PINN → PINN-EBM weight 50 → PINN-EBM weight 10 → PINN-EBM weight 1 | 27/27 완료, strict aggregate 통과 |
| RealPDEBench Controlled Cylinder, 10% 큰 이상치 | naPINN-MSE → naPINN-L1 → OrPINN q=2.9 → LAD-PINN → naPINN-q2.9 → PINN-EBM weight 50 → PINN-EBM weight 1 → MSE-PINN | naPINN-MSE → naPINN-L1 → OrPINN q=2.9 → LAD-PINN → naPINN-q2.9 → PINN-EBM weight 50 → MSE-PINN → PINN-EBM weight 1 | 24/24 완료; 3 seed 평균 |
| RealPDEBench Controlled Cylinder, 15% 큰 이상치 | naPINN-MSE → naPINN-L1 → OrPINN q=2.9 → PINN-EBM weight 50 → PINN-EBM weight 1 → naPINN-q2.9 → LAD-PINN → MSE-PINN | naPINN-MSE → naPINN-L1 → OrPINN q=2.9 → naPINN-q2.9 → LAD-PINN → PINN-EBM weight 50 → PINN-EBM weight 1 → MSE-PINN | 24/24 완료; 3 seed 평균 |
| RealPDEBench Foil, 10% 큰 이상치 | naPINN-MSE → naPINN-q2.9 → naPINN-L1 → OrPINN q=2.9 → LAD-PINN → PINN-EBM weight 1 → PINN-EBM weight 50 → MSE-PINN | naPINN-MSE → naPINN-q2.9 → naPINN-L1 → OrPINN q=2.9 → LAD-PINN → PINN-EBM weight 1 → PINN-EBM weight 50 → MSE-PINN | 24/24 완료; 3 seed 평균 |
| RealPDEBench Foil, 15% 큰 이상치 | naPINN-MSE → naPINN-L1 → OrPINN q=2.9 → PINN-EBM weight 1 → LAD-PINN → PINN-EBM weight 50 → MSE-PINN → naPINN-q2.9 | naPINN-MSE → naPINN-L1 → OrPINN q=2.9 → PINN-EBM weight 1 → LAD-PINN → PINN-EBM weight 50 → MSE-PINN → naPINN-q2.9 | 24/24 완료; naPINN-q2.9 seed 40 실패 포함 |
| RealPDEBench FSI, 10% 큰 이상치 | naPINN-MSE → OrPINN q=2.9 → LAD-PINN → naPINN-L1 → naPINN-q2.9 → MSE-PINN → PINN-EBM weight 50 → PINN-EBM weight 1 | OrPINN q=2.9 → LAD-PINN → naPINN-MSE → naPINN-L1 → MSE-PINN → naPINN-q2.9 → PINN-EBM weight 1 → PINN-EBM weight 50 | 24/24 완료; rMAE와 rMSE의 1위가 다름 |
| RealPDEBench FSI, 15% 큰 이상치 | naPINN-MSE → OrPINN q=2.9 → naPINN-L1 → LAD-PINN → naPINN-q2.9 → PINN-EBM weight 50 → MSE-PINN → PINN-EBM weight 1 | OrPINN q=2.9 → LAD-PINN → naPINN-MSE → naPINN-L1 → naPINN-q2.9 → PINN-EBM weight 50 → PINN-EBM weight 1 → MSE-PINN | 24/24 완료; rMAE와 rMSE의 1위가 다름 |
| Structured PIV, AR(1) 시간상관 오염 | LAD-PINN → OrPINN q=2.9 → naPINN → MAD-PINN → PINN-EBM → MSE-PINN | LAD-PINN → OrPINN q=2.9 → naPINN → MAD-PINN → MSE-PINN → PINN-EBM | 18/18 완료; 3 seed 평균 |
| Structured PIV, 10% persistent drift | LAD-PINN → naPINN → MAD-PINN → OrPINN q=2.9 → PINN-EBM → MSE-PINN | LAD-PINN → naPINN → MAD-PINN → OrPINN q=2.9 → MSE-PINN → PINN-EBM | 18/18 완료; 3 seed 평균 |
| Structured PIV, 20% persistent drift | naPINN → LAD-PINN → MAD-PINN → PINN-EBM → OrPINN q=2.9 → MSE-PINN | naPINN → LAD-PINN → MAD-PINN → PINN-EBM → OrPINN q=2.9 → MSE-PINN | 18/18 완료; 3 seed 평균 |
| Structured PIV, 30% persistent drift | naPINN → LAD-PINN → MAD-PINN → PINN-EBM → OrPINN q=2.9 → MSE-PINN | naPINN → LAD-PINN → MAD-PINN → PINN-EBM → OrPINN q=2.9 → MSE-PINN | 18/18 완료; 3 seed 평균 |
| Structured PIV, spatial burst | LAD-PINN → OrPINN q=2.9 → MAD-PINN → naPINN → MSE-PINN → PINN-EBM | LAD-PINN → OrPINN q=2.9 → MSE-PINN → naPINN → MAD-PINN → PINN-EBM | 18/18 완료; 3 seed 평균 |
| 추가 오염 없는 natural PIV의 과거 내부 진단 | LAD-PINN → MSE-PINN → MAD-PINN → naPINN → PINN-EBM | LAD-PINN → MSE-PINN → MAD-PINN → naPINN → PINN-EBM | active rebuttal evidence에서 제외; 원 집계 파일은 현재 checkout에 없음 |

## 실제 PIV: 사전에 고정한 기본 오염 강도

이 실험은 실제 Cylinder PIV 학습 측정에 다음 두 종류의 오염을 함께 적용했다.

- 모든 학습 측정값에는 제출 코드와 같은 four-Gaussian 배경 잡음을 적용했다.
- 선택된 학습 측정행의 두 속도 성분에는 큰 양의 이상치를 추가했다.

`b=1, o=1`은 모델이나 가중치를 전이했다는 뜻이 아니다. 제출 코드의 배경 잡음 크기와 큰 이상치 추가 크기를 배율 변경 없이 그대로 사용했다는 뜻이다.

| 큰 이상치를 넣은 학습행 비율 | rMAE 순위 | rMSE 순위 |
| --- | --- | --- |
| 10% | 1. naPINN-MSE 0.15517 → 2. naPINN-q2.9 0.20634 → 3. OrPINN q=2.9 0.21426 → 4. naPINN-L1 0.21958 → 5. PINN-EBM, PDE weight 50 0.22404 → 6. LAD-PINN 0.24617 → 7. PINN-EBM, PDE weight 1 0.37016 → 8. MSE-PINN 0.40576 | 1. naPINN-MSE 0.21532 → 2. OrPINN q=2.9 0.25026 → 3. naPINN-q2.9 0.25223 → 4. naPINN-L1 0.26167 → 5. LAD-PINN 0.27780 → 6. PINN-EBM, PDE weight 50 0.33065 → 7. PINN-EBM, PDE weight 1 0.42543 → 8. MSE-PINN 0.42918 |
| 15% | 1. naPINN-MSE 0.16023 → 2. naPINN-q2.9 0.21589 → 3. naPINN-L1 0.22629 → 4. OrPINN q=2.9 0.23106 → 5. LAD-PINN 0.26997 → 6. PINN-EBM, PDE weight 50 0.33611 → 7. PINN-EBM, PDE weight 1 0.42417 → 8. MSE-PINN 0.57380 | 1. naPINN-MSE 0.22266 → 2. naPINN-q2.9 0.26078 → 3. OrPINN q=2.9 0.26514 → 4. naPINN-L1 0.26952 → 5. LAD-PINN 0.30028 → 6. PINN-EBM, PDE weight 50 0.41420 → 7. PINN-EBM, PDE weight 1 0.46394 → 8. MSE-PINN 0.59140 |

출처: `analysis/results/runs/rebuttal_realpde_legacy_4g/aggregation.json`

이 표에서 주장할 수 있는 범위는 좁다. “이상치 비율이 10% 또는 15%이면 언제나 naPINN이 가장 좋다”가 아니라, **이 실제 PIV 자료와 이 오염 생성법과 이 기본 강도에서 naPINN-MSE의 평균 평가 오차가 가장 낮았다**고 말해야 한다.

## 실제 PIV: 오염 강도 후보의 새 seed 재확인

아래 세 조건은 seed 39의 전체 강도 격자에서 사전에 정한 규칙으로 고른 뒤, 후보 선정에 사용하지 않은 seeds 40–42에서 다시 실행한 결과다.

- `b`는 four-Gaussian 배경 잡음의 크기에 곱한 배율이다.
- `o`는 배경 잡음 크기를 기준으로 만든 큰 이상치 추가량에 다시 곱한 배율이다.
- 예를 들어 `b=2, o=2`는 배경 잡음을 기본값의 두 배로 하고, 큰 이상치 추가량도 그 커진 배경 잡음을 기준으로 두 배 적용한 매우 강한 오염이다.

| 오염 조건 | rMAE 순위 | rMSE 순위 | 해석 |
| --- | --- | --- | --- |
| 이상치 15%, b=2, o=2 | 1. naPINN-MSE 0.23278 → 2. naPINN-L1 0.40758 → 3. OrPINN q=2.9 0.43122 → 4. LAD-PINN 0.48595 → 5. naPINN-q2.9 0.62060 → 6. PINN-EBM, PDE weight 1 2.19705 → 7. PINN-EBM, PDE weight 50 2.19932 → 8. MSE-PINN 2.23968 | 1. naPINN-MSE 0.27909 → 2. naPINN-L1 0.44121 → 3. OrPINN q=2.9 0.45362 → 4. LAD-PINN 0.50653 → 5. naPINN-q2.9 0.64531 → 6. PINN-EBM, PDE weight 1 2.17524 → 7. PINN-EBM, PDE weight 50 2.18074 → 8. MSE-PINN 2.23432 | naPINN-MSE가 평균 1위이고 세 seed 모두 두 지표에서 비교 방법보다 좋았다. |
| 이상치 15%, b=2, o=1 | 1. naPINN-MSE 0.34638 → 2. OrPINN q=2.9 0.45725 → 3. LAD-PINN 0.48564 → 4. naPINN-L1 0.57159 → 5. naPINN-q2.9 0.64298 → 6. PINN-EBM, PDE weight 50 1.07398 → 7. PINN-EBM, PDE weight 1 1.16614 → 8. MSE-PINN 1.17399 | 1. naPINN-MSE 0.39434 → 2. OrPINN q=2.9 0.47849 → 3. LAD-PINN 0.50607 → 4. naPINN-L1 0.60507 → 5. naPINN-q2.9 0.66889 → 6. PINN-EBM, PDE weight 50 1.08488 → 7. PINN-EBM, PDE weight 1 1.17153 → 8. MSE-PINN 1.17996 | 평균은 naPINN-MSE 1위지만 두 지표가 함께 좋았던 seed는 2/3이다. |
| 이상치 15%, b=1, o=0.5 | 1. PINN-EBM, PDE weight 50 0.15629 → 2. naPINN-MSE 0.16402 → 3. naPINN-q2.9 0.22817 → 4. PINN-EBM, PDE weight 1 0.22889 → 5. naPINN-L1 0.23610 → 6. OrPINN q=2.9 0.25276 → 7. LAD-PINN 0.26813 → 8. MSE-PINN 0.31447 | 1. naPINN-MSE 0.22586 → 2. PINN-EBM, PDE weight 50 0.24024 → 3. naPINN-q2.9 0.27026 → 4. naPINN-L1 0.27753 → 5. OrPINN q=2.9 0.28384 → 6. LAD-PINN 0.29805 → 7. PINN-EBM, PDE weight 1 0.31714 → 8. MSE-PINN 0.34175 | rMAE는 PINN-EBM, rMSE는 naPINN-MSE가 1위인 혼합 결과다. |

출처: `analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/aggregation.json`, `analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/candidate_validation.json`

## 합성 PDE: four-Gaussian 배경 잡음과 큰 이상치를 함께 넣은 실험

이 실험은 실제 PIV가 아니라 Allen–Cahn, Burgers, lambda–omega 방정식에서 인공적으로 만든 자료를 사용했다. 모든 조건에서 four-Gaussian 배경 잡음과 큰 이상치를 함께 사용했다. 그러므로 이 표 역시 “잡음만 있는 조건”이나 “오염이 없는 원본 자료”의 결과가 아니다.

| PDE와 큰 이상치 비율 | rMAE 순위 | 현재 감사 가능한 rMSE 순위 |
| --- | --- | --- |
| Allen–Cahn, 5% | 1. direct PINN-EBM 0.09162 → 2. naPINN 0.13887 → 3. OrPINN q=2.9 0.16239 → 4. OrPINN q=1.9 0.22200 → 5. LAD-PINN 0.24124 → 6. MSE-PINN 0.34170 | 1. direct PINN-EBM 0.08931 → 2. naPINN 0.13334 → 3. OrPINN q=2.9 0.15061 → 4. OrPINN q=1.9 0.20260 → 5. LAD-PINN 0.20740 → 6. MSE-PINN 0.29860 |
| Allen–Cahn, 10% | 1. direct PINN-EBM 0.07850 → 2. naPINN 0.10426 → 3. OrPINN q=2.9 0.23904 → 4. LAD-PINN 0.28027 → 5. OrPINN q=1.9 0.40603 → 6. MSE-PINN 0.62634 | 1. direct PINN-EBM 0.07880 → 2. naPINN 0.10420 → 3. OrPINN q=2.9 0.20753 → 4. LAD-PINN 0.23688 → 5. OrPINN q=1.9 0.34741 → 6. MSE-PINN 0.52614 |
| Allen–Cahn, 15% | 1. naPINN 0.09202 → 2. direct PINN-EBM 0.10288 → 3. OrPINN q=2.9 0.26754 → 4. LAD-PINN 0.38339 → 5. OrPINN q=1.9 0.67317 → 6. MSE-PINN 0.85525 | 1. naPINN 0.09483 → 2. direct PINN-EBM 0.10169 → 3. OrPINN q=2.9 0.23081 → 4. LAD-PINN 0.32211 → 5. OrPINN q=1.9 0.56108 → 6. MSE-PINN 0.71225 |
| Burgers, 5% | 1. direct PINN-EBM 0.03932 → 2. naPINN 0.07995 → 3. OrPINN q=2.9 0.13535 → 4. OrPINN q=1.9 0.16502 → 5. LAD-PINN 0.19748 → 6. MSE-PINN 0.21481 | 1. direct PINN-EBM 0.05717 → 2. naPINN 0.09801 → 3. OrPINN q=2.9 0.14162 → 4. OrPINN q=1.9 0.16994 → 5. LAD-PINN 0.19375 → 6. MSE-PINN 0.21957 |
| Burgers, 10% | 1. direct PINN-EBM 0.04978 → 2. naPINN 0.08019 → 3. OrPINN q=2.9 0.16237 → 4. LAD-PINN 0.23106 → 5. OrPINN q=1.9 0.26624 → 6. MSE-PINN 0.40365 | 1. direct PINN-EBM 0.06608 → 2. naPINN 0.10150 → 3. OrPINN q=2.9 0.16707 → 4. LAD-PINN 0.22562 → 5. OrPINN q=1.9 0.26457 → 6. MSE-PINN 0.39331 |
| Burgers, 15% | 1. direct PINN-EBM 0.04738 → 2. naPINN 0.08186 → 3. OrPINN q=2.9 0.19082 → 4. LAD-PINN 0.25627 → 5. OrPINN q=1.9 0.39356 → 6. MSE-PINN 0.58642 | 1. direct PINN-EBM 0.06743 → 2. naPINN 0.09858 → 3. OrPINN q=2.9 0.19223 → 4. LAD-PINN 0.24634 → 5. OrPINN q=1.9 0.38632 → 6. MSE-PINN 0.56590 |
| lambda–omega, 5% | 1. direct PINN-EBM 0.04438 → 2. naPINN 0.07350 → 3. OrPINN q=1.9 0.08945 → 4. OrPINN q=2.9 0.12290 → 5. MSE-PINN 0.15286 → 6. LAD-PINN 0.15847 | 1. direct PINN-EBM 0.05843 → 2. naPINN 0.09313 → 3. OrPINN q=1.9 0.10633 → 4. OrPINN q=2.9 0.13693 → 5. LAD-PINN 0.16789 → 6. MSE-PINN 0.17001 |
| lambda–omega, 10% | 1. direct PINN-EBM 0.04975 → 2. naPINN 0.07531 → 3. OrPINN q=1.9 0.13010 → 4. OrPINN q=2.9 0.13906 → 5. LAD-PINN 0.17309 → 6. MSE-PINN 0.28855 | 1. direct PINN-EBM 0.06417 → 2. naPINN 0.09445 → 3. OrPINN q=1.9 0.14909 → 4. OrPINN q=2.9 0.15119 → 5. LAD-PINN 0.18209 → 6. MSE-PINN 0.30402 |
| lambda–omega, 15% | 1. direct PINN-EBM 0.04023 → 2. naPINN 0.07056 → 3. OrPINN q=2.9 0.15263 → 4. OrPINN q=1.9 0.17492 → 5. LAD-PINN 0.19097 → 6. MSE-PINN 0.42187 | 1. direct PINN-EBM 0.05564 → 2. naPINN 0.08890 → 3. OrPINN q=2.9 0.16514 → 4. OrPINN q=1.9 0.19818 → 5. LAD-PINN 0.19909 → 6. MSE-PINN 0.44110 |

출처: `outputs/rebuttal/synthetic_recovery_20260726/aggregation_strict.json`

새 strict aggregate는 162개 run, 54개 세-seed group을 모두 포함한다.
Allen–Cahn 15%에서는 naPINN이 seed 40과 42에서 direct PINN-EBM보다
rMAE가 낮고, PINN-EBM은 seed 41에서 낮았다. 평균과 seed별 방향이 모두
완전히 일방적이지 않으므로 작은 평균 차이를 보편적 우위로 확대하지
않는다. 이전 서버에서 “아홉 조건 모두 PINN-EBM 1위”라고 기록한 요약은
별도 계보로 보존한다.

## Structured PIV: 시간상관·지속 drift·공간 burst

이 실험은 실제 Cylinder PIV의 학습 센서에 구조가 있는 오염을 추가했다.
`AR(1)`은 한 센서의 오차가 다음 시간에도 이어지는 시간상관 오염이고,
`persistent drift`는 고장 난 센서의 bias가 시간에 따라 누적되는 조건이다.
`spatial burst`는 서로 가까운 센서 묶음이 함께 고장 나는 조건이다. 모든
평가는 오염하지 않은 별도 PIV 측정 위치에서 하며, 이 평가자료도 물리적
참값이 아니라 독립적인 실제 측정이다.

| 조건 | rMAE 순위 | rMSE 순위 |
| --- | --- | --- |
| AR(1) | 1. LAD 0.12291 → 2. OrPINN 0.14582 → 3. naPINN 0.15036 → 4. MAD 0.15158 → 5. PINN-EBM 0.20843 → 6. MSE 0.22005 | 1. LAD 0.19727 → 2. OrPINN 0.21065 → 3. naPINN 0.21544 → 4. MAD 0.23113 → 5. MSE 0.27981 → 6. PINN-EBM 0.31694 |
| 10% drift | 1. LAD 0.13199 → 2. naPINN 0.15538 → 3. MAD 0.15738 → 4. OrPINN 0.16528 → 5. PINN-EBM 0.20739 → 6. MSE 0.21017 | 1. LAD 0.20828 → 2. naPINN 0.23260 → 3. MAD 0.23950 → 4. OrPINN 0.24054 → 5. MSE 0.28448 → 6. PINN-EBM 0.31176 |
| 20% drift | 1. naPINN 0.17113 → 2. LAD 0.17190 → 3. MAD 0.17346 → 4. PINN-EBM 0.19577 → 5. OrPINN 0.23666 → 6. MSE 0.30046 | 1. naPINN 0.25266 → 2. LAD 0.25310 → 3. MAD 0.25574 → 4. PINN-EBM 0.29842 → 5. OrPINN 0.32521 → 6. MSE 0.38291 |
| 30% drift | 1. naPINN 0.21797 → 2. LAD 0.23411 → 3. MAD 0.24848 → 4. PINN-EBM 0.29558 → 5. OrPINN 0.29880 → 6. MSE 0.37077 | 1. naPINN 0.30886 → 2. LAD 0.35852 → 3. MAD 0.36792 → 4. PINN-EBM 0.39053 → 5. OrPINN 0.41375 → 6. MSE 0.49075 |
| Spatial burst | 1. LAD 0.11321 → 2. OrPINN 0.12539 → 3. MAD 0.14851 → 4. naPINN 0.15318 → 5. MSE 0.15975 → 6. PINN-EBM 0.25594 | 1. LAD 0.18545 → 2. OrPINN 0.18828 → 3. MSE 0.21735 → 4. naPINN 0.22605 → 5. MAD 0.22623 → 6. PINN-EBM 0.35479 |

LAD가 다섯 조건 중 세 조건에서 두 field 지표 1위이고, naPINN은 drift가
20%와 30%로 커진 두 조건에서 1위다. 이는 naPINN의 장점이 구조화 오염
전체가 아니라 **지속 고장 비율이 큰 조건**에서 가장 뚜렷하다는 뜻이다.
MAD-PINN은 어느 조건에서도 전체 1위가 아니며, 30,000회 LAD 학습 뒤
30,000회 MSE 재학습을 하므로 총 PINN update도 두 배다.

### 같은 30% drift에서 Reynolds 수까지 학습한 결과

고정-Re 실험과 별도로 Reynolds 수를 8,000에서 시작해 학습하고,
자료 metadata의 10,031과 비교했다. `Re 상대오차`는 학습한 Re와 10,031의
차이를 10,031로 나눈 비율이며 낮을수록 좋다.

| 방법 | 학습 Re, 평균 ± 표본 표준편차 | Re 상대오차 | field rMAE | field rMSE |
| --- | ---: | ---: | ---: | ---: |
| MSE | 1,331.81 ± 111.03 | 0.86723 | 0.37221 | 0.49041 |
| LAD | 3,634.24 ± 1,650.42 | 0.63770 | 0.23303 | 0.35739 |
| PINN-EBM | 7,681.08 ± 5,927.76 | **0.51537** | 0.28567 | 0.36947 |
| naPINN | 2,306.53 ± 550.97 | 0.77006 | **0.22825** | **0.31499** |

naPINN은 field 오차가 가장 낮지만 Re 오차는 77.0%다. PINN-EBM은 평균
Re 오차가 네 방법 중 가장 낮아도 51.5%이고, seed별 Re가 14,260.66,
2,757.24, 6,025.34로 크게 흔들렸다. 따라서 어느 방법도 이 조건에서
metadata coefficient를 안정적으로 복원했다고 볼 수 없다.

출처:
`outputs/rebuttal/realpde_recovery_20260726/injected_piv_aggregation_strict.json`

## 추가 오염 없는 natural PIV의 과거 진단

이 결과는 현재 reviewer-facing 근거에서 제외한 과거 내부 진단이다. 이번 Legacy-4G 캠페인의 일부가 아니며, 당시 원 집계 파일도 현재 checkout에는 없다. 다만 “원본 PIV에서 PINN-EBM이 1위였는가?”라는 질문에는 방향을 분명하게 답할 수 있다.

| 지표 | 과거 내부 진단 순위 |
| --- | --- |
| rMAE | 1. LAD-PINN 0.111378 → 2. MSE-PINN 0.139813 → 3. MAD-PINN 0.148101 → 4. naPINN 0.167193 → 5. PINN-EBM 0.246876 |
| rMSE | 1. LAD-PINN 0.183773 → 2. MSE-PINN 0.198120 → 3. MAD-PINN 0.225367 → 4. naPINN 0.266159 → 5. PINN-EBM 0.337063 |

출처: `rebuttal/archive/rebuttal_report_ko_natural_piv_draft.md`, `docs/PROGRESS.md`

MAD-PINN은 두 단계 합계 60,000 PINN update를 사용했으므로 30,000 update 방법들과 계산량이 맞지 않는다. 이 한계와 별개로, PINN-EBM이 1위가 아니라 마지막이라는 방향은 바뀌지 않는다.

## 현재 PINN-EBM 구현은 논문과 얼마나 같은가

### 감사 결론

현재 코드는 **PINN-EBM 논문의 핵심 목적함수와 gradient 경로를 구현한 비교 방법**이다. 그러나 논문과 저자 코드를 그대로 복제한 exact reproduction은 아니다.

즉, 다음 두 문장을 구분해야 한다.

- 말해도 되는 문장: “측정 residual의 EBM 음의 로그우도와 PDE residual을 함께 최소화하고, 공동학습에서 그 gradient가 PINN과 EBM 모두로 흐르는 PINN-EBM 목적함수를 구현했다.”
- 말하면 안 되는 문장: “저자 GitHub의 Navier–Stokes 실험 코드와 모든 구조·학습 설정을 그대로 재현했다.”

감사는 로컬 `reference/PINN-EBM.pdf`와 공식 저장소의 commit `0b74f6f9209d68c79ecb9608b71755977d08f578`을 기준으로 했다.

### 핵심 원리는 일치한다

| 항목 | 논문·저자 코드 | 현재 구현 | 판정 |
| --- | --- | --- | --- |
| 측정 residual | 관측값에서 PINN 예측값을 뺀 residual을 EBM 입력으로 사용 | `y_d - model(X_d)`를 계산하고 두 속도 성분을 하나의 scalar 표본열로 펼침 | 일치 |
| EBM 사용 전 단계 | 먼저 일반적인 제곱오차 PINN을 학습 | 5,000 PINN update 동안 MSE warm-up | 개념 일치 |
| EBM 초기화 | 현재 PINN residual을 고정하고 EBM만 별도로 학습 | residual을 detach한 뒤 EBM만 5,000회 학습 | 개념 일치 |
| 공동학습 data loss | EBM이 계산한 residual 음의 로그우도 | `estimator.mean_nll(..., detach_residual=False)` | 일치 |
| gradient 경로 | 공동학습에서 PINN과 EBM optimizer를 모두 update | 한 optimizer의 서로 다른 parameter group으로 PINN과 EBM을 함께 update | 수학적으로 일치 |
| 전체 목적함수 | `NLL + omega × PDE loss` | `data_weight × NLL + joint_pde_weight × PDE loss` | 일치 |
| Navier–Stokes의 강한 PDE 가중치 | 논문은 PINN-EBM에 `omega=50` 사용 | PDE weight 50 설정을 별도로 실행 | 일치 |
| Gate 사용 여부 | PINN-EBM 자체에는 naPINN gate가 없음 | `pinn_ebm` 분기에는 gate가 없음 | 일치 |

현재 구현의 공동학습 residual은 detach하지 않는다. 따라서 EBM의 음의 로그우도 gradient가 PINN 예측에도 전달된다. 이 부분은 PINN-EBM을 단순히 “EBM을 residual에 맞춘 뒤 점수만 읽는 방법”으로 잘못 구현한 것이 아니다.

코드 감사에서 direct `pinn_ebm` 경로의 심각한 detach 오류나 이중 update 오류는 발견하지 못했다.

### 실험 세부 설정은 다르다

| 항목 | 논문 | 공식 GitHub 코드 | 현재 구현 | 의미 |
| --- | --- | --- | --- | --- |
| 유동 자료 | `cylinder_nektar_wake.mat` 합성/수치해석 wake 자료 | 같은 자료를 읽는 코드 | RealPDEBench의 실제 Cylinder PIV | 같은 알고리즘을 다른 자료에 적용한 비교이지, 원 실험 재현이 아님 |
| PINN 출력 | potential `psi`와 pressure `p`; 미분으로 `u,v`를 구성 | `Net_NS`가 두 값을 출력하고 `u=psi_y`, `v=-psi_x` 사용 | `u,v,p`를 직접 출력하고 continuity residual을 따로 최소화 | PDE parameterization이 다름 |
| PINN 크기 | hidden layer 8개, width 20 | 설정 클래스에는 8×20이 적혀 있으나 실제 `Net_NS`는 4×30으로 hard-code | hidden layer 5개, width 80 | 논문과도, 실행 코드와도 다름 |
| EBM 크기 | hidden layer 3개, width 5, 마지막 층 전 dropout 0.5 | 3×5, dropout 0.5 | 실질적으로 hidden layer 2개, width 32, dropout 없음 | 밀도모형 용량과 regularization이 다름 |
| residual 스케일 | 원 residual 범위 사용 | batch residual의 최소값에서 5 표준편차 아래부터 최대값에서 5 표준편차 위까지 동적 적분 | running standard deviation으로 나눈 뒤 고정 구간 `[-10,10]`에서 적분 | NLL의 수치적 모양과 tail 처리가 다름 |
| 적분 격자 | 명시적 점 수는 본문보다 코드가 구체적 | 동적 구간 1,001점 | 고정 구간 256점 | 정규화 상수 근사 정확도가 다름 |
| PINN 학습량 | Navier–Stokes 100,000회 | 100,000회 설정 | warm-up 5,000 + joint 25,000 = 30,000회 | compute-matched rebuttal protocol에 맞춘 다른 예산 |
| EBM 단독 초기화 | 10,000 PINN update 후 2,000 EBM update | 같은 기본값, 실패 시 반복·증가 | 5,000 PINN update 후 5,000 EBM update, 자동 반복 없음 | 초기화 시점과 안정화 규칙이 다름 |
| learning rate | PINN과 EBM 모두 0.002; 80,000회 후 0.3배 | scheduler 구현 | PINN과 EBM 모두 0.001; scheduler 없음 | 최적화 궤적이 다름 |
| batch 크기 | data 200, collocation 100 | 같은 기본값 | data 2,048, collocation 8,192 | gradient noise와 PDE/data 상대 규모가 다름 |
| PDE parameter | `lambda1`, `lambda2`를 추정 | trainable parameter | 기본 실제 PIV 비교는 Reynolds 수를 고정 | inverse-problem 난이도가 다름 |
| EBM 초기화 합격 검사 | 분포 양 끝의 PDF가 충분히 0에 가까운지 검사하고 실패 시 재시도 | 구현되어 있음 | 구현되어 있지 않음 | 초기화 실패에 대한 방어가 더 약함 |

한 가지 수치적 위험은 별도로 추적할 필요가 있다. 현재 코드는 residual을 running standard deviation으로 나누면서, 원 residual 공간의 정규화된 likelihood로 환산할 때 필요한 `log(standard deviation)` 항을 NLL에 더하지 않는다. 이 항은 현재처럼 standard deviation을 gradient에서 분리하면 한 update 안의 PINN gradient를 직접 바꾸지는 않지만, 기록한 NLL을 원 residual 단위의 likelihood로 해석할 수 없게 한다. 또한 표준화한 residual이 `[-10,10]`을 벗어나더라도 partition function은 계속 그 고정 구간에서만 계산한다. 이 경우 구간 밖 residual의 score는 계산하면서 그 영역의 질량은 정규화 상수에 포함하지 않으므로, 엄밀한 확률밀도 NLL로 해석하기 어렵다.

완료된 실제 PIV 기본 강도의 PINN-EBM checkpoint 12개를 추가 검사한 결과, 전체 학습 residual에서 `|scaled residual| > 10`인 값은 12개 실행 모두 0개였다. 실행별 최대 절댓값은 4.84–6.02였다. 따라서 고정 적분구간 밖 residual 문제는 **이번 기본 강도 완료 결과에서는 실제로 발생하지 않았다**. 그래도 다른 오염 강도와 학습 중간 단계까지 자동으로 안전하다는 뜻은 아니다. 새 실제 PIV 실행의 `metrics.json`에는 최대 절댓값, 적분구간 밖 개수와 비율, 적분구간 경계를 자동으로 저장하도록 보완했다.

공식 GitHub 저장소 자체에도 논문과 실행 코드 사이의 차이가 있다. 데이터셋 설정 클래스에는 Navier–Stokes PINN을 8×20으로 지정하지만, 실제로 선택되는 `Net_NS`는 width 30과 네 hidden layer를 hard-code한다. 따라서 “논문과 공식 코드에 동시에 숫자 하나까지 일치”하는 기준은 원 저장소에서도 성립하지 않는다. 그래도 현재 구현은 양쪽 어느 쪽의 architecture도 그대로 사용하지 않으므로 exact reproduction으로 분류할 수 없다.

### 현재 성능 차이를 어떻게 해석해야 하는가

현재 PINN-EBM 성능이 과거 구현과 달라진 이유를 “이번 구현이 원 저자 코드와 완전히 같아졌기 때문”이라고 설명하면 안 된다. 실제로는 다음 변화가 함께 작용한다.

- 공동학습 NLL gradient가 PINN까지 흐르는 핵심 경로는 올바르게 구현되어 있다.
- PDE weight 50을 별도 조건으로 추가했다.
- 하지만 자료, 신경망, residual 정규화, 적분 격자, batch 크기, 학습량이 원 저자 실험과 크게 다르다.
- 특히 실제 PIV 기본 오염 실험에서는 PINN-EBM의 PDE weight 50이 weight 1보다 훨씬 좋았다. 새 합성 strict rerun에서는 direct PINN-EBM이 9개 중 8개 rMAE 조건에서 1위였고, Allen–Cahn 15%에서는 naPINN이 1위였다.

따라서 이 결과는 “PINN-EBM의 핵심 목적함수를 현재 rebuttal protocol에서 공정하게 비교한 결과”로 표현하고, “저자 코드의 정확한 재현 성능”으로 표현하지 않는 것이 안전하다.

## 결과는 어디에 저장되는가

### 이미 끝난 실험

이번 세션 전에 시작되어 완료된 Legacy-4G 실행은 다음 위치에 있다.

- 기본 강도 실제 PIV 실행과 모델: `analysis/results/runs/rebuttal_realpde_legacy_4g/`
- 강도 탐색 실행과 모델: `analysis/results/runs/rebuttal_realpde_legacy_4g_scale/`
- 새 seed 후보 재확인 실행과 모델: `analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/`

각 개별 실행 폴더에는 `config.yaml`, `run_metadata.json`, `train_history.jsonl`, 중간 `step_*.pt`, `final.pt`, `metrics.json`이 저장된다. 이 기존 파일들은 집계 JSON과 보고서가 현재 경로와 checksum을 기록하고 있으므로 이동하지 않았다. 기존 경로를 강제로 바꾸면 현재 evidence chain이 끊길 수 있다.

### 새 runner로 실행한 실험의 표준 위치

새 실행의 표준 위치를 `outputs/`로 변경했다.

| 실행 종류 | 기본 저장 위치 |
| --- | --- |
| 일반 `pinnlab.train` 실행 | `outputs/training/` |
| 합성 rebuttal 실행 | `outputs/rebuttal/synthetic/` |
| 일반 실제 PIV rebuttal 실행 | `outputs/rebuttal/realpde/` |
| Legacy-4G 기본 강도 | `outputs/rebuttal/realpde_legacy_4g/` |
| Legacy-4G 강도 탐색 | `outputs/rebuttal/realpde_legacy_4g_scale/` |
| Legacy-4G 후보 재확인 | `outputs/rebuttal/realpde_legacy_4g_candidates/` |
| 세 합성 PDE 5%·10%·15% 기본 비교 | `outputs/rebuttal/synthetic_recovery_20260726/` |
| 합성 EMA·cost·noise·selector 보충 실험 | `outputs/rebuttal/synthetic_supplement_recovery_20260726/` |
| PINN-EBM PDE weight held-out | `outputs/rebuttal/synthetic_pinn_ebm_weight_heldout_20260726/` |
| Synthetic MAD-PINN | `outputs/rebuttal/synthetic_mad_recovery_20260726/` |
| 35k PINN-update baseline | `outputs/rebuttal/synthetic_compute35_recovery_20260726/` |
| Controlled Cylinder·FSI·Foil 144-run | `outputs/rebuttal/realpdebench_multidataset/` |
| Structured PIV fixed-Re·inverse-Re·MAD | `outputs/rebuttal/realpde_recovery_20260726/` |
| 공식 PINN-EBM A/B | `outputs/rebuttal/pinn_ebm_upstream/` |
| queue 상태와 job log | `outputs/status/` |

실제 PIV runner에는 `--output-root`를 추가했고, 모든 queue가 자신이 검사하는 output root를 child runner에도 명시적으로 전달하도록 변경했다. 따라서 queue가 완료 여부를 확인하는 위치와 실제 모델이 저장되는 위치가 어긋나지 않는다.

일반 `pinnlab.train`도 `final.pt`뿐 아니라 최종 rMAE, rMSE, best rMSE, update 수, wall time, GPU memory를 `metrics.json`에 저장하도록 보완했다.

큰 checkpoint를 Git에 직접 넣으면 저장소가 급격히 커지므로 `outputs/`의 생성 파일은 Git에서 제외한다. 대신 폴더 구조와 보존 정책은 `outputs/README.md`에 기록한다. 서버 백업이나 장기 보존이 필요하면 `outputs/` 전체를 별도 artifact storage에 동기화해야 한다.

## Rebuttal 전략

1. “10%와 15% 이상치에서 naPINN이 가장 좋다” 앞에 반드시 **실제 PIV의 사전 고정 Legacy-4G 기본 강도**라는 조건을 붙인다.
2. 새 합성 PDE strict rerun에서 direct PINN-EBM이 9개 중 8개 조건의 rMAE·rMSE 1위였다는 불리한 결과와, Allen–Cahn 15%에서는 naPINN이 1위였다는 예외를 함께 밝힌다. 이전 서버에서 “아홉 조건 모두 PINN-EBM 1위”라고 기록한 요약도 별도 계보로 보존한다. 이 결과 때문에 보편적 우위 주장은 피하고, 자료와 오염 형태에 따른 조건부 장점으로 주장 범위를 줄인다.
3. `background-only`를 noise-only experiment라고 부르지 않는다. “같은 오염 학습자료 안에서 큰 이상치가 추가되지 않은 측정 성분”이라고 풀어 쓴다.
4. natural PIV에서 PINN-EBM이 1위였다고 쓰지 않는다. 과거 내부 진단은 LAD-PINN 1위, PINN-EBM 마지막이었다. 다만 이 진단은 현재 active rebuttal evidence에서 제외되어 있으므로 reviewer 답변의 주 근거로 되살리지 않는다.
5. PINN-EBM 비교는 “핵심 목적함수와 공동 gradient를 구현한 paper-aligned comparator”라고 설명한다. “공식 저장소를 그대로 실행한 exact reproduction”이라고 설명하지 않는다.
6. reviewer에게 직접 naPINN–PINN-EBM 수치를 공개할지는 여전히 저자 결정 사항이다. 공개한다면 실제 PIV의 긍정적 결과와 합성 PDE의 불리한 결과를 함께 제시해야 방어 가능하다.

## 감사에 사용한 주요 근거

- `reference/PINN-EBM.pdf`
- 공식 저장소: [ppilar/PINN-EBM](https://github.com/ppilar/PINN-EBM)
- `pinnlab/utils/ebm.py`
- `scripts/rebuttal/run_realpdebench.py`
- `pinnlab/experiments/realpdebench_cylinder.py`
- `configs/experiment/realpdebench_cylinder_common.yaml`
- `configs/experiment/realpdebench_cylinder_pinn_ebm.yaml`
- `configs/experiment/realpdebench_cylinder_pinn_ebm_equal_weight.yaml`
- `rebuttal/rebuttal_report_ko.md`
- `rebuttal/archive/rebuttal_report_ko_natural_piv_draft.md`
