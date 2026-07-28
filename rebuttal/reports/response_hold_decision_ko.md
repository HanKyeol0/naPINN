# RESPONSE HOLD 판단 근거: naPINN vs direct PINN-EBM 직접 비교 수치를 공개할 것인가

작성일: 2026-07-28
용도: 저자 결정용
관련: `rebuttal/response_matrix.md` 전략 #5, `rebuttal/responses/*.md`의 `[[HOLD]]` 구간

> **저자 결정 (2026-07-28): C안 채택.** 수치를 공개하되 regime 경계 발견으로
> 재구성한다. `rebuttal/responses/{ac,aoJS,6SDM}.md`가 이에 맞게 작성됐다.
> 아래 근거 분석은 결정 기록으로 보존한다.
>
> 결정 후 추가로 검증된 사실이 C안을 예상보다 크게 강화했다. 아래
> "C안을 뒷받침하는 검증 결과" 절을 참조하라.

## 결정해야 하는 것

`[[HOLD]]`로 표시된 구간, 즉 **세 합성 PDE × 세 이상치 비율 9조건 중 8조건에서
gate 없는 direct PINN-EBM(= Pilar–Wahlström 재현)이 naPINN보다 낮은 rMAE·rMSE를
기록했다**는 사실과 그 수치를 reviewer 답변에 쓸 것인가.

이것은 실험을 더 할지의 문제가 아니다. 실험은 끝났고 162/162 strict aggregate로
검증됐다. 순수하게 공개 범위의 문제다.

## 결정에 필요한 사실

### 불리한 쪽

| 항목 | 수치 | 근거 |
| --- | --- | --- |
| 합성 9조건 중 PINN-EBM 우세 | 8/9 (rMAE·rMSE 동시) | `synthetic_recovery_20260726/aggregation_strict.json` |
| 예외 | Allen–Cahn 15%만 naPINN 우세 (0.09202 vs 0.10288). 그마저 seed 41은 PINN-EBM 우세 | 동일 |
| 큰 이상치 없는 배경잡음만 | 3/3 조건 모두 PINN-EBM 우세 | `synthetic_background_only/aggregation.json` |
| 이상치 순위화 능력 | Gate AUROC 0.98766 ≈ raw EBM AUROC 0.98770 | `rebuttal_realpde_legacy_4g/aggregation.json` |
| PDE parameter 복원 | 9조건 중 PINN-EBM 5승, naPINN 2승 | `synthetic_recovery_20260726` |
| 실제 PIV inverse-Re | Re 상대오차 최저는 PINN-EBM 51.5%, naPINN 77.0% | `realpde_recovery_20260726` |

가장 위험한 문장은 이것이다: **"가장 가까운 선행 연구가 저자들 자신의 벤치마크
9조건 중 8조건에서 제안 방법을 이긴다."** 이 한 문장은 인용하기 쉽고, 이미 2점인
originality·significance 점수를 정당화하는 근거가 된다.

### 유리한 쪽

| 항목 | 수치 |
| --- | --- |
| 실제 Cylinder PIV, 사전 고정 기본 오염강도 10% | naPINN 0.15517/0.21532 vs 최강 baseline OrPINN-q2.9 0.21426/0.25026 (rMAE 27.6% 감소), PINN-EBM w50 0.22404/0.33065 |
| 동일 15% | naPINN 0.16023/0.22266 vs OrPINN 0.23106/0.26514 (30.7% 감소) |
| seed 재현성 | seeds 40·41·42 각각에서 두 지표 모두 우세 (3/3) |
| Cylinder·FSI·Foil 144-run | naPINN-MSE 평균 rMAE 6/6 조건 1위 |
| Structured drift 20%·30% | naPINN 두 지표 1위 |

## 비공개를 선택했을 때의 위험

1. **aoJS의 첫 번째 질문이 바로 이것이다.** "Can the authors compare directly
   against the closest EBM-based unknown-measurement-noise PINN baseline?
   ... this comparison is important to isolate the contribution of the proposed
   reliability gate and rejection-cost regularization." 질문에 답하지 않으면
   미응답으로 처리된다. aoJS는 confidence 4이고 rating 3(borderline)이라 유일하게
   설득 가능한 reviewer다.
2. **AC의 메타리뷰도 [31]을 명시적으로 지적했다.** AC는 "doesn't fairly account
   for existing prior work such as Pilar and Wahlstrom (ref 31)"을 두 개 약점 중
   하나로 꼽았다. 즉 이 항목은 AC가 건 조건이다.
3. **수치 없이 메커니즘 차이만 설명하면 역효과가 크다.** 두 reviewer가 명시적으로
   요구한 비교를, 실험을 수행한 상태에서 방법론 서술로만 답하면 "돌려봤는데 졌구나"로
   읽힐 가능성이 높다. 이는 숨긴 것으로 보이는 최악의 형태다.
4. **재현 가능한 실험이다.** PINN-EBM은 공개 논문이고 우리 코드로 재현 가능하다.
   reviewer나 후속 연구가 직접 돌리면 같은 결과가 나온다.

## 공개를 선택했을 때의 위험

1. 위의 "가장 위험한 문장"을 reviewer에게 직접 제공하게 된다.
2. rating 2인 6SDM·6XZg는 이 정보를 받고 점수를 올릴 유인이 적다.
3. 논문의 기여 주장을 rebuttal 중에 축소해야 하며, 축소된 주장이 NeurIPS
   기준을 넘는지는 별개 문제다.

## 판단에 결정적인 비대칭

AC 메타리뷰는 이미 **합성 벤치마크의 가치를 낮게 평가했다**: "The problem with
synthetic problems is that our imagination when dreaming up forms of observation
model-error is limited - nature can be much more perverse." 그리고 수락 조건을
명시했다: "If the authors are able to add a compelling real example, showing that
their approach can deliver a significant benefit in a real problem, that may move
the reviewers to reconsider their current scores."

즉 **AC가 요구한 것은 실제 데이터에서의 이득이고, 우리가 진 곳은 AC가 이미
평가절하한 합성 벤치마크다.** 이 비대칭이 공개 결정의 핵심이다. 합성에서 지는
것은 AC가 세운 수락 기준에 직접 해당하지 않는 반면, [31] 비교를 회피하는 것은
AC가 세운 두 조건 중 하나를 위반한다.

## 세 가지 답변 방향

### A안 — 공개하고 주장을 좁힌다 (권고)

합성 열세를 먼저 밝히고, 실제 PIV 결과를 주 근거로 삼으며, 기여를 메커니즘 수준
(detached density 목적함수, 명시적 포함 결정, gated 재구성 loss, rejection
regularization)과 특정 regime으로 한정한다. 현재
`rebuttal/responses/{ac,aoJS,6SDM}.md`의 초안이 이 형태다.

- 장점: AC·aoJS의 명시 요구를 모두 충족. 정직성이 방어 가능.
- 단점: 불리한 문장을 직접 제공.

### B안 — 비공개, 메커니즘 차이만 서술

`[[HOLD]]` 구간을 삭제하고 목적함수·gradient 경로 차이만 설명한다.

- 장점: 불리한 수치를 제공하지 않음.
- 단점: aoJS Q1 미응답. AC 조건 미충족. 발각 시 신뢰도 손상이 수치 자체보다 크다.

### C안 — 공개하되 regime 발견으로 재구성

A안과 같은 수치를 쓰되, 프레이밍을 "패배 인정"이 아니라 **경계 조건의 발견**으로
바꾼다. 즉 "직접 likelihood 최적화는 조밀한 배경잡음에서 우수하고, 명시적 gating은
희소하고 심한 실제 센서 오염에서 우수하다. 우리는 그 경계를 실제 PIV에서
정량화했다."

- 근거가 되는 사실: 배경잡음만 있을 때 3/3 PINN-EBM 우세 → 이상치 비율이 오를수록
  격차 축소 → Allen–Cahn 15%에서 역전 → 실제 PIV gross-outlier에서 naPINN 우세.
  이 단조적 경향은 우연이 아니라 해석 가능한 패턴이다.
- 장점: 불리한 데이터를 기여의 일부로 전환. 논문의 서사가 "더 좋은 방법"에서
  "언제 무엇이 좋은가"로 바뀌지만 NeurIPS에서 방어 가능한 형태다.
- 단점: 제출된 논문의 주장과 rebuttal의 주장이 달라진다는 지적을 받을 수 있다.

## 권고

**C안을 권고한다.** A안의 정직성을 유지하면서, 같은 데이터로 더 방어 가능한
주장을 만든다. B안은 권하지 않는다 — aoJS는 유일하게 설득 가능한 reviewer이고
그의 첫 질문이 바로 이 비교이기 때문이다.

C안을 택할 경우 현재 초안에서 바꿔야 할 것은 프레이밍 문장뿐이며 수치는 그대로다.
`aoJS.md`의 Q1 마지막 문단과 `ac.md`의 3절을 regime 서술로 다시 쓰면 된다.

## C안을 뒷받침하는 검증 결과 (결정 후 확인)

C안을 실제로 작성하기 위해 전 캠페인의 순위를 다시 계산한 결과, regime 경계가
예상보다 훨씬 선명했다. **완전한 역전**이다.

| 설정 | direct PINN-EBM | naPINN |
| --- | --- | --- |
| 합성 12조건 (PDE가 정확) | **11회 1위**, 1회 2위 | 11회 2위 |
| 실제 PIV 13조건 (nominal PDE) | **1위 없음, 3위도 없음**. 최고 4위 | 10회 1위 |

세부 근거:

- 합성: 9개 이상치 조건(6방법) + 3개 배경잡음만 조건(2방법) = 180 run.
  PINN-EBM이 11/12 최저 rMAE. 예외는 Allen–Cahn 15%.
- 실제 PIV 13조건에서 PINN-EBM 순위: Legacy-4G `[5,7]`·`[6,7]`,
  Cylinder `[6,7]`·`[4,5]`, Foil `[6,7]`·`[4,6]`, FSI `[7,8]`·`[6,8]`,
  structured `5·5·4·4·6`. **최고가 4위**이고 FSI 10%는 8/8, spatial burst는
  6/6으로 plain MSE보다도 나쁘다.
- naPINN은 gross-outlier 8조건 전부 1위, drift 20%·30% 1위 = 10/13.
- LAD는 structured 5조건 **전부**에서 direct PINN-EBM보다 좋다. 즉 "실제
  데이터에서는 잔차 밀도를 직접 최적화하는 쪽이 불리하다"는 방향이
  naPINN만의 효과가 아니라 일관된 패턴이다.

**이것이 AC의 지적과 정확히 맞물린다.** AC는 "물리 모형 불일치를 명시적으로
다루지 않으면 관측 오차 모형이 그 일을 떠안아 매우 복잡한 오차 분포가 생긴다"고
썼다. 우리 데이터는 그 예측 그대로다 — 오차 밀도에 가장 많이 의존하는 방법이
모형이 맞을 때 가장 강하고 모형이 틀릴 때 가장 크게 무너진다. 따라서 합성에서의
패배가 약점이 아니라 **경계를 정의하는 증거**가 된다.

주의: 관측 오차와 모형 불일치를 분리하지 못하므로 이는 25개 조건에 걸친 견고한
경험적 연관이지 인과 규명이 아니다. 답변 세 편 모두에 이 한계를 명시했다.

## 남은 작업

1. ~~A·B·C 중 선택~~ → C안 확정.
2. 실행 중인 캠페인 종료 후 `[[PENDING]]` 슬롯 치환. 여유 글자수는
   6SDM 1,031 / 6XZg 2,415 / aoJS 937 / AC 609자다. AC는 PENDING이 없다.
3. 제출 직전 HTML 주석(근거 경로)과 `[[HOLD]]`·`[[PENDING]]` 표시 제거.
4. `rebuttal/response_matrix.md` 전략 #5와 `docs/PROGRESS.md` blocker 1을
   C안 확정에 맞게 갱신 — 완료.
