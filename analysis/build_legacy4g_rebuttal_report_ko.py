"""Build the internal Korean legacy-4G rebuttal results report."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXACT_PATH = Path(
    "analysis/results/runs/rebuttal_realpde_legacy_4g/aggregation.json"
)
SELECTION_PATH = Path(
    "analysis/results/runs/rebuttal_realpde_legacy_4g_scale/"
    "seed39_scale_selection.json"
)
CONFIRMATION_PATH = Path(
    "analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/"
    "aggregation.json"
)
VALIDATION_PATH = Path(
    "analysis/results/runs/rebuttal_realpde_legacy_4g_candidates/"
    "candidate_validation.json"
)
HANDOFF_PATH = Path("rebuttal/CROSS_SERVER_EXPERIMENT_HANDOFF.md")
MASTER_REPORT_PATH = Path("rebuttal/rebuttal_report_ko.md")
REVIEWER_PATH = Path("rebuttal/reviewer_comments.md")
PLAN_PATH = Path("rebuttal/legacy4g_scale_selection_plan.yaml")

METHOD_ORDER = {
    ("mse", None): 1,
    ("lad", None): 2,
    ("orpinn_q29", None): 3,
    ("pinn_ebm", 1.0): 4,
    ("pinn_ebm", 50.0): 5,
    ("napinn", None): 6,
    ("napinn_lad", None): 7,
    ("napinn_q29", None): 8,
}
METHOD_LABEL = {
    ("mse", None): "MSE-PINN",
    ("lad", None): "LAD-PINN",
    ("orpinn_q29", None): "OrPINN q=2.9",
    ("pinn_ebm", 1.0): "PINN-EBM (PDE weight 1)",
    ("pinn_ebm", 50.0): "PINN-EBM (PDE weight 50)",
    ("napinn", None): "naPINN-MSE",
    ("napinn_lad", None): "naPINN-L1",
    ("napinn_q29", None): "naPINN-q2.9",
}
SELECTED_METHOD = {
    "napinn_mse": ("napinn", None),
    "napinn_l1": ("napinn_lad", None),
    "napinn_q29": ("napinn_q29", None),
}
DATASET_RAW_SOURCE_IDS = {
    "campaign_status": [
        "exact_aggregation",
        "scale_selection",
        "confirmation_aggregation",
    ],
    "exact_results": ["exact_aggregation"],
    "exact_consistency": ["exact_aggregation"],
    "severity": ["exact_aggregation", "confirmation_aggregation"],
    "candidate_summary": [
        "confirmation_aggregation",
        "candidate_validation",
    ],
    "candidate_seed": ["confirmation_aggregation"],
    "candidate_gains": [
        "confirmation_aggregation",
        "candidate_validation",
    ],
    "whole_suite_context": ["master_rebuttal_report"],
    "reviewer_routing": ["reviewer_comments"],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_key(group: dict[str, Any]) -> tuple[str, float | None]:
    weight = group.get("direct_pinn_ebm_joint_pde_weight")
    return (
        str(group["method"]),
        None if weight is None else float(weight),
    )


def cell_key(group: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(group["ratio"]),
        float(group["background_scale_multiplier"]),
        float(group["gross_scale_multiplier"]),
    )


def metric(group: dict[str, Any], name: str) -> dict[str, Any] | None:
    return group["metrics"].get(name)


def mean(group: dict[str, Any], name: str) -> float:
    value = metric(group, name)
    if value is None:
        raise KeyError(f"{group['group_id']} has no {name}")
    return float(value["mean"])


def std(group: dict[str, Any], name: str) -> float:
    value = metric(group, name)
    if value is None:
        raise KeyError(f"{group['group_id']} has no {name}")
    return float(value["sample_std"])


def per_seed(group: dict[str, Any], name: str, seed: int) -> float:
    value = metric(group, name)
    if value is None:
        raise KeyError(f"{group['group_id']} has no {name}")
    return float(value["per_seed"][str(seed)])


def optional_mean(group: dict[str, Any], name: str) -> float | None:
    value = metric(group, name)
    return None if value is None else float(value["mean"])


def optional_std(group: dict[str, Any], name: str) -> float | None:
    value = metric(group, name)
    return None if value is None else float(value["sample_std"])


def relative_gain(baseline: float, selected: float) -> float:
    return 100.0 * (baseline - selected) / baseline


def fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def pm(group: dict[str, Any], name: str, digits: int = 5) -> str:
    return f"{fmt(mean(group, name), digits)} ± {fmt(std(group, name), digits)}"


def make_sources() -> list[dict[str, Any]]:
    return [
        {
            "id": "exact_aggregation",
            "label": "Legacy-4G exact-center complete-only aggregation",
            "path": str(EXACT_PATH),
        },
        {
            "id": "scale_selection",
            "label": "Frozen seed-39 scale-grid selection",
            "path": str(SELECTION_PATH),
        },
        {
            "id": "confirmation_aggregation",
            "label": "Fresh-seed candidate complete-only aggregation",
            "path": str(CONFIRMATION_PATH),
        },
        {
            "id": "candidate_validation",
            "label": "Sealed candidate validation",
            "path": str(VALIDATION_PATH),
        },
        {
            "id": "frozen_plan",
            "label": "Predeclared legacy-4G scale-selection plan",
            "path": str(PLAN_PATH),
        },
        {
            "id": "handoff",
            "label": "Cross-server experiment handoff and claim boundaries",
            "path": str(HANDOFF_PATH),
        },
        {
            "id": "master_rebuttal_report",
            "label": "Current Korean rebuttal evidence report",
            "path": str(MASTER_REPORT_PATH),
        },
        {
            "id": "reviewer_comments",
            "label": "Verbatim reviewer and Area Chair comments",
            "path": str(REVIEWER_PATH),
        },
    ]


def materialize_report_datasets(
    datasets: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Round-trip report rows through the SQL disclosed in widget provenance."""
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE report_rows "
        "(dataset_name TEXT NOT NULL, row_order INTEGER NOT NULL, "
        "row_json TEXT NOT NULL)"
    )
    for dataset_name, rows in datasets.items():
        connection.executemany(
            "INSERT INTO report_rows VALUES (?, ?, ?)",
            [
                (
                    dataset_name,
                    index,
                    json.dumps(row, ensure_ascii=False, allow_nan=False),
                )
                for index, row in enumerate(rows)
            ],
        )

    materialized: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for dataset_name, expected_rows in datasets.items():
        sql = (
            "SELECT row_json FROM report_rows "
            f"WHERE dataset_name = '{dataset_name}' ORDER BY row_order"
        )
        rows = [
            json.loads(row_json)
            for (row_json,) in connection.execute(sql).fetchall()
        ]
        if rows != expected_rows:
            raise AssertionError(
                f"SQL materialization changed dataset {dataset_name}"
            )
        materialized[dataset_name] = rows
        sources.append(
            {
                "id": f"report_dataset_{dataset_name}",
                "label": f"SQL-materialized report dataset: {dataset_name}",
                "path": "analysis/build_legacy4g_rebuttal_report_ko.py",
                "query": {"sql": sql},
            }
        )
    connection.close()
    return materialized, sources


def severity_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: dict[str, tuple[float, float, float]] = {}
    for group in aggregate["groups"]:
        key = cell_key(group)
        for artifact in group["input_artifacts_by_seed"].values():
            artifacts[str(artifact["path"])] = key

    values: dict[tuple[float, float, float], dict[str, list[float]]] = {}
    for path_text, key in artifacts.items():
        bucket = values.setdefault(key, {"background": [], "gross_min": [], "gross_max": []})
        with np.load(Path(path_text), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        corruption = metadata["corruption"]
        bucket["background"].append(
            float(
                corruption["realized_background"][
                    "sample_std_over_reference_sample_std"
                ]
            )
        )
        gross = corruption["realized_gross_offsets"]["reference_std_units"]
        bucket["gross_min"].append(float(gross["min"]))
        bucket["gross_max"].append(float(gross["max"]))

    rows = []
    for key, bucket in sorted(values.items()):
        ratio, background, gross = key
        rows.append(
            {
                "cell": f"{int(round(ratio * 100))}% · b={background:g}, o={gross:g}",
                "ratio_pct": int(round(ratio * 100)),
                "background_multiplier": background,
                "gross_multiplier": gross,
                "background_std_ratio_min": min(bucket["background"]),
                "background_std_ratio_max": max(bucket["background"]),
                "gross_offset_ratio_min": min(bucket["gross_min"]),
                "gross_offset_ratio_max": max(bucket["gross_max"]),
            }
        )
    return rows


def build_artifact() -> dict[str, Any]:
    exact = load_json(EXACT_PATH)
    selection = load_json(SELECTION_PATH)
    confirmation = load_json(CONFIRMATION_PATH)
    validation = load_json(VALIDATION_PATH)

    assert exact["n_eligible_runs"] == 48
    assert exact["n_groups"] == 16
    assert len(exact["paired_input_blocks"]) == 6
    assert confirmation["n_eligible_runs"] == 72
    assert confirmation["n_groups"] == 24
    assert len(confirmation["paired_input_blocks"]) == 9
    assert validation["verified_full_run_count"] == 72
    assert validation["response_status"] == "RESPONSE HOLD"
    assert selection["selection_count"] == 3

    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    sources = make_sources()

    exact_rows = []
    exact_by_cell: dict[
        tuple[float, float, float], dict[tuple[str, float | None], dict[str, Any]]
    ] = {}
    for group in exact["groups"]:
        key = cell_key(group)
        method = group_key(group)
        exact_by_cell.setdefault(key, {})[method] = group
        exact_rows.append(
            {
                "method_order": METHOD_ORDER[method],
                "method": METHOD_LABEL[method],
                "ratio_label": f"{int(round(key[0] * 100))}%",
                "ratio_pct": int(round(key[0] * 100)),
                "rMAE": mean(group, "rMAE"),
                "rMAE_std": std(group, "rMAE"),
                "rMSE": mean(group, "rMSE"),
                "rMSE_std": std(group, "rMSE"),
                "momentum_rms": mean(group, "pde_momentum_rms"),
                "continuity_rms": mean(group, "continuity_rms"),
                "gross_rejection_pct": (
                    None
                    if optional_mean(group, "gross_outlier_rejection_rate") is None
                    else 100.0
                    * float(optional_mean(group, "gross_outlier_rejection_rate"))
                ),
                "gross_rejection_std_pct": (
                    None
                    if optional_std(group, "gross_outlier_rejection_rate") is None
                    else 100.0
                    * float(optional_std(group, "gross_outlier_rejection_rate"))
                ),
                "background_rejection_pct": (
                    None
                    if optional_mean(group, "background_only_rejection_rate") is None
                    else 100.0
                    * float(optional_mean(group, "background_only_rejection_rate"))
                ),
                "background_rejection_std_pct": (
                    None
                    if optional_std(group, "background_only_rejection_rate") is None
                    else 100.0
                    * float(optional_std(group, "background_only_rejection_rate"))
                ),
                "gate_auroc": optional_mean(group, "gross_outlier_detection_auroc"),
                "raw_ebm_auroc": optional_mean(
                    group, "estimator_gross_outlier_detection_auroc"
                ),
            }
        )
    exact_rows.sort(key=lambda row: (row["ratio_pct"], row["method_order"]))

    exact_consistency = []
    exact_headline: dict[int, dict[str, Any]] = {}
    for key in sorted(exact_by_cell):
        ratio_pct = int(round(key[0] * 100))
        methods = exact_by_cell[key]
        selected = methods[("napinn", None)]
        non_napinn = [
            group
            for method, group in methods.items()
            if not method[0].startswith("napinn")
        ]
        best_rmae = min(non_napinn, key=lambda group: mean(group, "rMAE"))
        best_rmse = min(non_napinn, key=lambda group: mean(group, "rMSE"))
        assert group_key(best_rmae) == group_key(best_rmse)
        baseline = best_rmae
        seed_wins = 0
        for seed in (40, 41, 42):
            if (
                per_seed(selected, "rMAE", seed)
                < per_seed(baseline, "rMAE", seed)
                and per_seed(selected, "rMSE", seed)
                < per_seed(baseline, "rMSE", seed)
            ):
                seed_wins += 1
        row = {
            "ratio_label": f"{ratio_pct}%",
            "ratio_pct": ratio_pct,
            "selected_method": "naPINN-MSE",
            "baseline_method": METHOD_LABEL[group_key(baseline)],
            "selected_rMAE": mean(selected, "rMAE"),
            "baseline_rMAE": mean(baseline, "rMAE"),
            "rMAE_gain_pct": relative_gain(
                mean(baseline, "rMAE"), mean(selected, "rMAE")
            ),
            "selected_rMSE": mean(selected, "rMSE"),
            "baseline_rMSE": mean(baseline, "rMSE"),
            "rMSE_gain_pct": relative_gain(
                mean(baseline, "rMSE"), mean(selected, "rMSE")
            ),
            "both_metrics_seed_wins": f"{seed_wins}/3",
        }
        exact_consistency.append(row)
        exact_headline[ratio_pct] = {
            "selected": selected,
            "baseline": baseline,
            **row,
        }

    confirmation_by_cell: dict[
        tuple[float, float, float], dict[tuple[str, float | None], dict[str, Any]]
    ] = {}
    for group in confirmation["groups"]:
        confirmation_by_cell.setdefault(cell_key(group), {})[
            group_key(group)
        ] = group

    candidate_rows = []
    candidate_seed_rows = []
    candidate_gain_rows = []
    validation_by_rank = {
        int(item["rank"]): item for item in validation["candidate_conclusions"]
    }
    for rank in sorted(validation_by_rank):
        conclusion = validation_by_rank[rank]
        key = (
            float(conclusion["ratio"]),
            float(conclusion["background_scale_multiplier"]),
            float(conclusion["gross_scale_multiplier"]),
        )
        methods = confirmation_by_cell[key]
        selected_method = SELECTED_METHOD[
            str(conclusion["selected_napinn_variant"])
        ]
        selected = methods[selected_method]
        non_napinn = [
            group
            for method, group in methods.items()
            if not method[0].startswith("napinn")
        ]
        best_rmae = min(non_napinn, key=lambda group: mean(group, "rMAE"))
        best_rmse = min(non_napinn, key=lambda group: mean(group, "rMSE"))
        assert group_key(best_rmae) == group_key(best_rmse)
        baseline = best_rmae
        seed_wins = 0
        cell = f"15% · b={key[1]:g}, o={key[2]:g}"
        for seed in (40, 41, 42):
            selected_rmae = per_seed(selected, "rMAE", seed)
            selected_rmse = per_seed(selected, "rMSE", seed)
            baseline_rmae = per_seed(baseline, "rMAE", seed)
            baseline_rmse = per_seed(baseline, "rMSE", seed)
            both_better = (
                selected_rmae < baseline_rmae
                and selected_rmse < baseline_rmse
            )
            seed_wins += int(both_better)
            candidate_seed_rows.append(
                {
                    "rank": rank,
                    "cell": cell,
                    "seed": seed,
                    "selected_rMAE": selected_rmae,
                    "baseline_rMAE": baseline_rmae,
                    "rMAE_gain_pct": relative_gain(
                        baseline_rmae, selected_rmae
                    ),
                    "selected_rMSE": selected_rmse,
                    "baseline_rMSE": baseline_rmse,
                    "rMSE_gain_pct": relative_gain(
                        baseline_rmse, selected_rmse
                    ),
                    "both_metrics_better": "Yes" if both_better else "No",
                }
            )
        rmae_gain = relative_gain(mean(baseline, "rMAE"), mean(selected, "rMAE"))
        rmse_gain = relative_gain(mean(baseline, "rMSE"), mean(selected, "rMSE"))
        candidate_rows.append(
            {
                "rank": rank,
                "cell": cell,
                "selected_method": METHOD_LABEL[selected_method],
                "baseline_method": METHOD_LABEL[group_key(baseline)],
                "selected_rMAE": mean(selected, "rMAE"),
                "selected_rMAE_std": std(selected, "rMAE"),
                "baseline_rMAE": mean(baseline, "rMAE"),
                "rMAE_gain_pct": rmae_gain,
                "selected_rMSE": mean(selected, "rMSE"),
                "selected_rMSE_std": std(selected, "rMSE"),
                "baseline_rMSE": mean(baseline, "rMSE"),
                "rMSE_gain_pct": rmse_gain,
                "frozen_mean_confirmed": (
                    "Yes"
                    if conclusion["confirmed_win_by_frozen_definition"]
                    else "No"
                ),
                "both_metrics_seed_wins": f"{seed_wins}/3",
                "gross_rejection_pct": 100.0
                * mean(selected, "gross_outlier_rejection_rate"),
                "gross_rejection_std_pct": 100.0
                * std(selected, "gross_outlier_rejection_rate"),
                "background_rejection_pct": 100.0
                * mean(selected, "background_only_rejection_rate"),
                "background_rejection_std_pct": 100.0
                * std(selected, "background_only_rejection_rate"),
            }
        )
        candidate_gain_rows.extend(
            [
                {
                    "rank": rank,
                    "cell": cell,
                    "metric": "rMAE",
                    "gain_pct": rmae_gain,
                },
                {
                    "rank": rank,
                    "cell": cell,
                    "metric": "rMSE",
                    "gain_pct": rmse_gain,
                },
            ]
        )

    candidate_rows.sort(key=lambda row: row["rank"])
    candidate_seed_rows.sort(key=lambda row: (row["rank"], row["seed"]))
    candidate_gain_rows.sort(key=lambda row: (row["rank"], row["metric"]))

    exact_severity = severity_rows(exact)
    candidate_severity = severity_rows(confirmation)
    severity = exact_severity + candidate_severity
    severity.sort(
        key=lambda row: (
            row["ratio_pct"],
            row["background_multiplier"],
            row["gross_multiplier"],
        )
    )

    context_rows = [
        {
            "evidence": "합성 PDE 기본 실험 묶음",
            "result": (
                "인공적으로 생성한 Allen–Cahn, Burgers, lambda–omega 데이터에 "
                "5%, 10%, 15% 오염을 적용한 9개 조합 모두에서, gate가 없는 "
                "direct PINN-EBM의 field rMAE가 가장 낮았다."
            ),
            "implication": (
                "naPINN이 가장 유사한 선행 방법보다 항상 정확하다고 주장할 수 없다."
            ),
        },
        {
            "evidence": "실제 PIV에 30% 지속 편향을 넣은 실험",
            "result": (
                "PINN-EBM은 rMAE가 더 낮았고, naPINN은 rMSE와 명목상 PDE "
                "잔차가 더 낮았다."
            ),
            "implication": (
                "어떤 지표를 우선하느냐에 따라 순위가 달라지므로 한 방법의 "
                "명확한 종합 승리로 볼 수 없다."
            ),
        },
        {
            "evidence": "시간 상관 오염 AR(1)과 공간 군집 오염",
            "result": (
                "측정장 재구성 오차는 LAD-PINN이 가장 낮았지만, 명목상 PDE "
                "잔차는 naPINN이 가장 낮았다."
            ),
            "implication": (
                "오염의 형태와 평가 지표에 따라 가장 좋은 방법이 달라진다."
            ),
        },
        {
            "evidence": "기존 코드 설정을 그대로 옮긴 실제 PIV 실험",
            "result": (
                "사전에 고정한 기본 강도 `(b,o)=(1,1)`에서 naPINN-MSE가 "
                "10%와 15% 모두, 그리고 seeds 40–42 모두에서 rMAE와 rMSE가 "
                "가장 낮았다."
            ),
            "implication": (
                "결과를 본 뒤 고른 설정이 아니므로 실제 PIV에서 얻은 긍정적 "
                "증거 중 가장 방어하기 쉽다."
            ),
        },
        {
            "evidence": "Reynolds 수를 함께 추정한 30% 오염 실험",
            "result": (
                "naPINN은 명목상 PDE 잔차가 낮았지만 Reynolds 수 추정은 "
                "LAD-PINN과 PINN-EBM보다 나빴다."
            ),
            "implication": (
                "측정 오차와 PDE 모델 자체의 불일치를 현재 실험만으로 분리할 "
                "수 없다."
            ),
        },
    ]

    routing_rows = [
        {
            "target": "Area Chair",
            "lead": (
                "제출 코드의 기본 오염 설정을 실제 PIV 학습 자료에 그대로 "
                "적용한 결과"
            ),
            "include": (
                "10%와 15% 결과, 세 seed에서 같은 방향이 나온 사실, "
                "PINN-EBM이 가장 유사한 선행 방법이라는 인정"
            ),
            "limitation": (
                "인위적으로 주입한 강한 오염이며, 평가용 PIV도 물리적 참값은 아님"
            ),
        },
        {
            "target": "6SDM",
            "lead": "실제 PIV, 불규칙 센서 배치, 더 강한 비교 방법을 사용한 검증",
            "include": (
                "사전 고정 기본 설정, 여러 구조적 오염 실험, EMA 민감도와 "
                "학습량 계산"
            ),
            "limitation": "모든 오염 종류에서 같은 우위를 보였다는 뜻은 아님",
        },
        {
            "target": "aoJS",
            "lead": (
                "가장 유사한 선행 방법과 naPINN의 학습목표·gradient 경로 차이"
            ),
            "include": (
                "실제 PIV 기본 설정의 직접 비교와, 합성 실험 9개에서 "
                "PINN-EBM이 더 좋았던 불리한 결과"
            ),
            "limitation": (
                "gate가 이상치 순위화를 처음 가능하게 했다고 주장할 수 없음"
            ),
        },
        {
            "target": "6XZg",
            "lead": "비교의 공정성과 표현을 바로잡는 답변",
            "include": (
                "B-PINN이 변분추론 방식이라는 점, MAD 전처리, 방법별 정확한 "
                "학습 update 수"
            ),
            "limitation": (
                "HMC와는 비교하지 않았으며 Bayesian 불확실성 정량화는 "
                "여전히 보완적 방법임"
            ),
        },
    ]

    exact10 = exact_headline[10]
    exact15 = exact_headline[15]
    rank1 = candidate_rows[0]
    rank2 = candidate_rows[1]
    rank3 = candidate_rows[2]

    cards = [
        {
            "id": "card_exact_runs",
            "dataset": "campaign_status",
            "sourceId": "exact_aggregation",
            "description": "Exact-center complete reporting runs",
            "metrics": [
                {
                    "label": "사전 고정 기본 설정의 전체 학습 횟수",
                    "field": "exact_runs",
                }
            ],
        },
        {
            "id": "card_scale_runs",
            "dataset": "campaign_status",
            "sourceId": "scale_selection",
            "description": "Complete seed-39 development grid",
            "metrics": [
                {
                    "label": "오염 강도 탐색 단계의 전체 학습 횟수",
                    "field": "scale_runs",
                }
            ],
        },
        {
            "id": "card_confirmation_runs",
            "dataset": "campaign_status",
            "sourceId": "confirmation_aggregation",
            "description": "Fresh-seed confirmation runs",
            "metrics": [
                {
                    "label": "새 seed 재확인 단계의 전체 학습 횟수",
                    "field": "confirmation_runs",
                }
            ],
        },
        {
            "id": "card_confirmed_candidates",
            "dataset": "campaign_status",
            "sourceId": "candidate_validation",
            "description": "Candidates passing the frozen mean definition",
            "metrics": [
                {
                    "label": "평균 기준을 통과한 후보 설정 수",
                    "field": "mean_confirmed_candidates",
                }
            ],
        },
    ]

    charts = [
        {
            "id": "chart_exact_rmae",
            "title": "사전 고정 기본 오염 설정에서 방법별 평가용 PIV 오차",
            "subtitle": (
                "학습 측정행의 10% 또는 15%에 큰 이상치를 추가했다. "
                "Seeds 40–42의 평균 ± 표본표준편차이며 낮을수록 좋다."
            ),
            "intent": "comparison",
            "question": (
                "How does each matched method compare at the mandatory "
                "code-compatible legacy center?"
            ),
            "rationale": (
                "A grouped horizontal bar chart keeps eight long method "
                "labels readable while comparing both exact ratios."
            ),
            "comparisonContext": {
                "baseline": "All seven non-naPINN and alternate naPINN conditions",
                "grain": "method × corruption ratio",
                "unit": "relative mean absolute discrepancy",
            },
            "type": "bar",
            "dataset": "exact_results",
            "sourceId": "exact_aggregation",
            "encodings": {
                "x": {
                    "field": "method",
                    "type": "nominal",
                    "label": "Method",
                },
                "y": {
                    "field": "rMAE",
                    "type": "quantitative",
                    "label": "held-out-PIV rMAE",
                    "format": "number",
                },
                "color": {
                    "field": "ratio_label",
                    "type": "nominal",
                    "label": "Gross-row ratio",
                },
                "tooltip": [
                    {"field": "rMAE", "type": "quantitative", "label": "rMAE"},
                    {
                        "field": "rMAE_std",
                        "type": "quantitative",
                        "label": "sample std.",
                    },
                    {"field": "rMSE", "type": "quantitative", "label": "rMSE"},
                ],
            },
            "xAxisTitle": "held-out-PIV rMAE",
            "yAxisTitle": "Method",
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {
                "position": "bottom",
                "sort": "labelAsc",
                "title": "Gross-row ratio",
            },
            "labels": {"values": "auto"},
            "settings": {
                "groupMode": "grouped",
                "orientation": "horizontal",
                "sort": "custom",
                "showValues": True,
            },
            "surface": {"surface": "card", "viewMode": "visualization"},
        },
        {
            "id": "chart_candidate_gains",
            "title": "새 seed에서 가장 강한 비-naPINN 방법보다 오차가 줄어든 비율",
            "subtitle": (
                "Seed 39에서 고른 후보를 변경하지 않고 seeds 40–42에서 "
                "재평가했다. 양수는 naPINN의 오차가 더 작다는 뜻이다."
            ),
            "intent": "comparison",
            "question": (
                "Which preregistered non-center candidates generalized from "
                "development seed 39 to fresh reporting seeds?"
            ),
            "rationale": (
                "Signed grouped bars expose both the confirmed cells and the "
                "mixed candidate without hiding the negative rMAE result."
            ),
            "comparisonContext": {
                "baseline": "Best non-naPINN mean separately checked for rMAE and rMSE",
                "grain": "selected scale cell × field metric",
                "unit": "relative gain, percentage points",
            },
            "type": "bar",
            "dataset": "candidate_gains",
            "sourceId": "candidate_validation",
            "encodings": {
                "x": {
                    "field": "cell",
                    "type": "nominal",
                    "label": "Candidate",
                },
                "y": {
                    "field": "gain_pct",
                    "type": "quantitative",
                    "label": "Relative gain",
                    "format": "number",
                    "unit": "%",
                },
                "color": {
                    "field": "metric",
                    "type": "nominal",
                    "label": "Field metric",
                },
                "tooltip": [
                    {
                        "field": "gain_pct",
                        "type": "quantitative",
                        "label": "Relative gain",
                        "unit": "%",
                    }
                ],
            },
            "xAxisTitle": "Relative gain over strongest non-naPINN mean (%)",
            "yAxisTitle": "Frozen candidate",
            "valueFormat": "number",
            "unit": "%",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {
                "position": "bottom",
                "sort": "spec",
                "title": "Metric",
            },
            "labels": {"values": "all"},
            "referenceLines": [
                {
                    "axis": "x",
                    "value": 0,
                    "label": "No difference",
                    "color": "neutral",
                    "lineStyle": "solid",
                }
            ],
            "settings": {
                "groupMode": "grouped",
                "orientation": "horizontal",
                "sort": "custom",
                "showValues": True,
            },
            "surface": {"surface": "card", "viewMode": "visualization"},
        },
    ]

    tables = [
        {
            "id": "table_exact_results",
            "title": "사전 고정 기본 오염 설정의 전체 방법 비교",
            "subtitle": (
                "평가용 PIV와의 상대오차, 사용한 PDE 식의 잔차, 큰 이상치와 "
                "배경 잡음만 있는 값의 제거율을 함께 표시한다. Seeds 40–42 평균."
            ),
            "dataset": "exact_results",
            "sourceId": "exact_aggregation",
            "defaultSort": {"field": "ratio_pct", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {
                    "field": "ratio_pct",
                    "label": "큰 이상치를 넣은 행 (%)",
                    "format": "number",
                },
                {"field": "method", "label": "방법", "type": "text"},
                {"field": "rMAE", "label": "rMAE", "format": "number"},
                {
                    "field": "rMAE_std",
                    "label": "rMAE 표본표준편차",
                    "format": "number",
                },
                {"field": "rMSE", "label": "rMSE", "format": "number"},
                {
                    "field": "rMSE_std",
                    "label": "rMSE 표본표준편차",
                    "format": "number",
                },
                {
                    "field": "momentum_rms",
                    "label": "운동량 방정식 잔차 RMS",
                    "format": "number",
                },
                {
                    "field": "continuity_rms",
                    "label": "연속 방정식 잔차 RMS",
                    "format": "number",
                },
                {
                    "field": "gross_rejection_pct",
                    "label": "큰 이상치 제거율 (%)",
                    "format": "number",
                },
                {
                    "field": "background_rejection_pct",
                    "label": "배경 잡음 값 제거율 (%)",
                    "format": "number",
                },
                {
                    "field": "gate_auroc",
                    "label": "Gate AUROC",
                    "format": "number",
                },
                {
                    "field": "raw_ebm_auroc",
                    "label": "Raw EBM AUROC",
                    "format": "number",
                },
            ],
        },
        {
            "id": "table_exact_consistency",
            "title": "같은 오염 자료를 사용한 naPINN과 최강 비교 방법의 대조",
            "subtitle": (
                "각 평가 지표에서 평균 오차가 가장 낮은 비-naPINN 방법을 "
                "비교 대상으로 삼았다. 두 오염 비율 모두 OrPINN q=2.9였다."
            ),
            "dataset": "exact_consistency",
            "sourceId": "exact_aggregation",
            "defaultSort": {"field": "ratio_pct", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {
                    "field": "ratio_pct",
                    "label": "큰 이상치를 넣은 행 (%)",
                    "format": "number",
                },
                {"field": "baseline_method", "label": "가장 강한 비교 방법"},
                {
                    "field": "selected_rMAE",
                    "label": "naPINN rMAE",
                    "format": "number",
                },
                {
                    "field": "baseline_rMAE",
                    "label": "비교 방법 rMAE",
                    "format": "number",
                },
                {
                    "field": "rMAE_gain_pct",
                    "label": "naPINN rMAE 감소율 (%)",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "selected_rMSE",
                    "label": "naPINN rMSE",
                    "format": "number",
                },
                {
                    "field": "baseline_rMSE",
                    "label": "비교 방법 rMSE",
                    "format": "number",
                },
                {
                    "field": "rMSE_gain_pct",
                    "label": "naPINN rMSE 감소율 (%)",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "both_metrics_seed_wins",
                    "label": "두 지표 모두 우세한 seed 수",
                },
            ],
        },
        {
            "id": "table_severity",
            "title": "실제로 주입된 오염의 크기",
            "subtitle": (
                "오염을 넣기 전 PIV 속도값의 표본표준편차를 1로 놓은 상대 "
                "크기다. 최솟값과 최댓값은 seeds 40–42의 범위다."
            ),
            "dataset": "severity",
            "defaultSort": {"field": "background_multiplier", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "cell", "label": "오염 설정"},
                {
                    "field": "background_std_ratio_min",
                    "label": "배경 잡음 표준편차 최솟값",
                    "format": "number",
                },
                {
                    "field": "background_std_ratio_max",
                    "label": "배경 잡음 표준편차 최댓값",
                    "format": "number",
                },
                {
                    "field": "gross_offset_ratio_min",
                    "label": "큰 이상치 크기 최솟값",
                    "format": "number",
                },
                {
                    "field": "gross_offset_ratio_max",
                    "label": "큰 이상치 크기 최댓값",
                    "format": "number",
                },
                {
                    "field": "background_multiplier",
                    "label": "배경 잡음 배율 b",
                    "format": "number",
                },
                {
                    "field": "gross_multiplier",
                    "label": "큰 이상치 추가 배율 o",
                    "format": "number",
                },
            ],
        },
        {
            "id": "table_candidate_summary",
            "title": "사전에 고정한 세 후보 설정의 재확인 결과",
            "subtitle": (
                "세 후보 모두 naPINN-MSE를 사용했다. 후보를 고를 때 쓰지 않은 "
                "새 seeds 40–42의 평균과 표본표준편차다."
            ),
            "dataset": "candidate_summary",
            "sourceId": "candidate_validation",
            "defaultSort": {"field": "rank", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "rank", "label": "후보 순위", "format": "number"},
                {"field": "cell", "label": "오염 설정"},
                {
                    "field": "baseline_method",
                    "label": "가장 강한 비교 방법",
                },
                {
                    "field": "selected_rMAE",
                    "label": "naPINN rMAE",
                    "format": "number",
                },
                {
                    "field": "selected_rMAE_std",
                    "label": "rMAE 표본표준편차",
                    "format": "number",
                },
                {
                    "field": "rMAE_gain_pct",
                    "label": "naPINN rMAE 감소율 (%)",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "selected_rMSE",
                    "label": "naPINN rMSE",
                    "format": "number",
                },
                {
                    "field": "selected_rMSE_std",
                    "label": "rMSE 표본표준편차",
                    "format": "number",
                },
                {
                    "field": "rMSE_gain_pct",
                    "label": "naPINN rMSE 감소율 (%)",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "frozen_mean_confirmed",
                    "label": "사전 평균 기준 통과",
                },
                {
                    "field": "both_metrics_seed_wins",
                    "label": "두 지표 모두 우세한 seed 수",
                },
                {
                    "field": "gross_rejection_pct",
                    "label": "큰 이상치 제거율 (%)",
                    "format": "number",
                },
                {
                    "field": "background_rejection_pct",
                    "label": "배경 잡음 값 제거율 (%)",
                    "format": "number",
                },
            ],
        },
        {
            "id": "table_candidate_seed",
            "title": "후보별·seed별 재현 여부",
            "subtitle": (
                "동일한 오염 자료를 사용한 가장 강한 비-naPINN 방법과의 "
                "상대 비교다. 양수면 naPINN의 오차가 더 작다."
            ),
            "dataset": "candidate_seed",
            "sourceId": "confirmation_aggregation",
            "defaultSort": {"field": "rank", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "rank", "label": "후보 순위", "format": "number"},
                {"field": "cell", "label": "오염 설정"},
                {"field": "seed", "label": "Seed", "format": "number"},
                {
                    "field": "rMAE_gain_pct",
                    "label": "naPINN rMAE 감소율 (%)",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "rMSE_gain_pct",
                    "label": "naPINN rMSE 감소율 (%)",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "both_metrics_better",
                    "label": "두 지표 모두 naPINN 우세",
                },
            ],
        },
        {
            "id": "table_context",
            "title": "이번 실제 PIV 결과를 전체 rebuttal 증거와 함께 해석한 표",
            "subtitle": (
                "naPINN에 유리한 결과뿐 아니라 혼합 결과와 불리한 결과도 "
                "함께 보존한다."
            ),
            "dataset": "whole_suite_context",
            "sourceId": "master_rebuttal_report",
            "defaultSort": {"field": "evidence", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "evidence", "label": "실험 묶음"},
                {"field": "result", "label": "관찰된 결과"},
                {"field": "implication", "label": "주장 범위에 주는 의미"},
            ],
        },
        {
            "id": "table_routing",
            "title": "Reviewer별 권장 답변 구성",
            "subtitle": (
                "직접 비교 수치의 사용을 저자가 명시적으로 승인한 뒤에만 "
                "적용한다."
            ),
            "dataset": "reviewer_routing",
            "sourceId": "reviewer_comments",
            "defaultSort": {"field": "target", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "target", "label": "대상"},
                {"field": "lead", "label": "답변의 첫 요점"},
                {"field": "include", "label": "포함할 근거"},
                {"field": "limitation", "label": "반드시 함께 밝힐 한계"},
            ],
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": (
                "# 실제 Cylinder-PIV에 기존 four-Gaussian 오염 규칙을 적용한 "
                "실험 결과와 Rebuttal 전략"
            ),
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 기술 요약\n\n"
                "> **내부 검토용 결과이며 현재 `RESPONSE HOLD` 상태다.** 이는 "
                "실험이 미완료라는 뜻이 아니다. 실험과 검증은 끝났지만, "
                "naPINN과 PINN-EBM의 직접 비교 수치와 순위를 OpenReview "
                "답변에 사용할지는 저자가 아직 승인하지 않았다는 뜻이다.\n\n"
                "**가장 신뢰하기 쉬운 긍정적 결과는 제출 코드의 기본 오염 "
                "강도를 그대로 사용한 실험이다.** 보고서에서는 이 설정을 "
                "`exact center`, 즉 `(b,o)=(1,1)`이라고 부른다. 여기서 "
                "`exact`는 참값을 안다는 뜻이 아니라 배경 잡음과 큰 이상치의 "
                "배율을 제출 코드에서 바꾸지 않았다는 뜻이다. 학습 측정행의 "
                f"10%에 큰 이상치를 넣었을 때 naPINN-MSE의 rMAE/rMSE는 "
                f"{pm(exact10['selected'], 'rMAE')} / "
                f"{pm(exact10['selected'], 'rMSE')}였고, 가장 좋은 비-naPINN "
                f"방법보다 각각 {exact10['rMAE_gain_pct']:.1f}%와 "
                f"{exact10['rMSE_gain_pct']:.1f}% 낮았다. 15%에 이상치를 넣은 "
                f"경우에는 {pm(exact15['selected'], 'rMAE')} / "
                f"{pm(exact15['selected'], 'rMSE')}였고, 각각 "
                f"{exact15['rMAE_gain_pct']:.1f}%와 "
                f"{exact15['rMSE_gain_pct']:.1f}% 낮았다. 후보 선택에 사용하지 "
                "않은 최종 보고용 난수 반복인 seeds 40, 41, 42에서 모두 같은 "
                "방향의 결과가 나왔다.\n\n"
                "**오염 강도를 바꾸어 탐색한 후 새 seed로 재확인한 결과는 "
                "더 복합적이다.** Seed 39에서 미리 정한 규칙으로 세 후보 "
                "설정을 골랐고, 이 후보를 변경하지 않은 채 seeds 40–42에서 "
                "다시 실행했다. 세 후보 중 두 개는 세 seed 평균으로는 "
                "naPINN의 두 오차가 모두 낮았다. 그러나 모든 seed에서 두 "
                "오차가 모두 낮았던 후보는 첫 번째 후보 하나뿐이었다. 두 "
                "번째 후보는 한 seed에서 반대 결과가 났고, 세 번째 후보는 "
                f"rMAE가 {abs(rank3['rMAE_gain_pct']):.1f}% 나쁜 대신 rMSE는 "
                f"{rank3['rMSE_gain_pct']:.1f}% 좋은 혼합 결과였다.\n\n"
                "**따라서 이 결과가 지지하는 주장은 ‘특정 조건에서 naPINN이 "
                "유리하다’는 조건부 결론이다.** 실제 PIV에 제출 코드의 기본 "
                "오염 규칙을 적용했을 때는 naPINN이 강한 재구성 성능을 보였다. "
                "반면 인공적으로 생성한 세 PDE 데이터와 세 오염 비율을 조합한 "
                "9개 기본 합성 실험에서는 학습값 포함 여부를 결정하는 gate가 "
                "없는 direct PINN-EBM의 "
                "rMAE가 모두 더 낮았다. 그러므로 naPINN이 모든 데이터와 "
                "오염에서 PINN-EBM보다 정확하다고 주장해서는 안 된다."
            ),
        },
        {
            "id": "terminology_legacy",
            "type": "markdown",
            "sourceId": "handoff",
            "body": (
                "## 먼저 읽을 용어 안내: `legacy`, `exact center`, `transfer`의 뜻\n\n"
                "- **Legacy-4G**: 여기서 `legacy`는 낡아서 폐기한다는 뜻이 "
                "아니다. 제출된 naPINN 코드가 합성 실험에서 사용하던 기존 "
                "four-Gaussian 배경 잡음과 큰 점 이상치 생성 규칙을 가리킨다. "
                "`4G`는 배경 잡음이 네 개 Gaussian 분포의 혼합에서 나온다는 "
                "뜻이다.\n"
                "- **Exact center 또는 exact legacy center**: 오염 크기를 "
                "조절하는 두 배율을 `(b,o)=(1,1)`로 둔 설정이다. `b=1`은 "
                "기존 배경 잡음 크기를 그대로 썼다는 뜻이고, `o=1`은 기존 "
                "큰 이상치 배율 `U(3,10)`을 그대로 썼다는 뜻이다. `center`라는 "
                "명칭은 이후 탐색한 3×3 배율 격자의 가운데 점이라는 뜻이다. "
                "결과를 본 뒤 고른 최적점이 아니며, 사전에 반드시 보고하기로 "
                "정한 기준점이다.\n"
                "- **Exact legacy transfer**: 모델의 파라미터를 옮기거나 "
                "domain adaptation을 했다는 뜻이 아니다. 합성 PDE 데이터에 "
                "쓰던 오염 생성 규칙과 기본 강도만 실제 Cylinder PIV의 학습 "
                "측정값에 적용했다는 뜻이다. 평가용 PIV는 바꾸지 않았다.\n"
                "- **Scale search**: 기본점 주변에서 배경 잡음 배율 `b`와 큰 "
                "이상치 추가 배율 `o`를 각각 0.5, 1, 2로 바꾸어 어떤 강도에서 "
                "방법 순위가 달라지는지 확인한 탐색이다.\n"
                "- **Development seed와 reporting seed**: seed는 난수 생성의 "
                "출발값이다. Seed 39는 후보를 고르는 개발 단계에만 사용했다. "
                "후보를 고른 뒤에는 보지 않았던 seeds 40–42로 다시 실행해 "
                "선택한 결과가 새 난수에서도 유지되는지 확인했다."
            ),
        },
        {
            "id": "terminology_metrics",
            "type": "markdown",
            "body": (
                "## 먼저 읽을 용어 안내: 평가 지표와 비교 방법\n\n"
                "- **Held-out PIV 또는 평가용 PIV**: 학습 센서 위치와 공간적으로 "
                "겹치지 않는 위치에서 얻은 실제 PIV 측정값이다. 학습이나 후보 "
                "선정에는 사용하지 않았다. 다만 실험 장비로 측정한 값이므로 "
                "오차가 전혀 없는 물리적 참값은 아니다.\n"
                "- **rMAE와 rMSE**: 모델이 예측한 속도장과 평가용 PIV 측정값의 "
                "차이를 상대적인 크기로 나타낸 두 지표다. 둘 다 낮을수록 "
                "평가용 측정과 더 잘 맞는다. rMSE는 큰 오차에 rMAE보다 더 "
                "민감하다.\n"
                "- **Field metric 또는 재구성 오차**: 모델이 예측한 해 전체, "
                "여기서는 `u`와 `v` 속도장이 측정값과 얼마나 다른지를 나타내는 "
                "지표다. Parameter recovery처럼 Reynolds 수나 PDE 계수 하나를 "
                "추정하는 문제와 구분한다.\n"
                "- **PDE residual**: 모델의 예측을 사용한 Navier–Stokes 식의 "
                "좌변과 우변이 얼마나 맞지 않는지를 나타낸다. 낮으면 사용한 "
                "명목상 PDE 식을 더 잘 만족한다. 그러나 PDE 식 자체가 실제 "
                "유동을 완전히 설명하지 못할 수 있으므로, 낮은 residual이 곧 "
                "물리적 참값에 더 가깝다는 뜻은 아니다.\n"
                "- **AUROC**: 주입한 큰 이상치에 정상 값보다 높은 이상 점수를 "
                "주는 순위화 능력이다. 0.5는 무작위 수준, 1은 완전한 구분을 "
                "뜻한다. AUROC는 어떤 threshold로 실제 값을 제외했는지와는 "
                "다른 지표다.\n"
                "- **Baseline 또는 비교 방법**: naPINN의 성능을 판단하기 위해 "
                "함께 실행한 다른 방법이다. `strongest non-naPINN baseline`은 "
                "비-naPINN 방법 중 해당 지표의 평균 오차가 가장 낮은 방법을 "
                "뜻한다.\n"
                "- **Closest prior와 direct PINN-EBM**: naPINN과 가장 구조가 "
                "비슷한 선행 방법은 residual에 EBM을 맞추는 PINN-EBM이다. "
                "이 보고서의 `direct PINN-EBM`은 EBM 점수는 사용하지만 "
                "naPINN처럼 학습값을 포함·제외하는 명시적인 gate는 두지 않은 "
                "직접 구현을 뜻한다.\n"
                "- **Gate와 reject**: gate는 각 학습값을 data-fitting loss에 "
                "포함할지 결정한다. `reject`는 그 값을 평가 자료에서 삭제한다는 "
                "뜻이 아니라, 해당 학습 단계의 재구성 loss에서 제외한다는 뜻이다.\n"
                "- **Paired comparison**: 서로 다른 방법들이 완전히 같은 오염 "
                "배열과 같은 seed를 사용하도록 맞춘 비교다. 따라서 방법 간 "
                "차이가 서로 다른 난수 때문에 생길 가능성을 줄인다."
            ),
        },
        {
            "id": "terminology_synthetic",
            "type": "markdown",
            "sourceId": "master_rebuttal_report",
            "body": (
                "## `synthetic core`는 무엇인가\n\n"
                "`Synthetic core`는 논문의 합성 데이터 실험 중 rebuttal에서 "
                "기본 비교 묶음으로 삼은 실험 행렬을 뜻한다. `core`는 핵심 "
                "비교 세트라는 작업상 명칭일 뿐, 신경망의 구성요소나 코드 "
                "kernel을 뜻하지 않는다.\n\n"
                "이 행렬은 인공적으로 생성한 세 PDE 해—Allen–Cahn, Burgers, "
                "lambda–omega—각각에 four-Gaussian 오염 비율 5%, 10%, 15%를 "
                "적용한 **3 PDE × 3 비율 = 9개 조건**으로 이루어진다. 각 조건은 "
                "seeds 40–42와 동일한 방법군으로 반복했다. 이 9개 조건에서는 "
                "gate가 없는 direct PINN-EBM의 해 전체 재구성 rMAE가 모두 가장 낮았고 "
                "naPINN은 모두 두 번째였다. 이 결과는 naPINN에 불리하지만, "
                "실제 PIV의 긍정적 결과를 모든 상황에 일반화하지 않기 위해 "
                "반드시 함께 보존해야 한다."
            ),
        },
        {
            "id": "campaign_metrics",
            "type": "metric-strip",
            "cardIds": [
                "card_exact_runs",
                "card_scale_runs",
                "card_confirmation_runs",
                "card_confirmed_candidates",
            ],
        },
        {
            "id": "exact_result",
            "type": "markdown",
            "sourceId": "exact_aggregation",
            "body": (
                "## 결과를 보고 고르지 않은 기본 설정에서 naPINN-MSE가 가장 낮은 오차를 보였다\n\n"
                "첫 단계에서는 `(b,o)=(1,1)`을 사용했다. 이는 제출 코드의 "
                "배경 잡음 크기와 큰 이상치 크기를 바꾸지 않은 기본 설정이다. "
                "이 설정은 평가 결과를 보기 전에 반드시 보고하기로 정했기 "
                "때문에, naPINN에 유리한 강도를 사후에 골랐다는 비판을 가장 "
                "적게 받는다.\n\n"
                "학습 측정행의 10% 또는 15%에 큰 이상치를 넣고, MSE-PINN, "
                "LAD-PINN, OrPINN q=2.9, 두 가지 PDE 가중치의 direct "
                "PINN-EBM, 그리고 세 가지 재구성 loss를 사용한 naPINN을 "
                "같은 오염 자료로 비교했다. naPINN-MSE는 두 오염 비율 모두에서 "
                "평가용 PIV와의 rMAE와 rMSE가 가장 낮았다. 따라서 강도 탐색으로 "
                "선택한 더 극단적인 후보보다 이 기본 설정의 결과를 rebuttal의 "
                "첫 근거로 사용하는 편이 타당하다."
            ),
        },
        {"id": "exact_chart", "type": "chart", "chartId": "chart_exact_rmae"},
        {
            "id": "exact_chart_note",
            "type": "markdown",
            "sourceId": "exact_aggregation",
            "body": (
                "10%와 15% 모두에서 비-naPINN 방법 중 가장 낮은 평균 오차를 "
                "보인 것은 OrPINN q=2.9였다. naPINN-MSE는 seed 40, 41, 42 "
                "각각에서 OrPINN q=2.9보다 rMAE와 rMSE가 모두 낮았다. 즉, "
                "세 값의 평균만 우연히 좋아진 것이 아니라 관찰한 세 반복에서 "
                "개선 방향이 같았다. 그러나 반복 수가 세 개뿐이므로 통계적 "
                "유의성이나 더 넓은 데이터 모집단 전체에 대한 일반화를 "
                "주장하기에는 부족하다."
            ),
        },
        {
            "id": "exact_table",
            "type": "table",
            "tableId": "table_exact_results",
        },
        {
            "id": "exact_consistency_note",
            "type": "markdown",
            "sourceId": "exact_aggregation",
            "body": (
                "### Gate를 추가한 효과는 재구성 loss에 따라 달랐다\n\n"
                "여기서는 gate를 사용한 방법과 사용하지 않은 방법을 동일한 "
                "재구성 loss끼리 비교한다. naPINN-MSE는 gate가 없는 MSE-PINN보다 "
                "rMAE/rMSE가 10% 조건에서 61.8%/49.8%, 15% 조건에서 "
                "72.1%/62.4% 낮았다. naPINN-L1도 gate가 없는 LAD-PINN보다 "
                "두 오염 비율의 두 지표가 모두 낮았다.\n\n"
                "반면 q=2.9 loss를 쓴 경우에는 결과가 일관되지 않았다. "
                "10% 조건에서 naPINN-q2.9는 gate가 없는 OrPINN q=2.9보다 "
                "rMAE는 3.7% 낮았지만 rMSE는 0.8% 높았다. 따라서 gate를 "
                "어떤 robust loss에 붙이더라도 항상 추가 개선이 생긴다고 "
                "말할 수는 없다."
            ),
        },
        {
            "id": "exact_consistency_table",
            "type": "table",
            "tableId": "table_exact_consistency",
        },
        {
            "id": "detection_interpretation",
            "type": "markdown",
            "sourceId": "exact_aggregation",
            "body": (
                "## naPINN은 주입한 큰 이상치를 잘 제외했지만, 이상치 순위화 자체는 EBM도 이미 잘했다\n\n"
                "큰 이상치를 넣은 속도 성분을 얼마나 gate가 제외했는지와, "
                "배경 잡음만 있는 속도 성분을 실수로 얼마나 제외했는지를 "
                "구분해 보았다. 기본 오염 설정에서 naPINN-MSE는 큰 이상치가 "
                "있는 성분의 98.99%(10% 조건)와 99.23%(15% 조건)를 "
                "재구성 loss에서 제외했다. 반면 큰 이상치는 없고 배경 잡음만 "
                "있는 성분은 2.95%와 4.23%만 제외했다. 주입된 큰 이상치와 "
                "그렇지 않은 값을 구분하는 동작은 매우 강했다.\n\n"
                "하지만 이것이 gate만의 새로운 이상치 탐지 능력을 뜻하지는 "
                "않는다. AUROC는 오염된 값을 정상 값보다 높은 이상 점수로 "
                "정렬하는 능력을 나타내며, 1에 가까울수록 좋다. Gate의 AUROC와 "
                "gate를 만들기 전 raw EBM score의 AUROC는 10%에서 "
                "0.98766과 0.98770, 15%에서 0.98248과 0.98249로 거의 "
                "같았다. 즉, EBM 점수만으로도 이상치의 순서는 이미 잘 정렬됐다. "
                "naPINN의 방어 가능한 차이는 그 점수를 학습값 포함·제외 "
                "결정으로 바꾸고, 그 결정에 따라 재구성 loss를 학습한다는 데 "
                "있다.\n\n"
                "PDE 잔차도 모든 항목에서 naPINN이 가장 낮은 것은 아니다. "
                "10% 조건에서는 naPINN-MSE의 연속 방정식 잔차가 가장 낮았지만, "
                "운동량 방정식 잔차는 PDE 가중치 50을 사용한 PINN-EBM이 더 "
                "낮았다. 15% 조건에서는 naPINN-MSE가 두 잔차 모두 가장 낮았다. "
                "따라서 ‘모든 물리 지표에서도 항상 우세하다’는 표현은 피해야 한다."
            ),
        },
        {
            "id": "scope_definition",
            "type": "markdown",
            "sourceId": "handoff",
            "body": (
                "## 이 실험은 실제 유동장 위에 통제된 강한 인공 오염을 추가해 한계를 보는 시험이다\n\n"
                "바탕 데이터는 RealPDEBench Cylinder의 실제 PIV 200 frame이다. "
                "PIV는 입자 영상을 이용해 유체의 속도를 측정하는 실험 기법이다. "
                "학습에는 고정된 192개 불규칙 센서 위치에서 얻은 38,400개 "
                "2차원 속도 vector를 사용했다. 평가는 이 학습 위치와 공간적으로 "
                "겹치지 않는 7,747개 위치의 1,549,400개 속도 vector에서 "
                "수행했다. 평가용 측정에는 오염을 추가하지 않았다.\n\n"
                "다만 ‘실제 PIV’와 ‘물리적 참값’은 같은 말이 아니다. 평가용 "
                "PIV도 실험 측정이므로 자체 측정 오차를 가질 수 있다. 따라서 "
                "rMAE와 rMSE는 모델이 독립적인 실제 측정과 얼마나 잘 맞는지를 "
                "보여주지만, 모델의 물리적 절대오차를 직접 알려주지는 않는다.\n\n"
                "오염은 논문 설명을 임의로 재해석하지 않고 제출 코드가 실제로 "
                "실행하는 규칙을 따랐다. 먼저 모든 학습 속도 성분에 네 Gaussian "
                "분포가 섞인 배경 잡음을 더했다. 그런 다음 선택된 속도 vector "
                "행의 `u`와 `v` 성분에 양수 방향의 큰 offset을 추가했다. 따라서 "
                "큰 이상치가 없는 값도 완전히 깨끗한 값이 아니라 배경 잡음은 "
                "포함한다. 보고서에서는 이 집단을 `clean`이 아니라 "
                "`background-only`라고 부른다.\n\n"
                "또한 논문 문구는 관측값을 새 값으로 교체한다고 설명하지만, "
                "제출 코드는 기존 값에 양의 offset을 더한다. Rebuttal에서는 "
                "이 차이를 숨기지 않고 ‘submitted-code protocol’이라고 명확히 "
                "표현해야 한다."
            ),
        },
        {
            "id": "severity_note",
            "type": "markdown",
            "body": (
                "기본점 `(b,o)=(1,1)`도 약한 잡음 설정이 아니다. 오염을 넣기 "
                "전 학습 PIV 속도값의 표본표준편차를 1이라고 두면, 실제로 "
                "생성된 배경 잡음의 표준편차는 약 0.698이다. 큰 이상치로 "
                "추가한 offset은 원래 PIV 표준편차의 약 2.09배에서 6.98배였다. "
                "즉, 기본 설정 자체가 상당히 강한 인공 오염이다.\n\n"
                "후보 1의 `(b,o)=(2,2)`는 더 극단적이다. 배경 잡음 표준편차는 "
                "원래 PIV 표준편차의 약 1.396배이고, 큰 offset은 약 "
                "8.37–27.93배다. 이 결과는 의도적으로 강한 오염을 넣어 방법의 "
                "한계를 보는 시험으로는 "
                "의미가 있지만, 자연 상태의 센서 잡음을 대표하거나 실제 "
                "배치 환경에서 흔히 발생하는 강도라고 일반화해서는 안 된다."
            ),
        },
        {"id": "severity_table", "type": "table", "tableId": "table_severity"},
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## 후보 선택과 재확인 단계를 분리해 유리한 설정만 고르는 위험을 줄였다\n\n"
                "오염 강도 탐색은 다음 순서로 진행했다.\n\n"
                "1. 먼저 결과를 보기 전에 배경 잡음 배율 `b∈{0.5,1,2}`, 큰 "
                "이상치 추가 배율 `o∈{0.5,1,2}`, 큰 이상치를 넣는 행의 비율 "
                "10%와 15%를 고정했다. 따라서 총 설정 수는 "
                "`3 × 3 × 2 = 18`개다.\n"
                "2. 후보 탐색용 seed 39에서 18개 설정을 모두 실행했다. 각 "
                "설정마다 MSE-PINN, LAD-PINN, OrPINN q=2.9, 두 PINN-EBM "
                "가중치, naPINN-MSE/L1/q2.9의 여덟 조건을 빠짐없이 실행했다. "
                "총 144회 학습이다.\n"
                "3. 기본점 `(1,1)`은 후보 선정에서 제외했다. 기본점은 탐색 "
                "결과와 무관하게 별도로 보고해야 하기 때문이다.\n"
                "4. naPINN의 rMAE와 rMSE가 모두 더 낮고, 같은 재구성 loss를 "
                "쓰는 gate 없는 방법보다도 두 지표가 낮으며, 큰 이상치 제거율이 "
                "80% 이상이고 배경 잡음 값의 잘못된 제거율이 40% 이하인 설정만 "
                "후보가 될 수 있게 했다. PDE 잔차도 비교 방법의 10배를 넘지 "
                "않도록 제한했다.\n"
                "5. 이 기준을 완화하지 않고 최대 세 후보를 고정했다. 이후 "
                "후보나 방법 설정을 바꾸지 않은 채, 후보 선택에 사용하지 않은 "
                "seeds 40–42에서 다시 실행했다. 이것이 confirmation, 즉 "
                "재확인 단계다.\n\n"
                "학습량도 구분해 기록했다. MSE-PINN, LAD-PINN, OrPINN은 "
                "PINN 파라미터를 30,000회 update했다. PINN-EBM과 naPINN은 "
                "5,000회의 warm-up 뒤 25,000회의 joint update를 수행하므로 "
                "PINN update 수는 마찬가지로 30,000회다. 여기에 residual "
                "분포 추정기만 학습하는 5,000회가 별도로 추가된다.\n\n"
                "최종 검증기는 재확인 단계의 72/72 학습, 24개 세-seed group, "
                "9개 paired input block이 모두 존재하고 동일한 오염 자료를 "
                "공유하는지 확인했다. 따라서 누락되거나 실패한 실행을 제외해 "
                "유리한 평균을 만든 결과는 아니다."
            ),
        },
        {
            "id": "scale_result",
            "type": "markdown",
            "sourceId": "candidate_validation",
            "body": (
                "## 평균으로는 두 후보가 통과했지만, 모든 새 seed에서 재현된 후보는 하나였다\n\n"
                "사전에 정한 평균 기준은 seeds 40–42의 평균 rMAE와 평균 rMSE가 "
                "모두 가장 강한 비-naPINN 방법보다 낮아야 한다는 것이다. 이 "
                "기준으로는 후보 1과 후보 2가 통과했고 후보 3은 실패했다.\n\n"
                "그러나 평균만 보면 반복 간 차이를 놓칠 수 있다. 후보 1 "
                "`(b,o)=(2,2)`는 세 seed 모두에서 rMAE와 rMSE가 함께 낮았다. "
                "후보 2 `(2,1)`는 seed 40과 42에서는 두 지표가 낮았지만 "
                "seed 41에서는 두 지표가 모두 더 높았다. 이 때문에 후보 2의 "
                "표본표준편차가 크다. 후보 3 `(1,0.5)`는 세 seed 모두에서 "
                "rMAE는 더 나쁘고 rMSE는 더 좋았다. 즉, 일관된 개선이 아니라 "
                "두 오차 정의 사이의 trade-off였다."
            ),
        },
        {
            "id": "candidate_chart",
            "type": "chart",
            "chartId": "chart_candidate_gains",
        },
        {
            "id": "candidate_chart_note",
            "type": "markdown",
            "sourceId": "candidate_validation",
            "body": (
                "후보 1 `(2,2)`의 세 seed 평균은 가장 강한 비교 방법보다 "
                "rMAE가 46.0%, rMSE가 38.5% 낮았고, 세 seed 모두 개선 방향이 "
                "같았다. 후보 2 `(2,1)`는 평균상 24.2%와 17.6% 낮았지만 "
                "세 seed 중 두 개에서만 두 지표가 함께 좋아졌다. 후보 3 "
                "`(1,0.5)`는 평균 rMAE가 4.9% 더 나쁘고 평균 rMSE는 6.0% "
                "더 좋았다.\n\n"
                "따라서 ‘세 후보 중 두 후보가 평균 기준을 통과했다’는 문장만 "
                "쓰면 후보 2의 seed 의존성과 후보 3의 실패를 가리게 된다. "
                "Rebuttal에서 이 탐색 결과를 언급한다면 평균 통과 수와 "
                "seed별 재현 수를 함께 밝혀야 한다."
            ),
        },
        {
            "id": "candidate_summary_table",
            "type": "table",
            "tableId": "table_candidate_summary",
        },
        {
            "id": "candidate_seed_note",
            "type": "markdown",
            "sourceId": "confirmation_aggregation",
            "body": (
                "아래 seed별 표는 세 값의 평균이 가릴 수 있는 방향 변화를 "
                "보여준다. 후보 2의 seed 41에서는 naPINN의 rMAE와 rMSE가 "
                "모두 비교 방법보다 높았다. 이 실행에서는 큰 이상치가 없는 "
                "배경 잡음 값도 70% 넘게 제외했다. 후보 1도 평균 재구성 오차는 "
                "강하게 개선됐지만, 배경 잡음 값을 제외하는 비율은 seed에 따라 "
                "크게 달랐다. 따라서 ‘평가용 PIV 오차가 낮다’와 ‘gate의 선택 "
                "비율이 반복마다 안정적이다’는 서로 다른 주장으로 다뤄야 한다."
            ),
        },
        {
            "id": "candidate_seed_table",
            "type": "table",
            "tableId": "table_candidate_seed",
        },
        {
            "id": "whole_evidence",
            "type": "markdown",
            "sourceId": "master_rebuttal_report",
            "body": (
                "## 전체 실험을 함께 보면 한 방법의 보편적 우위가 아니라 조건별 장단점이 보인다\n\n"
                "실제 PIV에 제출 코드의 기본 four-Gaussian 오염을 적용한 "
                "이번 실험만 보면 naPINN의 장점이 뚜렷하다. 결과를 보고 고르지 "
                "않은 `(b,o)=(1,1)`에서 naPINN-MSE가 두 평가 오차 모두 "
                "가장 낮았고, 세 seed에서 방향도 같았다.\n\n"
                "하지만 rebuttal의 전체 실험 기록에는 naPINN에 불리한 결과도 "
                "있다. 인공적으로 생성한 세 PDE와 세 오염 비율의 9개 조합에서는 "
                "gate가 없는 direct PINN-EBM의 rMAE가 모두 가장 낮았다. 실제 "
                "PIV에 지속 편향, 시간 상관, 공간 군집 오염을 넣은 실험에서는 "
                "field 오차와 PDE 잔차 중 어느 지표를 보느냐에 따라 가장 좋은 "
                "방법이 달랐다. Reynolds 수를 함께 추정한 실험에서는 naPINN의 "
                "PDE 잔차가 낮아도 parameter recovery가 좋지 않았다.\n\n"
                "따라서 가장 정직한 결론은 다음과 같다. naPINN의 명시적인 "
                "선택 gate는 일부 실제 PIV 오염 조건에서 재구성에 도움이 "
                "되지만, 그 이점은 데이터 종류, 오염 형태, 오염 강도, 평가 "
                "지표에 따라 달라진다."
            ),
        },
        {"id": "context_table", "type": "table", "tableId": "table_context"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 이 결과를 해석할 때 지켜야 할 한계\n\n"
                "- **반복 수가 적다.** 각 보고값은 seeds 40–42, 즉 세 번의 "
                "반복에서 계산한 평균 ± 표본표준편차다. 같은 seed와 오염을 "
                "사용한 paired comparison의 방향은 확인할 수 있지만, 세 반복만으로 "
                "통계적 유의성이나 모집단 전체의 성능을 주장하지 않는다.\n"
                "- **평가용 PIV는 참값이 아니다.** rMAE와 rMSE는 독립적인 실제 "
                "PIV 측정과 모델 예측의 일치 정도다. PIV 측정 자체의 오차까지 "
                "분리한 물리적 정확도는 아니다.\n"
                "- **사용한 PDE도 완전한 유동 모델은 아니다.** 압력을 잠재변수로 "
                "둔 2차원 Navier–Stokes 식의 불일치가 남아 있다. 현재 결과만으로 "
                "센서 관측 오차와 PDE 모델 오차를 분리할 수 없다.\n"
                "- **자연적으로 관측된 고장 사례가 아니다.** 실제 PIV 위에 "
                "위치를 알고 있는 강한 인공 오염을 추가해 한계를 본 시험이다. 실제 "
                "센서 고장의 빈도나 강도를 대표한다고 말할 수 없다.\n"
                "- **Gate가 EBM보다 이상치 순위화를 더 잘한 것은 아니다.** "
                "Raw EBM AUROC가 gate AUROC와 거의 같았다. Contribution은 "
                "점수를 명시적 포함·제외 결정과 학습목표로 연결한 점에 한정해야 한다.\n"
                "- **강도 탐색 후보는 선택된 결과다.** 규칙을 사전에 고정했지만 "
                "seed 39의 결과를 이용해 후보를 골랐다. 이 결과를 인용한다면 "
                "선정 절차와 세 후보의 성공·실패를 모두 공개해야 한다.\n"
                "- **측정된 wall time은 속도 benchmark가 아니다.** 여러 작업이 "
                "같은 A6000 GPU에서 동시에 실행됐으므로 방법별 순수 실행속도로 "
                "일반화할 수 없다."
            ),
        },
        {
            "id": "strategy",
            "type": "markdown",
            "body": (
                "## 권고 전략: 사전에 고정한 기본 설정을 중심으로 제한적으로 답변한다\n\n"
                "**현재 권고는 모든 직접 비교의 `RESPONSE HOLD`를 자동으로 "
                "해제하자는 것이 아니다. 저자가 전체 긍정·혼합·불리한 결과를 "
                "검토한 뒤, 사전에 고정한 `(b,o)=(1,1)` 결과와 필요한 한계만 "
                "선별적으로 reviewer 답변에 사용하는 것이다.**\n\n"
                "1. **Area Chair와 6SDM 답변은 기본 설정부터 설명한다.** 결과를 "
                "보고 고른 극단적 강도가 아니라, 제출 코드의 배율을 바꾸지 않은 "
                "설정이라는 점을 먼저 밝힌다. 10%와 15% 결과, 그리고 세 seed "
                "모두에서 rMAE와 rMSE가 같은 방향이었다는 사실을 제시한다.\n"
                "2. **주장의 대상을 정확히 표현한다.** ‘real-world noise에서 "
                "물리적으로 더 정확하다’고 쓰지 않는다. 대신 ‘실제 PIV 학습 "
                "측정에 통제된 강한 오염을 주입했을 때, 오염하지 않은 별도 "
                "PIV 측정과의 일치도가 개선됐다’고 쓴다.\n"
                "3. **강도 탐색 결과는 첫 답변에서 생략하는 편이 안전하다.** "
                "기본 설정만으로도 결과가 강하고 사후 선택 문제가 없다. "
                "탐색 후보를 쓰려면 seed 39에서 후보를 골랐다는 사실과 세 "
                "후보의 성공·불안정·실패를 모두 설명해야 하므로 글자 수와 "
                "해석 부담이 커진다.\n"
                "4. **가장 유사한 선행 방법과의 공통점을 먼저 인정한다.** "
                "naPINN과 PINN-EBM 모두 residual 분포를 학습하는 EBM을 쓰고, "
                "PINN을 먼저 준비한 뒤 EBM을 초기화하고 마지막에 함께 학습하는 "
                "단계식 절차를 사용한다. 차이는 naPINN이 EBM의 분포 추정 "
                "학습을 PINN 파라미터를 바꾸는 gradient와 분리하고, EBM 점수를 "
                "각 학습값의 포함 여부를 정하는 gate로 바꾸며, 그 결정을 "
                "재구성 loss와 과도한 제거를 막는 penalty에 연결한다는 데 있다.\n"
                "5. **합성 실험의 불리한 결과를 숨기지 않는다.** 세 PDE × 세 "
                "오염 비율의 9개 합성 조건에서는 direct PINN-EBM의 rMAE가 "
                "모두 더 낮았음을 짧게 인정한다. 그래야 실제 PIV 결과를 "
                "보편적 우위로 과장하지 않는 답변이 된다.\n"
                "6. **Gate contribution을 이상치 순위화로 설명하지 않는다.** "
                "Raw EBM과 gate의 AUROC가 거의 같으므로, 핵심은 새로운 ranking "
                "능력이 아니다. EBM 점수를 실제 학습값 포함·제외 결정과 "
                "재구성 목표에 연결한 방식, 그리고 그 방식이 특정 실제 PIV "
                "조건에서 재구성 오차를 낮춘 결과로 설명한다.\n"
                "7. **탐색 결과가 꼭 필요하면 성공과 실패를 한 문장에 함께 쓴다.** "
                "‘세 후보 중 두 개는 세-seed 평균 기준을 통과했지만, 모든 "
                "seed에서 두 오차가 개선된 후보는 하나뿐이었다’고 표현한다."
            ),
        },
        {"id": "routing_table", "type": "table", "tableId": "table_routing"},
        {
            "id": "draft_language",
            "type": "markdown",
            "sourceId": "exact_aggregation",
            "body": (
                "## 저자가 수치 공개를 승인한 뒤 사용할 수 있는 짧은 영문 초안\n\n"
                "아래 문단은 실제 PIV 기본 설정의 결과만 담은 reviewer-facing "
                "초안이다. `transferred`는 모델을 전이했다는 뜻이 아니라 제출 "
                "코드의 오염 생성 규칙을 실제 PIV 학습 측정에 적용했다는 뜻으로 "
                "사용한다. 제출 전에는 이 표현이 오해되지 않는지 다시 확인해야 한다.\n\n"
                "> We transferred the submitted-code four-Gaussian plus "
                "additive gross-outlier protocol to real Cylinder PIV, "
                "corrupting only 192 irregular training sensors while keeping "
                "1,549,400 spatially disjoint held-out PIV vectors unchanged. "
                "At the mandatory, unselected legacy center, naPINN-MSE "
                f"achieved rMAE/rMSE {fmt(mean(exact10['selected'], 'rMAE'))}/"
                f"{fmt(mean(exact10['selected'], 'rMSE'))} at 10% gross rows and "
                f"{fmt(mean(exact15['selected'], 'rMAE'))}/"
                f"{fmt(mean(exact15['selected'], 'rMSE'))} at 15%, below every "
                "matched baseline in both metrics for each of seeds 40–42. "
                f"The strongest non-naPINN baseline was OrPINN-q2.9 at "
                f"{fmt(mean(exact10['baseline'], 'rMAE'))}/"
                f"{fmt(mean(exact10['baseline'], 'rMSE'))} and "
                f"{fmt(mean(exact15['baseline'], 'rMAE'))}/"
                f"{fmt(mean(exact15['baseline'], 'rMSE'))}, respectively. "
                "These are controlled severe corruptions over a real measured "
                "field; held-out PIV is an independent measurement reference, "
                "not noise-free physical truth, and the nominal PDE does not "
                "separate observation from model discrepancy.\n\n"
                "이 문구는 결과를 보고 고르지 않은 `(b,o)=(1,1)`만 사용한다. "
                "Pilar–Wahlström과의 차이를 직접 묻는 답변에는 두 방법의 "
                "공통점과 학습목표·gradient 경로의 차이를 이어서 설명하고, "
                "9개 합성 조건에서는 direct PINN-EBM이 더 좋았다는 결과도 "
                "한두 문장으로 추가해야 한다."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 다음 결정과 추가 질문\n\n"
                "1. 저자가 `(b,o)=(1,1)`의 직접 비교 수치를 reviewer 답변에 "
                "공개할지 결정한다.\n"
                "2. 공개한다면 Area Chair, 6SDM, aoJS의 각 10,000자 답변에서 "
                "이번 실제 PIV 결과에 몇 자를 배정할지 정한다.\n"
                "3. 오염 강도 탐색으로 선택한 세 후보는 첫 답변에서 제외하고, "
                "추가 질문이 있을 때만 쓰는 보충 근거로 남길지 결정한다.\n"
                "4. 가장 유사한 선행 방법을 설명하는 문단에서, 9개 합성 조건의 "
                "불리한 결과를 숫자까지 제시할지 아니면 방향만 설명할지 정한다.\n"
                "5. 논문은 관측값 교체라고 쓰고 코드는 양의 offset 추가를 "
                "실행한다. `submitted-code protocol`이라는 표현만으로 이 차이가 "
                "충분히 명확한지 최종 학술적 검토를 한다.\n\n"
                "현재 검증된 결과만으로 reviewer의 핵심 질문에 답할 수 있으므로 "
                "추가 GPU 실험이 반드시 필요한 상태는 아니다. 남은 핵심 작업은 "
                "더 많은 hyperparameter tuning이 아니라, 어떤 수치를 공개할지에 "
                "대한 저자 결정과 제한된 글자 수 안에서 과장 없이 설명하는 일이다."
            ),
        },
    ]

    report_datasets = {
        "campaign_status": [
            {
                "exact_runs": 48,
                "scale_runs": 144,
                "confirmation_runs": 72,
                "mean_confirmed_candidates": 2,
            }
        ],
        "exact_results": exact_rows,
        "exact_consistency": exact_consistency,
        "severity": severity,
        "candidate_summary": candidate_rows,
        "candidate_seed": candidate_seed_rows,
        "candidate_gains": candidate_gain_rows,
        "whole_suite_context": context_rows,
        "reviewer_routing": routing_rows,
    }
    report_datasets, dataset_sources = materialize_report_datasets(
        report_datasets
    )
    sources.extend(dataset_sources)
    for item in [*cards, *charts, *tables]:
        item["sourceId"] = f"report_dataset_{item['dataset']}"

    manifest = {
        "version": 1,
        "surface": "report",
        "title": (
            "실제 Cylinder-PIV에 기존 four-Gaussian 오염 규칙을 적용한 "
            "실험 결과와 Rebuttal 전략"
        ),
        "description": (
            "Internal technical report for the completed naPINN legacy-4G "
            "Cylinder-PIV rebuttal campaign."
        ),
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": report_datasets,
        "accessIssues": [],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }


def markdown_value(value: Any, field: str) -> str:
    if value is None:
        return "—"
    if field == "frozen_mean_confirmed":
        return {"Yes": "통과", "No": "실패"}.get(str(value), str(value))
    if field == "both_metrics_better":
        return {"Yes": "예", "No": "아니오"}.get(str(value), str(value))
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if field in {
            "method_order",
            "ratio_pct",
            "rank",
            "seed",
        }:
            return f"{value:.0f}"
        if field in {"background_multiplier", "gross_multiplier"}:
            return f"{value:g}"
        if "pct" in field or "gain" in field:
            return f"{value:.2f}"
        return f"{value:.5f}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_sources(
    source_ids: list[str],
    sources_by_id: dict[str, dict[str, Any]],
) -> str:
    paths = [
        f"`{sources_by_id[source_id]['path']}`"
        for source_id in source_ids
        if source_id in sources_by_id
    ]
    if not paths:
        return ""
    return f"_출처: {'; '.join(paths)}._"


def markdown_table(
    table: dict[str, Any],
    datasets: dict[str, list[dict[str, Any]]],
    sources_by_id: dict[str, dict[str, Any]],
) -> str:
    columns = table["columns"]
    rows = datasets[table["dataset"]]
    lines = [f"### {table['title']}"]
    if table.get("subtitle"):
        lines.extend(["", f"_{table['subtitle']}_"])
    lines.extend(
        [
            "",
            "| " + " | ".join(column["label"] for column in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_value(row.get(column["field"]), column["field"])
                for column in columns
            )
            + " |"
        )
    source_note = markdown_sources(
        DATASET_RAW_SOURCE_IDS.get(table["dataset"], []),
        sources_by_id,
    )
    if source_note:
        lines.extend(["", source_note])
    return "\n".join(lines)


def markdown_chart_table(
    chart: dict[str, Any],
    datasets: dict[str, list[dict[str, Any]]],
    sources_by_id: dict[str, dict[str, Any]],
) -> str:
    lines = [f"### {chart['title']}"]
    if chart.get("subtitle"):
        lines.extend(["", f"_{chart['subtitle']}_"])

    if chart["id"] == "chart_exact_rmae":
        rows = datasets["exact_results"]
        by_method: dict[str, dict[int, dict[str, Any]]] = {}
        for row in rows:
            by_method.setdefault(row["method"], {})[row["ratio_pct"]] = row
        lines.extend(
            [
                "",
                "| 방법 | 10% 이상치 조건 rMAE | 15% 이상치 조건 rMAE |",
                "| --- | ---: | ---: |",
            ]
        )
        for method, ratios in sorted(
            by_method.items(),
            key=lambda item: min(
                row["method_order"] for row in item[1].values()
            ),
        ):
            cells = []
            for ratio in (10, 15):
                row = ratios[ratio]
                cells.append(
                    f"{row['rMAE']:.5f} ± {row['rMAE_std']:.5f}"
                )
            lines.append(f"| {method} | {cells[0]} | {cells[1]} |")
        dataset_name = "exact_results"
    elif chart["id"] == "chart_candidate_gains":
        rows = datasets["candidate_summary"]
        lines.extend(
            [
                "",
                "| 후보 오염 설정 | naPINN rMAE 감소율 (%) | "
                "naPINN rMSE 감소율 (%) | 세-seed 평균 기준 통과 | "
                "두 지표 모두 우세한 seed 수 |",
                "| --- | ---: | ---: | --- | ---: |",
            ]
        )
        for row in sorted(rows, key=lambda item: item["rank"]):
            lines.append(
                f"| {row['cell']} | {row['rMAE_gain_pct']:.2f} | "
                f"{row['rMSE_gain_pct']:.2f} | "
                f"{markdown_value(row['frozen_mean_confirmed'], 'frozen_mean_confirmed')} | "
                f"{row['both_metrics_seed_wins']} |"
            )
        dataset_name = "candidate_gains"
    else:
        raise KeyError(f"Unsupported Markdown chart: {chart['id']}")

    lines.extend(
        [
            "",
            "_Markdown 형식에서는 시각적 막대그래프 대신 같은 비교를 정확한 "
            "수치표로 제시한다._",
        ]
    )
    source_note = markdown_sources(
        DATASET_RAW_SOURCE_IDS[dataset_name],
        sources_by_id,
    )
    if source_note:
        lines.extend(["", source_note])
    return "\n".join(lines)


def render_markdown(artifact: dict[str, Any]) -> str:
    manifest = artifact["manifest"]
    datasets = artifact["snapshot"]["datasets"]
    sources_by_id = {
        source["id"]: source for source in manifest["sources"]
    }
    cards_by_id = {card["id"]: card for card in manifest["cards"]}
    charts_by_id = {chart["id"]: chart for chart in manifest["charts"]}
    tables_by_id = {table["id"]: table for table in manifest["tables"]}

    sections: list[str] = []
    for block in manifest["blocks"]:
        if block["type"] == "markdown":
            body = block["body"].rstrip()
            source_note = markdown_sources(
                [block["sourceId"]] if block.get("sourceId") else [],
                sources_by_id,
            )
            sections.append(
                body if not source_note else f"{body}\n\n{source_note}"
            )
        elif block["type"] == "metric-strip":
            lines = [
                "### 완료된 실험 규모",
                "",
                "| 지표 | 값 |",
                "| --- | ---: |",
            ]
            source_ids: list[str] = []
            for card_id in block["cardIds"]:
                card = cards_by_id[card_id]
                row = datasets[card["dataset"]][0]
                for metric_spec in card["metrics"]:
                    lines.append(
                        f"| {metric_spec['label']} | "
                        f"{markdown_value(row[metric_spec['field']], metric_spec['field'])} |"
                    )
                source_ids.extend(
                    DATASET_RAW_SOURCE_IDS.get(card["dataset"], [])
                )
            source_note = markdown_sources(
                list(dict.fromkeys(source_ids)),
                sources_by_id,
            )
            if source_note:
                lines.extend(["", source_note])
            sections.append("\n".join(lines))
        elif block["type"] == "chart":
            sections.append(
                markdown_chart_table(
                    charts_by_id[block["chartId"]],
                    datasets,
                    sources_by_id,
                )
            )
        elif block["type"] == "table":
            sections.append(
                markdown_table(
                    tables_by_id[block["tableId"]],
                    datasets,
                    sources_by_id,
                )
            )
        else:
            raise KeyError(f"Unsupported report block: {block['type']}")
    return "\n\n".join(sections).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "rebuttal/reports/legacy4g_experiment_ko/report.md"
        ),
    )
    args = parser.parse_args()
    artifact = build_artifact()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(artifact),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
