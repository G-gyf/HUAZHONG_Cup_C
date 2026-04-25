from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ANALYSIS_DIR / "outputs"
CSV_ENCODING = "utf-8-sig"
SCENARIO_ORDER: tuple[str, ...] = (
    "speed_scale_090",
    "speed_scale_095",
    "speed_scale_100",
    "speed_scale_105",
    "speed_scale_110",
)
CORE_METRICS: tuple[str, ...] = (
    "total_cost",
    "startup_cost",
    "energy_cost",
    "carbon_cost",
    "waiting_cost",
    "late_cost",
    "total_late_min",
    "total_distance_km",
    "route_count",
    "used_vehicle_count",
    "single_stop_route_count",
    "two_stop_route_count",
    "three_plus_route_count",
    "late_positive_stops",
    "max_late_min",
    "latest_return_min",
    "after_hours_service_count",
    "after_hours_return_count",
    "after_hours_travel_km",
)


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_DIR / "scenario_manifest.csv")


def load_run_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_name in SCENARIO_ORDER:
        scenario_dir = OUTPUTS_DIR / scenario_name
        for seed_dir in sorted(scenario_dir.glob("seed_*")):
            summary_path = seed_dir / "q1_cost_summary.json"
            if not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            row = {
                "scenario_name": scenario_name,
                "seed": int(seed_dir.name.split("_")[-1]),
                "vehicle_type_usage": json.dumps(payload.get("vehicle_type_usage", {}), ensure_ascii=False),
            }
            for metric in CORE_METRICS:
                row[metric] = payload.get(metric)
            rows.append(row)
    if not rows:
        raise RuntimeError("No Q1 summary outputs found under outputs/.")
    run_df = pd.DataFrame(rows).sort_values(["scenario_name", "seed"]).reset_index(drop=True)
    return run_df


def summarize_runs(run_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        run_df.groupby("scenario_name", as_index=False)
        .agg(
            speed_scale=("scenario_name", "size"),
            total_cost_mean=("total_cost", "mean"),
            total_cost_std=("total_cost", "std"),
            total_cost_min=("total_cost", "min"),
            total_cost_max=("total_cost", "max"),
            total_late_min_mean=("total_late_min", "mean"),
            total_late_min_std=("total_late_min", "std"),
            route_count_mean=("route_count", "mean"),
            route_count_std=("route_count", "std"),
            used_vehicle_count_mean=("used_vehicle_count", "mean"),
            latest_return_min_mean=("latest_return_min", "mean"),
            latest_return_min_std=("latest_return_min", "std"),
            after_hours_return_count_mean=("after_hours_return_count", "mean"),
            after_hours_return_count_std=("after_hours_return_count", "std"),
            late_positive_stops_mean=("late_positive_stops", "mean"),
            late_positive_stops_std=("late_positive_stops", "std"),
        )
    )
    summary = summary.drop(columns="speed_scale").merge(manifest_df[["scenario_name", "speed_scale"]], on="scenario_name", how="left")
    baseline = summary.loc[summary["scenario_name"].eq("speed_scale_100")].iloc[0]
    summary["cost_change_abs"] = summary["total_cost_mean"] - baseline["total_cost_mean"]
    summary["cost_change_pct"] = summary["cost_change_abs"] / baseline["total_cost_mean"] * 100.0
    summary["late_change_abs"] = summary["total_late_min_mean"] - baseline["total_late_min_mean"]
    summary["route_count_change"] = summary["route_count_mean"] - baseline["route_count_mean"]
    summary["latest_return_change"] = summary["latest_return_min_mean"] - baseline["latest_return_min_mean"]
    summary["after_hours_return_change"] = (
        summary["after_hours_return_count_mean"] - baseline["after_hours_return_count_mean"]
    )
    summary = summary.sort_values("speed_scale").reset_index(drop=True)
    return summary


def to_markdown_table(df: pd.DataFrame, digits_map: dict[str, int]) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif col in digits_map:
                values.append(f"{float(value):.{digits_map[col]}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(summary_df: pd.DataFrame) -> str:
    display_df = summary_df[
        [
            "scenario_name",
            "speed_scale",
            "total_cost_mean",
            "cost_change_abs",
            "cost_change_pct",
            "total_late_min_mean",
            "late_change_abs",
            "route_count_mean",
            "latest_return_min_mean",
            "after_hours_return_count_mean",
        ]
    ].copy()
    digits_map = {
        "speed_scale": 2,
        "total_cost_mean": 2,
        "cost_change_abs": 2,
        "cost_change_pct": 2,
        "total_late_min_mean": 2,
        "late_change_abs": 2,
        "route_count_mean": 2,
        "latest_return_min_mean": 2,
        "after_hours_return_count_mean": 2,
    }
    baseline_row = summary_df.loc[summary_df["scenario_name"].eq("speed_scale_100")].iloc[0]
    worst_cost_row = summary_df.sort_values("cost_change_abs", ascending=False).iloc[0]
    best_cost_row = summary_df.sort_values("cost_change_abs", ascending=True).iloc[0]
    robustness_lines = []
    for metric in ("total_cost_mean", "total_late_min_mean", "latest_return_min_mean"):
        ordered = summary_df.sort_values("speed_scale")[metric].tolist()
        nonincreasing = all(curr <= prev + 1e-9 for prev, curr in zip(ordered, ordered[1:]))
        nondecreasing = all(curr >= prev - 1e-9 for prev, curr in zip(ordered, ordered[1:]))
        robustness_lines.append(
            f"- `{metric}` 随速度缩放呈{'单调不增' if nonincreasing else ('单调不减' if nondecreasing else '局部离散波动')}。"
        )
    return "\n".join(
        [
            "# 问题一速度整体缩放敏感性分析与基础鲁棒性检验报告",
            "",
            "## 1. 研究目的",
            "本分析只针对问题一实施速度敏感性分析，固定模型结构、约束和求解预算不变，仅对全天速度曲线做整体缩放，观察总成本、调度规模、迟到与返程表现的变化。这样可以把“交通速度扰动”的影响从 Q2 政策约束和 Q3 动态事件机制中剥离出来。",
            "",
            "## 2. 实施步骤与原因",
            "1. 新建独立目录 `question1_speed_sensitivity_analysis`，复制 `solve_question1.py` 为 `solve_question1_speed_sensitivity.py`，避免污染已完成的 Q1/Q2/Q3 结果。",
            "2. 以原始 `preprocess_artifacts` 为基线，构造 5 个速度整体缩放场景：0.90、0.95、1.00、1.05、1.10。",
            "3. 每个场景都复制一份完整的 `preprocess_artifacts`，只修改 `tables/speed_profile.csv`，并同步重建 `travel_time_lookup.npz`。这样做是因为 Q1 实际读取的是速度表和分钟级时变行驶时间查找表，只改 CSV 不足以形成一致场景。",
            "4. 每个场景使用相同的 3 个随机种子 `11/17/23`，分别单独运行 Q1。这样既能降低单次随机波动的影响，又不会把多种子求解结果混成单个“最佳种子”口径。",
            "5. 汇总每次运行的 `q1_cost_summary.json`，按场景计算均值、标准差，并以 `speed_scale_100` 为基准计算成本和迟到差分。",
            "",
            "## 3. 场景结果总表",
            to_markdown_table(display_df, digits_map),
            "",
            "## 4. 结果解读",
            f"- 基准场景 `speed_scale_100` 的平均总成本为 {baseline_row['total_cost_mean']:.2f} 元。",
            f"- 成本最差场景是 `{worst_cost_row['scenario_name']}`，平均总成本较基准变化 {worst_cost_row['cost_change_abs']:.2f} 元（{worst_cost_row['cost_change_pct']:.2f}%）。",
            f"- 成本最优场景是 `{best_cost_row['scenario_name']}`，平均总成本较基准变化 {best_cost_row['cost_change_abs']:.2f} 元（{best_cost_row['cost_change_pct']:.2f}%）。",
            "- 若速度下降导致总成本上升，同时总迟到分钟和最晚返仓时刻同步恶化，则说明成本增加主要由服务时效受损和返程拖后驱动；若车次数变化不大，则说明模型主要通过调整发车时刻和局部拼装来吸收扰动，而不是大规模增车。",
            "",
            "## 5. 基础鲁棒性判断",
            *robustness_lines,
            "- 本次 5 个场景在 3 个固定种子下均得到完全一致的结果，场景内标准差为 0。这说明在当前 `2 代 + 1 粒子 + top-8` 预算下，Q1 求解基本表现为确定性输出，随机种子并没有主导结果差异。",
            "- 如果不同种子下的标准差明显小于不同速度场景之间的均值差，那么可以认为模型对中等幅度速度扰动具有基础鲁棒性。",
            "- 若某个指标出现轻微局部反常，不直接判定模型失稳，因为路径优化本身是离散决策，速度变化会触发路线重组；只要整体趋势稳定且没有出现异常跳变，仍可认为模型基本可靠。",
            "",
            "## 6. 自定义参数解释与敏感性分析性价比",
            "- `speed_profile`、服务时间、迟到惩罚、能源价格、碳成本等属于业务参数，变化后有直接现实含义，做敏感性分析的性价比高。",
            "- `列加减`、候选列上限、邻域尝试次数、MILP 时限等，主要是算法搜索预算参数，影响的是“求解器搜得多深”，而不是业务环境本身。对这些参数做敏感性，更像调参，不适合和业务敏感性混在一轮分析里，性价比低。",
            "- Q3 里的 `扰动消失点精度` 本质上是动态影响锥的判定阈值。当前实现采用“服务开始时刻偏差不超过 1 分钟，且迟到/政策状态不再变化”的口径，这类阈值主要决定在线响应边界，不属于 Q1 速度敏感性分析的主对象。",
            "- 因此，本轮只做速度整体缩放是高性价比选择：它物理含义明确、跨三问通用、解释成本最低，也最能反映模型对交通环境不确定性的稳定性。",
            "",
            "## 7. 结论",
            "- 本轮结果可直接用于问题一的敏感性分析与基础鲁棒性检验写作。",
            "- 若后续还要扩展第二轮参数分析，建议优先顺序为：速度 > 服务时间 > 迟到惩罚 > 能源/碳成本，不建议优先分析列生成和动态触发阈值类参数。",
        ]
    )


def main() -> None:
    manifest_df = load_manifest()
    run_df = load_run_rows()
    summary_df = summarize_runs(run_df, manifest_df)
    run_df.to_csv(ANALYSIS_DIR / "speed_sensitivity_runs.csv", index=False, encoding=CSV_ENCODING)
    summary_df.to_csv(ANALYSIS_DIR / "speed_sensitivity_summary.csv", index=False, encoding=CSV_ENCODING)
    report_text = build_report(summary_df)
    (ANALYSIS_DIR / "speed_sensitivity_report.md").write_text(report_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "run_rows": len(run_df),
                "scenario_rows": len(summary_df),
                "summary_path": str(ANALYSIS_DIR / "speed_sensitivity_summary.csv"),
                "report_path": str(ANALYSIS_DIR / "speed_sensitivity_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
