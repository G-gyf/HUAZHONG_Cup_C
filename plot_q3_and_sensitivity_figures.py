from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


Q2_BASELINE_TOTAL_COST = 89850.0134399558
Q2_BASELINE_ROUTE_COUNT = 132


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures for Q3 dynamic results and Q1 speed sensitivity.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--q3-root", type=Path, default=Path.cwd() / "question3_artifacts_dynamic_s11")
    parser.add_argument(
        "--q1-sensitivity-root",
        type=Path,
        default=Path.cwd() / "question1_speed_sensitivity_analysis",
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "paper_compare_figures" / "q3_sensitivity_figures")
    return parser.parse_args()


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 260,
            "axes.facecolor": "#fbfdff",
            "figure.facecolor": "#fbfdff",
            "savefig.facecolor": "#fbfdff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#dbe4ee",
            "axes.titleweight": "bold",
            "font.size": 10,
            "legend.frameon": True,
        }
    )


def save_figure(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_no_ext.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_no_ext.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def minute_to_clock_text(minute_value: float) -> str:
    total_minutes = int(round(float(minute_value)))
    hour = 8 + total_minutes // 60
    minute = total_minutes % 60
    if hour >= 24:
        return f"D+1 {hour - 24:02d}:{minute:02d}"
    return f"{hour:02d}:{minute:02d}"


def annotate_line(ax: plt.Axes, x: list[float], y: list[float], fmt: str, dy: float = 0.0) -> None:
    for xi, yi in zip(x, y):
        ax.text(xi, yi + dy, fmt.format(yi), ha="center", va="bottom", fontsize=9, color="#0f172a")


def load_inputs(q3_root: Path, q1_sensitivity_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q3_metrics = pd.read_csv(q3_root / "q3_event_metrics.csv")
    q3_log = pd.read_csv(q3_root / "q3_event_log.csv")
    q1_summary = pd.read_csv(q1_sensitivity_root / "speed_sensitivity_summary.csv")
    return q3_metrics, q3_log, q1_summary


def plot_q3_cost_and_routes(q3_metrics: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(q3_metrics))
    labels = [f"{eid}\n{hhmm}" for eid, hhmm in zip(q3_metrics["event_id"], q3_metrics["event_time_hhmm"])]

    fig, ax1 = plt.subplots(figsize=(9.2, 5.4))
    ax2 = ax1.twinx()

    cost_line = ax1.plot(
        x,
        q3_metrics["projected_full_day_cost"],
        color="#1d4ed8",
        marker="o",
        linewidth=2.2,
        label="Projected full-day cost",
    )
    ax1.axhline(Q2_BASELINE_TOTAL_COST, color="#94a3b8", linestyle="--", linewidth=1.8, label="Q2 static baseline cost")
    annotate_line(ax1, list(x), q3_metrics["projected_full_day_cost"].tolist(), "{:.0f}", dy=120)
    ax1.set_ylabel("Projected full-day cost")

    route_bars = ax2.bar(
        x,
        q3_metrics["projected_full_day_route_count"],
        width=0.38,
        color="#f59e0b",
        alpha=0.33,
        edgecolor="#b45309",
        linewidth=1.2,
        label="Projected route count",
    )
    ax2.axhline(Q2_BASELINE_ROUTE_COUNT, color="#cbd5e1", linestyle=":", linewidth=1.6, label="Q2 static route count")
    for idx, value in enumerate(q3_metrics["projected_full_day_route_count"]):
        ax2.text(idx, value + 0.22, f"{int(value)}", ha="center", va="bottom", fontsize=9, color="#7c2d12")
    ax2.set_ylabel("Projected route count")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Q3 Dynamic Response: Full-Day Cost and Route Count by Event")

    handles = cost_line + [route_bars]
    labels_legend = ["Projected full-day cost", "Projected route count"]
    ax1.legend(handles, labels_legend, loc="upper left", facecolor="#ffffff", edgecolor="#e2e8f0")
    save_figure(fig, output_dir / "q3_cost_route_progression")


def plot_q3_response_structure(q3_metrics: pd.DataFrame, q3_log: pd.DataFrame, output_dir: Path) -> None:
    merged = q3_log.merge(
        q3_metrics[
            [
                "event_id",
                "modified_vehicle_count",
                "switched_depot_unit_count",
            ]
        ],
        on="event_id",
        how="left",
    )
    plot_df = pd.DataFrame(
        {
            "event": merged["event_id"],
            "Onboard routes": merged["direct_onboard_route_count"],
            "Depot routes": merged["direct_depot_route_count"],
            "Modified vehicles": merged["modified_vehicle_count"],
            "Switched depot units": merged["switched_depot_unit_count"],
        }
    )

    long_df = plot_df.melt(id_vars="event", var_name="metric", value_name="value")
    palette = {
        "Onboard routes": "#2563eb",
        "Depot routes": "#10b981",
        "Modified vehicles": "#f59e0b",
        "Switched depot units": "#ef4444",
    }
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    sns.barplot(data=long_df, x="event", y="value", hue="metric", palette=palette, ax=ax)
    ax.set_title("Q3 Dynamic Response: Local Re-optimization Load by Event")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", ncol=2, facecolor="#ffffff", edgecolor="#e2e8f0")
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            height + 0.18,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#0f172a",
        )
    save_figure(fig, output_dir / "q3_response_structure")


def plot_q3_lateness_and_cone(q3_metrics: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(q3_metrics))
    labels = [f"{eid}\n{hhmm}" for eid, hhmm in zip(q3_metrics["event_id"], q3_metrics["event_time_hhmm"])]

    fig, ax1 = plt.subplots(figsize=(9.2, 5.4))
    ax2 = ax1.twinx()

    bars = ax1.bar(
        x,
        q3_metrics["remaining_total_late_min"],
        color="#e11d48",
        alpha=0.78,
        edgecolor="#9f1239",
        linewidth=1.1,
        label="Remaining late minutes",
    )
    for idx, value in enumerate(q3_metrics["remaining_total_late_min"]):
        ax1.text(idx, value + 10, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5, color="#881337")
    ax1.set_ylabel("Remaining late minutes")

    t0_line = ax2.plot(
        x,
        q3_metrics["t0_min"],
        color="#0f766e",
        marker="D",
        linewidth=2.0,
        label="Adaptive cone boundary T0",
    )
    for idx, value in enumerate(q3_metrics["t0_min"]):
        ax2.text(idx, value + 12, minute_to_clock_text(value), ha="center", va="bottom", fontsize=8.5, color="#115e59")
    ax2.set_ylabel("T0 minute (from 08:00)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Q3 Dynamic Response: Lateness Pressure and Adaptive Cone Boundary")
    ax1.legend(bars.patches[:1] + t0_line, ["Remaining late minutes", "Adaptive cone boundary T0"], loc="upper left")
    save_figure(fig, output_dir / "q3_lateness_cone")


def plot_q1_sensitivity_cost_and_late(q1_summary: pd.DataFrame, output_dir: Path) -> None:
    df = q1_summary.sort_values("speed_scale").copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    axes[0].plot(df["speed_scale"], df["total_cost_mean"], color="#2563eb", marker="o", linewidth=2.2)
    axes[0].axvline(1.0, color="#cbd5e1", linestyle="--", linewidth=1.5)
    annotate_line(axes[0], df["speed_scale"].tolist(), df["total_cost_mean"].tolist(), "{:.0f}", dy=10)
    axes[0].set_title("Q1 Speed Sensitivity: Total Cost")
    axes[0].set_xlabel("Speed scaling factor")
    axes[0].set_ylabel("Average total cost")

    axes[1].plot(df["speed_scale"], df["total_late_min_mean"], color="#dc2626", marker="o", linewidth=2.2)
    axes[1].axvline(1.0, color="#cbd5e1", linestyle="--", linewidth=1.5)
    annotate_line(axes[1], df["speed_scale"].tolist(), df["total_late_min_mean"].tolist(), "{:.0f}", dy=8)
    axes[1].set_title("Q1 Speed Sensitivity: Total Late Minutes")
    axes[1].set_xlabel("Speed scaling factor")
    axes[1].set_ylabel("Average late minutes")

    save_figure(fig, output_dir / "q1_speed_cost_late")


def plot_q1_sensitivity_return(q1_summary: pd.DataFrame, output_dir: Path) -> None:
    df = q1_summary.sort_values("speed_scale").copy()
    x = np.arange(len(df))
    labels = [f"{scale:.2f}" for scale in df["speed_scale"]]

    fig, ax1 = plt.subplots(figsize=(9.0, 5.2))
    ax2 = ax1.twinx()

    line = ax1.plot(
        x,
        df["latest_return_min_mean"],
        color="#7c3aed",
        marker="o",
        linewidth=2.2,
        label="Latest return minute",
    )
    for idx, value in enumerate(df["latest_return_min_mean"]):
        ax1.text(idx, value + 6, minute_to_clock_text(value), ha="center", va="bottom", fontsize=8.5, color="#5b21b6")
    ax1.set_ylabel("Latest return minute (from 08:00)")

    bars = ax2.bar(
        x,
        df["after_hours_return_count_mean"],
        width=0.38,
        color="#14b8a6",
        alpha=0.35,
        edgecolor="#0f766e",
        linewidth=1.1,
        label="After-hours return count",
    )
    for idx, value in enumerate(df["after_hours_return_count_mean"]):
        ax2.text(idx, value + 0.1, f"{value:.0f}", ha="center", va="bottom", fontsize=8.5, color="#115e59")
    ax2.set_ylabel("After-hours return count")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Speed scaling factor")
    ax1.set_title("Q1 Speed Sensitivity: Return-Time Pressure")
    ax1.legend(line + [bars], ["Latest return minute", "After-hours return count"], loc="upper left")
    save_figure(fig, output_dir / "q1_speed_return_pressure")


def plot_q1_sensitivity_delta(q1_summary: pd.DataFrame, output_dir: Path) -> None:
    df = q1_summary.sort_values("speed_scale").copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    bars1 = axes[0].bar(df["speed_scale"].astype(str), df["cost_change_pct"], color="#0f766e", alpha=0.82)
    axes[0].axhline(0.0, color="#94a3b8", linewidth=1.4)
    axes[0].set_title("Q1 Speed Sensitivity: Cost Change vs Baseline")
    axes[0].set_xlabel("Speed scaling factor")
    axes[0].set_ylabel("Cost change (%)")
    for patch, value in zip(bars1, df["cost_change_pct"]):
        axes[0].text(
            patch.get_x() + patch.get_width() / 2.0,
            value + (0.006 if value >= 0 else -0.02),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8.5,
        )

    bars2 = axes[1].bar(df["speed_scale"].astype(str), df["route_count_change"], color="#f59e0b", alpha=0.82)
    axes[1].axhline(0.0, color="#94a3b8", linewidth=1.4)
    axes[1].set_title("Q1 Speed Sensitivity: Route Count Change vs Baseline")
    axes[1].set_xlabel("Speed scaling factor")
    axes[1].set_ylabel("Route count change")
    for patch, value in zip(bars2, df["route_count_change"]):
        axes[1].text(
            patch.get_x() + patch.get_width() / 2.0,
            value + 0.03,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    save_figure(fig, output_dir / "q1_speed_delta_summary")


def write_caption_notes(output_dir: Path) -> None:
    text = "\n".join(
        [
            "# Paper Figure Notes",
            "",
            "## Q3 Dynamic Scheduling Figures",
            "1. `q3_cost_route_progression`: 插入第三问结果分析部分，用于说明四次事件触发后，全天预测总成本和总调度车次如何逐步上升，并与 Q2 静态基线比较。",
            "2. `q3_response_structure`: 插入第三问方法效果分析部分，用于说明各事件触发的 onboard/depot 局部重优化负载，以及实际被改动车辆和换车单元数量。",
            "3. `q3_lateness_cone`: 插入第三问机制解释部分，用于说明事件推进后剩余迟到压力的累积，以及自适应影响锥边界 T0 的扩张位置。",
            "",
            "## Q1 Sensitivity / Robustness Figures",
            "4. `q1_speed_cost_late`: 插入敏感性分析部分，用于同时展示整体速度缩放对总成本和总迟到分钟的影响。",
            "5. `q1_speed_return_pressure`: 插入鲁棒性分析部分，用于展示速度变化对最晚返仓时刻和超时返仓车次数的影响。",
            "6. `q1_speed_delta_summary`: 插入结论补充部分，用于强调速度变化对成本和车次数的相对影响幅度，突出“成本稳、时效更敏感”的结论。",
            "",
            "## Usage Note",
            "- 所有图均同时导出为 PNG 和 SVG，可直接插入论文；若正文需要中文图题，建议在论文软件中直接改为中文题注，保留图内英文坐标即可。",
        ]
    )
    (output_dir / "figure_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_plot_style()
    q3_metrics, q3_log, q1_summary = load_inputs(args.q3_root, args.q1_sensitivity_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_q3_cost_and_routes(q3_metrics, args.output_dir)
    plot_q3_response_structure(q3_metrics, q3_log, args.output_dir)
    plot_q3_lateness_and_cone(q3_metrics, args.output_dir)
    plot_q1_sensitivity_cost_and_late(q1_summary, args.output_dir)
    plot_q1_sensitivity_return(q1_summary, args.output_dir)
    plot_q1_sensitivity_delta(q1_summary, args.output_dir)
    write_caption_notes(args.output_dir)
    print(f"Figures written to: {args.output_dir}")


if __name__ == "__main__":
    main()
