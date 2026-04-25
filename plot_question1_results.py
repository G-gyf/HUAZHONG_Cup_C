from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects as pe


GREEN_ZONE_RADIUS_KM = 10.0
GREEN_ZONE_FILL = "#d9f0e5"
GREEN_ZONE_EDGE = "#66a182"

VEHICLE_COLORS = {
    "fuel_3000": "#4c566a",
    "fuel_1500": "#5b8fb9",
    "ev_3000": "#2a9d8f",
    "ev_1250": "#7cae5a",
    "fuel_1250": "#c98c64",
}

FAMILY_COLORS = {
    "Rigid Big": "#4c566a",
    "Piggyback Big": "#c98c64",
    "Promotion-Like Big": "#2a9d8f",
    "Blocking Big": "#7a516b",
    "Flex Small": "#5b8fb9",
    "Support": "#94a3b8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for question 1 results.")
    parser.add_argument("--artifacts-root", type=Path, default=Path.cwd() / "question1_artifacts_hybrid_coupled_heavy_s11")
    parser.add_argument("--preprocess-root", type=Path, default=Path.cwd() / "preprocess_artifacts")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 220,
            "axes.facecolor": "#fbfdff",
            "figure.facecolor": "#fbfdff",
            "savefig.facecolor": "#fbfdff",
            "axes.edgecolor": "#cbd5e1",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#dbe4ee",
            "font.size": 10,
        }
    )


def format_minutes_axis(ax: plt.Axes) -> None:
    ticks = np.arange(0, 24 * 60 + 1, 120)
    labels = [f"{int(tick // 60):02d}:{int(tick % 60):02d}" for tick in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0f172a",
        )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_soft_route(
    ax: plt.Axes,
    x: list[float],
    y: list[float],
    color: str,
    linewidth: float,
    alpha: float,
    zorder: float,
    halo_width: float = 1.2,
) -> None:
    line = ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        solid_capstyle="round",
        zorder=zorder,
    )[0]
    if halo_width > 0.0:
        line.set_path_effects(
            [
                pe.Stroke(linewidth=linewidth + halo_width, foreground="#fbfdff"),
                pe.Normal(),
            ]
        )


def family_label(row: pd.Series) -> str:
    if int(row.get("blocking_big_flexible_route_flag", 0)) == 1:
        return "Blocking Big"
    if int(row.get("piggyback_big_route_flag", 0)) == 1:
        return "Piggyback Big"
    if int(row.get("promotion_like_big_route_flag", 0)) == 1:
        return "Promotion-Like Big"
    if int(row.get("big_route_flag", 0)) == 1:
        return "Rigid Big"
    if str(row.get("candidate_family", "")) == "flex_small":
        return "Flex Small"
    return "Support"


def build_plot_context(artifacts_root: Path, preprocess_root: Path) -> dict[str, object]:
    cost_summary = json.loads((artifacts_root / "q1_hybrid_cost_summary.json").read_text(encoding="utf-8"))
    route_df = pd.read_csv(artifacts_root / "q1_hybrid_route_summary.csv")
    stop_df = pd.read_csv(artifacts_root / "q1_hybrid_stop_schedule.csv")
    route_pool_df = pd.read_csv(artifacts_root / "q1_hybrid_route_pool_summary.csv")
    customer_df = pd.read_csv(preprocess_root / "tables" / "customer_master_98.csv")

    route_df["unit_count"] = route_df["unit_sequence"].astype(str).apply(lambda text: len(text.split(",")))
    route_df["route_key"] = route_df["vehicle_type"].astype(str) + "|" + route_df["unit_sequence"].astype(str)
    route_df["vehicle_label"] = route_df["vehicle_type"].astype(str)

    route_pool_df["family"] = route_pool_df.apply(family_label, axis=1)
    family_by_key = route_pool_df.set_index("route_key")["family"].to_dict()
    route_df["family"] = route_df["route_key"].map(family_by_key).fillna("Support")

    vehicle_rank = {name: idx for idx, name in enumerate(VEHICLE_COLORS)}
    route_df["vehicle_rank"] = route_df["vehicle_type"].map(vehicle_rank).fillna(99)
    route_df["return_span_min"] = route_df["return_min"] - route_df["departure_min"]

    active_customers = customer_df.loc[customer_df["has_orders"]].copy()
    customer_coords = customer_df[["cust_id", "x_km", "y_km"]].copy().rename(columns={"cust_id": "orig_cust_id"})
    stop_df = stop_df.merge(customer_coords, on="orig_cust_id", how="left")
    dominant_vehicle = (
        stop_df.groupby(["orig_cust_id", "vehicle_type"], as_index=False)
        .size()
        .sort_values(["orig_cust_id", "size", "vehicle_type"], ascending=[True, False, True])
        .drop_duplicates("orig_cust_id")
        .rename(columns={"size": "visit_count"})
    )
    customer_visit_count = (
        stop_df.groupby("orig_cust_id", as_index=False)
        .agg(
            stop_visit_count=("route_id", "size"),
            route_visit_count=("route_id", "nunique"),
            delivered_weight_kg=("delivered_weight_kg", "sum"),
        )
    )
    active_customers = active_customers.merge(
        dominant_vehicle[["orig_cust_id", "vehicle_type"]].rename(columns={"orig_cust_id": "cust_id"}),
        on="cust_id",
        how="left",
    )
    active_customers = active_customers.merge(
        customer_visit_count.rename(columns={"orig_cust_id": "cust_id"}),
        on="cust_id",
        how="left",
    )
    active_customers["vehicle_type"] = active_customers["vehicle_type"].fillna("unserved")
    active_customers["route_visit_count"] = active_customers["route_visit_count"].fillna(0)
    active_customers["stop_visit_count"] = active_customers["stop_visit_count"].fillna(0)
    active_customers["delivered_weight_kg"] = active_customers["delivered_weight_kg"].fillna(0.0)

    return {
        "cost_summary": cost_summary,
        "route_df": route_df,
        "stop_df": stop_df,
        "route_pool_df": route_pool_df,
        "active_customers": active_customers,
    }


def build_route_paths(route_df: pd.DataFrame, stop_df: pd.DataFrame) -> list[dict[str, object]]:
    route_meta = route_df.set_index("route_id").to_dict(orient="index")
    route_paths: list[dict[str, object]] = []
    for route_id, sub_df in stop_df.sort_values(["route_id", "stop_index"]).groupby("route_id"):
        points_x = [0.0, *sub_df["x_km"].astype(float).tolist(), 0.0]
        points_y = [0.0, *sub_df["y_km"].astype(float).tolist(), 0.0]
        meta = route_meta.get(route_id, {})
        route_paths.append(
            {
                "route_id": int(route_id),
                "vehicle_type": str(meta.get("vehicle_type", "")),
                "family": str(meta.get("family", "Support")),
                "route_cost": float(meta.get("route_cost", 0.0)),
                "late_positive_stop_count": int(meta.get("late_positive_stop_count", 0)),
                "after_hours_return_flag": int(meta.get("after_hours_return_flag", 0)),
                "x": points_x,
                "y": points_y,
            }
        )
    return route_paths


def draw_base_map(ax: plt.Axes, active_customers: pd.DataFrame, subtitle: str | None = None) -> None:
    ax.add_patch(plt.Circle((0, 0), GREEN_ZONE_RADIUS_KM, color=GREEN_ZONE_FILL, alpha=0.24, ec=GREEN_ZONE_EDGE, lw=2))
    ax.scatter(
        active_customers["x_km"],
        active_customers["y_km"],
        s=18,
        c="#cbd5e1",
        alpha=0.58,
        edgecolor="none",
        zorder=1,
    )
    ax.scatter([0], [0], s=150, c="#0f172a", marker="s", zorder=5)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal", adjustable="box")
    if subtitle is not None:
        ax.text(
            0.02,
            0.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "edgecolor": "#e2e8f0"},
        )


def plot_total_cost_progress(cost_summary: dict[str, object], figures_dir: Path) -> None:
    phase_rows = cost_summary["global_phase_statuses"]
    labels = ["Baseline", *[str(row["pass_label"]).upper() for row in phase_rows], "Final"]
    values = [
        float(cost_summary["baseline_total_cost"]),
        *[float(row["objective_value"]) for row in phase_rows],
        float(cost_summary["total_cost"]),
    ]
    colors = ["#94a3b8", "#60a5fa", "#34d399", "#0f766e"][: len(labels)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{values[idx]:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    improvement = float(cost_summary["baseline_total_cost"]) - float(cost_summary["total_cost"])
    ax.set_title("Total Cost Improvement Through Cost-First Reoptimization")
    ax.set_ylabel("Total cost")
    ax.text(
        0.99,
        0.96,
        f"Net improvement: {improvement:,.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffffff", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, figures_dir / "01_total_cost_progress.png")


def plot_final_cost_composition(cost_summary: dict[str, object], figures_dir: Path) -> None:
    labels = ["Startup", "Energy", "Carbon", "Waiting", "Late"]
    values = [
        float(cost_summary["startup_cost"]),
        float(cost_summary["energy_cost"]),
        float(cost_summary["carbon_cost"]),
        float(cost_summary["waiting_cost"]),
        float(cost_summary["late_cost"]),
    ]
    colors = ["#1f2937", "#1d4ed8", "#0f766e", "#f59e0b", "#dc2626"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    wedges, _ = axes[0].pie(
        values,
        colors=colors,
        startangle=110,
        wedgeprops={"width": 0.42, "edgecolor": "#f8fafc"},
    )
    axes[0].legend(wedges, [f"{label}: {value:,.1f}" for label, value in zip(labels, values)], loc="center left", bbox_to_anchor=(1.0, 0.5))
    axes[0].set_title("Final Cost Composition")

    share_df = pd.DataFrame(
        {
            "component": labels,
            "share": [value / float(cost_summary["total_cost"]) * 100.0 for value in values],
        }
    )
    sns.barplot(
        data=share_df,
        x="share",
        y="component",
        hue="component",
        palette=colors,
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Share of Total Cost")
    axes[1].set_xlabel("Share (%)")
    axes[1].set_ylabel("")
    for patch, share in zip(axes[1].patches, share_df["share"]):
        axes[1].text(patch.get_width(), patch.get_y() + patch.get_height() / 2.0, f" {share:.2f}%", va="center")
    save_figure(fig, figures_dir / "02_final_cost_composition.png")


def plot_key_metrics(cost_summary: dict[str, object], figures_dir: Path) -> None:
    metrics = [
        ("Total Cost", float(cost_summary["baseline_total_cost"]), float(cost_summary["total_cost"]), "{:,.0f}"),
        ("Single-Stop Routes", float(cost_summary["baseline_single_stop_route_count"]), float(cost_summary["single_stop_route_count"]), "{:.0f}"),
        ("Mixed Big Routes", float(cost_summary["baseline_routes_with_flexible_units_on_big"]), float(cost_summary["final_routes_with_flexible_units_on_big"]), "{:.0f}"),
        ("Blocking Big Routes", float(cost_summary["baseline_blocking_big_flexible_count"]), float(cost_summary["final_blocking_big_flexible_count"]), "{:.0f}"),
        ("Piggyback Big Routes", float(cost_summary["baseline_piggyback_big_count"]), float(cost_summary["final_piggyback_big_count"]), "{:.0f}"),
        ("Mixed Big Flexible Units", float(cost_summary["baseline_flexible_units_on_big_routes"]), float(cost_summary["final_flexible_units_on_big_routes"]), "{:.0f}"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8))
    for ax, (title, baseline_value, final_value, fmt) in zip(axes.flat, metrics):
        values = [baseline_value, final_value]
        ax.bar(["Baseline", "Final"], values, color=["#94a3b8", "#0f766e"], width=0.58)
        for idx, value in enumerate(values):
            ax.text(idx, value, fmt.format(value), ha="center", va="bottom", fontsize=10)
        delta = final_value - baseline_value
        ax.set_title(title)
        ax.text(
            0.98,
            0.93,
            f"Delta: {delta:+,.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": "#e2e8f0"},
        )
    fig.suptitle("Baseline vs Final Key Metrics", y=1.01, fontsize=15, fontweight="bold")
    save_figure(fig, figures_dir / "03_baseline_vs_final_metrics.png")


def plot_vehicle_mix_and_stops(route_df: pd.DataFrame, figures_dir: Path) -> None:
    stop_pattern_df = route_df.copy()
    stop_pattern_df["stop_pattern"] = np.select(
        [stop_pattern_df["unit_count"].eq(1), stop_pattern_df["unit_count"].eq(2)],
        ["1 stop", "2 stops"],
        default="3+ stops",
    )
    order = (
        route_df.groupby("vehicle_type", as_index=False)
        .size()
        .sort_values(["size", "vehicle_type"], ascending=[False, True])["vehicle_type"]
        .tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    vehicle_counts = route_df.groupby("vehicle_type", as_index=False).size()
    sns.barplot(
        data=vehicle_counts,
        x="vehicle_type",
        y="size",
        hue="vehicle_type",
        order=order,
        palette=[VEHICLE_COLORS.get(name, "#64748b") for name in order],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Route Count by Vehicle Type")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Route count")
    axes[0].tick_params(axis="x", rotation=15)
    annotate_bars(axes[0])

    stop_mix = (
        stop_pattern_df.groupby(["vehicle_type", "stop_pattern"], as_index=False)
        .size()
        .pivot(index="vehicle_type", columns="stop_pattern", values="size")
        .fillna(0.0)
        .reindex(order)
    )
    stop_mix = stop_mix[["1 stop", "2 stops", "3+ stops"]]
    stop_mix.plot(
        kind="bar",
        stacked=True,
        color=["#cbd5e1", "#9db6d5", "#6b9a95"],
        ax=axes[1],
        width=0.68,
    )
    axes[1].set_title("Stop Pattern by Vehicle Type")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Route count")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(title="", loc="upper right")

    save_figure(fig, figures_dir / "04_vehicle_mix_and_stop_pattern.png")


def plot_route_utilization(route_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    for vehicle_type, sub_df in route_df.groupby("vehicle_type"):
        ax.scatter(
            sub_df["weight_utilization"] * 100.0,
            sub_df["volume_utilization"] * 100.0,
            s=np.clip(sub_df["route_cost"] / 3.2, 36, 260),
            alpha=0.75,
            color=VEHICLE_COLORS.get(vehicle_type, "#64748b"),
            edgecolor="#ffffff",
            linewidth=0.8,
            label=vehicle_type,
        )
    ax.axvline(100.0, color="#cbd5e1", linestyle="--", linewidth=1.1)
    ax.axhline(100.0, color="#cbd5e1", linestyle="--", linewidth=1.1)
    ax.set_title("Route Utilization by Vehicle Type")
    ax.set_xlabel("Weight utilization (%)")
    ax.set_ylabel("Volume utilization (%)")
    ax.legend(title="", loc="upper left", frameon=True)
    ax.set_xlim(0, max(105, route_df["weight_utilization"].max() * 100.0 + 3.0))
    ax.set_ylim(0, max(105, route_df["volume_utilization"].max() * 100.0 + 5.0))
    save_figure(fig, figures_dir / "05_route_utilization_scatter.png")


def plot_route_timeline(route_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = route_df.sort_values(["departure_min", "vehicle_rank", "route_cost"], ascending=[True, True, False]).reset_index(drop=True)
    fig_height = max(10.0, len(plot_df) * 0.14)
    fig, ax = plt.subplots(figsize=(14.5, fig_height))

    for y_pos, row in plot_df.iterrows():
        color = VEHICLE_COLORS.get(row["vehicle_type"], "#64748b")
        ax.barh(
            y=y_pos,
            width=float(row["return_span_min"]),
            left=float(row["departure_min"]),
            height=0.74,
            color=color,
            alpha=0.86,
            edgecolor="#ffffff",
            linewidth=0.6,
        )
        if int(row["late_positive_stop_count"]) > 0:
            ax.scatter(float(row["return_min"]), y_pos, s=22, color="#dc2626", zorder=3)
        if int(row["after_hours_return_flag"]) == 1:
            ax.scatter(float(row["return_min"]), y_pos, s=42, color="#f59e0b", marker="*", zorder=4)

    ax.set_title("Daily Route Timeline")
    ax.set_xlabel("Clock time")
    ax.set_ylabel("Routes ordered by departure")
    ax.set_yticks([])
    format_minutes_axis(ax)
    ax.set_xlim(0, max(960, float(plot_df["return_min"].max()) + 20.0))
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=name)
        for name, color in VEHICLE_COLORS.items()
        if name in set(plot_df["vehicle_type"])
    ]
    legend_handles.extend(
        [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", markersize=7, label="Late-positive route"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#f59e0b", markersize=10, label="After-hours return"),
        ]
    )
    ax.legend(handles=legend_handles, ncol=3, loc="upper left")
    save_figure(fig, figures_dir / "06_route_timeline.png")


def plot_big_route_structure(cost_summary: dict[str, object], route_pool_df: pd.DataFrame, figures_dir: Path) -> None:
    families = ["Rigid Big", "Piggyback Big", "Promotion-Like Big", "Blocking Big"]
    pass1_counts = (
        route_pool_df.loc[route_pool_df["selected_in_pass1"].eq(1) & route_pool_df["family"].isin(families)]
        .groupby("family")
        .size()
        .reindex(families, fill_value=0)
        .tolist()
    )
    final_counts = (
        route_pool_df.loc[route_pool_df["selected_in_final"].eq(1) & route_pool_df["family"].isin(families)]
        .groupby("family")
        .size()
        .reindex(families, fill_value=0)
        .tolist()
    )

    selected_final = route_pool_df.loc[route_pool_df["selected_in_final"].eq(1)].copy()
    saving_df = (
        selected_final.groupby("family", as_index=False)["current_cost_saving"]
        .sum()
        .query("family in @families")
        .set_index("family")
        .reindex(families, fill_value=0.0)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    x = np.arange(len(families))
    width = 0.35
    axes[0].bar(x - width / 2.0, pass1_counts, width=width, color="#94a3b8", label="Pass 1")
    axes[0].bar(x + width / 2.0, final_counts, width=width, color="#0f766e", label="Final")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(families, rotation=12)
    axes[0].set_ylabel("Route count")
    axes[0].set_title("Big-Route Structure: Pass 1 vs Final")
    axes[0].legend(loc="upper right")

    sns.barplot(
        data=saving_df,
        x="current_cost_saving",
        y="family",
        hue="family",
        palette=[FAMILY_COLORS.get(name, "#64748b") for name in saving_df["family"]],
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Selected Route-Column Local Savings")
    axes[1].set_xlabel("Current-cover saving")
    axes[1].set_ylabel("")
    for patch, value in zip(axes[1].patches, saving_df["current_cost_saving"]):
        axes[1].text(patch.get_width(), patch.get_y() + patch.get_height() / 2.0, f" {value:,.0f}", va="center")

    save_figure(fig, figures_dir / "07_big_route_structure.png")


def plot_route_cost_pareto(route_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = route_df.sort_values("route_cost", ascending=False).reset_index(drop=True)
    plot_df["cum_cost_share"] = plot_df["route_cost"].cumsum() / plot_df["route_cost"].sum() * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))

    axes[0].bar(
        np.arange(1, len(plot_df) + 1),
        plot_df["route_cost"],
        color="#60a5fa",
        alpha=0.88,
        width=0.9,
    )
    twin = axes[0].twinx()
    twin.plot(np.arange(1, len(plot_df) + 1), plot_df["cum_cost_share"], color="#0f172a", linewidth=2.0)
    twin.set_ylabel("Cumulative share (%)")
    twin.set_ylim(0, 102)
    axes[0].set_title("Route Cost Pareto Curve")
    axes[0].set_xlabel("Route rank (descending cost)")
    axes[0].set_ylabel("Route cost")

    sns.boxplot(
        data=route_df,
        x="vehicle_type",
        y="route_cost",
        hue="vehicle_type",
        palette=[VEHICLE_COLORS.get(name, "#64748b") for name in route_df["vehicle_type"].unique()],
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Route Cost Distribution by Vehicle Type")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylabel("Route cost")

    save_figure(fig, figures_dir / "08_route_cost_pareto.png")


def plot_spatial_map(active_customers: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.8))
    ax.add_patch(plt.Circle((0, 0), GREEN_ZONE_RADIUS_KM, color=GREEN_ZONE_FILL, alpha=0.24, ec=GREEN_ZONE_EDGE, lw=2))
    ax.scatter([0], [0], s=140, c="#0f172a", marker="s", label="Depot / origin")

    for vehicle_type, sub_df in active_customers.groupby("vehicle_type"):
        color = VEHICLE_COLORS.get(vehicle_type, "#64748b")
        split_mask = sub_df["route_visit_count"] > 1
        ax.scatter(
            sub_df.loc[~split_mask, "x_km"],
            sub_df.loc[~split_mask, "y_km"],
            s=np.clip(sub_df.loc[~split_mask, "total_weight"] / 9.0 + 18.0, 24.0, 220.0),
            c=color,
            alpha=0.78,
            edgecolor="#ffffff",
            linewidth=0.8,
            label=vehicle_type,
        )
        if split_mask.any():
            ax.scatter(
                sub_df.loc[split_mask, "x_km"],
                sub_df.loc[split_mask, "y_km"],
                s=np.clip(sub_df.loc[split_mask, "total_weight"] / 9.0 + 22.0, 34.0, 240.0),
                c=color,
                alpha=0.92,
                edgecolor="#1f2937",
                linewidth=1.35,
            )

    ax.set_title("Customer Spatial Pattern by Dominant Assigned Vehicle")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(title="", loc="upper right", ncol=2)
    ax.text(
        0.02,
        0.02,
        "Dark outline marks customers visited by more than one route.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "edgecolor": "#e2e8f0"},
    )
    save_figure(fig, figures_dir / "09_customer_spatial_map.png")


def plot_candidate_pool_mix(route_pool_df: pd.DataFrame, figures_dir: Path) -> None:
    family_order = ["Support", "Flex Small", "Piggyback Big", "Promotion-Like Big", "Blocking Big", "Rigid Big"]
    pool_view = route_pool_df.copy()
    pool_view["family"] = pool_view["family"].replace({"Rigid Big": "Rigid Big"})
    family_counts = (
        pool_view.groupby("family", as_index=False)
        .size()
        .set_index("family")
        .reindex(family_order, fill_value=0.0)
        .reset_index()
    )
    selected_counts = (
        pool_view.loc[pool_view["selected_in_final"].eq(1)]
        .groupby("family", as_index=False)
        .size()
        .set_index("family")
        .reindex(family_order, fill_value=0.0)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    sns.barplot(
        data=family_counts,
        x="size",
        y="family",
        hue="family",
        palette=[FAMILY_COLORS.get(name, "#64748b") for name in family_counts["family"]],
        dodge=False,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Candidate Pool by Family")
    axes[0].set_xlabel("Candidate count")
    axes[0].set_ylabel("")

    sns.barplot(
        data=selected_counts,
        x="size",
        y="family",
        hue="family",
        palette=[FAMILY_COLORS.get(name, "#64748b") for name in selected_counts["family"]],
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Selected Final Routes by Family")
    axes[1].set_xlabel("Selected count")
    axes[1].set_ylabel("")

    save_figure(fig, figures_dir / "10_candidate_pool_family_mix.png")


def plot_route_network_by_vehicle(route_paths: list[dict[str, object]], active_customers: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 9.2))
    draw_base_map(
        ax,
        active_customers,
        subtitle="All routes are drawn from depot to customer sequence and back to depot.",
    )
    for path in route_paths:
        color = VEHICLE_COLORS.get(path["vehicle_type"], "#64748b")
        draw_soft_route(
            ax,
            path["x"],
            path["y"],
            color=color,
            linewidth=1.35,
            alpha=0.26,
            zorder=2.4,
            halo_width=0.0,
        )
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=vehicle_type)
        for vehicle_type, color in VEHICLE_COLORS.items()
        if any(path["vehicle_type"] == vehicle_type for path in route_paths)
    ]
    ax.legend(handles=legend_handles, title="", loc="upper right", ncol=2)
    ax.set_title("Distribution Route Network by Vehicle Type")
    save_figure(fig, figures_dir / "11_route_network_by_vehicle.png")


def plot_big_route_network(route_paths: list[dict[str, object]], active_customers: pd.DataFrame, figures_dir: Path) -> None:
    big_families = {"Rigid Big", "Piggyback Big", "Promotion-Like Big", "Blocking Big"}
    big_paths = [path for path in route_paths if path["family"] in big_families]

    fig, ax = plt.subplots(figsize=(10.4, 9.2))
    draw_base_map(
        ax,
        active_customers,
        subtitle="Big-route families are separated into rigid, piggyback, promotion-like, and blocking.",
    )
    for path in big_paths:
        draw_soft_route(
            ax,
            path["x"],
            path["y"],
            color="#d8e0ea",
            linewidth=1.3,
            alpha=0.22,
            zorder=2.2,
            halo_width=0.0,
        )
    for family in ["Rigid Big", "Piggyback Big", "Promotion-Like Big", "Blocking Big"]:
        family_paths = [path for path in big_paths if path["family"] == family]
        for path in family_paths:
            draw_soft_route(
                ax,
                path["x"],
                path["y"],
                color=FAMILY_COLORS.get(family, "#64748b"),
                alpha=0.42 if family != "Blocking Big" else 0.8,
                linewidth=1.55 if family != "Blocking Big" else 2.3,
                zorder=3.3,
            )
    legend_handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[family], lw=3, label=family)
        for family in ["Rigid Big", "Piggyback Big", "Promotion-Like Big", "Blocking Big"]
        if any(path["family"] == family for path in big_paths)
    ]
    ax.legend(handles=legend_handles, title="", loc="upper right")
    ax.set_title("Big-Route Network by Structure Family")
    save_figure(fig, figures_dir / "12_big_route_network_by_family.png")


def plot_top_cost_route_map(route_paths: list[dict[str, object]], active_customers: pd.DataFrame, figures_dir: Path) -> None:
    top_paths = sorted(route_paths, key=lambda item: (-item["route_cost"], item["route_id"]))[:18]

    fig, ax = plt.subplots(figsize=(10.8, 9.4))
    draw_base_map(
        ax,
        active_customers,
        subtitle="Top 18 costly routes are highlighted and labeled by route id.",
    )
    for path in route_paths:
        draw_soft_route(
            ax,
            path["x"],
            path["y"],
            color="#d7e1ec",
            linewidth=0.95,
            alpha=0.12,
            zorder=1.4,
            halo_width=0.0,
        )
    for path in top_paths:
        color = VEHICLE_COLORS.get(path["vehicle_type"], "#64748b")
        draw_soft_route(
            ax,
            path["x"],
            path["y"],
            color=color,
            linewidth=2.35,
            alpha=0.9,
            zorder=3.8,
        )
        if len(path["x"]) >= 3:
            label_x = float(path["x"][1])
            label_y = float(path["y"][1])
        else:
            label_x = float(np.mean(path["x"]))
            label_y = float(np.mean(path["y"]))
        ax.text(
            label_x,
            label_y,
            str(path["route_id"]),
            fontsize=8.5,
            color="#0f172a",
            zorder=4,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "#ffffff", "edgecolor": "#e2e8f0", "alpha": 0.92},
        )
    ax.set_title("Top-Cost Route Map")
    save_figure(fig, figures_dir / "13_top_cost_route_map.png")


def write_visual_summary(figures_dir: Path) -> None:
    figure_files = sorted(path.name for path in figures_dir.glob("*.png"))
    lines = [
        "# Question 1 Visual Summary",
        "",
        "Generated figures:",
        *[f"- `{file_name}`" for file_name in figure_files],
        "",
        "Reading order recommendation:",
        "- 01-03: overall optimization result and key metric change",
        "- 04-08: route structure, utilization, and schedule profile",
        "- 09-10: spatial and candidate-pool diagnostics",
        "- 11-13: publication-ready route-network figures",
    ]
    (figures_dir / "q1_visual_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    figures_dir = args.output_dir or (args.artifacts_root / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    context = build_plot_context(args.artifacts_root, args.preprocess_root)

    plot_total_cost_progress(context["cost_summary"], figures_dir)
    plot_final_cost_composition(context["cost_summary"], figures_dir)
    plot_key_metrics(context["cost_summary"], figures_dir)
    plot_vehicle_mix_and_stops(context["route_df"], figures_dir)
    plot_route_utilization(context["route_df"], figures_dir)
    plot_route_timeline(context["route_df"], figures_dir)
    plot_big_route_structure(context["cost_summary"], context["route_pool_df"], figures_dir)
    plot_route_cost_pareto(context["route_df"], figures_dir)
    plot_spatial_map(context["active_customers"], figures_dir)
    plot_candidate_pool_mix(context["route_pool_df"], figures_dir)
    route_paths = build_route_paths(context["route_df"], context["stop_df"])
    plot_route_network_by_vehicle(route_paths, context["active_customers"], figures_dir)
    plot_big_route_network(route_paths, context["active_customers"], figures_dir)
    plot_top_cost_route_map(route_paths, context["active_customers"], figures_dir)
    write_visual_summary(figures_dir)


if __name__ == "__main__":
    main()
