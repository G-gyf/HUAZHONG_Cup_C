from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D


SCENARIO_COLORS = {
    "Q1 s11": "#7b8ea8",
    "Q2 s11": "#1f4e5f",
}

SEED_COLORS = {
    "s11": "#1f4e5f",
    "s17": "#5b8fb9",
    "s23": "#7a6aa9",
}

VEHICLE_COLORS = {
    "fuel_3000": "#4c566a",
    "fuel_1500": "#5b8fb9",
    "ev_3000": "#2a9d8f",
    "ev_1250": "#7cae5a",
    "fuel_1250": "#c98c64",
}

Q1_BACKGROUND_COLOR = "#8ea2bb"
Q2_SHARED_COLOR = "#475569"
Q1_REMOVED_COLOR = "#9aa9bc"
GREEN_ZONE_FILL = "#d9f0e5"
GREEN_ZONE_EDGE = "#66a182"
GREEN_ZONE_RADIUS_KM = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures comparing Q1 and Q2 results.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--q1-root", type=Path, default=Path.cwd() / "question1_artifacts_hybrid_coupled_heavy_s11")
    parser.add_argument("--q2-root", type=Path, default=Path.cwd() / "question2_artifacts_hybrid_standard_s11")
    parser.add_argument(
        "--q2-seed-roots",
        type=Path,
        nargs="*",
        default=[
            Path.cwd() / "question2_artifacts_hybrid_standard_s11",
            Path.cwd() / "question2_artifacts_hybrid_standard_s17",
            Path.cwd() / "question2_artifacts_hybrid_standard_s23",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "paper_compare_figures")
    return parser.parse_args()


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
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
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_q1_q2_context(q1_root: Path, q2_root: Path) -> dict[str, object]:
    q1_summary = load_json(q1_root / "q1_hybrid_cost_summary.json")
    q2_summary = load_json(q2_root / "q2_hybrid_cost_summary.json")

    q1_route_df = pd.read_csv(q1_root / "q1_hybrid_route_summary.csv")
    q2_route_df = pd.read_csv(q2_root / "q2_hybrid_route_summary.csv")
    q1_customer_df = pd.read_csv(q1_root / "q1_hybrid_customer_aggregate.csv")
    q2_customer_df = pd.read_csv(q2_root / "q2_hybrid_customer_aggregate.csv")
    q1_stop_df = pd.read_csv(q1_root / "q1_hybrid_stop_schedule.csv")
    q2_stop_df = pd.read_csv(q2_root / "q2_hybrid_stop_schedule.csv")

    customer_coords = pd.read_csv(q1_root.parents[0] / "preprocess_artifacts" / "tables" / "customer_master_98.csv")[
        ["cust_id", "x_km", "y_km", "in_green_zone", "has_orders"]
    ].rename(columns={"cust_id": "orig_cust_id"})
    stop_coords = customer_coords[["orig_cust_id", "x_km", "y_km"]]
    q1_stop_df = q1_stop_df.merge(stop_coords, on="orig_cust_id", how="left")
    q2_stop_df = q2_stop_df.merge(stop_coords, on="orig_cust_id", how="left")
    if "in_green_zone" not in q1_stop_df.columns:
        q1_stop_df = q1_stop_df.merge(customer_coords[["orig_cust_id", "in_green_zone"]], on="orig_cust_id", how="left")
    if "in_green_zone" not in q2_stop_df.columns:
        q2_stop_df = q2_stop_df.merge(customer_coords[["orig_cust_id", "in_green_zone"]], on="orig_cust_id", how="left")

    q1_route_df["scenario"] = "Q1 s11"
    q2_route_df["scenario"] = "Q2 s11"
    q1_route_df["stop_count"] = q1_route_df["unit_sequence"].astype(str).apply(lambda text: len(str(text).split(",")))
    q2_route_df["stop_count"] = q2_route_df["unit_sequence"].astype(str).apply(lambda text: len(str(text).split(",")))

    q1_customer_df["scenario"] = "Q1 s11"
    q2_customer_df["scenario"] = "Q2 s11"
    if "served_vehicle_types" not in q1_customer_df.columns:
        q1_customer_df["served_vehicle_types"] = ""
    if "served_by_ev" not in q1_customer_df.columns:
        q1_customer_df["served_by_ev"] = q1_customer_df["served_vehicle_types"].astype(str).str.contains("ev").astype(int)
    if "served_by_fuel" not in q1_customer_df.columns:
        q1_customer_df["served_by_fuel"] = q1_customer_df["served_vehicle_types"].astype(str).str.contains("fuel").astype(int)

    return {
        "q1_summary": q1_summary,
        "q2_summary": q2_summary,
        "route_df": pd.concat([q1_route_df, q2_route_df], ignore_index=True),
        "customer_df": pd.concat([q1_customer_df, q2_customer_df], ignore_index=True, sort=False),
        "q1_stop_df": q1_stop_df,
        "q2_stop_df": q2_stop_df,
        "customer_coords": customer_coords,
    }


def build_q2_seed_df(seed_roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in seed_roots:
        summary = load_json(root / "q2_hybrid_cost_summary.json")
        rows.append(
            {
                "seed": f"s{int(summary['best_seed'])}",
                "total_cost": float(summary["total_cost"]),
                "route_count": int(summary["route_count"]),
                "single_stop_route_count": int(summary["single_stop_route_count"]),
                "late_positive_stops": int(summary["late_positive_stops"]),
                "max_late_min": float(summary["max_late_min"]),
                "ordinary_customers_served_by_ev": int(summary["ordinary_customers_served_by_ev"]),
                "elapsed_sec": float(summary["elapsed_sec"]),
            }
        )
    return pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)


def plot_q1_q2_overview(q1_summary: dict[str, object], q2_summary: dict[str, object], output_dir: Path) -> None:
    compare_df = pd.DataFrame(
        {
            "metric": [
                "Total cost",
                "Route count",
                "Single-stop routes",
                "Late stops",
                "Fuel (L)",
                "Carbon (kg)",
            ],
            "Q1 s11": [
                float(q1_summary["total_cost"]),
                float(q1_summary["route_count"]),
                float(q1_summary["single_stop_route_count"]),
                float(q1_summary["late_positive_stops"]),
                float(q1_summary["total_fuel_l"]),
                float(q1_summary["total_carbon_kg"]),
            ],
            "Q2 s11": [
                float(q2_summary["total_cost"]),
                float(q2_summary["route_count"]),
                float(q2_summary["single_stop_route_count"]),
                float(q2_summary["late_positive_stops"]),
                float(q2_summary["total_fuel_l"]),
                float(q2_summary["total_carbon_kg"]),
            ],
        }
    )
    long_df = compare_df.melt(id_vars="metric", var_name="scenario", value_name="value")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    left_df = long_df[long_df["metric"].isin(["Total cost", "Route count", "Single-stop routes", "Late stops"])]
    right_df = long_df[long_df["metric"].isin(["Fuel (L)", "Carbon (kg)"])]

    sns.barplot(
        data=left_df,
        x="metric",
        y="value",
        hue="scenario",
        palette=SCENARIO_COLORS,
        ax=axes[0],
    )
    axes[0].set_title("Q1 vs Q2: Economic and Routing Indicators")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Value")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    sns.barplot(
        data=right_df,
        x="metric",
        y="value",
        hue="scenario",
        palette=SCENARIO_COLORS,
        ax=axes[1],
    )
    axes[1].set_title("Q1 vs Q2: Fuel and Carbon Footprint")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Value")
    axes[1].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    save_figure(fig, output_dir / "q1_q2_overview.png")


def plot_q1_q2_vehicle_mix(route_df: pd.DataFrame, output_dir: Path) -> None:
    vehicle_mix = (
        route_df.groupby(["scenario", "vehicle_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    sns.barplot(
        data=vehicle_mix,
        x="vehicle_type",
        y="count",
        hue="scenario",
        palette=SCENARIO_COLORS,
        ax=axes[0],
    )
    axes[0].set_title("Vehicle-Type Mix Before and After Policy Constraints")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Route count")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    structure_df = route_df.copy()
    structure_df["route_class"] = np.select(
        [
            structure_df["stop_count"] == 1,
            structure_df["stop_count"] == 2,
        ],
        [
            "Single-stop",
            "Two-stop",
        ],
        default="Three-plus",
    )
    structure_count = (
        structure_df.groupby(["scenario", "route_class"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    sns.barplot(
        data=structure_count,
        x="route_class",
        y="count",
        hue="scenario",
        palette=SCENARIO_COLORS,
        ax=axes[1],
    )
    axes[1].set_title("Route Structure Shift Under Policy Constraints")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    save_figure(fig, output_dir / "q1_q2_vehicle_structure.png")


def plot_q1_q2_customer_change(customer_df: pd.DataFrame, q2_summary: dict[str, object], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    split_df = (
        customer_df.groupby(["scenario", "unit_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    sns.barplot(
        data=split_df,
        x="unit_type",
        y="count",
        hue="scenario",
        palette=SCENARIO_COLORS,
        ax=axes[0],
    )
    axes[0].set_title("Customer Splitting Pattern: Q1 vs Q2")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Customer count")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    policy_df = pd.DataFrame(
        {
            "metric": [
                "Green-zone customers used",
                "Mandatory-EV active",
                "Fuel-pre16 violations",
                "Policy violations",
            ],
            "value": [
                int(q2_summary["green_zone_customer_count_used"]),
                int(q2_summary["mandatory_ev_active_customer_count"]),
                int(q2_summary["fuel_route_green_zone_pre16_visit_count"]),
                int(q2_summary["policy_violation_count"]),
            ],
        }
    )
    sns.barplot(data=policy_df, x="metric", y="value", palette=["#16a34a", "#0f766e", "#dc2626", "#b91c1c"], ax=axes[1], hue="metric", legend=False)
    axes[1].set_title("Q2 Policy Constraint Outcomes")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=10)

    save_figure(fig, output_dir / "q1_q2_customer_policy_change.png")


def plot_q2_seed_stability(seed_df: pd.DataFrame, baseline_total_cost: float, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    sns.barplot(
        data=seed_df,
        x="seed",
        y="total_cost",
        palette=SEED_COLORS,
        ax=axes[0],
        hue="seed",
        legend=False,
    )
    axes[0].axhline(baseline_total_cost, color="#b91c1c", linestyle="--", linewidth=1.8, label="Q2 v2 baseline")
    axes[0].set_title("Q2 Hybrid Standard: Seed Stability in Total Cost")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Total cost")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    for patch in axes[0].patches:
        h = patch.get_height()
        axes[0].text(patch.get_x() + patch.get_width() / 2.0, h, f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    metric_df = seed_df.melt(
        id_vars="seed",
        value_vars=["route_count", "single_stop_route_count", "late_positive_stops", "ordinary_customers_served_by_ev"],
        var_name="metric",
        value_name="value",
    )
    label_map = {
        "route_count": "Routes",
        "single_stop_route_count": "Single-stop",
        "late_positive_stops": "Late stops",
        "ordinary_customers_served_by_ev": "Ordinary by EV",
    }
    metric_df["metric"] = metric_df["metric"].map(label_map)
    sns.barplot(
        data=metric_df,
        x="metric",
        y="value",
        hue="seed",
        palette=SEED_COLORS,
        ax=axes[1],
    )
    axes[1].set_title("Q2 Seed Stability in Structural and Policy Indicators")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    save_figure(fig, output_dir / "q2_seed_stability.png")


def plot_q1_q2_route_cost_profile(route_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    sns.scatterplot(
        data=route_df,
        x="return_min",
        y="route_cost",
        hue="scenario",
        style="vehicle_type",
        palette=SCENARIO_COLORS,
        alpha=0.75,
        s=55,
        ax=axes[0],
    )
    axes[0].set_title("Route Cost vs Return Time: Q1 and Q2")
    axes[0].set_xlabel("Return time (min)")
    axes[0].set_ylabel("Route cost")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8)

    late_df = (
        route_df.groupby("scenario", as_index=False)
        .agg(
            after_hours_return_count=("after_hours_return_flag", "sum"),
            late_route_count=("late_positive_stop_count", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
        .melt(id_vars="scenario", var_name="metric", value_name="count")
    )
    sns.barplot(
        data=late_df,
        x="metric",
        y="count",
        hue="scenario",
        palette=SCENARIO_COLORS,
        ax=axes[1],
    )
    axes[1].set_title("Schedule Pressure: Q1 vs Q2")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    save_figure(fig, output_dir / "q1_q2_schedule_profile.png")


def lighten_color(hex_color: str, factor: float = 0.55) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(1.0 - factor * (1.0 - channel) for channel in rgb)


def draw_q1_underlay(
    ax: plt.Axes,
    x: list[float],
    y: list[float],
    linewidth: float,
    alpha: float = 0.78,
    zorder: float = 2.0,
    line_color: str = Q1_BACKGROUND_COLOR,
) -> None:
    ax.plot(
        x,
        y,
        color="#eef4fa",
        linewidth=linewidth + 1.9,
        alpha=0.95,
        solid_capstyle="round",
        zorder=max(1.0, zorder - 0.2),
    )
    ax.plot(
        x,
        y,
        color=line_color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=(0, (3.2, 2.4)),
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=zorder,
    )


def draw_q2_highlight(
    ax: plt.Axes,
    x: list[float],
    y: list[float],
    color: str,
    linewidth: float,
    alpha: float = 0.9,
    zorder: float = 5.0,
    halo_width: float = 1.8,
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


def draw_base_customers(ax: plt.Axes, customer_coords: pd.DataFrame) -> pd.DataFrame:
    active_customers = customer_coords.loc[customer_coords["has_orders"].astype(bool)].copy()
    green_zone_customers = active_customers.loc[active_customers["in_green_zone"].astype(bool)]
    ax.add_patch(plt.Circle((0.0, 0.0), GREEN_ZONE_RADIUS_KM, color=GREEN_ZONE_FILL, alpha=0.22, ec=GREEN_ZONE_EDGE, lw=1.6))
    ax.scatter(
        active_customers["x_km"],
        active_customers["y_km"],
        s=14,
        c="#cbd5e1",
        alpha=0.32,
        edgecolor="none",
        zorder=1,
    )
    ax.scatter(
        green_zone_customers["x_km"],
        green_zone_customers["y_km"],
        s=24,
        c=GREEN_ZONE_EDGE,
        alpha=0.22,
        edgecolor="none",
        zorder=1,
    )
    ax.scatter([0], [0], s=180, c="#0f172a", marker="s", zorder=8)
    return active_customers


def route_paths_from_stop_df(stop_df: pd.DataFrame) -> list[dict[str, object]]:
    paths: list[dict[str, object]] = []
    for route_id, sub_df in stop_df.sort_values(["route_id", "stop_index"]).groupby("route_id"):
        ordered = sub_df.sort_values("stop_index")
        points_x = [0.0, *ordered["x_km"].astype(float).tolist(), 0.0]
        points_y = [0.0, *ordered["y_km"].astype(float).tolist(), 0.0]
        node_seq = [0, *ordered["orig_cust_id"].astype(int).tolist(), 0]
        paths.append(
            {
                "route_id": int(route_id),
                "vehicle_type": str(ordered["vehicle_type"].iloc[0]),
                "x": points_x,
                "y": points_y,
                "nodes": node_seq,
                "touches_green_zone": bool(ordered["in_green_zone"].fillna(False).astype(bool).any()) if "in_green_zone" in ordered.columns else False,
            }
        )
    return paths


def edge_records_from_paths(paths: list[dict[str, object]], scenario: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        for idx in range(len(path["nodes"]) - 1):
            left_node = int(path["nodes"][idx])
            right_node = int(path["nodes"][idx + 1])
            rows.append(
                {
                    "scenario": scenario,
                    "vehicle_type": path["vehicle_type"],
                    "route_id": path["route_id"],
                    "left_node": left_node,
                    "right_node": right_node,
                    "edge_key": tuple(sorted((left_node, right_node))),
                    "x0": float(path["x"][idx]),
                    "y0": float(path["y"][idx]),
                    "x1": float(path["x"][idx + 1]),
                    "y1": float(path["y"][idx + 1]),
                }
            )
    return rows


def edge_df_from_stop_df(stop_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    paths = route_paths_from_stop_df(stop_df)
    edge_df = pd.DataFrame(edge_records_from_paths(paths, scenario))
    if edge_df.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "vehicle_type",
                "route_id",
                "left_node",
                "right_node",
                "edge_key",
                "x0",
                "y0",
                "x1",
                "y1",
            ]
        )
    return edge_df


def aggregate_edges(edge_df: pd.DataFrame, keep_vehicle: bool = False) -> pd.DataFrame:
    if edge_df.empty:
        columns = ["x0", "y0", "x1", "y1", "edge_count"]
        if keep_vehicle:
            columns.append("vehicle_type")
        return pd.DataFrame(columns=columns)

    work = edge_df.copy()
    x0 = work["x0"].round(4).to_numpy()
    y0 = work["y0"].round(4).to_numpy()
    x1 = work["x1"].round(4).to_numpy()
    y1 = work["y1"].round(4).to_numpy()
    swap_mask = (x0 > x1) | ((x0 == x1) & (y0 > y1))

    work["gx0"] = np.where(swap_mask, x1, x0)
    work["gy0"] = np.where(swap_mask, y1, y0)
    work["gx1"] = np.where(swap_mask, x0, x1)
    work["gy1"] = np.where(swap_mask, y0, y1)

    group_cols = ["gx0", "gy0", "gx1", "gy1"]
    if keep_vehicle:
        group_cols.append("vehicle_type")

    aggregated = (
        work.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "edge_count", "gx0": "x0", "gy0": "y0", "gx1": "x1", "gy1": "y1"})
        .sort_values(["edge_count", "vehicle_type"] if keep_vehicle else ["edge_count"], ascending=[False, True] if keep_vehicle else [False])
        .reset_index(drop=True)
    )
    return aggregated


def filter_edges_to_box(edge_df: pd.DataFrame, radius: float) -> pd.DataFrame:
    if edge_df.empty:
        return edge_df.copy()
    mask = edge_df[["x0", "y0", "x1", "y1"]].abs().le(radius).all(axis=1)
    return edge_df.loc[mask].copy()


def build_edge_difference(
    q1_stop_df: pd.DataFrame, q2_stop_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q1_edges = edge_df_from_stop_df(q1_stop_df, "Q1")
    q2_edges = edge_df_from_stop_df(q2_stop_df, "Q2")

    q1_edge_keys = set(q1_edges["edge_key"])
    q2_edge_keys = set(q2_edges["edge_key"])
    shared_keys = q1_edge_keys & q2_edge_keys
    q1_only_keys = q1_edge_keys - q2_edge_keys
    q2_only_keys = q2_edge_keys - q1_edge_keys

    q1_only_edges = q1_edges.loc[q1_edges["edge_key"].isin(q1_only_keys)].copy()
    q2_only_edges = q2_edges.loc[q2_edges["edge_key"].isin(q2_only_keys)].copy()
    q2_shared_edges = q2_edges.loc[q2_edges["edge_key"].isin(shared_keys)].copy()
    return q1_edges, q2_edges, q1_only_edges, q2_only_edges, q2_shared_edges


def plot_q1_q2_route_overlay(q1_stop_df: pd.DataFrame, q2_stop_df: pd.DataFrame, customer_coords: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 9.2))
    draw_base_customers(ax, customer_coords)

    q1_edges, _, _, q2_only_edges, q2_shared_edges = build_edge_difference(q1_stop_df, q2_stop_df)
    q1_network = aggregate_edges(q1_edges, keep_vehicle=False)
    q2_shared_network = aggregate_edges(q2_shared_edges, keep_vehicle=False)
    q2_changed_network = aggregate_edges(q2_only_edges, keep_vehicle=True)

    for row in q1_network.itertuples(index=False):
        draw_q1_underlay(
            ax,
            [float(row.x0), float(row.x1)],
            [float(row.y0), float(row.y1)],
            linewidth=1.05 + 0.18 * np.log1p(float(row.edge_count)),
            alpha=0.82,
            zorder=2.0,
        )

    for row in q2_shared_network.itertuples(index=False):
        draw_q2_highlight(
            ax,
            [float(row.x0), float(row.x1)],
            [float(row.y0), float(row.y1)],
            color=Q2_SHARED_COLOR,
            linewidth=0.95 + 0.18 * np.log1p(float(row.edge_count)),
            alpha=0.28,
            zorder=3.6,
            halo_width=0.0,
        )

    for row in q2_changed_network.itertuples(index=False):
        draw_q2_highlight(
            ax,
            [float(row.x0), float(row.x1)],
            [float(row.y0), float(row.y1)],
            color=VEHICLE_COLORS.get(str(row.vehicle_type), "#334155"),
            linewidth=1.65 + 0.28 * np.log1p(float(row.edge_count)),
            alpha=0.94,
            zorder=5.5,
        )

    present_vehicle_types = list(dict.fromkeys(q2_changed_network.get("vehicle_type", pd.Series(dtype=str)).astype(str).tolist()))
    legend_handles = [
        Line2D([0], [0], color=Q1_BACKGROUND_COLOR, lw=2.2, linestyle=(0, (3.2, 2.4)), label="Q1 baseline network"),
        Line2D([0], [0], color=Q2_SHARED_COLOR, lw=2.4, alpha=0.45, label="Q2 segments retained from Q1"),
        Line2D([0], [0], color="#111827", lw=2.8, label="Q2 changed segments"),
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color=VEHICLE_COLORS[vehicle_type], lw=2.6, label=vehicle_type)
            for vehicle_type in present_vehicle_types
        ]
    )
    ax.legend(handles=legend_handles, loc="upper left", ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.5)
    ax.set_title("Q1 Background Network and Q2 Policy-Driven Route Adjustments")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02,
        0.02,
        "Q1 is kept as a cool dashed baseline.\nOnly the Q2 segments that differ from Q1 use strong vehicle colors; shared Q2 arcs stay muted.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffffff", "edgecolor": "#e2e8f0"},
    )
    save_figure(fig, output_dir / "q1_q2_route_overlay.png")


def plot_q1_q2_route_difference_focus(q1_stop_df: pd.DataFrame, q2_stop_df: pd.DataFrame, customer_coords: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 9.2))
    draw_base_customers(ax, customer_coords)
    _, _, q1_only_edges, q2_only_edges, _ = build_edge_difference(q1_stop_df, q2_stop_df)
    q1_removed = aggregate_edges(q1_only_edges, keep_vehicle=False)
    q2_added = aggregate_edges(q2_only_edges, keep_vehicle=True)

    for row in q1_removed.itertuples(index=False):
        draw_q1_underlay(
            ax,
            [float(row.x0), float(row.x1)],
            [float(row.y0), float(row.y1)],
            linewidth=1.05 + 0.18 * np.log1p(float(row.edge_count)),
            alpha=0.9,
            zorder=2.2,
            line_color=Q1_REMOVED_COLOR,
        )
    for row in q2_added.itertuples(index=False):
        draw_q2_highlight(
            ax,
            [float(row.x0), float(row.x1)],
            [float(row.y0), float(row.y1)],
            color=VEHICLE_COLORS.get(str(row.vehicle_type), "#334155"),
            linewidth=1.8 + 0.28 * np.log1p(float(row.edge_count)),
            alpha=0.95,
            zorder=5.6,
        )

    legend_handles = [
        Line2D([0], [0], color=Q1_REMOVED_COLOR, lw=2.2, linestyle=(0, (3.2, 2.4)), label="Segments used in Q1 only"),
        Line2D([0], [0], color="#111827", lw=2.8, label="Segments added in Q2"),
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color=VEHICLE_COLORS[vehicle_type], lw=2.8, label=vehicle_type)
            for vehicle_type in dict.fromkeys(q2_added.get("vehicle_type", pd.Series(dtype=str)).astype(str).tolist())
        ]
    )
    ax.legend(handles=legend_handles, loc="upper left", ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.4)
    ax.set_title("Route Differences Between Q1 and Q2")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02,
        0.02,
        f"Q1-only segments: {len(q1_removed)}\nQ2-only segments: {len(q2_added)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffffff", "edgecolor": "#e2e8f0"},
    )
    save_figure(fig, output_dir / "q1_q2_route_difference_focus.png")


def plot_q1_q2_green_zone_zoom(q1_stop_df: pd.DataFrame, q2_stop_df: pd.DataFrame, customer_coords: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    draw_base_customers(ax, customer_coords)
    zoom_radius = 12.8
    q1_green_route_ids = set(q1_stop_df.loc[q1_stop_df["in_green_zone"].fillna(False).astype(bool), "route_id"].astype(int))
    q2_green_route_ids = set(q2_stop_df.loc[q2_stop_df["in_green_zone"].fillna(False).astype(bool), "route_id"].astype(int))
    q1_green_edges = filter_edges_to_box(edge_df_from_stop_df(q1_stop_df.loc[q1_stop_df["route_id"].isin(q1_green_route_ids)], "Q1"), zoom_radius)
    q2_green_edges = filter_edges_to_box(edge_df_from_stop_df(q2_stop_df.loc[q2_stop_df["route_id"].isin(q2_green_route_ids)], "Q2"), zoom_radius)

    q1_green_network = aggregate_edges(q1_green_edges, keep_vehicle=False)
    q2_green_network = aggregate_edges(q2_green_edges, keep_vehicle=True)

    for row in q1_green_network.itertuples(index=False):
        draw_q1_underlay(
            ax,
            [float(row.x0), float(row.x1)],
            [float(row.y0), float(row.y1)],
            linewidth=1.12 + 0.18 * np.log1p(float(row.edge_count)),
            alpha=0.88,
            zorder=2.4,
        )
    for row in q2_green_network.itertuples(index=False):
        draw_q2_highlight(
            ax,
            [float(row.x0), float(row.x1)],
            [float(row.y0), float(row.y1)],
            color=VEHICLE_COLORS.get(str(row.vehicle_type), "#334155"),
            linewidth=1.75 + 0.28 * np.log1p(float(row.edge_count)),
            alpha=0.94,
            zorder=5.6,
        )

    ax.set_xlim(-zoom_radius, zoom_radius)
    ax.set_ylim(-zoom_radius, zoom_radius)
    ax.set_title("Green-Zone Zoom: Local Route Changes Around the Policy-Controlled Area")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02,
        0.02,
        "Only routes touching green-zone customers are shown here,\nwith segments clipped to the local policy area.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "edgecolor": "#e2e8f0"},
    )
    save_figure(fig, output_dir / "q1_q2_green_zone_zoom.png")


def write_notes(q1_summary: dict[str, object], q2_summary: dict[str, object], seed_df: pd.DataFrame, output_dir: Path) -> None:
    best_seed_row = seed_df.sort_values("total_cost").iloc[0]
    lines = [
        "# Q1 Q2 Comparison Figure Notes",
        "",
        f"- Q1 reference run: seed 11, total cost {float(q1_summary['total_cost']):.6f}.",
        f"- Q2 reference run: seed 11, total cost {float(q2_summary['total_cost']):.6f}.",
        f"- Q2 policy violations: {int(q2_summary['policy_violation_count'])}.",
        f"- Q2 best seed among standard runs: {best_seed_row['seed']} with total cost {best_seed_row['total_cost']:.6f}.",
        f"- Q2 ordinary customers served by EV in seed 11: {int(q2_summary['ordinary_customers_served_by_ev'])}.",
    ]
    (output_dir / "q1_q2_compare_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    context = load_q1_q2_context(args.q1_root, args.q2_root)
    seed_df = build_q2_seed_df(args.q2_seed_roots)

    plot_q1_q2_overview(context["q1_summary"], context["q2_summary"], args.output_dir)
    plot_q1_q2_vehicle_mix(context["route_df"], args.output_dir)
    plot_q1_q2_customer_change(context["customer_df"], context["q2_summary"], args.output_dir)
    plot_q2_seed_stability(seed_df, float(context["q2_summary"]["baseline_total_cost"]), args.output_dir)
    plot_q1_q2_route_cost_profile(context["route_df"], args.output_dir)
    plot_q1_q2_route_overlay(context["q1_stop_df"], context["q2_stop_df"], context["customer_coords"], args.output_dir)
    plot_q1_q2_route_difference_focus(context["q1_stop_df"], context["q2_stop_df"], context["customer_coords"], args.output_dir)
    plot_q1_q2_green_zone_zoom(context["q1_stop_df"], context["q2_stop_df"], context["customer_coords"], args.output_dir)
    write_notes(context["q1_summary"], context["q2_summary"], seed_df, args.output_dir)


if __name__ == "__main__":
    main()
