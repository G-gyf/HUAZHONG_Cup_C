from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


VEHICLE_COLORS = {
    "fuel_3000": "#c2410c",
    "ev_3000": "#0f766e",
    "fuel_1500": "#b45309",
    "fuel_1250": "#92400e",
    "ev_1250": "#2563eb",
}

SEED_COLORS = {
    "s11": "#0f766e",
    "s17": "#b45309",
    "s23": "#7c3aed",
}

POLICY_COLORS = {
    "Must-EV only": "#0f766e",
    "Fuel-after-16 served by fuel": "#b45309",
    "Ordinary served by EV": "#2563eb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-style figures for question 2 results.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--preprocess-root", type=Path, default=Path.cwd() / "preprocess_artifacts")
    parser.add_argument("--best-root", type=Path, default=Path.cwd() / "question2_artifacts_hybrid_standard_s11")
    parser.add_argument(
        "--comparison-roots",
        type=Path,
        nargs="*",
        default=[
            Path.cwd() / "question2_artifacts_hybrid_standard_s11",
            Path.cwd() / "question2_artifacts_hybrid_standard_s17",
            Path.cwd() / "question2_artifacts_hybrid_standard_s23",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "#f8fafc",
            "savefig.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#e2e8f0",
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}", rotation: int = 0) -> None:
    for patch in ax.patches:
        if hasattr(patch, "get_height"):
            value = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2.0
            y = value
            va = "bottom"
        else:
            continue
        if np.isnan(value):
            continue
        ax.text(x, y, fmt.format(value), ha="center", va=va, fontsize=9, color="#0f172a", rotation=rotation)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_seed_comparison(comparison_roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in comparison_roots:
        summary = load_json(root / "q2_hybrid_cost_summary.json")
        seed = int(summary["best_seed"])
        rows.append(
            {
                "seed": f"s{seed}",
                "total_cost": float(summary["total_cost"]),
                "route_count": int(summary["route_count"]),
                "single_stop_route_count": int(summary["single_stop_route_count"]),
                "late_positive_stops": int(summary["late_positive_stops"]),
                "max_late_min": float(summary["max_late_min"]),
                "elapsed_sec": float(summary["elapsed_sec"]),
                "policy_violation_count": int(summary["policy_violation_count"]),
            }
        )
    df = pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)
    return df


def build_best_context(best_root: Path, preprocess_root: Path) -> dict[str, object]:
    summary = load_json(best_root / "q2_hybrid_cost_summary.json")
    compare_summary = load_json(best_root / "q2_hybrid_compare_to_baseline.json")
    route_df = pd.read_csv(best_root / "q2_hybrid_route_summary.csv")
    stop_df = pd.read_csv(best_root / "q2_hybrid_stop_schedule.csv")
    customer_df = pd.read_csv(best_root / "q2_hybrid_customer_aggregate.csv")
    trace_df = pd.read_csv(best_root / "q2_hybrid_outer_search_trace.csv")
    preprocess_customer_df = pd.read_csv(preprocess_root / "tables" / "customer_master_98.csv")

    customer_coords = preprocess_customer_df[["cust_id", "x_km", "y_km"]].rename(columns={"cust_id": "orig_cust_id"})
    stop_df = stop_df.merge(customer_coords, on="orig_cust_id", how="left")

    route_df["stop_count"] = route_df["unit_sequence"].astype(str).apply(lambda text: len(str(text).split(",")))
    route_df["return_span_min"] = route_df["return_min"] - route_df["departure_min"]
    route_df["vehicle_group"] = route_df["vehicle_type"].map(
        {
            "ev_1250": "EV",
            "ev_3000": "EV",
            "fuel_1500": "Fuel",
            "fuel_3000": "Fuel",
            "fuel_1250": "Fuel",
        }
    )

    return {
        "summary": summary,
        "compare_summary": compare_summary,
        "route_df": route_df,
        "stop_df": stop_df,
        "customer_df": customer_df,
        "trace_df": trace_df,
    }


def plot_seed_comparison(seed_df: pd.DataFrame, baseline_total_cost: float, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    bars = axes[0].bar(
        seed_df["seed"],
        seed_df["total_cost"],
        color=[SEED_COLORS.get(seed, "#64748b") for seed in seed_df["seed"]],
        edgecolor="#ffffff",
        linewidth=1.5,
    )
    axes[0].axhline(baseline_total_cost, color="#b91c1c", linestyle="--", linewidth=1.8, label="Q2 v2 baseline")
    axes[0].set_title("Q2 Hybrid Standard: Seed Comparison by Total Cost")
    axes[0].set_ylabel("Total cost")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    for bar, value in zip(bars, seed_df["total_cost"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    metric_df = seed_df.melt(
        id_vars="seed",
        value_vars=["route_count", "single_stop_route_count", "late_positive_stops"],
        var_name="metric",
        value_name="value",
    )
    metric_label = {
        "route_count": "Routes",
        "single_stop_route_count": "Single-stop routes",
        "late_positive_stops": "Late stops",
    }
    metric_df["metric"] = metric_df["metric"].map(metric_label)
    sns.barplot(data=metric_df, x="metric", y="value", hue="seed", palette=SEED_COLORS, ax=axes[1])
    axes[1].set_title("Structural Indicators Across Seeds")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="Seed", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    save_figure(fig, output_dir / "q2_seed_comparison.png")


def plot_cost_breakdown(summary: dict[str, object], output_dir: Path) -> None:
    labels = ["Startup", "Energy", "Carbon", "Waiting", "Late"]
    values = [
        float(summary["startup_cost"]),
        float(summary["energy_cost"]),
        float(summary["carbon_cost"]),
        float(summary["waiting_cost"]),
        float(summary["late_cost"]),
    ]
    colors = ["#334155", "#0f766e", "#2563eb", "#d97706", "#dc2626"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="#ffffff", linewidth=1.5)
    ax.set_title("Cost Composition of Final Q2 Hybrid Solution (Seed 11)")
    ax.set_ylabel("Cost")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    total_cost = float(summary["total_cost"])
    ax.text(
        0.98,
        0.96,
        f"Total cost = {total_cost:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#ffffff", "edgecolor": "#e2e8f0"},
    )
    save_figure(fig, output_dir / "q2_cost_breakdown.png")


def plot_vehicle_and_route_structure(route_df: pd.DataFrame, summary: dict[str, object], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    vehicle_usage = pd.DataFrame(
        {
            "vehicle_type": list(summary["vehicle_type_usage"].keys()),
            "count": list(summary["vehicle_type_usage"].values()),
        }
    )
    sns.barplot(
        data=vehicle_usage,
        x="vehicle_type",
        y="count",
        hue="vehicle_type",
        palette=VEHICLE_COLORS,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Vehicle Usage by Type")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Route count")
    annotate_bars(axes[0], "{:.0f}")

    structure_df = pd.DataFrame(
        {
            "route class": ["Single-stop", "Two-stop", "Three-plus"],
            "count": [
                int(summary["single_stop_route_count"]),
                int(summary["two_stop_route_count"]),
                int(summary["three_plus_route_count"]),
            ],
        }
    )
    sns.barplot(
        data=structure_df,
        x="route class",
        y="count",
        hue="route class",
        palette={"Single-stop": "#475569", "Two-stop": "#0ea5e9", "Three-plus": "#16a34a"},
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Route Structure of Final Solution")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    annotate_bars(axes[1], "{:.0f}")

    save_figure(fig, output_dir / "q2_vehicle_route_structure.png")


def plot_policy_usage(summary: dict[str, object], customer_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    policy_usage = pd.DataFrame(
        {
            "category": ["Must-EV only", "Fuel-after-16 served by fuel", "Ordinary served by EV"],
            "count": [
                int(summary["must_use_ev_customers_served_by_ev_only"]),
                int(summary["fuel_after_16_service_customer_count"]),
                int(summary["ev_usage_by_policy_class"]["ordinary_customers_served_by_ev"]),
            ],
        }
    )
    sns.barplot(
        data=policy_usage,
        x="category",
        y="count",
        hue="category",
        palette=[POLICY_COLORS[item] for item in policy_usage["category"]],
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Policy-Class Service Outcomes")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Customer count")
    axes[0].tick_params(axis="x", rotation=12)
    annotate_bars(axes[0], "{:.0f}")

    customer_df["policy_group"] = np.select(
        [
            customer_df["must_use_ev_under_policy"] == 1,
            customer_df["fuel_allowed_after_16"] == 1,
        ],
        [
            "Must-EV customers",
            "Fuel-after-16 customers",
        ],
        default="Ordinary customers",
    )
    served_mix = (
        customer_df.groupby("policy_group", as_index=False)
        .agg(
            served_by_ev=("served_by_ev", "sum"),
            served_by_fuel=("served_by_fuel", "sum"),
        )
        .melt(id_vars="policy_group", var_name="service_mode", value_name="count")
    )
    service_palette = {"served_by_ev": "#0f766e", "served_by_fuel": "#c2410c"}
    sns.barplot(
        data=served_mix,
        x="policy_group",
        y="count",
        hue="service_mode",
        palette=service_palette,
        ax=axes[1],
    )
    axes[1].set_title("Vehicle-Energy Mode by Policy Group")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Customer count")
    axes[1].legend(
        title="Service mode",
        labels=["Served by EV", "Served by fuel"],
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#e2e8f0",
    )

    save_figure(fig, output_dir / "q2_policy_service_mix.png")


def plot_search_trace(trace_df: pd.DataFrame, summary: dict[str, object], output_dir: Path) -> None:
    trace_df = trace_df.copy()
    trace_df["iteration"] = np.arange(len(trace_df))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    axes[0].plot(
        trace_df["iteration"],
        trace_df["candidate_total_cost"],
        color="#0f766e",
        linewidth=2.0,
        alpha=0.85,
        label="Candidate cost",
    )
    axes[0].axhline(float(summary["total_cost"]), color="#b91c1c", linestyle="--", linewidth=1.8, label="Final best")
    axes[0].set_title("Hybrid Outer Search Trace")
    axes[0].set_xlabel("Outer iteration record")
    axes[0].set_ylabel("Total cost")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    operator_counts = (
        trace_df["operator_name"]
        .fillna("construct")
        .value_counts()
        .rename_axis("operator_name")
        .reset_index(name="count")
    )
    sns.barplot(
        data=operator_counts,
        x="operator_name",
        y="count",
        hue="operator_name",
        palette="crest",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Destroy/Repair Operator Usage")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=22)
    annotate_bars(axes[1], "{:.0f}")

    save_figure(fig, output_dir / "q2_hybrid_search_trace.png")


def plot_route_cost_profile(route_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    sns.scatterplot(
        data=route_df,
        x="return_min",
        y="route_cost",
        hue="vehicle_type",
        size="stop_count",
        palette=VEHICLE_COLORS,
        sizes=(35, 180),
        alpha=0.88,
        ax=axes[0],
    )
    axes[0].set_title("Route Cost vs. Return Time")
    axes[0].set_xlabel("Return time (min)")
    axes[0].set_ylabel("Route cost")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8, title="Vehicle / stop count")

    top_routes = route_df.nlargest(12, "route_cost").sort_values("route_cost", ascending=True)
    sns.barplot(
        data=top_routes,
        x="route_cost",
        y=top_routes["route_id"].astype(str),
        hue=top_routes["route_id"].astype(str),
        palette=[VEHICLE_COLORS.get(item, "#64748b") for item in top_routes["vehicle_type"]],
        ax=axes[1],
        orient="h",
        legend=False,
    )
    axes[1].set_title("Top 12 Most Expensive Routes")
    axes[1].set_xlabel("Route cost")
    axes[1].set_ylabel("Route ID")

    save_figure(fig, output_dir / "q2_route_cost_profile.png")


def plot_route_map(stop_df: pd.DataFrame, summary: dict[str, object], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 8.0))

    ax.scatter(stop_df["x_km"], stop_df["y_km"], s=18, c="#cbd5e1", alpha=0.45, edgecolor="none", zorder=1)
    ax.scatter([0], [0], s=180, c="#0f172a", marker="s", zorder=6)

    for route_id, sub_df in stop_df.sort_values(["route_id", "stop_index"]).groupby("route_id"):
        points_x = [0.0, *sub_df["x_km"].astype(float).tolist(), 0.0]
        points_y = [0.0, *sub_df["y_km"].astype(float).tolist(), 0.0]
        vehicle_type = str(sub_df["vehicle_type"].iloc[0])
        ax.plot(
            points_x,
            points_y,
            color=VEHICLE_COLORS.get(vehicle_type, "#64748b"),
            linewidth=1.2,
            alpha=0.42,
            zorder=2,
        )

    handles = [
        plt.Line2D([0], [0], color=color, lw=2.2, label=vehicle_type)
        for vehicle_type, color in VEHICLE_COLORS.items()
        if vehicle_type in set(stop_df["vehicle_type"].astype(str))
    ]
    ax.legend(handles=handles, title="Vehicle type", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.set_title("Spatial Layout of Final Q2 Hybrid Routes (Seed 11)")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02,
        0.02,
        f"Total routes: {int(summary['route_count'])}\nPolicy violations: {int(summary['policy_violation_count'])}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffffff", "edgecolor": "#e2e8f0"},
    )

    save_figure(fig, output_dir / "q2_route_map.png")


def plot_baseline_vs_final(summary: dict[str, object], compare_summary: dict[str, object], output_dir: Path) -> None:
    compare_df = pd.DataFrame(
        {
            "metric": ["Total cost", "Route count", "Late stops", "Fuel (L)", "Carbon (kg)"],
            "Baseline": [
                float(summary["baseline_total_cost"]),
                float(summary["baseline_route_count"]),
                float(compare_summary["baseline_late_positive_stops"]),
                float(summary["reference_total_fuel_l"]),
                float(summary["reference_total_carbon_kg"]),
            ],
            "Final": [
                float(summary["total_cost"]),
                float(summary["route_count"]),
                float(summary["late_positive_stops"]),
                float(summary["total_fuel_l"]),
                float(summary["total_carbon_kg"]),
            ],
        }
    )
    long_df = compare_df.melt(id_vars="metric", var_name="solution", value_name="value")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    left_df = long_df[long_df["metric"].isin(["Total cost", "Route count", "Late stops"])]
    sns.barplot(
        data=left_df,
        x="metric",
        y="value",
        hue="solution",
        palette={"Baseline": "#94a3b8", "Final": "#0f766e"},
        ax=axes[0],
    )
    axes[0].set_title("Baseline vs Final: Economic and Structural Metrics")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Value")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    right_df = long_df[long_df["metric"].isin(["Fuel (L)", "Carbon (kg)"])]
    sns.barplot(
        data=right_df,
        x="metric",
        y="value",
        hue="solution",
        palette={"Baseline": "#94a3b8", "Final": "#2563eb"},
        ax=axes[1],
    )
    axes[1].set_title("Baseline vs Final: Energy and Carbon Metrics")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Value")
    axes[1].legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    save_figure(fig, output_dir / "q2_baseline_vs_final.png")


def write_figure_notes(seed_df: pd.DataFrame, best_summary: dict[str, object], output_dir: Path) -> None:
    best_seed = seed_df.sort_values("total_cost").iloc[0]
    note_lines = [
        "# Q2 Paper Figures Notes",
        "",
        f"- Best standard-seed run: {best_seed['seed']} with total cost {best_seed['total_cost']:.6f}.",
        f"- Q2 v2 baseline total cost: {float(best_summary['baseline_total_cost']):.6f}.",
        f"- Improvement of final selected seed-11 result vs baseline: "
        f"{float(best_summary['baseline_total_cost']) - float(best_summary['total_cost']):.6f}.",
        f"- Policy violation count: {int(best_summary['policy_violation_count'])}.",
        f"- Mandatory EV served by non-EV: {int(best_summary['mandatory_ev_served_by_non_ev_count'])}.",
        f"- Fuel visits in green zone before 16:00: {int(best_summary['fuel_route_green_zone_pre16_visit_count'])}.",
    ]
    (output_dir / "q2_figure_notes.md").write_text("\n".join(note_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.best_root / "paper_figures")

    set_plot_style()
    seed_df = build_seed_comparison(args.comparison_roots)
    best_context = build_best_context(args.best_root, args.preprocess_root)
    best_summary = best_context["summary"]

    plot_seed_comparison(seed_df, float(best_summary["baseline_total_cost"]), output_dir)
    plot_cost_breakdown(best_summary, output_dir)
    plot_vehicle_and_route_structure(best_context["route_df"], best_summary, output_dir)
    plot_policy_usage(best_summary, best_context["customer_df"], output_dir)
    plot_search_trace(best_context["trace_df"], best_summary, output_dir)
    plot_route_cost_profile(best_context["route_df"], output_dir)
    plot_route_map(best_context["stop_df"], best_summary, output_dir)
    plot_baseline_vs_final(best_summary, best_context["compare_summary"], output_dir)
    write_figure_notes(seed_df, best_summary, output_dir)


if __name__ == "__main__":
    main()
