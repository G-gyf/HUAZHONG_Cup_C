from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


GREEN_ZONE_RADIUS_KM = 10.0
SERVICE_TIME_MIN = 20
POLICY_BAN_START_MIN = 0
POLICY_BAN_END_MIN = 480
DAY_END_MIN = 780
CSV_ENCODING = "utf-8-sig"


@dataclass(frozen=True)
class SpeedSegment:
    segment_id: int
    traffic_state: str
    start_min: int
    end_min: int
    speed_kmh: float


SPEED_SEGMENTS: tuple[SpeedSegment, ...] = (
    SpeedSegment(1, "congested", 0, 60, 9.8),
    SpeedSegment(2, "free_flow", 60, 120, 55.3),
    SpeedSegment(3, "normal", 120, 210, 35.4),
    SpeedSegment(4, "congested", 210, 300, 9.8),
    SpeedSegment(5, "free_flow", 300, 420, 55.3),
    SpeedSegment(6, "normal", 420, 540, 35.4),
    SpeedSegment(7, "congested", 540, 630, 9.8),
    SpeedSegment(8, "normal", 630, 720, 35.4),
    SpeedSegment(9, "free_flow", 720, 780, 55.3),
)


VEHICLE_CAPACITY_REFERENCE = pd.DataFrame(
    [
        {"vehicle_type": "fuel_3000", "power_type": "fuel", "capacity_kg": 3000, "capacity_m3": 13.5, "vehicle_count": 60},
        {"vehicle_type": "fuel_1500", "power_type": "fuel", "capacity_kg": 1500, "capacity_m3": 10.8, "vehicle_count": 50},
        {"vehicle_type": "fuel_1250", "power_type": "fuel", "capacity_kg": 1250, "capacity_m3": 6.5, "vehicle_count": 50},
        {"vehicle_type": "ev_3000", "power_type": "ev", "capacity_kg": 3000, "capacity_m3": 15.0, "vehicle_count": 10},
        {"vehicle_type": "ev_1250", "power_type": "ev", "capacity_kg": 1250, "capacity_m3": 8.5, "vehicle_count": 15},
    ]
)


class PreprocessError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the logistics case data.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace containing the PDF and attachment folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "preprocess_artifacts",
        help="Directory for generated tables, figures, and reports.",
    )
    return parser.parse_args()


def minutes_to_hhmm(minutes: float | int | None) -> str:
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return ""
    total_minutes = int(round(float(minutes)))
    hour = 8 + total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def format_minutes_axis(ax: plt.Axes) -> None:
    ticks = np.arange(0, DAY_END_MIN + 1, 60)
    ax.set_xticks(ticks)
    ax.set_xticklabels([minutes_to_hhmm(tick) for tick in ticks], rotation=45, ha="right")


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    for directory in (output_dir, tables_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "tables": tables_dir, "figures": figures_dir, "reports": reports_dir}


def discover_excel_files(workspace: Path) -> list[Path]:
    attachment_dirs = [path for path in workspace.iterdir() if path.is_dir()]
    if not attachment_dirs:
        raise PreprocessError("No attachment directory found in the workspace.")
    excel_files = []
    for directory in attachment_dirs:
        for path in directory.glob("*.xlsx"):
            if path.name.startswith("~$"):
                continue
            excel_files.append(path)
    if not excel_files:
        raise PreprocessError("No .xlsx input files were found in the attachment directory.")
    return sorted(excel_files)


def identify_input_tables(excel_files: Iterable[Path]) -> dict[str, tuple[Path, pd.DataFrame]]:
    identified: dict[str, tuple[Path, pd.DataFrame]] = {}
    for path in excel_files:
        df = pd.read_excel(path)
        shape = df.shape
        object_cols = int(df.select_dtypes(include="object").shape[1])
        if shape == (99, 100):
            identified["distance"] = (path, df)
        elif shape == (98, 3):
            identified["time"] = (path, df)
        elif shape == (2169, 4):
            identified["orders"] = (path, df)
        elif shape == (99, 4) and object_cols == 1:
            identified["coord"] = (path, df)

    required = {"coord", "time", "orders", "distance"}
    if set(identified) != required:
        missing = sorted(required - set(identified))
        raise PreprocessError(f"Failed to identify all required input tables. Missing: {missing}")
    return identified


def standardize_inputs(identified: dict[str, tuple[Path, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    coord = identified["coord"][1].copy()
    coord.columns = ["raw_type", "id", "x_km", "y_km"]
    coord = coord.sort_values("id").reset_index(drop=True)
    coord["node_type"] = np.where(coord["id"].eq(0), "depot", "customer")
    coord = coord[["id", "node_type", "x_km", "y_km", "raw_type"]]

    time_windows = identified["time"][1].copy()
    time_windows.columns = ["cust_id", "tw_start_raw", "tw_end_raw"]
    time_windows = time_windows.sort_values("cust_id").reset_index(drop=True)

    orders = identified["orders"][1].copy()
    orders.columns = ["order_id", "weight_original", "volume_original", "cust_id"]
    orders = orders.sort_values(["cust_id", "order_id"]).reset_index(drop=True)

    distance = identified["distance"][1].copy()
    first_col = distance.columns[0]
    distance = distance.rename(columns={first_col: "origin_id"})
    distance.columns = ["origin_id"] + [int(col) for col in distance.columns[1:]]
    distance = distance.sort_values("origin_id").reset_index(drop=True)
    ordered_cols = ["origin_id"] + sorted(int(col) for col in distance.columns[1:])
    distance = distance[ordered_cols]
    return {"coord": coord, "time": time_windows, "orders": orders, "distance": distance}


def validate_inputs(inputs: dict[str, pd.DataFrame]) -> None:
    coord = inputs["coord"]
    time_windows = inputs["time"]
    orders = inputs["orders"]
    distance = inputs["distance"]

    coord_ids = coord["id"].tolist()
    if coord["id"].duplicated().any():
        raise PreprocessError("Coordinate table contains duplicate node IDs.")
    if coord_ids != list(range(99)):
        raise PreprocessError("Coordinate table IDs are expected to be exactly 0..98.")
    if not coord.loc[coord["id"].eq(0), "node_type"].eq("depot").all():
        raise PreprocessError("Node 0 must be the depot.")

    if time_windows["cust_id"].duplicated().any():
        raise PreprocessError("Time window table contains duplicate customer IDs.")
    if set(time_windows["cust_id"]) != set(range(1, 99)):
        raise PreprocessError("Time window customer IDs are expected to be exactly 1..98.")

    if orders["order_id"].duplicated().any():
        raise PreprocessError("Orders table contains duplicate order IDs.")
    if not set(orders["cust_id"]).issubset(set(range(1, 99))):
        raise PreprocessError("Orders table contains invalid customer IDs.")

    invalid_weight = orders["weight_original"].notna() & orders["weight_original"].le(0)
    invalid_volume = orders["volume_original"].notna() & orders["volume_original"].le(0)
    if invalid_weight.any():
        raise PreprocessError("Orders table contains non-positive weights.")
    if invalid_volume.any():
        raise PreprocessError("Orders table contains non-positive volumes.")

    both_missing = orders["weight_original"].isna() & orders["volume_original"].isna()
    if both_missing.any():
        raise PreprocessError("At least one order row has both weight and volume missing.")

    parsed_start = pd.to_datetime(time_windows["tw_start_raw"], format="%H:%M", errors="coerce")
    parsed_end = pd.to_datetime(time_windows["tw_end_raw"], format="%H:%M", errors="coerce")
    if parsed_start.isna().any() or parsed_end.isna().any():
        raise PreprocessError("Time window parsing failed for at least one row.")
    widths = (parsed_end - parsed_start).dt.total_seconds() / 60
    if (widths <= 0).any():
        raise PreprocessError("Time windows must have positive width.")

    row_ids = distance["origin_id"].tolist()
    col_ids = [int(col) for col in distance.columns[1:]]
    if row_ids != list(range(99)) or col_ids != list(range(99)):
        raise PreprocessError("Distance matrix IDs are expected to be exactly 0..98.")
    matrix = distance.drop(columns="origin_id").to_numpy(dtype=float)
    if not np.allclose(np.diag(matrix), 0.0):
        raise PreprocessError("Distance matrix diagonal must be zero.")
    if np.any(matrix < 0):
        raise PreprocessError("Distance matrix contains negative distances.")
    if not np.allclose(matrix, matrix.T):
        raise PreprocessError("Distance matrix must be symmetric.")


def impute_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    clean = orders.copy()
    clean["weight"] = clean["weight_original"]
    clean["volume"] = clean["volume_original"]
    clean["is_imputed_weight"] = False
    clean["is_imputed_volume"] = False
    clean["weight_impute_method"] = ""
    clean["volume_impute_method"] = ""

    complete = clean.dropna(subset=["weight_original", "volume_original"]).copy()
    density_by_customer = (
        (complete["weight_original"] / complete["volume_original"])
        .groupby(complete["cust_id"])
        .median()
    )

    weight_coef = np.polyfit(complete["volume_original"], complete["weight_original"], deg=1)
    volume_coef = np.polyfit(complete["weight_original"], complete["volume_original"], deg=1)

    weight_missing_idx = clean.index[clean["weight_original"].isna() & clean["volume_original"].notna()]
    for idx in weight_missing_idx:
        cust_id = int(clean.at[idx, "cust_id"])
        volume = float(clean.at[idx, "volume_original"])
        if cust_id in density_by_customer.index:
            density = float(density_by_customer.loc[cust_id])
            predicted_weight = volume * density
            method = "customer_density"
        else:
            predicted_weight = float(np.polyval(weight_coef, volume))
            method = "global_regression"
        clean.at[idx, "weight"] = max(predicted_weight, 1e-6)
        clean.at[idx, "is_imputed_weight"] = True
        clean.at[idx, "weight_impute_method"] = method

    volume_missing_idx = clean.index[clean["volume_original"].isna() & clean["weight_original"].notna()]
    for idx in volume_missing_idx:
        cust_id = int(clean.at[idx, "cust_id"])
        weight = float(clean.at[idx, "weight_original"])
        if cust_id in density_by_customer.index:
            density = float(density_by_customer.loc[cust_id])
            predicted_volume = weight / density
            method = "customer_density"
        else:
            predicted_volume = float(np.polyval(volume_coef, weight))
            method = "global_regression"
        clean.at[idx, "volume"] = max(predicted_volume, 1e-6)
        clean.at[idx, "is_imputed_volume"] = True
        clean.at[idx, "volume_impute_method"] = method

    if clean["weight"].isna().any() or clean["volume"].isna().any():
        raise PreprocessError("Order imputation did not remove all missing values.")

    clean["is_imputed_any"] = clean["is_imputed_weight"] | clean["is_imputed_volume"]
    clean["is_oversize_order"] = clean["weight"].gt(3000) | clean["volume"].gt(15)
    impute_stats = {
        "weight_fit_slope": float(weight_coef[0]),
        "weight_fit_intercept": float(weight_coef[1]),
        "volume_fit_slope": float(volume_coef[0]),
        "volume_fit_intercept": float(volume_coef[1]),
    }
    return clean, impute_stats


def build_time_windows_numeric(time_windows: pd.DataFrame) -> pd.DataFrame:
    tw = time_windows.copy()
    start_dt = pd.to_datetime(tw["tw_start_raw"], format="%H:%M")
    end_dt = pd.to_datetime(tw["tw_end_raw"], format="%H:%M")
    base_dt = pd.Timestamp("1900-01-01 08:00")
    tw["tw_start_min"] = ((start_dt - base_dt).dt.total_seconds() / 60).astype(int)
    tw["tw_end_min"] = ((end_dt - base_dt).dt.total_seconds() / 60).astype(int)
    tw["tw_width_min"] = tw["tw_end_min"] - tw["tw_start_min"]
    tw["service_time_min"] = SERVICE_TIME_MIN
    return tw


def build_node_master(coord: pd.DataFrame) -> pd.DataFrame:
    node_master = coord.copy()
    node_master["radius_to_center_km"] = np.sqrt(node_master["x_km"] ** 2 + node_master["y_km"] ** 2)
    node_master["in_green_zone"] = node_master["node_type"].eq("customer") & node_master["radius_to_center_km"].le(GREEN_ZONE_RADIUS_KM)
    return node_master


def build_customer_demand(orders_clean: pd.DataFrame, customer_ids: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregated = (
        orders_clean.groupby("cust_id", as_index=False)
        .agg(
            order_count=("order_id", "count"),
            total_weight=("weight", "sum"),
            total_volume=("volume", "sum"),
            imputed_order_count=("is_imputed_any", "sum"),
            oversize_order_count=("is_oversize_order", "sum"),
        )
    )
    base = pd.DataFrame({"cust_id": list(customer_ids)})
    customer_demand = base.merge(aggregated, how="left", on="cust_id").fillna(
        {
            "order_count": 0,
            "total_weight": 0.0,
            "total_volume": 0.0,
            "imputed_order_count": 0,
            "oversize_order_count": 0,
        }
    )
    integer_cols = ["order_count", "imputed_order_count", "oversize_order_count"]
    customer_demand[integer_cols] = customer_demand[integer_cols].astype(int)
    customer_demand["has_orders"] = customer_demand["order_count"].gt(0)
    customer_demand["customer_split_required"] = customer_demand["total_weight"].gt(3000) | customer_demand["total_volume"].gt(15)
    active_customer_demand = customer_demand.loc[customer_demand["has_orders"]].copy()
    return customer_demand, active_customer_demand


def build_speed_profile() -> pd.DataFrame:
    records = []
    for segment in SPEED_SEGMENTS:
        records.append(
            {
                "segment_id": segment.segment_id,
                "traffic_state": segment.traffic_state,
                "start_min": segment.start_min,
                "end_min": segment.end_min,
                "start_hhmm": minutes_to_hhmm(segment.start_min),
                "end_hhmm": minutes_to_hhmm(segment.end_min),
                "duration_min": segment.end_min - segment.start_min,
                "speed_kmh": segment.speed_kmh,
                "speed_km_per_min": segment.speed_kmh / 60.0,
            }
        )
    return pd.DataFrame(records)


def compute_travel_time_lookup(distance_df: pd.DataFrame, speed_profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance_matrix = distance_df.drop(columns="origin_id").to_numpy(dtype=np.float32)
    node_ids = distance_df["origin_id"].to_numpy(dtype=np.int16)
    departure_minutes = np.arange(0, DAY_END_MIN + 1, dtype=np.int16)
    lookup = np.full((len(departure_minutes), len(node_ids), len(node_ids)), np.nan, dtype=np.float32)

    starts = speed_profile["start_min"].to_numpy(dtype=np.int32)
    ends = speed_profile["end_min"].to_numpy(dtype=np.int32)
    speeds = speed_profile["speed_kmh"].to_numpy(dtype=np.float64)
    diag_mask = np.eye(distance_matrix.shape[0], dtype=bool)

    for minute_index, depart_min in enumerate(departure_minutes):
        if minute_index % 120 == 0:
            print(f"Precomputing travel times for departure minute {depart_min:03d}/{DAY_END_MIN}")

        travel = np.zeros_like(distance_matrix, dtype=np.float64)
        remaining = distance_matrix.astype(np.float64).copy()
        if depart_min >= DAY_END_MIN:
            travel[:] = np.nan
            travel[diag_mask] = 0.0
            lookup[minute_index] = travel.astype(np.float32)
            continue

        segment_index = int(np.searchsorted(ends, depart_min, side="right"))
        for idx in range(segment_index, len(starts)):
            unfinished = remaining > 1e-9
            if not np.any(unfinished):
                break
            segment_start = depart_min if idx == segment_index else starts[idx]
            available_min = int(ends[idx] - segment_start)
            if available_min <= 0:
                continue

            distance_capacity = speeds[idx] * available_min / 60.0
            finishes_here = unfinished & (remaining <= distance_capacity + 1e-9)
            if np.any(finishes_here):
                travel[finishes_here] += remaining[finishes_here] / speeds[idx] * 60.0
                remaining[finishes_here] = 0.0

            continue_to_next = unfinished & ~finishes_here
            if np.any(continue_to_next):
                travel[continue_to_next] += available_min
                remaining[continue_to_next] -= distance_capacity

        travel[remaining > 1e-6] = np.nan
        travel[diag_mask] = 0.0
        lookup[minute_index] = travel.astype(np.float32)

    return lookup, departure_minutes, node_ids


def build_customer_master(
    node_master: pd.DataFrame,
    time_windows_numeric: pd.DataFrame,
    customer_demand: pd.DataFrame,
) -> pd.DataFrame:
    customers = node_master.loc[node_master["node_type"].eq("customer")].copy()
    customers = customers.rename(columns={"id": "cust_id"})
    customers = customers.drop(columns=["raw_type"], errors="ignore")
    customer_master = customers.merge(time_windows_numeric, how="left", on="cust_id")
    customer_master = customer_master.merge(customer_demand, how="left", on="cust_id")
    customer_master["fuel_forbidden_all_window"] = (
        customer_master["in_green_zone"]
        & customer_master["tw_start_min"].ge(POLICY_BAN_START_MIN)
        & customer_master["tw_end_min"].le(POLICY_BAN_END_MIN)
    )
    customer_master["fuel_allowed_after_16"] = customer_master["in_green_zone"] & customer_master["tw_end_min"].gt(POLICY_BAN_END_MIN)
    customer_master["fuel_partial_overlap_ban"] = (
        customer_master["in_green_zone"]
        & customer_master["tw_start_min"].lt(POLICY_BAN_END_MIN)
        & customer_master["tw_end_min"].gt(POLICY_BAN_END_MIN)
    )
    customer_master["must_use_ev_under_policy"] = customer_master["fuel_forbidden_all_window"]

    fuel_window_start = np.where(
        customer_master["fuel_allowed_after_16"],
        np.maximum(customer_master["tw_start_min"], POLICY_BAN_END_MIN),
        np.nan,
    )
    fuel_window_end = np.where(customer_master["fuel_allowed_after_16"], customer_master["tw_end_min"], np.nan)
    customer_master["fuel_service_window_start_min"] = fuel_window_start
    customer_master["fuel_service_window_end_min"] = fuel_window_end
    return customer_master


def build_policy_tables(customer_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_feasibility = customer_master[
        [
            "cust_id",
            "x_km",
            "y_km",
            "radius_to_center_km",
            "in_green_zone",
            "tw_start_raw",
            "tw_end_raw",
            "tw_start_min",
            "tw_end_min",
            "order_count",
            "total_weight",
            "total_volume",
            "fuel_forbidden_all_window",
            "fuel_allowed_after_16",
            "fuel_partial_overlap_ban",
            "must_use_ev_under_policy",
            "fuel_service_window_start_min",
            "fuel_service_window_end_min",
        ]
    ].copy()

    mandatory_ev = policy_feasibility.loc[policy_feasibility["must_use_ev_under_policy"]].copy()
    total_weight = float(mandatory_ev["total_weight"].sum())
    total_volume = float(mandatory_ev["total_volume"].sum())
    total_orders = int(mandatory_ev["order_count"].sum())
    ev_weight_cap = float((VEHICLE_CAPACITY_REFERENCE.query("power_type == 'ev'")["capacity_kg"] * VEHICLE_CAPACITY_REFERENCE.query("power_type == 'ev'")["vehicle_count"]).sum())
    ev_volume_cap = float((VEHICLE_CAPACITY_REFERENCE.query("power_type == 'ev'")["capacity_m3"] * VEHICLE_CAPACITY_REFERENCE.query("power_type == 'ev'")["vehicle_count"]).sum())

    ev_policy_summary = pd.DataFrame(
        [
            {
                "mandatory_ev_customer_count": int(mandatory_ev.shape[0]),
                "mandatory_ev_active_customer_count": int(mandatory_ev["order_count"].gt(0).sum()),
                "mandatory_ev_total_orders": total_orders,
                "mandatory_ev_total_weight": total_weight,
                "mandatory_ev_total_volume": total_volume,
                "optimistic_lb_trips_weight": int(math.ceil(total_weight / 3000)) if total_weight > 0 else 0,
                "optimistic_lb_trips_volume": int(math.ceil(total_volume / 15.0)) if total_volume > 0 else 0,
                "one_wave_ev_fleet_weight_capacity": ev_weight_cap,
                "one_wave_ev_fleet_volume_capacity": ev_volume_cap,
                "within_one_wave_ev_fleet_capacity": bool(total_weight <= ev_weight_cap and total_volume <= ev_volume_cap),
            }
        ]
    )
    return policy_feasibility, ev_policy_summary


def build_distance_matrix_clean(distance_df: pd.DataFrame) -> pd.DataFrame:
    matrix = distance_df.copy()
    matrix = matrix.set_index("origin_id")
    matrix.index.name = "origin_id"
    return matrix


def build_route_template() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "route_id",
            "vehicle_type",
            "node_sequence",
            "departure_time_min",
            "arrival_time_min",
            "load_used_kg",
            "load_used_m3",
        ]
    )


def build_summary_metrics(
    inputs: dict[str, pd.DataFrame],
    orders_clean: pd.DataFrame,
    node_master: pd.DataFrame,
    customer_master: pd.DataFrame,
    ev_policy_summary: pd.DataFrame,
) -> dict[str, object]:
    in_zone_count = int(node_master["in_green_zone"].sum())
    metrics = {
        "source_order_rows": int(inputs["orders"].shape[0]),
        "source_customer_rows": int(inputs["time"].shape[0]),
        "source_node_rows": int(inputs["coord"].shape[0]),
        "source_distance_shape": list(inputs["distance"].shape),
        "missing_weight_rows": int(inputs["orders"]["weight_original"].isna().sum()),
        "missing_volume_rows": int(inputs["orders"]["volume_original"].isna().sum()),
        "imputed_weight_rows": int(orders_clean["is_imputed_weight"].sum()),
        "imputed_volume_rows": int(orders_clean["is_imputed_volume"].sum()),
        "total_imputed_rows": int(orders_clean["is_imputed_any"].sum()),
        "oversize_order_count": int(orders_clean["is_oversize_order"].sum()),
        "zero_demand_customer_count": int(customer_master["has_orders"].eq(False).sum()),
        "active_customer_count": int(customer_master["has_orders"].sum()),
        "green_zone_customer_count_geometry": in_zone_count,
        "problem_statement_green_zone_count": 30,
        "green_zone_count_conflict": bool(in_zone_count != 30),
        "split_required_customer_count": int(customer_master["customer_split_required"].sum()),
        "fuel_forbidden_all_window_count": int(customer_master["fuel_forbidden_all_window"].sum()),
        "fuel_allowed_after_16_count": int(customer_master["fuel_allowed_after_16"].sum()),
    }
    metrics.update(ev_policy_summary.iloc[0].to_dict())
    return metrics


def save_tables(
    output_dirs: dict[str, Path],
    tables: dict[str, pd.DataFrame],
    travel_time_lookup: np.ndarray,
    departure_minutes: np.ndarray,
    node_ids: np.ndarray,
    summary_metrics: dict[str, object],
    impute_stats: dict[str, float],
) -> None:
    tables_dir = output_dirs["tables"]
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=True if name == "distance_matrix_clean" else False, encoding=CSV_ENCODING)

    workbook_path = tables_dir / "preprocess_tables.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=True if name == "distance_matrix_clean" else False)

    np.savez_compressed(
        output_dirs["root"] / "travel_time_lookup.npz",
        travel_time_minutes=travel_time_lookup.astype(np.float32),
        departure_minutes=departure_minutes.astype(np.int16),
        node_ids=node_ids.astype(np.int16),
    )

    config = {
        "green_zone_radius_km": GREEN_ZONE_RADIUS_KM,
        "service_time_min": SERVICE_TIME_MIN,
        "policy_ban_start_min": POLICY_BAN_START_MIN,
        "policy_ban_end_min": POLICY_BAN_END_MIN,
        "day_end_min": DAY_END_MIN,
        "speed_segments": [asdict(segment) for segment in SPEED_SEGMENTS],
        "vehicle_capacity_reference": VEHICLE_CAPACITY_REFERENCE.to_dict(orient="records"),
        "imputation_models": impute_stats,
    }
    (output_dirs["root"] / "preprocess_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dirs["reports"] / "preprocess_metrics.json").write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")


def create_plots(
    output_dirs: dict[str, Path],
    node_master: pd.DataFrame,
    customer_master: pd.DataFrame,
    orders_clean: pd.DataFrame,
    distance_matrix_clean: pd.DataFrame,
    speed_profile: pd.DataFrame,
    travel_time_lookup: np.ndarray,
    departure_minutes: np.ndarray,
) -> None:
    figures_dir = output_dirs["figures"]
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 160

    customers = node_master.loc[node_master["node_type"].eq("customer")].copy()
    depot = node_master.loc[node_master["node_type"].eq("depot")].iloc[0]
    in_zone = customers.loc[customers["in_green_zone"]]
    out_zone = customers.loc[~customers["in_green_zone"]]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(plt.Circle((0, 0), GREEN_ZONE_RADIUS_KM, color="#7ccba2", alpha=0.18, ec="#2f855a", lw=2))
    ax.scatter(out_zone["x_km"], out_zone["y_km"], s=28, c="#718096", alpha=0.8, label="Outside green zone")
    ax.scatter(in_zone["x_km"], in_zone["y_km"], s=36, c="#2f855a", alpha=0.9, label="Inside green zone")
    ax.scatter([depot["x_km"]], [depot["y_km"]], s=120, marker="s", c="#c53030", label="Depot")
    ax.scatter([0], [0], s=120, marker="*", c="#1a202c", label="City center")
    ax.set_title("Customer spatial distribution and 10 km green zone")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    ax.text(-32, 35, f"Geometry-based green-zone customers: {len(in_zone)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "customer_spatial_distribution.png")
    plt.close(fig)

    active = customer_master.loc[customer_master["has_orders"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    weight_capacity_handle = Line2D([0], [0], color="#c53030", linestyle="--", linewidth=1.2, label="Vehicle weight capacity")
    volume_capacity_handle = Line2D([0], [0], color="#c53030", linestyle="--", linewidth=1.2, label="Vehicle volume capacity")
    sns.histplot(active["total_weight"], bins=18, ax=axes[0], color="#2b6cb0")
    for cap in (1250, 1500, 3000):
        axes[0].axvline(cap, color="#c53030", linestyle="--", linewidth=1)
    axes[0].set_title("Customer total weight")
    axes[0].set_xlabel("Weight (kg)")
    axes[0].legend(handles=[weight_capacity_handle], loc="upper right", frameon=True)

    sns.histplot(active["total_volume"], bins=18, ax=axes[1], color="#2f855a")
    for cap in (6.5, 8.5, 10.8, 13.5, 15.0):
        axes[1].axvline(cap, color="#c53030", linestyle="--", linewidth=1)
    axes[1].set_title("Customer total volume")
    axes[1].set_xlabel("Volume (m^3)")
    axes[1].legend(handles=[volume_capacity_handle], loc="upper right", frameon=True)

    sns.histplot(active["order_count"], bins=18, ax=axes[2], color="#b7791f")
    axes[2].set_title("Orders per active customer")
    axes[2].set_xlabel("Order count")
    axes[2].text(
        0.98,
        0.95,
        "No red line here:\norder count has no direct\nvehicle-capacity threshold.",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d2d6dc", "alpha": 0.95},
    )
    fig.suptitle("Customer demand distributions and vehicle-capacity references", fontsize=14, y=0.98)
    fig.text(
        0.5,
        0.02,
        "Dashed red lines are capacity thresholds and only apply within the weight/volume panels; the three panels use different x-axis units.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(figures_dir / "customer_demand_distribution.png")
    plt.close(fig)

    observed = orders_clean.loc[~orders_clean["is_imputed_any"]]
    imputed = orders_clean.loc[orders_clean["is_imputed_any"]]
    oversize = orders_clean.loc[orders_clean["is_oversize_order"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(observed["weight"], observed["volume"], s=12, alpha=0.35, c="#2b6cb0", label="Observed orders")
    if not imputed.empty:
        ax.scatter(imputed["weight"], imputed["volume"], s=45, alpha=0.9, c="#ed8936", marker="D", label="Imputed orders")
    if not oversize.empty:
        ax.scatter(oversize["weight"], oversize["volume"], s=120, alpha=0.95, c="#c53030", marker="x", label="Oversize orders")
    ax.axvline(3000, color="#c53030", linestyle="--", linewidth=1)
    ax.axhline(15, color="#c53030", linestyle="--", linewidth=1)
    ax.set_title("Order weight-volume profile")
    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Volume (m^3)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "order_weight_volume_scatter.png")
    plt.close(fig)

    tw_plot = customer_master.sort_values(["tw_start_min", "tw_end_min"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    color_map = {"congested": "#f6ad55", "normal": "#90cdf4", "free_flow": "#9ae6b4"}
    for _, segment in speed_profile.iterrows():
        ax.axvspan(segment["start_min"], segment["end_min"], color=color_map[segment["traffic_state"]], alpha=0.22)
    for row_index, row in tw_plot.iterrows():
        color = "#2f855a" if row["in_green_zone"] else "#4a5568"
        ax.hlines(row_index, row["tw_start_min"], row["tw_end_min"], color=color, linewidth=2.2)
    ax.axvline(POLICY_BAN_END_MIN, color="#c53030", linestyle="--", linewidth=1.4, label="Fuel ban ends at 16:00")
    ax.set_title("Customer time windows with traffic-state background")
    ax.set_xlabel("Clock time")
    ax.set_ylabel("Customers sorted by time window")
    format_minutes_axis(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "customer_time_windows.png")
    plt.close(fig)

    policy_plot = customer_master.loc[customer_master["in_green_zone"]].sort_values(["must_use_ev_under_policy", "tw_start_min", "tw_end_min"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for row_index, row in policy_plot.iterrows():
        color = "#c53030" if row["must_use_ev_under_policy"] else "#2b6cb0"
        ax.hlines(row_index, row["tw_start_min"], row["tw_end_min"], color=color, linewidth=2.6)
    ax.axvspan(POLICY_BAN_START_MIN, POLICY_BAN_END_MIN, color="#fed7d7", alpha=0.35)
    ax.axvline(POLICY_BAN_END_MIN, color="#c53030", linestyle="--", linewidth=1.4)
    ax.set_title("Green-zone customer policy feasibility")
    ax.set_xlabel("Clock time")
    ax.set_ylabel("Green-zone customers")
    format_minutes_axis(ax)
    ax.text(6, max(len(policy_plot) - 1, 0), "Red: EV-only under policy\nBlue: fuel can serve after 16:00", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "policy_feasibility_green_zone.png")
    plt.close(fig)

    depot_distances = distance_matrix_clean.loc[0].drop(labels=0)
    candidate_ids = [
        int(depot_distances.sort_values().index[0]),
        int(depot_distances.sort_values().index[len(depot_distances) // 2]),
        int(depot_distances.sort_values().index[-1]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(distance_matrix_clean, ax=axes[0], cmap="viridis", cbar_kws={"label": "Distance (km)"})
    axes[0].set_title("Road distance matrix")
    axes[0].set_xlabel("Destination ID")
    axes[0].set_ylabel("Origin ID")
    for cust_id in candidate_ids:
        travel_series = travel_time_lookup[:, 0, cust_id]
        axes[1].plot(departure_minutes, travel_series, linewidth=2, label=f"Depot -> customer {cust_id}")
    axes[1].set_title("Departure-time travel curves for sample arcs")
    axes[1].set_xlabel("Departure time")
    axes[1].set_ylabel("Travel time (min)")
    format_minutes_axis(axes[1])
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "distance_travel_time_overview.png")
    plt.close(fig)


def write_reports(
    output_dirs: dict[str, Path],
    input_files: dict[str, tuple[Path, pd.DataFrame]],
    summary_metrics: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> None:
    reports_dir = output_dirs["reports"]
    summary_lines = [
        "# Preprocessing Summary",
        "",
        "## Inputs",
        f"- Coordinate file: `{input_files['coord'][0].name}`",
        f"- Time-window file: `{input_files['time'][0].name}`",
        f"- Orders file: `{input_files['orders'][0].name}`",
        f"- Distance file: `{input_files['distance'][0].name}`",
        "",
        "## Data quality snapshot",
        f"- Order rows: {summary_metrics['source_order_rows']}",
        f"- Missing weight rows repaired: {summary_metrics['imputed_weight_rows']}",
        f"- Missing volume rows repaired: {summary_metrics['imputed_volume_rows']}",
        f"- Total imputed rows: {summary_metrics['total_imputed_rows']}",
        f"- Oversize orders: {summary_metrics['oversize_order_count']}",
        f"- Active customers with orders: {summary_metrics['active_customer_count']}",
        f"- Zero-demand customers: {summary_metrics['zero_demand_customer_count']}",
        f"- Customers requiring split service: {summary_metrics['split_required_customer_count']}",
        "",
        "## Green-zone note",
        f"- Geometry-based count within 10 km of (0,0): {summary_metrics['green_zone_customer_count_geometry']}",
        "- Problem statement states 30 customers in the green zone.",
        "- This implementation follows the attachment geometry and flags the discrepancy for reporting.",
        "",
        "## Policy preview",
        f"- Green-zone customers that are EV-only under the ban: {summary_metrics['fuel_forbidden_all_window_count']}",
        f"- Green-zone customers that fuel vehicles may serve after 16:00: {summary_metrics['fuel_allowed_after_16_count']}",
        f"- Mandatory-EV active customers: {summary_metrics['mandatory_ev_active_customer_count']}",
        f"- Mandatory-EV total weight: {summary_metrics['mandatory_ev_total_weight']:.3f} kg",
        f"- Mandatory-EV total volume: {summary_metrics['mandatory_ev_total_volume']:.3f} m^3",
        f"- Optimistic lower bound on EV trips by weight: {summary_metrics['optimistic_lb_trips_weight']}",
        f"- Optimistic lower bound on EV trips by volume: {summary_metrics['optimistic_lb_trips_volume']}",
        "",
        "## Outputs",
        "- Clean tables are stored under `tables/` in CSV and Excel workbook form.",
        "- Minute-level travel time lookup is stored in `travel_time_lookup.npz`.",
        "- Visualizations are stored under `figures/`.",
        "",
    ]
    (reports_dir / "preprocessing_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    dictionary_lines = [
        "# Output Data Dictionary",
        "",
        "## Tables",
        "- `node_master`: all 99 nodes with coordinates, node type, and geometry-based green-zone label.",
        "- `time_windows_numeric`: customer time windows in both raw HH:MM and minute form.",
        "- `orders_clean`: cleaned order-level demand with imputation flags and oversize markers.",
        "- `customer_demand_98`: 98-customer demand table including zero-demand customers.",
        "- `active_customer_demand_88`: subset of customers with at least one order.",
        "- `customer_master_98`: merged customer coordinates, time windows, demand, and policy feasibility flags.",
        "- `policy_feasibility`: policy-focused customer summary for the green-zone restriction analysis.",
        "- `speed_profile`: piecewise deterministic speed segments used for travel-time lookup.",
        "- `distance_matrix_clean`: cleaned 99x99 road distance matrix indexed by node ID.",
        "- `oversize_order`: order-level records above the largest single-vehicle capacity.",
        "- `ev_policy_summary`: one-row lower-bound summary for mandatory-EV demand under the policy.",
        "- `route_solution_template`: empty schema for later route-level post-solution analysis.",
        "",
        "## Key fields",
        "- `tw_start_min` / `tw_end_min`: minutes after 08:00.",
        "- `in_green_zone`: geometry-based label using radius <= 10 km around (0,0).",
        "- `customer_split_required`: customer demand exceeds 3000 kg or 15 m^3.",
        "- `must_use_ev_under_policy`: green-zone customer whose full time window is inside [08:00, 16:00].",
        "- `fuel_service_window_start_min` / `fuel_service_window_end_min`: feasible service interval for fuel vehicles after the ban ends.",
        "",
        "## Travel-time lookup file",
        "- `travel_time_lookup.npz` contains arrays `travel_time_minutes`, `departure_minutes`, and `node_ids`.",
        "- `travel_time_minutes[k, i, j]` is the travel time in minutes from node `i` to node `j` when departing at `departure_minutes[k]`.",
        "- Values become `NaN` if the trip cannot be completed before 21:00 under the chosen speed profile.",
        "",
    ]
    (reports_dir / "data_dictionary.md").write_text("\n".join(dictionary_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dirs = ensure_output_dirs(args.output_dir)

    excel_files = discover_excel_files(args.workspace)
    identified = identify_input_tables(excel_files)
    standardized = standardize_inputs(identified)
    validate_inputs(standardized)

    orders_clean, impute_stats = impute_orders(standardized["orders"])
    time_windows_numeric = build_time_windows_numeric(standardized["time"])
    node_master = build_node_master(standardized["coord"])
    customer_ids = range(1, 99)
    customer_demand_98, active_customer_demand_88 = build_customer_demand(orders_clean, customer_ids)
    speed_profile = build_speed_profile()
    customer_master_98 = build_customer_master(node_master, time_windows_numeric, customer_demand_98)
    policy_feasibility, ev_policy_summary = build_policy_tables(customer_master_98)
    distance_matrix_clean = build_distance_matrix_clean(standardized["distance"])
    oversize_order = orders_clean.loc[orders_clean["is_oversize_order"]].copy()
    route_solution_template = build_route_template()
    travel_time_lookup, departure_minutes, node_ids = compute_travel_time_lookup(standardized["distance"], speed_profile)

    summary_metrics = build_summary_metrics(
        standardized,
        orders_clean,
        node_master,
        customer_master_98,
        ev_policy_summary,
    )

    tables = {
        "node_master": node_master.drop(columns="raw_type"),
        "time_windows_numeric": time_windows_numeric,
        "orders_clean": orders_clean,
        "customer_demand_98": customer_demand_98,
        "active_customer_demand_88": active_customer_demand_88,
        "customer_master_98": customer_master_98,
        "policy_feasibility": policy_feasibility,
        "speed_profile": speed_profile,
        "distance_matrix_clean": distance_matrix_clean,
        "oversize_order": oversize_order,
        "ev_policy_summary": ev_policy_summary,
        "vehicle_capacity_reference": VEHICLE_CAPACITY_REFERENCE,
        "route_solution_template": route_solution_template,
    }

    save_tables(output_dirs, tables, travel_time_lookup, departure_minutes, node_ids, summary_metrics, impute_stats)
    create_plots(
        output_dirs,
        node_master,
        customer_master_98,
        orders_clean,
        distance_matrix_clean,
        speed_profile,
        travel_time_lookup,
        departure_minutes,
    )
    write_reports(output_dirs, identified, summary_metrics, tables)

    print("Preprocessing completed.")
    print(f"Artifacts written to: {output_dirs['root']}")
    print(json.dumps(summary_metrics, indent=2))


if __name__ == "__main__":
    main()
