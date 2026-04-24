from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


DAY_END_MIN = 780
CSV_ENCODING = "utf-8-sig"
AFTER_HOURS_SPEED_KMH = 55.3
MAX_SINGLE_WEIGHT = 3000.0
MAX_SINGLE_VOLUME = 15.0
SERVICE_TIME_MIN = 20.0
START_COST = 400.0
WAIT_COST_PER_MIN = 20.0 / 60.0
LATE_COST_PER_MIN = 50.0 / 60.0
FUEL_PRICE = 7.61
ELECTRICITY_PRICE = 1.64
FUEL_CARBON_FACTOR = 2.547
ELECTRICITY_CARBON_FACTOR = 0.501
CARBON_COST = 0.65
CONSTRUCTION_NEIGHBOR_LIMIT = 18
REPAIR_ROUTE_SHORTLIST = 12
REMOVE_MIN = 4
REMOVE_MAX = 20
BASE_REMOVE_COUNT = 10
ACCEPT_TEMPERATURE = 1000.0
PACKING_ATTEMPTS = 96
BIG_VEHICLE_RESERVE = 5
MERGE_ROUTE_LIMIT = 40
MERGE_PAIR_LIMIT = 120
RELOCATE_ROUTE_LIMIT = 60
ROUTE_MERGE_COST_ALLOWANCE = 250.0
RELOCATE_REMOVE_COST_ALLOWANCE = 200.0
ROUTE_TYPE_CHANGE_COST_ALLOWANCE = 80.0
FUEL_3000_SEARCH_RESERVE = 1
BATCH_MERGE_LIMIT = 1
SINGLE_MERGE_SAMPLE_LIMIT = 24
ROUTE_POOL_MAX_COLUMNS = 4000
ROUTE_POOL_PROMOTION_LIMIT = 2000
ROUTE_POOL_FLEX_SMALL_LIMIT = 1200
ROUTE_POOL_PASS2_MAX_COLUMNS = 5000
BASE_FLEX_SPATIAL_NEIGHBOR_LIMIT = 8
BASE_FLEX_TW_NEIGHBOR_LIMIT = 8
BASE_FLEX_NEIGHBOR_LIMIT = 12
BASE_FLEX_TRIPLE_SEED_LIMIT = 5
ROUTE_POOL_RESIDUAL_PROMOTION_LIMIT = 600
ROUTE_POOL_RESIDUAL_FLEX_SMALL_LIMIT = 400
RESIDUAL_SOURCE_LIMIT = 40
RESIDUAL_SPATIAL_NEIGHBOR_LIMIT = 8
RESIDUAL_TW_NEIGHBOR_LIMIT = 8
RESIDUAL_NEIGHBOR_LIMIT = 16
RESIDUAL_TRIPLE_PARTNER_LIMIT = 6
RESIDUAL_PROMOTION_PER_SOURCE_LIMIT = 12
RESIDUAL_FLEX_SMALL_PER_SOURCE_LIMIT = 8
RESIDUAL_TW_OVERLAP_MIN = 30.0
PASS3_SOURCE_LIMIT = 64
PASS3_TRIPLE_PARTNER_LIMIT = 5
PASS3_FLEX_SMALL_PAIR_PER_SOURCE_LIMIT = 8
PASS3_FLEX_SMALL_TRIPLE_PER_SOURCE_LIMIT = 6
PASS3_FLEX_SMALL_QUAD_PER_SOURCE_LIMIT = 4
PASS3_CLUSTER_ROUTE_LIMIT = 4
PASS3_CLUSTER_PARTNER_UNIT_LIMIT = 7
BIG_ROUTE_SOFT_PENALTY = 10.0
MILP_PHASE_TIME_LIMIT_SEC = 60
MILP_REL_GAP = 0.0
ROUTE_POOL_PIGGYBACK_BIG_LIMIT = 500
COST_FIRST_ROUTE_POOL_ITERATIONS = 2
COST_IMPROVEMENT_EPS = 1e-6


@dataclass(frozen=True)
class VehicleType:
    vehicle_type: str
    power_type: str
    capacity_kg: float
    capacity_m3: float
    vehicle_count: int


@dataclass(frozen=True)
class SpeedSegment:
    start_min: int
    end_min: int
    speed_kmh: float


@dataclass(frozen=True)
class OrderFragment:
    fragment_id: str
    source_order_id: int
    weight: float
    volume: float


@dataclass(frozen=True)
class ServiceUnit:
    unit_id: int
    orig_cust_id: int
    unit_type: str
    visit_index: int
    weight: float
    volume: float
    tw_start_min: int
    tw_end_min: int
    x_km: float
    y_km: float
    eligible_vehicle_types: tuple[str, ...]
    source_order_ids: tuple[int, ...]
    required_visit_count: int


@dataclass(frozen=True)
class TypedRoute:
    vehicle_type: str
    unit_ids: tuple[int, ...]


@dataclass
class RouteEvaluation:
    vehicle_type: str
    unit_ids: tuple[int, ...]
    feasible: bool
    total_weight: float
    total_volume: float
    best_cost: float
    best_start: int
    best_return: float


@dataclass
class CandidateRouteSpec:
    route: TypedRoute
    route_eval: RouteEvaluation
    roles: set[str]


@dataclass
class AssignedRoute:
    route_index: int
    vehicle_type: str
    power_type: str
    vehicle_instance: int
    unit_ids: tuple[int, ...]
    departure_min: float
    return_min: float
    route_cost: float
    energy_cost: float
    carbon_cost: float
    waiting_cost: float
    late_cost: float
    startup_cost: float
    total_wait_min: float
    total_late_min: float
    total_fuel_l: float
    total_electricity_kwh: float
    route_distance_km: float
    after_hours_travel_km: float
    after_hours_service_count: int
    after_hours_return_flag: bool
    late_positive_stop_count: int
    max_late_min: float
    units: tuple[ServiceUnit, ...]
    stop_rows: list[dict[str, object]]


@dataclass
class SolutionEvaluation:
    total_cost: float
    total_energy_cost: float
    total_carbon_cost: float
    total_waiting_cost: float
    total_late_cost: float
    total_late_min: float
    total_startup_cost: float
    total_fuel_l: float
    total_electricity_kwh: float
    total_carbon_kg: float
    total_distance_km: float
    route_count: int
    used_vehicle_count: int
    split_customer_count: int
    mandatory_split_customer_count: int
    mandatory_split_visit_count: int
    normal_customer_count: int
    single_stop_route_count: int
    two_stop_route_count: int
    three_plus_route_count: int
    late_positive_stops: int
    max_late_min: float
    latest_return_min: float
    after_hours_service_count: int
    after_hours_return_count: int
    after_hours_travel_km: float
    vehicle_type_usage: dict[str, int]
    assigned_routes: list[AssignedRoute]


class RouteCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[int, ...]], RouteEvaluation] = {}

    def get(self, key: tuple[str, tuple[int, ...]]) -> RouteEvaluation | None:
        return self._cache.get(key)

    def set(self, key: tuple[str, tuple[int, ...]], value: RouteEvaluation) -> None:
        self._cache[key] = value


class Question1Solver:
    def __init__(
        self,
        workspace: Path,
        input_root: Path,
        output_root: Path,
        seed_list: list[int],
        max_generations: int,
        particle_count: int,
        top_route_candidates: int,
        packing_strategy: str = "baseline",
        enable_split_packing_sensitivity: bool = True,
    ) -> None:
        self.workspace = workspace
        self.input_root = input_root
        self.output_root = output_root
        self.seed_list = seed_list
        self.max_generations = max_generations
        self.particle_count = particle_count
        self.top_route_candidates = top_route_candidates
        self.packing_strategy = packing_strategy
        self.enable_split_packing_sensitivity = enable_split_packing_sensitivity

        tables_root = self.input_root / "tables"
        self.customer_master = pd.read_csv(tables_root / "customer_master_98.csv")
        self.orders = pd.read_csv(tables_root / "orders_clean.csv")
        self.distance_df = pd.read_csv(tables_root / "distance_matrix_clean.csv")
        self.vehicle_df = pd.read_csv(tables_root / "vehicle_capacity_reference.csv")
        self.speed_profile_df = pd.read_csv(tables_root / "speed_profile.csv")

        config = json.loads((self.input_root / "preprocess_config.json").read_text(encoding="utf-8"))
        self.service_time_min = float(config.get("service_time_min", SERVICE_TIME_MIN))

        lookup = np.load(self.input_root / "travel_time_lookup.npz")
        self.raw_travel_time_lookup = lookup["travel_time_minutes"].astype(np.float32)
        self.node_ids = lookup["node_ids"].astype(np.int16)
        self.node_to_idx = {int(node_id): idx for idx, node_id in enumerate(self.node_ids)}
        self.route_start_grid = np.arange(DAY_END_MIN + 1, dtype=np.float64)

        self.vehicles = [
            VehicleType(
                vehicle_type=str(row.vehicle_type),
                power_type=str(row.power_type),
                capacity_kg=float(row.capacity_kg),
                capacity_m3=float(row.capacity_m3),
                vehicle_count=int(row.vehicle_count),
            )
            for row in self.vehicle_df.itertuples(index=False)
        ]
        self.vehicle_by_name = {vehicle.vehicle_type: vehicle for vehicle in self.vehicles}
        self.vehicle_size_rank = {
            vehicle.vehicle_type: (vehicle.capacity_kg, vehicle.capacity_m3, 0 if vehicle.power_type == "fuel" else 1)
            for vehicle in self.vehicles
        }

        self.segments = [
            SpeedSegment(
                start_min=int(row.start_min),
                end_min=int(row.end_min),
                speed_kmh=float(row.speed_kmh),
            )
            for row in self.speed_profile_df.itertuples(index=False)
        ]
        self.segment_starts = np.array([segment.start_min for segment in self.segments], dtype=np.int32)
        self.segment_ends = np.array([segment.end_min for segment in self.segments], dtype=np.int32)
        self.segment_speeds = np.array([segment.speed_kmh for segment in self.segments], dtype=np.float64)
        self.segment_fpk = 0.0025 * self.segment_speeds**2 - 0.2554 * self.segment_speeds + 31.75
        self.segment_epk = 0.0014 * self.segment_speeds**2 - 0.12 * self.segment_speeds + 36.19
        self.after_hours_speed = AFTER_HOURS_SPEED_KMH
        self.after_hours_fpk = float(0.0025 * self.after_hours_speed**2 - 0.2554 * self.after_hours_speed + 31.75)
        self.after_hours_epk = float(0.0014 * self.after_hours_speed**2 - 0.12 * self.after_hours_speed + 36.19)

        self.distance_matrix = self.distance_df.drop(columns=["origin_id"]).to_numpy(dtype=np.float32)
        self.distance_lookup = self.distance_df.set_index("origin_id")
        self.active_customer_df = self.customer_master.loc[self.customer_master["has_orders"]].copy().sort_values("cust_id")
        self.customer_ids = [0] + self.active_customer_df["cust_id"].astype(int).tolist()
        self.customer_index = {cust_id: idx for idx, cust_id in enumerate(self.customer_ids)}
        self.customer_affinity = self._build_customer_affinity()
        self.customer_neighbors = self._build_customer_neighbors()

        self.energy_lookup_dir = self.output_root / "cache"
        self.energy_lookup_dir.mkdir(parents=True, exist_ok=True)
        self.travel_time_lookup, self.base_fuel_lookup, self.base_electric_lookup = self._load_or_build_day_night_arc_lookups()

        self.service_units, self.split_plan_rows = self._build_service_units()
        self.unit_by_id = {unit.unit_id: unit for unit in self.service_units}
        self.active_unit_ids = [unit.unit_id for unit in self.service_units]
        self.route_cache = RouteCache()
        self.enable_fuel_3000_reserve = False

    def _packing_candidate_sort_key(self, item: dict[str, object]) -> tuple[int, int, int, int]:
        if self.packing_strategy == "reduced_big_dependency":
            return (
                int(item["heavy_big_only_count"]),
                int(item["big_only_count"]),
                int(item["visit_count"]),
                -int(item["eligible_sum"]),
            )
        return (
            int(item["visit_count"]),
            int(item["heavy_big_only_count"]),
            int(item["big_only_count"]),
            -int(item["eligible_sum"]),
        )

    def _route_counts(self, routes: list[TypedRoute]) -> Counter[str]:
        return Counter(route.vehicle_type for route in routes)

    def _fuel_3000_free_count(self, route_counts: Counter[str]) -> int:
        return self.vehicle_by_name["fuel_3000"].vehicle_count - route_counts.get("fuel_3000", 0)

    def _route_has_heavy_big_only_unit(self, route: TypedRoute) -> bool:
        return any(
            self._is_heavy_big_only_unit(unit.weight, unit.volume, unit.eligible_vehicle_types)
            for unit in (self.unit_by_id[unit_id] for unit_id in route.unit_ids)
        )

    def _route_has_flexible_unit(self, route: TypedRoute) -> bool:
        return any(
            not self._is_heavy_big_only_unit(unit.weight, unit.volume, unit.eligible_vehicle_types)
            for unit in (self.unit_by_id[unit_id] for unit_id in route.unit_ids)
        )

    def _unit_is_heavy_big_only(self, unit_id: int) -> bool:
        unit = self.unit_by_id[unit_id]
        return self._is_heavy_big_only_unit(unit.weight, unit.volume, unit.eligible_vehicle_types)

    def _route_big_structure_metrics(self, route: TypedRoute) -> tuple[int, int, int, int, int, int, int]:
        if route.vehicle_type not in {"fuel_3000", "ev_3000"}:
            return 0, 0, 0, 0, 0, 0, 0
        flexible_unit_count = sum(1 for unit_id in route.unit_ids if not self._unit_is_heavy_big_only(unit_id))
        heavy_big_only_unit_count = len(route.unit_ids) - flexible_unit_count
        promotion_like_big_route_flag = int(
            len(route.unit_ids) >= 2
            and flexible_unit_count == len(route.unit_ids)
            and flexible_unit_count > 0
        )
        piggyback_big_route_flag = int(
            heavy_big_only_unit_count >= 1
            and 1 <= flexible_unit_count <= 1
        )
        blocking_big_flexible_route_flag = int(
            flexible_unit_count > 0
            and not promotion_like_big_route_flag
            and not piggyback_big_route_flag
        )
        blocking_big_flexible_unit_count = flexible_unit_count if blocking_big_flexible_route_flag else 0
        return (
            promotion_like_big_route_flag,
            piggyback_big_route_flag,
            blocking_big_flexible_route_flag,
            blocking_big_flexible_unit_count,
            1,
            flexible_unit_count,
            heavy_big_only_unit_count,
        )

    def _route_big_flexible_metrics(self, route: TypedRoute) -> tuple[int, int, int]:
        if route.vehicle_type not in {"fuel_3000", "ev_3000"}:
            return 0, 0, 0
        flexible_unit_count = sum(1 for unit_id in route.unit_ids if not self._unit_is_heavy_big_only(unit_id))
        return int(flexible_unit_count > 0), flexible_unit_count, 1

    def _route_metric_tuple(self, route: TypedRoute, route_eval: RouteEvaluation | None = None) -> tuple[int, int, int, int, int, float]:
        routes_with_flexible_on_big, flexible_units_on_big, big_route_count = self._route_big_flexible_metrics(route)
        if route_eval is None:
            route_eval = self.evaluate_route(route)
        return (
            routes_with_flexible_on_big,
            flexible_units_on_big,
            big_route_count,
            1,
            int(len(route.unit_ids) == 1),
            route_eval.best_cost,
        )

    def _aggregate_route_metric_tuple(
        self,
        routes: Iterable[TypedRoute],
    ) -> tuple[int, int, int, int, int, float]:
        metrics = [self._route_metric_tuple(route) for route in routes]
        return (
            sum(item[0] for item in metrics),
            sum(item[1] for item in metrics),
            sum(item[2] for item in metrics),
            sum(item[3] for item in metrics),
            sum(item[4] for item in metrics),
            float(sum(item[5] for item in metrics)),
        )

    def _release_metric_key(self, metric_tuple: tuple[int, int, int, int, int, float]) -> tuple[int, int, int, int, int, float]:
        return metric_tuple

    def _promotion_metric_key(self, metric_tuple: tuple[int, int, int, int, int, float]) -> tuple[int, int, float, int, int, int]:
        return (
            metric_tuple[3],
            metric_tuple[4],
            round(metric_tuple[5], 6),
            metric_tuple[0],
            metric_tuple[1],
            metric_tuple[2],
        )

    def _solution_big_route_diagnostics(self, routes: Iterable[TypedRoute]) -> dict[str, int]:
        route_metrics = [self._route_big_structure_metrics(route) for route in routes]
        return {
            "promotion_like_big_count": int(sum(item[0] for item in route_metrics)),
            "piggyback_big_count": int(sum(item[1] for item in route_metrics)),
            "blocking_big_flexible_count": int(sum(item[2] for item in route_metrics)),
            "blocking_big_flexible_unit_count": int(sum(item[3] for item in route_metrics)),
            "big_route_count": int(sum(item[4] for item in route_metrics)),
            "mixed_big_route_count": int(sum(int(item[5] > 0) for item in route_metrics)),
            "mixed_big_flexible_unit_count": int(sum(item[5] for item in route_metrics)),
            "heavy_big_only_unit_count": int(sum(item[6] for item in route_metrics)),
        }

    def _final_solution_structure_metrics(self, routes: list[TypedRoute]) -> tuple[int, int]:
        diagnostics = self._solution_big_route_diagnostics(routes)
        return (
            diagnostics["blocking_big_flexible_count"],
            diagnostics["blocking_big_flexible_unit_count"],
        )

    def _preserves_fuel_3000_reserve(
        self,
        route_counts: Counter[str],
        current_vehicle_type: str | None,
        candidate_vehicle_type: str,
        unit_is_heavy_big_only: bool,
    ) -> bool:
        if unit_is_heavy_big_only:
            return True
        projected_counts = route_counts.copy()
        if current_vehicle_type is not None:
            projected_counts[current_vehicle_type] -= 1
            if projected_counts[current_vehicle_type] <= 0:
                projected_counts.pop(current_vehicle_type, None)
        projected_counts[candidate_vehicle_type] += 1
        return self._fuel_3000_free_count(projected_counts) >= FUEL_3000_SEARCH_RESERVE

    def _choice_rank_key(
        self,
        route_counts: Counter[str],
        current_vehicle_type: str | None,
        candidate_vehicle_type: str,
        unit_is_heavy_big_only: bool,
        mixes_flexible_into_big: bool,
        delta_cost: float,
    ) -> tuple[int, int, float, tuple[float, float, int]]:
        if not self.enable_fuel_3000_reserve:
            return (
                0,
                0,
                round(delta_cost, 6),
                self.vehicle_size_rank[candidate_vehicle_type],
            )
        preserves_reserve = self._preserves_fuel_3000_reserve(
            route_counts=route_counts,
            current_vehicle_type=current_vehicle_type,
            candidate_vehicle_type=candidate_vehicle_type,
            unit_is_heavy_big_only=unit_is_heavy_big_only,
        )
        return (
            0 if preserves_reserve else 1,
            1 if mixes_flexible_into_big else 0,
            round(delta_cost, 6),
            self.vehicle_size_rank[candidate_vehicle_type],
        )

    def _build_customer_affinity(self) -> np.ndarray:
        affinity = np.zeros((len(self.customer_ids), len(self.customer_ids)), dtype=np.float64)
        tw_mid = {0: 0.0}
        for row in self.active_customer_df[["cust_id", "tw_start_min", "tw_end_min"]].itertuples(index=False):
            tw_mid[int(row.cust_id)] = (float(row.tw_start_min) + float(row.tw_end_min)) / 2.0
        for left in self.customer_ids:
            for right in self.customer_ids:
                if left == right:
                    continue
                left_idx = self.node_to_idx[left]
                right_idx = self.node_to_idx[right]
                distance = float(self.distance_matrix[left_idx, right_idx])
                time_gap = abs(tw_mid.get(left, 0.0) - tw_mid.get(right, 0.0)) / 60.0
                affinity[self.customer_index[left], self.customer_index[right]] = distance + 2.5 * time_gap
        return affinity

    def _time_window_overlap_minutes(self, left_unit_id: int, right_unit_id: int) -> float:
        left_unit = self.unit_by_id[left_unit_id]
        right_unit = self.unit_by_id[right_unit_id]
        return max(
            0.0,
            min(float(left_unit.tw_end_min), float(right_unit.tw_end_min))
            - max(float(left_unit.tw_start_min), float(right_unit.tw_start_min)),
        )

    def _average_time_window_overlap_minutes(self, unit_ids: Iterable[int]) -> float:
        unit_ids = tuple(unit_ids)
        if len(unit_ids) <= 1:
            return 0.0
        overlap_values = [
            self._time_window_overlap_minutes(left_unit_id, right_unit_id)
            for left_unit_id, right_unit_id in combinations(unit_ids, 2)
        ]
        if not overlap_values:
            return 0.0
        return float(sum(overlap_values) / len(overlap_values))

    def _average_customer_distance_km(self, unit_ids: Iterable[int]) -> float:
        customer_ids = tuple(self.unit_by_id[unit_id].orig_cust_id for unit_id in unit_ids)
        if len(customer_ids) <= 1:
            return 0.0
        distance_values = [
            self._distance_between(left_customer_id, right_customer_id)
            for left_customer_id, right_customer_id in combinations(customer_ids, 2)
        ]
        if not distance_values:
            return 0.0
        return float(sum(distance_values) / len(distance_values))

    def _build_customer_neighbors(self) -> dict[int, list[int]]:
        neighbors: dict[int, list[int]] = {}
        active_customers = [cust_id for cust_id in self.customer_ids if cust_id != 0]
        for cust_id in active_customers:
            left_idx = self.customer_index[cust_id]
            scored = [
                (
                    self.customer_affinity[left_idx, self.customer_index[other_id]],
                    other_id,
                )
                for other_id in active_customers
                if other_id != cust_id
            ]
            scored.sort(key=lambda item: item[0])
            neighbors[cust_id] = [other_id for _, other_id in scored[:CONSTRUCTION_NEIGHBOR_LIMIT]]
        neighbors[0] = active_customers[:]
        return neighbors

    def _eligible_vehicle_types_for_load(self, weight: float, volume: float) -> tuple[str, ...]:
        return tuple(
            vehicle.vehicle_type
            for vehicle in self.vehicles
            if weight <= vehicle.capacity_kg + 1e-6 and volume <= vehicle.capacity_m3 + 1e-6
        )

    def _split_order_into_fragments(self, order_id: int, weight: float, volume: float) -> list[OrderFragment]:
        if weight <= MAX_SINGLE_WEIGHT + 1e-6 and volume <= MAX_SINGLE_VOLUME + 1e-6:
            return [OrderFragment(fragment_id=f"{order_id}-1", source_order_id=order_id, weight=weight, volume=volume)]
        split_count = max(math.ceil(weight / MAX_SINGLE_WEIGHT), math.ceil(volume / MAX_SINGLE_VOLUME))
        split_count = max(split_count, 1)
        fragments: list[OrderFragment] = []
        residual_weight = weight
        residual_volume = volume
        for idx in range(split_count):
            if idx == split_count - 1:
                frag_weight = residual_weight
                frag_volume = residual_volume
            else:
                frag_weight = weight / split_count
                frag_volume = volume / split_count
            residual_weight -= frag_weight
            residual_volume -= frag_volume
            fragments.append(
                OrderFragment(
                    fragment_id=f"{order_id}-{idx + 1}",
                    source_order_id=order_id,
                    weight=float(frag_weight),
                    volume=float(frag_volume),
                )
            )
        return fragments

    @staticmethod
    def _is_big_only_unit(eligible_vehicle_types: tuple[str, ...]) -> bool:
        return set(eligible_vehicle_types) <= {"fuel_3000", "ev_3000"}

    def _is_heavy_big_only_unit(self, weight: float, volume: float, eligible_vehicle_types: tuple[str, ...]) -> bool:
        return self._is_big_only_unit(eligible_vehicle_types) and (weight > 1500.0 + 1e-6 or volume > 10.8 + 1e-6)

    def _packing_score(self, bins: list[dict[str, object]]) -> tuple[float, float]:
        non_empty = [bin_state for bin_state in bins if bin_state["fragments"]]
        eligible_sum = sum(len(bin_state["eligible_vehicle_types"]) for bin_state in non_empty)
        slack = sum(
            (MAX_SINGLE_WEIGHT - float(bin_state["weight"])) / MAX_SINGLE_WEIGHT
            + (MAX_SINGLE_VOLUME - float(bin_state["volume"])) / MAX_SINGLE_VOLUME
            for bin_state in non_empty
        )
        return (-float(eligible_sum), float(slack))

    def _greedy_pack_items(
        self,
        fragments: list[OrderFragment],
        bin_count: int,
        rng: random.Random,
    ) -> list[dict[str, object]] | None:
        bins = [
            {
                "weight": 0.0,
                "volume": 0.0,
                "fragments": [],
                "order_ids": [],
                "eligible_vehicle_types": tuple(vehicle.vehicle_type for vehicle in self.vehicles),
            }
            for _ in range(bin_count)
        ]
        order = list(fragments)
        order.sort(
            key=lambda fragment: (
                max(fragment.weight / MAX_SINGLE_WEIGHT, fragment.volume / MAX_SINGLE_VOLUME),
                fragment.weight,
                fragment.volume,
                rng.random(),
            ),
            reverse=True,
        )
        for fragment in order:
            options: list[tuple[tuple[float, bool, float, float], int, tuple[str, ...]]] = []
            for idx, bin_state in enumerate(bins):
                new_weight = float(bin_state["weight"]) + fragment.weight
                new_volume = float(bin_state["volume"]) + fragment.volume
                eligible = self._eligible_vehicle_types_for_load(new_weight, new_volume)
                if not eligible:
                    continue
                slack_weight = (MAX_SINGLE_WEIGHT - new_weight) / MAX_SINGLE_WEIGHT
                slack_volume = (MAX_SINGLE_VOLUME - new_volume) / MAX_SINGLE_VOLUME
                score = (
                    -float(len(eligible)),
                    len(bin_state["fragments"]) == 0,
                    max(slack_weight, slack_volume),
                    slack_weight + slack_volume,
                )
                options.append((score, idx, eligible))
            if not options:
                return None
            best_score = min(option[0] for option in options)
            shortlisted = [option for option in options if option[0] == best_score]
            _, best_idx, eligible = rng.choice(shortlisted)
            bins[best_idx]["weight"] = float(bins[best_idx]["weight"]) + fragment.weight
            bins[best_idx]["volume"] = float(bins[best_idx]["volume"]) + fragment.volume
            bins[best_idx]["fragments"].append(fragment)
            bins[best_idx]["order_ids"].append(fragment.source_order_id)
            bins[best_idx]["eligible_vehicle_types"] = eligible
        return [bin_state for bin_state in bins if bin_state["fragments"]]

    def _pack_customer_fragments_for_count(
        self,
        cust_id: int,
        fragments: list[OrderFragment],
        bin_count: int,
    ) -> list[dict[str, object]] | None:
        best_bins: list[dict[str, object]] | None = None
        best_score: tuple[float, float] | None = None
        for attempt in range(PACKING_ATTEMPTS):
            rng = random.Random(cust_id * 1000 + bin_count * 100 + attempt)
            candidate = self._greedy_pack_items(fragments, bin_count, rng)
            if candidate is None:
                continue
            score = self._packing_score(candidate)
            if best_bins is None or score < best_score:
                best_bins = candidate
                best_score = score
        return best_bins

    def _build_service_units(self) -> tuple[list[ServiceUnit], list[dict[str, object]]]:
        service_units: list[ServiceUnit] = []
        split_rows: list[dict[str, object]] = []
        unit_id = 0
        mandatory_customers: list[dict[str, object]] = []
        normal_heavy_big_only_count = 0
        for row in self.active_customer_df.itertuples(index=False):
            cust_id = int(row.cust_id)
            total_weight = float(row.total_weight)
            total_volume = float(row.total_volume)
            eligible_vehicle_types = self._eligible_vehicle_types_for_load(total_weight, total_volume)
            customer_orders = self.orders.loc[self.orders["cust_id"] == cust_id].copy()
            source_order_ids = tuple(sorted(customer_orders["order_id"].astype(int).tolist()))
            if eligible_vehicle_types:
                service_units.append(
                    ServiceUnit(
                        unit_id=unit_id,
                        orig_cust_id=cust_id,
                        unit_type="normal",
                        visit_index=1,
                        weight=total_weight,
                        volume=total_volume,
                        tw_start_min=int(row.tw_start_min),
                        tw_end_min=int(row.tw_end_min),
                        x_km=float(row.x_km),
                        y_km=float(row.y_km),
                        eligible_vehicle_types=eligible_vehicle_types,
                        source_order_ids=source_order_ids,
                        required_visit_count=1,
                    )
                )
                if self._is_heavy_big_only_unit(total_weight, total_volume, eligible_vehicle_types):
                    normal_heavy_big_only_count += 1
                unit_id += 1
                continue

            fragments: list[OrderFragment] = []
            for order in customer_orders.itertuples(index=False):
                fragments.extend(
                    self._split_order_into_fragments(
                        order_id=int(order.order_id),
                        weight=float(order.weight),
                        volume=float(order.volume),
                    )
                )
            lower_bound = max(math.ceil(total_weight / MAX_SINGLE_WEIGHT), math.ceil(total_volume / MAX_SINGLE_VOLUME))
            upper_bound = min(
                len(fragments),
                max(
                    lower_bound + 6,
                    math.ceil(total_weight / 1500.0),
                    math.ceil(total_volume / 10.8),
                    math.ceil(total_weight / 1250.0),
                    math.ceil(total_volume / 8.5),
                ),
            )
            candidate_packings: list[dict[str, object]] = []
            for bin_count in range(lower_bound, upper_bound + 1):
                packed_bins = self._pack_customer_fragments_for_count(cust_id, fragments, bin_count)
                if packed_bins is None:
                    continue
                heavy_big_only_count = 0
                big_only_count = 0
                eligible_sum = 0
                for bin_state in packed_bins:
                    eligible = tuple(bin_state["eligible_vehicle_types"])
                    eligible_sum += len(eligible)
                    if self._is_big_only_unit(eligible):
                        big_only_count += 1
                    if self._is_heavy_big_only_unit(float(bin_state["weight"]), float(bin_state["volume"]), eligible):
                        heavy_big_only_count += 1
                candidate_packings.append(
                    {
                        "visit_count": len(packed_bins),
                        "heavy_big_only_count": heavy_big_only_count,
                        "big_only_count": big_only_count,
                        "eligible_sum": eligible_sum,
                        "bins": packed_bins,
                    }
                )
            if not candidate_packings:
                raise RuntimeError(f"Unable to build packing candidates for mandatory split customer {cust_id}")
            candidate_packings.sort(key=self._packing_candidate_sort_key)
            mandatory_customers.append(
                {
                    "cust_id": cust_id,
                    "row": row,
                    "candidates": candidate_packings,
                    "selected_index": 0,
                    "source_order_ids": source_order_ids,
                }
            )

        big_vehicle_inventory = sum(vehicle.vehicle_count for vehicle in self.vehicles if vehicle.capacity_kg >= 3000.0)
        heavy_big_only_capacity = max(big_vehicle_inventory - BIG_VEHICLE_RESERVE - normal_heavy_big_only_count, 0)
        current_heavy_big_only = sum(
            int(customer["candidates"][customer["selected_index"]]["heavy_big_only_count"])
            for customer in mandatory_customers
        )
        while current_heavy_big_only > heavy_big_only_capacity:
            best_customer_idx = None
            best_candidate_idx = None
            best_score = None
            for customer_idx, customer in enumerate(mandatory_customers):
                current_candidate = customer["candidates"][customer["selected_index"]]
                for candidate_idx in range(customer["selected_index"] + 1, len(customer["candidates"])):
                    next_candidate = customer["candidates"][candidate_idx]
                    heavy_reduction = int(current_candidate["heavy_big_only_count"]) - int(next_candidate["heavy_big_only_count"])
                    extra_visits = int(next_candidate["visit_count"]) - int(current_candidate["visit_count"])
                    if heavy_reduction <= 0 or extra_visits <= 0:
                        continue
                    score = (
                        -(heavy_reduction / extra_visits),
                        extra_visits,
                        int(next_candidate["heavy_big_only_count"]),
                        int(next_candidate["big_only_count"]),
                        -int(next_candidate["eligible_sum"]),
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_customer_idx = customer_idx
                        best_candidate_idx = candidate_idx
            if best_customer_idx is None or best_candidate_idx is None:
                break
            current_candidate = mandatory_customers[best_customer_idx]["candidates"][mandatory_customers[best_customer_idx]["selected_index"]]
            next_candidate = mandatory_customers[best_customer_idx]["candidates"][best_candidate_idx]
            current_heavy_big_only += int(next_candidate["heavy_big_only_count"]) - int(current_candidate["heavy_big_only_count"])
            mandatory_customers[best_customer_idx]["selected_index"] = best_candidate_idx

        for customer in mandatory_customers:
            row = customer["row"]
            cust_id = int(row.cust_id)
            selected = customer["candidates"][customer["selected_index"]]
            packed_bins = selected["bins"]
            required_visit_count = len(packed_bins)
            for visit_index, bin_state in enumerate(packed_bins, start=1):
                unit_weight = float(bin_state["weight"])
                unit_volume = float(bin_state["volume"])
                eligible = tuple(bin_state["eligible_vehicle_types"])
                order_ids = tuple(sorted(set(int(order_id) for order_id in bin_state["order_ids"])))
                service_units.append(
                    ServiceUnit(
                        unit_id=unit_id,
                        orig_cust_id=cust_id,
                        unit_type="mandatory_split",
                        visit_index=visit_index,
                        weight=unit_weight,
                        volume=unit_volume,
                        tw_start_min=int(row.tw_start_min),
                        tw_end_min=int(row.tw_end_min),
                        x_km=float(row.x_km),
                        y_km=float(row.y_km),
                        eligible_vehicle_types=eligible,
                        source_order_ids=order_ids,
                        required_visit_count=required_visit_count,
                    )
                )
                split_rows.append(
                    {
                        "cust_id": cust_id,
                        "unit_id": unit_id,
                        "visit_index": visit_index,
                        "required_visit_count": required_visit_count,
                        "unit_weight_kg": round(unit_weight, 6),
                        "unit_volume_m3": round(unit_volume, 6),
                        "eligible_vehicle_types": ",".join(eligible),
                        "source_order_ids": ",".join(str(order_id) for order_id in order_ids),
                    }
                )
                unit_id += 1
        self.service_unit_summary = {
            "mandatory_split_customer_count": len(mandatory_customers),
            "normal_heavy_big_only_count": normal_heavy_big_only_count,
            "mandatory_heavy_big_only_count": current_heavy_big_only,
            "heavy_big_only_count": normal_heavy_big_only_count + current_heavy_big_only,
            "big_vehicle_inventory": big_vehicle_inventory,
            "big_vehicle_reserve": BIG_VEHICLE_RESERVE,
            "heavy_big_only_capacity": heavy_big_only_capacity,
        }
        return service_units, split_rows

    def _load_or_build_day_night_arc_lookups(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_path = self.energy_lookup_dir / "question1_day_night_arc_lookup_v3.npz"
        if cache_path.exists():
            try:
                arrays = np.load(cache_path)
                return arrays["travel_time_minutes"], arrays["base_fuel_l"], arrays["base_electric_kwh"]
            except (EOFError, ValueError, OSError):
                cache_path.unlink(missing_ok=True)

        distance_matrix = self.distance_matrix.astype(np.float64)
        diag_mask = np.eye(distance_matrix.shape[0], dtype=bool)
        travel_time = np.full((DAY_END_MIN + 1, distance_matrix.shape[0], distance_matrix.shape[1]), np.nan, dtype=np.float32)
        base_fuel = np.full_like(travel_time, np.nan)
        base_electric = np.full_like(base_fuel, np.nan)

        for depart_min in range(DAY_END_MIN + 1):
            remaining = distance_matrix.copy()
            travel = np.zeros_like(distance_matrix, dtype=np.float64)
            fuel = np.zeros_like(distance_matrix, dtype=np.float64)
            electric = np.zeros_like(distance_matrix, dtype=np.float64)
            segment_index = int(np.searchsorted(self.segment_ends, depart_min, side="right"))
            for idx in range(segment_index, len(self.segment_starts)):
                unfinished = remaining > 1e-9
                if not np.any(unfinished):
                    break
                segment_start = depart_min if idx == segment_index else self.segment_starts[idx]
                available_min = int(self.segment_ends[idx] - segment_start)
                if available_min <= 0:
                    continue
                speed = self.segment_speeds[idx]
                distance_capacity = speed * available_min / 60.0
                finishes_here = unfinished & (remaining <= distance_capacity + 1e-9)
                if np.any(finishes_here):
                    travelled = remaining[finishes_here]
                    travel[finishes_here] += travelled / speed * 60.0
                    fuel[finishes_here] += travelled / 100.0 * self.segment_fpk[idx]
                    electric[finishes_here] += travelled / 100.0 * self.segment_epk[idx]
                    remaining[finishes_here] = 0.0
                continue_to_next = unfinished & ~finishes_here
                if np.any(continue_to_next):
                    travel[continue_to_next] += available_min
                    fuel[continue_to_next] += distance_capacity / 100.0 * self.segment_fpk[idx]
                    electric[continue_to_next] += distance_capacity / 100.0 * self.segment_epk[idx]
                    remaining[continue_to_next] -= distance_capacity

            unfinished = remaining > 1e-9
            if np.any(unfinished):
                travel[unfinished] += remaining[unfinished] / self.after_hours_speed * 60.0
                fuel[unfinished] += remaining[unfinished] / 100.0 * self.after_hours_fpk
                electric[unfinished] += remaining[unfinished] / 100.0 * self.after_hours_epk

            raw_slice = self.raw_travel_time_lookup[depart_min].astype(np.float64)
            raw_finite = np.isfinite(raw_slice)
            travel[raw_finite] = raw_slice[raw_finite]

            travel[diag_mask] = 0.0
            fuel[diag_mask] = 0.0
            electric[diag_mask] = 0.0
            travel_time[depart_min] = travel.astype(np.float32)
            base_fuel[depart_min] = fuel.astype(np.float32)
            base_electric[depart_min] = electric.astype(np.float32)

        np.savez_compressed(
            cache_path,
            travel_time_minutes=travel_time,
            base_fuel_l=base_fuel,
            base_electric_kwh=base_electric,
        )
        return travel_time, base_fuel, base_electric

    @staticmethod
    def _minutes_to_hhmm(minutes: float) -> str:
        total_minutes = int(round(minutes))
        hour = 8 + total_minutes // 60
        minute = total_minutes % 60
        return f"{hour:02d}:{minute:02d}"

    @staticmethod
    def _safe_exp(value: float) -> float:
        return math.exp(max(min(value, 60.0), -60.0))

    def _distance_between(self, origin_id: int, dest_id: int) -> float:
        return float(self.distance_matrix[self.node_to_idx[origin_id], self.node_to_idx[dest_id]])

    def _after_hours_full_values(self, origin_id: int, dest_id: int) -> tuple[float, float, float]:
        distance = self._distance_between(origin_id, dest_id)
        travel = distance / self.after_hours_speed * 60.0
        fuel = distance / 100.0 * self.after_hours_fpk
        electric = distance / 100.0 * self.after_hours_epk
        return travel, fuel, electric

    def _interpolate_metric(
        self,
        lookup_array: np.ndarray,
        origin_id: int,
        dest_id: int,
        departure_vec: np.ndarray,
        after_hours_value: float,
    ) -> np.ndarray:
        origin_idx = self.node_to_idx[origin_id]
        dest_idx = self.node_to_idx[dest_id]
        base_series = lookup_array[:, origin_idx, dest_idx]
        valid_departure = np.isfinite(departure_vec)
        result = np.full(departure_vec.shape, np.nan, dtype=np.float64)
        if np.any(valid_departure):
            day_mask = valid_departure & (departure_vec <= DAY_END_MIN)
            if np.any(day_mask):
                day_departure = departure_vec[day_mask]
                floor_idx = np.floor(day_departure).astype(np.int32)
                floor_idx = np.clip(floor_idx, 0, DAY_END_MIN)
                ceil_idx = np.clip(floor_idx + 1, 0, DAY_END_MIN)
                frac = np.clip(day_departure - floor_idx, 0.0, 1.0)
                left = base_series[floor_idx]
                right = base_series[ceil_idx]
                result[day_mask] = left + (right - left) * frac
            after_mask = valid_departure & (departure_vec > DAY_END_MIN)
            if np.any(after_mask):
                result[after_mask] = after_hours_value
        return result

    def _scalar_after_hours_distance(self, origin_id: int, dest_id: int, departure_min: float) -> float:
        distance = self._distance_between(origin_id, dest_id)
        if departure_min >= DAY_END_MIN:
            return distance
        remaining = distance
        current_time = departure_min
        segment_index = int(np.searchsorted(self.segment_ends, current_time, side="right"))
        for idx in range(segment_index, len(self.segment_starts)):
            if remaining <= 1e-9:
                return 0.0
            segment_start = current_time if idx == segment_index else self.segment_starts[idx]
            available_min = float(self.segment_ends[idx] - segment_start)
            if available_min <= 0:
                continue
            distance_capacity = self.segment_speeds[idx] * available_min / 60.0
            if remaining <= distance_capacity + 1e-9:
                return 0.0
            remaining -= distance_capacity
        return max(remaining, 0.0)

    def _route_customers(self, route: TypedRoute) -> set[int]:
        return {self.unit_by_id[unit_id].orig_cust_id for unit_id in route.unit_ids}

    def _route_affinity(self, route: TypedRoute, unit_id: int) -> float:
        unit = self.unit_by_id[unit_id]
        if not route.unit_ids:
            return 0.0
        cust_idx = self.customer_index[unit.orig_cust_id]
        return min(
            self.customer_affinity[cust_idx, self.customer_index[self.unit_by_id[other_id].orig_cust_id]]
            for other_id in route.unit_ids
        )

    def _route_key(self, route: TypedRoute) -> tuple[str, tuple[int, ...]]:
        return (route.vehicle_type, route.unit_ids)

    def evaluate_route(self, route: TypedRoute) -> RouteEvaluation:
        cache_key = self._route_key(route)
        cached = self.route_cache.get(cache_key)
        if cached is not None:
            return cached

        vehicle = self.vehicle_by_name[route.vehicle_type]
        units = tuple(self.unit_by_id[unit_id] for unit_id in route.unit_ids)
        total_weight = float(sum(unit.weight for unit in units))
        total_volume = float(sum(unit.volume for unit in units))
        customer_ids = [unit.orig_cust_id for unit in units]
        unique_customer_count = len(set(customer_ids))
        if (
            total_weight > vehicle.capacity_kg + 1e-6
            or total_volume > vehicle.capacity_m3 + 1e-6
            or unique_customer_count != len(customer_ids)
            or any(route.vehicle_type not in unit.eligible_vehicle_types for unit in units)
        ):
            evaluation = RouteEvaluation(
                vehicle_type=route.vehicle_type,
                unit_ids=route.unit_ids,
                feasible=False,
                total_weight=total_weight,
                total_volume=total_volume,
                best_cost=np.inf,
                best_start=-1,
                best_return=np.inf,
            )
            self.route_cache.set(cache_key, evaluation)
            return evaluation

        departure_vec = self.route_start_grid.copy()
        total_wait = np.zeros_like(departure_vec)
        total_late = np.zeros_like(departure_vec)
        total_energy = np.zeros_like(departure_vec)
        valid_mask = np.ones_like(departure_vec, dtype=bool)
        remaining_weight = total_weight
        previous_node = 0
        for unit in units:
            after_hours_travel, after_hours_fuel, after_hours_electric = self._after_hours_full_values(previous_node, unit.orig_cust_id)
            travel = self._interpolate_metric(
                self.travel_time_lookup,
                previous_node,
                unit.orig_cust_id,
                departure_vec,
                after_hours_travel,
            )
            base_energy_lookup = self.base_fuel_lookup if vehicle.power_type == "fuel" else self.base_electric_lookup
            base_energy = self._interpolate_metric(
                base_energy_lookup,
                previous_node,
                unit.orig_cust_id,
                departure_vec,
                after_hours_fuel if vehicle.power_type == "fuel" else after_hours_electric,
            )
            load_ratio = max(0.0, min(1.0, remaining_weight / vehicle.capacity_kg))
            load_multiplier = 1.0 + (0.40 if vehicle.power_type == "fuel" else 0.35) * load_ratio
            arrival = departure_vec + travel
            wait = np.maximum(unit.tw_start_min - arrival, 0.0)
            service_start = arrival + wait
            late = np.maximum(service_start - unit.tw_end_min, 0.0)
            departure_vec = service_start + self.service_time_min
            valid_mask &= np.isfinite(travel) & np.isfinite(base_energy) & np.isfinite(departure_vec)
            total_wait += np.where(valid_mask, wait, 0.0)
            total_late += np.where(valid_mask, late, 0.0)
            total_energy += np.where(valid_mask, base_energy * load_multiplier, 0.0)
            previous_node = unit.orig_cust_id
            remaining_weight -= unit.weight

        after_hours_travel_back, after_hours_fuel_back, after_hours_electric_back = self._after_hours_full_values(previous_node, 0)
        travel_back = self._interpolate_metric(
            self.travel_time_lookup,
            previous_node,
            0,
            departure_vec,
            after_hours_travel_back,
        )
        base_energy_lookup = self.base_fuel_lookup if vehicle.power_type == "fuel" else self.base_electric_lookup
        base_energy_back = self._interpolate_metric(
            base_energy_lookup,
            previous_node,
            0,
            departure_vec,
            after_hours_fuel_back if vehicle.power_type == "fuel" else after_hours_electric_back,
        )
        completion = departure_vec + travel_back
        valid_mask &= np.isfinite(travel_back) & np.isfinite(base_energy_back) & np.isfinite(completion)
        total_energy += np.where(valid_mask, base_energy_back, 0.0)

        energy_cost = total_energy * (FUEL_PRICE if vehicle.power_type == "fuel" else ELECTRICITY_PRICE)
        carbon_factor = FUEL_CARBON_FACTOR if vehicle.power_type == "fuel" else ELECTRICITY_CARBON_FACTOR
        carbon_cost = total_energy * carbon_factor * CARBON_COST
        total_cost = START_COST + energy_cost + carbon_cost + total_wait * WAIT_COST_PER_MIN + total_late * LATE_COST_PER_MIN
        total_cost = np.where(valid_mask, total_cost, np.inf)
        if not np.isfinite(total_cost).any():
            evaluation = RouteEvaluation(
                vehicle_type=route.vehicle_type,
                unit_ids=route.unit_ids,
                feasible=False,
                total_weight=total_weight,
                total_volume=total_volume,
                best_cost=np.inf,
                best_start=-1,
                best_return=np.inf,
            )
            self.route_cache.set(cache_key, evaluation)
            return evaluation

        best_start = int(np.nanargmin(total_cost))
        evaluation = RouteEvaluation(
            vehicle_type=route.vehicle_type,
            unit_ids=route.unit_ids,
            feasible=True,
            total_weight=total_weight,
            total_volume=total_volume,
            best_cost=float(total_cost[best_start]),
            best_start=best_start,
            best_return=float(completion[best_start]),
        )
        self.route_cache.set(cache_key, evaluation)
        return evaluation

    def _simulate_route_scalar(self, route: TypedRoute, route_index: int, vehicle_instance: int) -> AssignedRoute:
        evaluation = self.evaluate_route(route)
        if not evaluation.feasible:
            raise ValueError("Route is infeasible")
        vehicle = self.vehicle_by_name[route.vehicle_type]
        units = tuple(self.unit_by_id[unit_id] for unit_id in route.unit_ids)

        previous_node = 0
        current_departure = float(evaluation.best_start)
        remaining_weight = evaluation.total_weight
        remaining_volume = evaluation.total_volume
        total_wait = 0.0
        total_late = 0.0
        total_fuel = 0.0
        total_electricity = 0.0
        route_distance = 0.0
        after_hours_travel_km = 0.0
        after_hours_service_count = 0
        late_positive_stop_count = 0
        max_late_min = 0.0
        stop_rows: list[dict[str, object]] = []

        for stop_index, unit in enumerate(units, start=1):
            departure_array = np.array([current_departure], dtype=np.float64)
            after_hours_travel, after_hours_fuel, after_hours_electric = self._after_hours_full_values(previous_node, unit.orig_cust_id)
            travel = float(
                self._interpolate_metric(
                    self.travel_time_lookup,
                    previous_node,
                    unit.orig_cust_id,
                    departure_array,
                    after_hours_travel,
                )[0]
            )
            base_fuel = float(
                self._interpolate_metric(
                    self.base_fuel_lookup,
                    previous_node,
                    unit.orig_cust_id,
                    departure_array,
                    after_hours_fuel,
                )[0]
            )
            base_electric = float(
                self._interpolate_metric(
                    self.base_electric_lookup,
                    previous_node,
                    unit.orig_cust_id,
                    departure_array,
                    after_hours_electric,
                )[0]
            )
            arrival = current_departure + travel
            wait = max(unit.tw_start_min - arrival, 0.0)
            service_start = arrival + wait
            late = max(service_start - unit.tw_end_min, 0.0)
            service_end = service_start + self.service_time_min
            load_ratio = max(0.0, min(1.0, remaining_weight / vehicle.capacity_kg))
            load_multiplier = 1.0 + (0.40 if vehicle.power_type == "fuel" else 0.35) * load_ratio
            fuel = base_fuel * load_multiplier
            electricity = base_electric * load_multiplier
            total_wait += wait
            total_late += late
            total_fuel += fuel
            total_electricity += electricity
            route_distance += self._distance_between(previous_node, unit.orig_cust_id)
            after_hours_travel_km += self._scalar_after_hours_distance(previous_node, unit.orig_cust_id, current_departure)
            if service_start > DAY_END_MIN + 1e-9:
                after_hours_service_count += 1
            if late > 1e-9:
                late_positive_stop_count += 1
                max_late_min = max(max_late_min, late)
            stop_rows.append(
                {
                    "route_id": route_index,
                    "vehicle_type": route.vehicle_type,
                    "vehicle_instance": vehicle_instance,
                    "stop_index": stop_index,
                    "unit_id": unit.unit_id,
                    "orig_cust_id": unit.orig_cust_id,
                    "unit_type": unit.unit_type,
                    "visit_index": unit.visit_index,
                    "required_visit_count": unit.required_visit_count,
                    "source_order_ids": ",".join(str(order_id) for order_id in unit.source_order_ids),
                    "arrival_min": round(arrival, 1),
                    "arrival_hhmm": self._minutes_to_hhmm(arrival),
                    "service_start_min": round(service_start, 1),
                    "service_start_hhmm": self._minutes_to_hhmm(service_start),
                    "service_end_min": round(service_end, 1),
                    "service_end_hhmm": self._minutes_to_hhmm(service_end),
                    "waiting_min": round(wait, 1),
                    "late_min": round(late, 1),
                    "delivered_weight_kg": round(unit.weight, 6),
                    "delivered_volume_m3": round(unit.volume, 6),
                    "remaining_weight_after_stop_kg": round(max(remaining_weight - unit.weight, 0.0), 6),
                    "remaining_volume_after_stop_m3": round(max(remaining_volume - unit.volume, 0.0), 6),
                    "tw_start_min": unit.tw_start_min,
                    "tw_start_hhmm": self._minutes_to_hhmm(unit.tw_start_min),
                    "tw_end_min": unit.tw_end_min,
                    "tw_end_hhmm": self._minutes_to_hhmm(unit.tw_end_min),
                    "after_hours_service_flag": int(service_start > DAY_END_MIN + 1e-9),
                    "late_positive_flag": int(late > 1e-9),
                }
            )
            current_departure = service_end
            previous_node = unit.orig_cust_id
            remaining_weight -= unit.weight
            remaining_volume -= unit.volume

        departure_array = np.array([current_departure], dtype=np.float64)
        after_hours_travel_back, after_hours_fuel_back, after_hours_electric_back = self._after_hours_full_values(previous_node, 0)
        travel_back = float(
            self._interpolate_metric(
                self.travel_time_lookup,
                previous_node,
                0,
                departure_array,
                after_hours_travel_back,
            )[0]
        )
        base_fuel_back = float(
            self._interpolate_metric(
                self.base_fuel_lookup,
                previous_node,
                0,
                departure_array,
                after_hours_fuel_back,
            )[0]
        )
        base_electric_back = float(
            self._interpolate_metric(
                self.base_electric_lookup,
                previous_node,
                0,
                departure_array,
                after_hours_electric_back,
            )[0]
        )
        total_fuel += base_fuel_back
        total_electricity += base_electric_back
        route_distance += self._distance_between(previous_node, 0)
        after_hours_travel_km += self._scalar_after_hours_distance(previous_node, 0, current_departure)
        return_min = current_departure + travel_back
        after_hours_return_flag = return_min > DAY_END_MIN + 1e-9

        if vehicle.power_type == "fuel":
            energy_cost = total_fuel * FUEL_PRICE
            carbon_kg = total_fuel * FUEL_CARBON_FACTOR
        else:
            energy_cost = total_electricity * ELECTRICITY_PRICE
            carbon_kg = total_electricity * ELECTRICITY_CARBON_FACTOR
        carbon_cost = carbon_kg * CARBON_COST
        waiting_cost = total_wait * WAIT_COST_PER_MIN
        late_cost = total_late * LATE_COST_PER_MIN
        route_cost = START_COST + energy_cost + carbon_cost + waiting_cost + late_cost

        return AssignedRoute(
            route_index=route_index,
            vehicle_type=route.vehicle_type,
            power_type=vehicle.power_type,
            vehicle_instance=vehicle_instance,
            unit_ids=route.unit_ids,
            departure_min=round(evaluation.best_start, 1),
            return_min=round(return_min, 1),
            route_cost=float(route_cost),
            energy_cost=float(energy_cost),
            carbon_cost=float(carbon_cost),
            waiting_cost=float(waiting_cost),
            late_cost=float(late_cost),
            startup_cost=START_COST,
            total_wait_min=float(total_wait),
            total_late_min=float(total_late),
            total_fuel_l=float(total_fuel),
            total_electricity_kwh=float(total_electricity),
            route_distance_km=float(route_distance),
            after_hours_travel_km=float(after_hours_travel_km),
            after_hours_service_count=after_hours_service_count,
            after_hours_return_flag=after_hours_return_flag,
            late_positive_stop_count=late_positive_stop_count,
            max_late_min=float(max_late_min),
            units=units,
            stop_rows=stop_rows,
        )

    def evaluate_solution(self, solution: list[TypedRoute]) -> SolutionEvaluation | None:
        if any(not route.unit_ids for route in solution):
            solution = [route for route in solution if route.unit_ids]
        vehicle_type_usage = Counter(route.vehicle_type for route in solution)
        for vehicle_type, count in vehicle_type_usage.items():
            if count > self.vehicle_by_name[vehicle_type].vehicle_count:
                return None

        assigned_routes: list[AssignedRoute] = []
        instance_counter = Counter[str]()
        for route_index, route in enumerate(solution, start=1):
            evaluation = self.evaluate_route(route)
            if not evaluation.feasible:
                return None
            instance_counter[route.vehicle_type] += 1
            assigned_routes.append(
                self._simulate_route_scalar(
                    route=route,
                    route_index=route_index,
                    vehicle_instance=instance_counter[route.vehicle_type],
                )
            )

        total_cost = float(sum(route.route_cost for route in assigned_routes))
        total_energy_cost = float(sum(route.energy_cost for route in assigned_routes))
        total_carbon_cost = float(sum(route.carbon_cost for route in assigned_routes))
        total_waiting_cost = float(sum(route.waiting_cost for route in assigned_routes))
        total_late_cost = float(sum(route.late_cost for route in assigned_routes))
        total_late_min = float(sum(route.total_late_min for route in assigned_routes))
        total_startup_cost = float(sum(route.startup_cost for route in assigned_routes))
        total_fuel_l = float(sum(route.total_fuel_l for route in assigned_routes))
        total_electricity_kwh = float(sum(route.total_electricity_kwh for route in assigned_routes))
        total_carbon_kg = total_fuel_l * FUEL_CARBON_FACTOR + total_electricity_kwh * ELECTRICITY_CARBON_FACTOR
        total_distance_km = float(sum(route.route_distance_km for route in assigned_routes))
        late_positive_stops = int(sum(route.late_positive_stop_count for route in assigned_routes))
        max_late_min = float(max((route.max_late_min for route in assigned_routes), default=0.0))
        latest_return_min = float(max((route.return_min for route in assigned_routes), default=0.0))
        after_hours_service_count = int(sum(route.after_hours_service_count for route in assigned_routes))
        after_hours_return_count = int(sum(int(route.after_hours_return_flag) for route in assigned_routes))
        after_hours_travel_km = float(sum(route.after_hours_travel_km for route in assigned_routes))

        customer_route_map: dict[int, set[int]] = defaultdict(set)
        mandatory_split_customers: set[int] = set()
        normal_customers: set[int] = set()
        mandatory_split_visit_count = 0
        for assigned in assigned_routes:
            for unit in assigned.units:
                customer_route_map[unit.orig_cust_id].add(assigned.route_index)
                if unit.unit_type == "mandatory_split":
                    mandatory_split_customers.add(unit.orig_cust_id)
                    mandatory_split_visit_count += 1
                else:
                    normal_customers.add(unit.orig_cust_id)

        split_customer_count = int(sum(1 for route_ids in customer_route_map.values() if len(route_ids) > 1))
        single_stop_route_count = int(sum(1 for route in assigned_routes if len(route.unit_ids) == 1))
        two_stop_route_count = int(sum(1 for route in assigned_routes if len(route.unit_ids) == 2))
        three_plus_route_count = int(sum(1 for route in assigned_routes if len(route.unit_ids) >= 3))
        return SolutionEvaluation(
            total_cost=total_cost,
            total_energy_cost=total_energy_cost,
            total_carbon_cost=total_carbon_cost,
            total_waiting_cost=total_waiting_cost,
            total_late_cost=total_late_cost,
            total_late_min=total_late_min,
            total_startup_cost=total_startup_cost,
            total_fuel_l=total_fuel_l,
            total_electricity_kwh=total_electricity_kwh,
            total_carbon_kg=total_carbon_kg,
            total_distance_km=total_distance_km,
            route_count=len(assigned_routes),
            used_vehicle_count=len(assigned_routes),
            split_customer_count=split_customer_count,
            mandatory_split_customer_count=len(mandatory_split_customers),
            mandatory_split_visit_count=mandatory_split_visit_count,
            normal_customer_count=len(normal_customers),
            single_stop_route_count=single_stop_route_count,
            two_stop_route_count=two_stop_route_count,
            three_plus_route_count=three_plus_route_count,
            late_positive_stops=late_positive_stops,
            max_late_min=max_late_min,
            latest_return_min=latest_return_min,
            after_hours_service_count=after_hours_service_count,
            after_hours_return_count=after_hours_return_count,
            after_hours_travel_km=after_hours_travel_km,
            vehicle_type_usage=dict(vehicle_type_usage),
            assigned_routes=assigned_routes,
        )

    def _solution_rank_key(self, solution_eval: SolutionEvaluation) -> tuple[float, int, int, int, float, float]:
        structural_split_gap = abs(
            solution_eval.split_customer_count - solution_eval.mandatory_split_customer_count
        )
        return (
            round(solution_eval.total_cost, 6),
            structural_split_gap,
            solution_eval.route_count,
            solution_eval.single_stop_route_count,
            round(solution_eval.total_late_min, 6),
            round(solution_eval.total_carbon_kg, 6),
        )

    def _unit_priority_key(self, unit_id: int) -> tuple[int, int, float, int, int]:
        unit = self.unit_by_id[unit_id]
        normalized_size = max(unit.weight / MAX_SINGLE_WEIGHT, unit.volume / MAX_SINGLE_VOLUME)
        return (
            len(unit.eligible_vehicle_types),
            int(unit.tw_end_min),
            -normalized_size,
            unit.orig_cust_id,
            unit.visit_index,
        )

    def _candidate_route_indices_for_unit(self, routes: list[TypedRoute], unit_id: int) -> list[int]:
        unit = self.unit_by_id[unit_id]
        priority_customers = {unit.orig_cust_id, *self.customer_neighbors.get(unit.orig_cust_id, [])}
        scored = []
        for idx, route in enumerate(routes):
            route_customers = self._route_customers(route)
            boost = 0 if route_customers & priority_customers else 1
            affinity = self._route_affinity(route, unit_id)
            scored.append((boost, affinity, len(route.unit_ids), idx))
        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [idx for _, _, _, idx in scored[: max(1, min(len(routes), self.top_route_candidates, REPAIR_ROUTE_SHORTLIST))]]

    def _choose_best_new_route(self, unit_id: int, route_counts: Counter[str]) -> tuple[TypedRoute, RouteEvaluation] | None:
        unit = self.unit_by_id[unit_id]
        unit_is_heavy_big_only = self._is_heavy_big_only_unit(unit.weight, unit.volume, unit.eligible_vehicle_types)
        options: list[tuple[tuple[int, int, float, tuple[float, float, int]], TypedRoute, RouteEvaluation]] = []
        for vehicle_type in unit.eligible_vehicle_types:
            if route_counts[vehicle_type] >= self.vehicle_by_name[vehicle_type].vehicle_count:
                continue
            candidate_route = TypedRoute(vehicle_type=vehicle_type, unit_ids=(unit_id,))
            candidate_eval = self.evaluate_route(candidate_route)
            if candidate_eval.feasible:
                rank_key = self._choice_rank_key(
                    route_counts=route_counts,
                    current_vehicle_type=None,
                    candidate_vehicle_type=vehicle_type,
                    unit_is_heavy_big_only=unit_is_heavy_big_only,
                    mixes_flexible_into_big=(vehicle_type == "fuel_3000" and not unit_is_heavy_big_only),
                    delta_cost=candidate_eval.best_cost,
                )
                options.append((rank_key, candidate_route, candidate_eval))
        if not options:
            return None
        options.sort(key=lambda item: item[0])
        _, best_route, best_eval = options[0]
        return best_route, best_eval

    def _insert_unit_best(self, routes: list[TypedRoute], unit_id: int) -> list[TypedRoute] | None:
        unit = self.unit_by_id[unit_id]
        route_counts = self._route_counts(routes)
        unit_is_heavy_big_only = self._is_heavy_big_only_unit(unit.weight, unit.volume, unit.eligible_vehicle_types)
        best_rank: tuple[int, int, float, tuple[float, float, int]] | None = None
        best_routes: list[TypedRoute] | None = None
        candidate_groups = [self._candidate_route_indices_for_unit(routes, unit_id) if routes else []]
        if routes:
            fallback_indices = [idx for idx in range(len(routes)) if idx not in set(candidate_groups[0])]
            candidate_groups.append(fallback_indices)
        for candidate_indices in candidate_groups:
            for route_idx in candidate_indices:
                route = routes[route_idx]
                if route.vehicle_type not in unit.eligible_vehicle_types:
                    continue
                if unit.orig_cust_id in self._route_customers(route):
                    continue
                base_eval = self.evaluate_route(route)
                if not base_eval.feasible:
                    continue
                for position in range(len(route.unit_ids) + 1):
                    new_unit_ids = route.unit_ids[:position] + (unit_id,) + route.unit_ids[position:]
                    candidate_route = TypedRoute(vehicle_type=route.vehicle_type, unit_ids=new_unit_ids)
                    candidate_eval = self.evaluate_route(candidate_route)
                    if not candidate_eval.feasible:
                        continue
                    delta = candidate_eval.best_cost - base_eval.best_cost
                    mixes_flexible_into_big = (
                        route.vehicle_type == "fuel_3000"
                        and not unit_is_heavy_big_only
                        and self._route_has_flexible_unit(candidate_route)
                    )
                    rank_key = self._choice_rank_key(
                        route_counts=route_counts,
                        current_vehicle_type=route.vehicle_type,
                        candidate_vehicle_type=route.vehicle_type,
                        unit_is_heavy_big_only=unit_is_heavy_big_only,
                        mixes_flexible_into_big=mixes_flexible_into_big,
                        delta_cost=delta,
                    )
                    if best_rank is None or rank_key < best_rank:
                        updated = list(routes)
                        updated[route_idx] = candidate_route
                        best_rank = rank_key
                        best_routes = updated
            if best_routes is not None:
                break
        new_route_choice = self._choose_best_new_route(unit_id, route_counts)
        if new_route_choice is not None:
            new_route, new_eval = new_route_choice
            new_route_rank = self._choice_rank_key(
                route_counts=route_counts,
                current_vehicle_type=None,
                candidate_vehicle_type=new_route.vehicle_type,
                unit_is_heavy_big_only=unit_is_heavy_big_only,
                mixes_flexible_into_big=(new_route.vehicle_type == "fuel_3000" and not unit_is_heavy_big_only),
                delta_cost=new_eval.best_cost,
            )
            if best_rank is None or new_route_rank < best_rank:
                best_routes = list(routes) + [new_route]
        return best_routes

    def _build_initial_solution(self, rng: random.Random) -> list[TypedRoute]:
        ordered_units = sorted(
            self.active_unit_ids,
            key=lambda unit_id: (*self._unit_priority_key(unit_id), rng.random()),
        )
        routes: list[TypedRoute] = []
        for unit_id in ordered_units:
            updated_routes = self._insert_unit_best(routes, unit_id)
            if updated_routes is None:
                raise RuntimeError(f"Unable to insert service unit {unit_id}")
            routes = updated_routes
        return routes

    def _flatten_solution(self, solution: list[TypedRoute]) -> list[int]:
        return [unit_id for route in solution for unit_id in route.unit_ids]

    def _random_remove(self, solution: list[TypedRoute], q_remove: int, rng: random.Random) -> tuple[list[TypedRoute], list[int]]:
        flat = self._flatten_solution(solution)
        remove_ids = set(rng.sample(flat, min(q_remove, len(flat))))
        partial_routes = []
        removed = []
        for route in solution:
            kept = tuple(unit_id for unit_id in route.unit_ids if unit_id not in remove_ids)
            removed.extend(unit_id for unit_id in route.unit_ids if unit_id in remove_ids)
            if kept:
                partial_routes.append(TypedRoute(route.vehicle_type, kept))
        return partial_routes, removed

    def _worst_cost_remove(self, solution_eval: SolutionEvaluation, q_remove: int, rng: random.Random) -> tuple[list[TypedRoute], list[int]]:
        scored_units: list[tuple[float, int]] = []
        for route in solution_eval.assigned_routes:
            score = route.route_cost / max(len(route.unit_ids), 1) + route.total_late_min / max(len(route.unit_ids), 1)
            for unit_id in route.unit_ids:
                scored_units.append((score + rng.random() * 1e-3, unit_id))
        scored_units.sort(reverse=True)
        remove_ids = {unit_id for _, unit_id in scored_units[: min(q_remove, len(scored_units))]}
        partial_routes = []
        removed = []
        for route in solution_eval.assigned_routes:
            kept = tuple(unit_id for unit_id in route.unit_ids if unit_id not in remove_ids)
            removed.extend(unit_id for unit_id in route.unit_ids if unit_id in remove_ids)
            if kept:
                partial_routes.append(TypedRoute(route.vehicle_type, kept))
        return partial_routes, removed

    def _late_route_remove(self, solution_eval: SolutionEvaluation, q_remove: int) -> tuple[list[TypedRoute], list[int]]:
        ordered_routes = sorted(solution_eval.assigned_routes, key=lambda route: route.total_late_min, reverse=True)
        remove_ids: list[int] = []
        for route in ordered_routes:
            remove_ids.extend(route.unit_ids)
            if len(remove_ids) >= q_remove:
                break
        remove_set = set(remove_ids[:q_remove])
        partial_routes = []
        removed = []
        for route in solution_eval.assigned_routes:
            kept = tuple(unit_id for unit_id in route.unit_ids if unit_id not in remove_set)
            removed.extend(unit_id for unit_id in route.unit_ids if unit_id in remove_set)
            if kept:
                partial_routes.append(TypedRoute(route.vehicle_type, kept))
        return partial_routes, removed

    def _typed_route_merge_remove(self, solution_eval: SolutionEvaluation, q_remove: int) -> tuple[list[TypedRoute], list[int]]:
        def utilization(route: AssignedRoute) -> float:
            vehicle = self.vehicle_by_name[route.vehicle_type]
            used_weight = sum(unit.weight for unit in route.units)
            used_volume = sum(unit.volume for unit in route.units)
            return max(used_weight / vehicle.capacity_kg, used_volume / vehicle.capacity_m3)

        ordered_routes = sorted(
            solution_eval.assigned_routes,
            key=lambda route: (len(route.unit_ids), utilization(route), route.route_cost),
        )
        remove_ids: list[int] = []
        for route in ordered_routes:
            remove_ids.extend(route.unit_ids)
            if len(remove_ids) >= q_remove:
                break
        remove_set = set(remove_ids[:q_remove])
        partial_routes = []
        removed = []
        for route in solution_eval.assigned_routes:
            kept = tuple(unit_id for unit_id in route.unit_ids if unit_id not in remove_set)
            removed.extend(unit_id for unit_id in route.unit_ids if unit_id in remove_set)
            if kept:
                partial_routes.append(TypedRoute(route.vehicle_type, kept))
        return partial_routes, removed

    def _mandatory_split_cluster_remove(self, solution_eval: SolutionEvaluation, q_remove: int, rng: random.Random) -> tuple[list[TypedRoute], list[int]]:
        candidate_customers = []
        for route in solution_eval.assigned_routes:
            for unit in route.units:
                if unit.unit_type != "mandatory_split":
                    continue
                candidate_customers.append(unit.orig_cust_id)
        if not candidate_customers:
            return self._random_remove([TypedRoute(route.vehicle_type, route.unit_ids) for route in solution_eval.assigned_routes], q_remove, rng)
        customer_counts = Counter(candidate_customers)
        ordered_customers = [cust_id for cust_id, _ in customer_counts.most_common()]
        remove_ids: list[int] = []
        for cust_id in ordered_customers:
            for route in solution_eval.assigned_routes:
                for unit_id in route.unit_ids:
                    if self.unit_by_id[unit_id].orig_cust_id == cust_id:
                        remove_ids.append(unit_id)
            if len(remove_ids) >= q_remove:
                break
        remove_set = set(remove_ids[:q_remove])
        partial_routes = []
        removed = []
        for route in solution_eval.assigned_routes:
            kept = tuple(unit_id for unit_id in route.unit_ids if unit_id not in remove_set)
            removed.extend(unit_id for unit_id in route.unit_ids if unit_id in remove_set)
            if kept:
                partial_routes.append(TypedRoute(route.vehicle_type, kept))
        return partial_routes, removed

    def repair_solution(self, partial_routes: list[TypedRoute], removed_units: list[int]) -> list[TypedRoute] | None:
        routes = list(partial_routes)
        for unit_id in sorted(set(removed_units), key=self._unit_priority_key):
            updated_routes = self._insert_unit_best(routes, unit_id)
            if updated_routes is None:
                return None
            routes = updated_routes
        return [route for route in routes if route.unit_ids]

    def _try_reserve_repair_rebuild(self, routes: list[TypedRoute]) -> tuple[bool, list[TypedRoute]]:
        route_counts = self._route_counts(routes)
        if self._fuel_3000_free_count(route_counts) >= FUEL_3000_SEARCH_RESERVE:
            return False, routes
        partial_routes: list[TypedRoute] = []
        removed_units: list[int] = []
        for route in routes:
            if route.vehicle_type == "fuel_3000" and self._route_has_flexible_unit(route):
                removed_units.extend(
                    unit_id
                    for unit_id in route.unit_ids
                    if not self._is_heavy_big_only_unit(
                        self.unit_by_id[unit_id].weight,
                        self.unit_by_id[unit_id].volume,
                        self.unit_by_id[unit_id].eligible_vehicle_types,
                    )
                )
                kept_unit_ids = tuple(
                    unit_id
                    for unit_id in route.unit_ids
                    if self._is_heavy_big_only_unit(
                        self.unit_by_id[unit_id].weight,
                        self.unit_by_id[unit_id].volume,
                        self.unit_by_id[unit_id].eligible_vehicle_types,
                    )
                )
                if kept_unit_ids:
                    partial_routes.append(TypedRoute(route.vehicle_type, kept_unit_ids))
            else:
                partial_routes.append(route)
        if not removed_units:
            return False, routes
        self.enable_fuel_3000_reserve = True
        try:
            rebuilt = self.repair_solution(partial_routes, removed_units)
        finally:
            self.enable_fuel_3000_reserve = False
        if rebuilt is None:
            return False, routes
        rebuilt_counts = self._route_counts(rebuilt)
        if self._fuel_3000_free_count(rebuilt_counts) >= FUEL_3000_SEARCH_RESERVE:
            return True, rebuilt
        return False, routes

    def _single_stop_merge_candidates(
        self,
        routes: list[TypedRoute],
    ) -> tuple[list[dict[str, object]], int, int]:
        route_counts = self._route_counts(routes)
        single_routes = [
            (idx, route)
            for idx, route in enumerate(routes)
            if len(route.unit_ids) == 1 and route.vehicle_type in {"fuel_1500", "fuel_1250", "ev_1250"}
        ]
        candidates: list[dict[str, object]] = []
        feasible_pair_count = 0
        inventory_blocked_pair_count = 0
        for left_pos, (left_idx, left_route) in enumerate(single_routes):
            left_unit = self.unit_by_id[left_route.unit_ids[0]]
            left_eval = self.evaluate_route(left_route)
            for right_idx, right_route in single_routes[left_pos + 1 :]:
                right_unit = self.unit_by_id[right_route.unit_ids[0]]
                if left_unit.orig_cust_id == right_unit.orig_cust_id:
                    continue
                shared_vehicle_types = set(left_unit.eligible_vehicle_types) & set(right_unit.eligible_vehicle_types)
                if "fuel_3000" not in shared_vehicle_types:
                    continue
                best_candidate: dict[str, object] | None = None
                for unit_ids in ((left_route.unit_ids[0], right_route.unit_ids[0]), (right_route.unit_ids[0], left_route.unit_ids[0])):
                    candidate_route = TypedRoute(vehicle_type="fuel_3000", unit_ids=unit_ids)
                    candidate_eval = self.evaluate_route(candidate_route)
                    if not candidate_eval.feasible:
                        continue
                    feasible_pair_count += 1
                    projected_count = route_counts["fuel_3000"] + 1
                    inventory_ok = projected_count <= self.vehicle_by_name["fuel_3000"].vehicle_count
                    if not inventory_ok:
                        inventory_blocked_pair_count += 1
                    separate_cost = left_eval.best_cost + self.evaluate_route(right_route).best_cost
                    saving = separate_cost - candidate_eval.best_cost
                    candidate = {
                        "left_route_idx": left_idx,
                        "right_route_idx": right_idx,
                        "left_route_id": left_idx + 1,
                        "right_route_id": right_idx + 1,
                        "left_unit_id": left_route.unit_ids[0],
                        "right_unit_id": right_route.unit_ids[0],
                        "left_customer": left_unit.orig_cust_id,
                        "right_customer": right_unit.orig_cust_id,
                        "vehicle_type": "fuel_3000",
                        "merged_unit_ids": unit_ids,
                        "separate_cost": separate_cost,
                        "merged_cost": candidate_eval.best_cost,
                        "saving": saving,
                        "distance_km": self._distance_between(left_unit.orig_cust_id, right_unit.orig_cust_id),
                        "tw_gap_min": max(
                            0.0,
                            max(left_unit.tw_start_min, right_unit.tw_start_min)
                            - min(left_unit.tw_end_min, right_unit.tw_end_min),
                        ),
                        "inventory_ok": inventory_ok,
                    }
                    if best_candidate is None or (
                        candidate["merged_cost"],
                        candidate["distance_km"],
                        candidate["tw_gap_min"],
                    ) < (
                        best_candidate["merged_cost"],
                        best_candidate["distance_km"],
                        best_candidate["tw_gap_min"],
                    ):
                        best_candidate = candidate
                if best_candidate is not None:
                    candidates.append(best_candidate)
        candidates.sort(
            key=lambda item: (
                0 if bool(item["inventory_ok"]) else 1,
                -float(item["saving"]),
                float(item["distance_km"]),
                float(item["tw_gap_min"]),
            )
        )
        return candidates, feasible_pair_count, inventory_blocked_pair_count

    def _apply_batch_small_to_fuel3000_merge(
        self,
        routes: list[TypedRoute],
    ) -> tuple[list[TypedRoute], dict[str, int], list[dict[str, object]]]:
        updated_routes = list(routes)
        success_count = 0
        diagnostics: list[dict[str, object]] = []
        for candidate in self._single_stop_merge_candidates(updated_routes)[0]:
            if success_count >= BATCH_MERGE_LIMIT:
                break
            if not bool(candidate["inventory_ok"]) or float(candidate["saving"]) <= 1e-6:
                diagnostics.append(candidate)
                continue
            left_idx = int(candidate["left_route_idx"])
            right_idx = int(candidate["right_route_idx"])
            if left_idx >= len(updated_routes) or right_idx >= len(updated_routes):
                continue
            if updated_routes[left_idx].unit_ids != (int(candidate["left_unit_id"]),):
                continue
            if updated_routes[right_idx].unit_ids != (int(candidate["right_unit_id"]),):
                continue
            merged_route = TypedRoute("fuel_3000", tuple(int(unit_id) for unit_id in candidate["merged_unit_ids"]))
            merged_eval = self.evaluate_route(merged_route)
            if not merged_eval.feasible:
                continue
            kept = [
                route
                for idx, route in enumerate(updated_routes)
                if idx not in {left_idx, right_idx}
            ]
            kept.append(merged_route)
            if self.evaluate_solution(kept) is None:
                continue
            updated_routes = kept
            success_count += 1
            diagnostics.append({**candidate, "accepted": 1})
        return updated_routes, {"batch_merge_success_count": success_count}, diagnostics

    def _register_candidate_route(
        self,
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec],
        vehicle_type: str,
        unit_ids: tuple[int, ...],
        role: str,
    ) -> None:
        key = (vehicle_type, tuple(unit_ids))
        if key in route_pool:
            route_pool[key].roles.add(role)
            return
        candidate_route = TypedRoute(vehicle_type=vehicle_type, unit_ids=tuple(unit_ids))
        candidate_eval = self.evaluate_route(candidate_route)
        if not candidate_eval.feasible:
            return
        route_pool[key] = CandidateRouteSpec(
            route=candidate_route,
            route_eval=candidate_eval,
            roles={role},
        )

    def _generate_route_pool(
        self,
        seed_routes: list[TypedRoute],
    ) -> dict[tuple[str, tuple[int, ...]], CandidateRouteSpec]:
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec] = {}
        small_vehicle_types = ("fuel_1250", "ev_1250", "fuel_1500")
        big_vehicle_types = ("fuel_3000", "ev_3000")
        units_by_customer: dict[int, list[int]] = defaultdict(list)
        for unit in self.service_units:
            units_by_customer[unit.orig_cust_id].append(unit.unit_id)

        for route in seed_routes:
            self._register_candidate_route(route_pool, route.vehicle_type, route.unit_ids, "seed")

        for unit in self.service_units:
            for vehicle_type in unit.eligible_vehicle_types:
                self._register_candidate_route(route_pool, vehicle_type, (unit.unit_id,), "singleton")

        flexible_unit_ids = [unit_id for unit_id in self.active_unit_ids if not self._unit_is_heavy_big_only(unit_id)]
        heavy_unit_ids = [unit_id for unit_id in self.active_unit_ids if self._unit_is_heavy_big_only(unit_id)]

        flexible_neighbors: dict[int, list[int]] = {}
        for unit_id in flexible_unit_ids:
            unit = self.unit_by_id[unit_id]
            spatial_neighbor_ids: list[int] = []
            for cust_id in self.customer_neighbors.get(unit.orig_cust_id, []):
                for neighbor_unit_id in units_by_customer.get(cust_id, []):
                    if neighbor_unit_id != unit_id and not self._unit_is_heavy_big_only(neighbor_unit_id):
                        spatial_neighbor_ids.append(neighbor_unit_id)
                if len(dict.fromkeys(spatial_neighbor_ids)) >= BASE_FLEX_SPATIAL_NEIGHBOR_LIMIT:
                    break
            spatial_neighbor_ids = list(dict.fromkeys(spatial_neighbor_ids))[:BASE_FLEX_SPATIAL_NEIGHBOR_LIMIT]
            tw_neighbor_ids = [
                neighbor_unit_id
                for neighbor_unit_id in flexible_unit_ids
                if neighbor_unit_id != unit_id
                and self.unit_by_id[neighbor_unit_id].orig_cust_id != unit.orig_cust_id
                and self._time_window_overlap_minutes(unit_id, neighbor_unit_id) >= RESIDUAL_TW_OVERLAP_MIN
            ]
            tw_neighbor_ids.sort(
                key=lambda neighbor_unit_id: (
                    -self._time_window_overlap_minutes(unit_id, neighbor_unit_id),
                    self._distance_between(unit.orig_cust_id, self.unit_by_id[neighbor_unit_id].orig_cust_id),
                    neighbor_unit_id,
                )
            )
            tw_neighbor_ids = tw_neighbor_ids[:BASE_FLEX_TW_NEIGHBOR_LIMIT]
            flexible_neighbors[unit_id] = list(dict.fromkeys(spatial_neighbor_ids + tw_neighbor_ids))[:BASE_FLEX_NEIGHBOR_LIMIT]

        heavy_neighbors: dict[int, list[int]] = {}
        for unit_id in heavy_unit_ids:
            unit = self.unit_by_id[unit_id]
            neighbor_units: list[int] = []
            for cust_id in self.customer_neighbors.get(unit.orig_cust_id, [])[:4]:
                for neighbor_unit_id in units_by_customer.get(cust_id, []):
                    if neighbor_unit_id != unit_id and self._unit_is_heavy_big_only(neighbor_unit_id):
                        neighbor_units.append(neighbor_unit_id)
            heavy_neighbors[unit_id] = list(dict.fromkeys(neighbor_units))[:4]

        heavy_flexible_neighbors: dict[int, list[int]] = {}
        for unit_id in heavy_unit_ids:
            unit = self.unit_by_id[unit_id]
            spatial_neighbor_ids: list[int] = []
            for cust_id in self.customer_neighbors.get(unit.orig_cust_id, []):
                for neighbor_unit_id in units_by_customer.get(cust_id, []):
                    if neighbor_unit_id != unit_id and not self._unit_is_heavy_big_only(neighbor_unit_id):
                        spatial_neighbor_ids.append(neighbor_unit_id)
                if len(dict.fromkeys(spatial_neighbor_ids)) >= BASE_FLEX_SPATIAL_NEIGHBOR_LIMIT:
                    break
            spatial_neighbor_ids = list(dict.fromkeys(spatial_neighbor_ids))[:BASE_FLEX_SPATIAL_NEIGHBOR_LIMIT]
            tw_neighbor_ids = [
                neighbor_unit_id
                for neighbor_unit_id in flexible_unit_ids
                if self.unit_by_id[neighbor_unit_id].orig_cust_id != unit.orig_cust_id
                and self._time_window_overlap_minutes(unit_id, neighbor_unit_id) >= RESIDUAL_TW_OVERLAP_MIN
            ]
            tw_neighbor_ids.sort(
                key=lambda neighbor_unit_id: (
                    -self._time_window_overlap_minutes(unit_id, neighbor_unit_id),
                    self._distance_between(unit.orig_cust_id, self.unit_by_id[neighbor_unit_id].orig_cust_id),
                    neighbor_unit_id,
                )
            )
            tw_neighbor_ids = tw_neighbor_ids[:BASE_FLEX_TW_NEIGHBOR_LIMIT]
            heavy_flexible_neighbors[unit_id] = list(
                dict.fromkeys(spatial_neighbor_ids + tw_neighbor_ids)
            )[:BASE_FLEX_NEIGHBOR_LIMIT]

        for unit_id in flexible_unit_ids:
            unit = self.unit_by_id[unit_id]
            for neighbor_unit_id in flexible_neighbors[unit_id]:
                if unit_id >= neighbor_unit_id:
                    continue
                neighbor_unit = self.unit_by_id[neighbor_unit_id]
                if unit.orig_cust_id == neighbor_unit.orig_cust_id:
                    continue
                common_small_types = [
                    vehicle_type
                    for vehicle_type in small_vehicle_types
                    if vehicle_type in unit.eligible_vehicle_types and vehicle_type in neighbor_unit.eligible_vehicle_types
                ]
                for vehicle_type in common_small_types:
                    for ordered_unit_ids in ((unit_id, neighbor_unit_id), (neighbor_unit_id, unit_id)):
                        self._register_candidate_route(route_pool, vehicle_type, ordered_unit_ids, "flex_small")
                common_big_types = [
                    vehicle_type
                    for vehicle_type in big_vehicle_types
                    if vehicle_type in unit.eligible_vehicle_types and vehicle_type in neighbor_unit.eligible_vehicle_types
                ]
                for vehicle_type in common_big_types:
                    for ordered_unit_ids in ((unit_id, neighbor_unit_id), (neighbor_unit_id, unit_id)):
                        self._register_candidate_route(route_pool, vehicle_type, ordered_unit_ids, "promotion")

        for unit_id in flexible_unit_ids:
            unit = self.unit_by_id[unit_id]
            triple_candidates = [neighbor_unit_id for neighbor_unit_id in flexible_neighbors[unit_id] if neighbor_unit_id > unit_id]
            for left_neighbor_id, right_neighbor_id in combinations(triple_candidates[:BASE_FLEX_TRIPLE_SEED_LIMIT], 2):
                left_neighbor = self.unit_by_id[left_neighbor_id]
                right_neighbor = self.unit_by_id[right_neighbor_id]
                if len({unit.orig_cust_id, left_neighbor.orig_cust_id, right_neighbor.orig_cust_id}) < 3:
                    continue
                common_small_types = [
                    vehicle_type
                    for vehicle_type in small_vehicle_types
                    if vehicle_type in unit.eligible_vehicle_types
                    and vehicle_type in left_neighbor.eligible_vehicle_types
                    and vehicle_type in right_neighbor.eligible_vehicle_types
                ]
                common_big_types = [
                    vehicle_type
                    for vehicle_type in big_vehicle_types
                    if vehicle_type in unit.eligible_vehicle_types
                    and vehicle_type in left_neighbor.eligible_vehicle_types
                    and vehicle_type in right_neighbor.eligible_vehicle_types
                ]
                for ordered_unit_ids in permutations((unit_id, left_neighbor_id, right_neighbor_id)):
                    for vehicle_type in common_small_types[:2]:
                        self._register_candidate_route(route_pool, vehicle_type, tuple(ordered_unit_ids), "flex_small")
                    for vehicle_type in common_big_types:
                        self._register_candidate_route(route_pool, vehicle_type, tuple(ordered_unit_ids), "promotion")

        for unit_id in heavy_unit_ids:
            unit = self.unit_by_id[unit_id]
            for vehicle_type in [name for name in big_vehicle_types if name in unit.eligible_vehicle_types]:
                self._register_candidate_route(route_pool, vehicle_type, (unit_id,), "rigid_big")
            for neighbor_unit_id in heavy_neighbors[unit_id]:
                if unit_id >= neighbor_unit_id:
                    continue
                neighbor_unit = self.unit_by_id[neighbor_unit_id]
                common_big_types = [
                    vehicle_type
                    for vehicle_type in big_vehicle_types
                    if vehicle_type in unit.eligible_vehicle_types and vehicle_type in neighbor_unit.eligible_vehicle_types
                ]
                for vehicle_type in common_big_types:
                    for ordered_unit_ids in ((unit_id, neighbor_unit_id), (neighbor_unit_id, unit_id)):
                        self._register_candidate_route(route_pool, vehicle_type, ordered_unit_ids, "rigid_big")
            for flexible_unit_id in heavy_flexible_neighbors[unit_id]:
                flexible_unit = self.unit_by_id[flexible_unit_id]
                common_big_types = [
                    vehicle_type
                    for vehicle_type in big_vehicle_types
                    if vehicle_type in unit.eligible_vehicle_types and vehicle_type in flexible_unit.eligible_vehicle_types
                ]
                for vehicle_type in common_big_types:
                    for ordered_unit_ids in ((unit_id, flexible_unit_id), (flexible_unit_id, unit_id)):
                        self._register_candidate_route(route_pool, vehicle_type, ordered_unit_ids, "piggyback_big")
                for heavy_neighbor_id in heavy_neighbors[unit_id]:
                    if unit_id >= heavy_neighbor_id:
                        continue
                    heavy_neighbor = self.unit_by_id[heavy_neighbor_id]
                    triple_vehicle_types = [
                        vehicle_type
                        for vehicle_type in big_vehicle_types
                        if vehicle_type in unit.eligible_vehicle_types
                        and vehicle_type in heavy_neighbor.eligible_vehicle_types
                        and vehicle_type in flexible_unit.eligible_vehicle_types
                    ]
                    for vehicle_type in triple_vehicle_types:
                        for ordered_unit_ids in permutations((unit_id, heavy_neighbor_id, flexible_unit_id)):
                            self._register_candidate_route(route_pool, vehicle_type, tuple(ordered_unit_ids), "piggyback_big")
        return route_pool

    def _route_pool_role_counts(
        self,
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec],
    ) -> dict[str, int]:
        role_counts = Counter[str]()
        for spec in route_pool.values():
            for role in spec.roles:
                role_counts[role] += 1
        return dict(role_counts)

    @staticmethod
    def _column_has_any_role(column: dict[str, object], role_names: set[str]) -> bool:
        return bool(set(column["roles"]) & role_names)

    @staticmethod
    def _column_pool_pass(roles: Iterable[str]) -> str:
        return "residual" if any(role.startswith("residual_") for role in roles) else "base"

    def _column_effective_saving(self, column: dict[str, object]) -> float:
        return float(column["current_cost_saving"])

    def _column_candidate_score(self, column: dict[str, object]) -> float:
        return float(
            self._column_effective_saving(column)
            + 0.25 * float(column["avg_time_window_overlap_min"])
            - 0.10 * float(column["avg_customer_distance_km"])
        )

    def _column_candidate_sort_key(self, column: dict[str, object]) -> tuple[float, float, float, int, float, str, tuple[int, ...]]:
        return (
            -float(column["current_cost_saving"]),
            -float(column["avg_time_window_overlap_min"]),
            float(column["avg_customer_distance_km"]),
            -int(column["unit_count"]),
            float(column["best_cost"]),
            str(column["vehicle_type"]),
            tuple(int(unit_id) for unit_id in column["unit_ids"]),
        )

    def _flex_small_count_by_size(
        self,
        columns: Iterable[dict[str, object]],
    ) -> dict[str, int]:
        counts = Counter[int]()
        for column in columns:
            if self._column_has_any_role(column, {"flex_small", "residual_flex_small", "cluster_flex_small"}):
                counts[int(column["unit_count"])] += 1
        return {str(size): count for size, count in sorted(counts.items())}

    def _best_route_candidate_for_units(
        self,
        vehicle_types: Iterable[str],
        unit_ids: tuple[int, ...],
    ) -> tuple[str, tuple[int, ...], RouteEvaluation] | None:
        best_candidate: tuple[str, tuple[int, ...], RouteEvaluation] | None = None
        for vehicle_type in vehicle_types:
            for ordered_unit_ids in permutations(unit_ids):
                route = TypedRoute(vehicle_type=vehicle_type, unit_ids=tuple(ordered_unit_ids))
                route_eval = self.evaluate_route(route)
                if not route_eval.feasible:
                    continue
                candidate = (vehicle_type, tuple(ordered_unit_ids), route_eval)
                if best_candidate is None or (
                    route_eval.best_cost,
                    vehicle_type,
                    tuple(ordered_unit_ids),
                ) < (
                    best_candidate[2].best_cost,
                    best_candidate[0],
                    best_candidate[1],
                ):
                    best_candidate = candidate
        return best_candidate

    def _rank_residual_source_unit_ids(
        self,
        source_unit_ids: Iterable[int],
        singleton_cost_by_unit: dict[int, float],
        source_limit: int,
    ) -> list[int]:
        source_unit_ids = list(source_unit_ids)
        overlap_potential_cache: dict[int, float] = {}
        distance_centrality_cache: dict[int, float] = {}
        for unit_id in source_unit_ids:
            other_unit_ids = [other_unit_id for other_unit_id in source_unit_ids if other_unit_id != unit_id]
            overlap_potential_cache[unit_id] = max(
                (self._time_window_overlap_minutes(unit_id, other_unit_id) for other_unit_id in other_unit_ids),
                default=0.0,
            )
            nearest_distances = sorted(
                self._distance_between(self.unit_by_id[unit_id].orig_cust_id, self.unit_by_id[other_unit_id].orig_cust_id)
                for other_unit_id in other_unit_ids
            )[:3]
            distance_centrality_cache[unit_id] = (
                float(sum(nearest_distances) / len(nearest_distances))
                if nearest_distances
                else np.inf
            )
        return sorted(
            source_unit_ids,
            key=lambda unit_id: (
                -singleton_cost_by_unit[unit_id],
                -overlap_potential_cache[unit_id],
                distance_centrality_cache[unit_id],
                self.unit_by_id[unit_id].orig_cust_id,
                unit_id,
            ),
        )[:source_limit]

    def _residual_candidate_neighbor_ids(
        self,
        source_unit_id: int,
        candidate_unit_ids: Iterable[int],
        units_by_customer: dict[int, list[int]],
        singleton_cost_by_unit: dict[int, float],
        spatial_limit: int,
        tw_limit: int,
        neighbor_limit: int,
    ) -> list[int]:
        source_unit = self.unit_by_id[source_unit_id]
        spatial_neighbor_ids: list[int] = []
        for cust_id in self.customer_neighbors.get(source_unit.orig_cust_id, []):
            for neighbor_unit_id in units_by_customer.get(cust_id, []):
                if neighbor_unit_id == source_unit_id:
                    continue
                if self.unit_by_id[neighbor_unit_id].orig_cust_id == source_unit.orig_cust_id:
                    continue
                spatial_neighbor_ids.append(neighbor_unit_id)
            if len(dict.fromkeys(spatial_neighbor_ids)) >= spatial_limit:
                break
        spatial_neighbor_ids = list(dict.fromkeys(spatial_neighbor_ids))[:spatial_limit]

        tw_neighbor_ids = [
            unit_id
            for unit_id in candidate_unit_ids
            if unit_id != source_unit_id
            and self.unit_by_id[unit_id].orig_cust_id != source_unit.orig_cust_id
            and self._time_window_overlap_minutes(source_unit_id, unit_id) >= RESIDUAL_TW_OVERLAP_MIN
        ]
        tw_neighbor_ids.sort(
            key=lambda unit_id: (
                -self._time_window_overlap_minutes(source_unit_id, unit_id),
                self._distance_between(source_unit.orig_cust_id, self.unit_by_id[unit_id].orig_cust_id),
                -singleton_cost_by_unit[unit_id],
                unit_id,
            )
        )
        tw_neighbor_ids = tw_neighbor_ids[:tw_limit]
        return list(dict.fromkeys(spatial_neighbor_ids + tw_neighbor_ids))[:neighbor_limit]

    def _augment_route_pool_with_residual_candidates(
        self,
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec],
        residual_source_routes: list[TypedRoute],
    ) -> dict[tuple[str, tuple[int, ...]], float]:
        singleton_cost_by_unit = {
            route.unit_ids[0]: self.evaluate_route(route).best_cost
            for route in residual_source_routes
            if len(route.unit_ids) == 1 and not self._unit_is_heavy_big_only(route.unit_ids[0])
        }
        residual_unit_ids = list(singleton_cost_by_unit)
        residual_units_by_customer: dict[int, list[int]] = defaultdict(list)
        for unit_id in residual_unit_ids:
            residual_units_by_customer[self.unit_by_id[unit_id].orig_cust_id].append(unit_id)

        ranked_source_unit_ids = self._rank_residual_source_unit_ids(
            source_unit_ids=residual_unit_ids,
            singleton_cost_by_unit=singleton_cost_by_unit,
            source_limit=RESIDUAL_SOURCE_LIMIT,
        )

        residual_saving_by_key: dict[tuple[str, tuple[int, ...]], float] = {}
        small_vehicle_types = ("fuel_1250", "ev_1250", "fuel_1500")
        big_vehicle_types = ("fuel_3000", "ev_3000")

        for source_unit_id in ranked_source_unit_ids:
            source_unit = self.unit_by_id[source_unit_id]
            candidate_neighbor_ids = self._residual_candidate_neighbor_ids(
                source_unit_id=source_unit_id,
                candidate_unit_ids=residual_unit_ids,
                units_by_customer=residual_units_by_customer,
                singleton_cost_by_unit=singleton_cost_by_unit,
                spatial_limit=RESIDUAL_SPATIAL_NEIGHBOR_LIMIT,
                tw_limit=RESIDUAL_TW_NEIGHBOR_LIMIT,
                neighbor_limit=RESIDUAL_NEIGHBOR_LIMIT,
            )
            if not candidate_neighbor_ids:
                continue

            residual_promotion_records: list[tuple[float, float, float, float, str, tuple[int, ...]]] = []
            residual_flex_small_records: list[tuple[float, float, float, float, str, tuple[int, ...]]] = []
            partner_best_saving: dict[int, float] = {}

            for neighbor_unit_id in candidate_neighbor_ids:
                neighbor_unit = self.unit_by_id[neighbor_unit_id]
                avg_overlap = self._average_time_window_overlap_minutes((source_unit_id, neighbor_unit_id))
                avg_distance = self._average_customer_distance_km((source_unit_id, neighbor_unit_id))
                common_small_types = [
                    vehicle_type
                    for vehicle_type in small_vehicle_types
                    if vehicle_type in source_unit.eligible_vehicle_types and vehicle_type in neighbor_unit.eligible_vehicle_types
                ]
                common_big_types = [
                    vehicle_type
                    for vehicle_type in big_vehicle_types
                    if vehicle_type in source_unit.eligible_vehicle_types and vehicle_type in neighbor_unit.eligible_vehicle_types
                ]
                if common_small_types:
                    best_candidate = self._best_route_candidate_for_units(common_small_types, (source_unit_id, neighbor_unit_id))
                    if best_candidate is not None:
                        vehicle_type, ordered_unit_ids, route_eval = best_candidate
                        saving = singleton_cost_by_unit[source_unit_id] + singleton_cost_by_unit[neighbor_unit_id] - route_eval.best_cost
                        residual_flex_small_records.append((saving, avg_overlap, avg_distance, route_eval.best_cost, vehicle_type, ordered_unit_ids))
                        partner_best_saving[neighbor_unit_id] = max(partner_best_saving.get(neighbor_unit_id, -np.inf), saving)
                if common_big_types:
                    best_candidate = self._best_route_candidate_for_units(common_big_types, (source_unit_id, neighbor_unit_id))
                    if best_candidate is not None:
                        vehicle_type, ordered_unit_ids, route_eval = best_candidate
                        saving = singleton_cost_by_unit[source_unit_id] + singleton_cost_by_unit[neighbor_unit_id] - route_eval.best_cost
                        residual_promotion_records.append((saving, avg_overlap, avg_distance, route_eval.best_cost, vehicle_type, ordered_unit_ids))
                        partner_best_saving[neighbor_unit_id] = max(partner_best_saving.get(neighbor_unit_id, -np.inf), saving)

            top_partner_ids = [
                neighbor_unit_id
                for neighbor_unit_id, _ in sorted(
                    partner_best_saving.items(),
                    key=lambda item: (
                        -item[1],
                        self._distance_between(source_unit.orig_cust_id, self.unit_by_id[item[0]].orig_cust_id),
                        item[0],
                    ),
                )[:RESIDUAL_TRIPLE_PARTNER_LIMIT]
            ]

            for left_neighbor_id, right_neighbor_id in combinations(top_partner_ids, 2):
                left_neighbor = self.unit_by_id[left_neighbor_id]
                right_neighbor = self.unit_by_id[right_neighbor_id]
                if len({source_unit.orig_cust_id, left_neighbor.orig_cust_id, right_neighbor.orig_cust_id}) < 3:
                    continue
                common_small_types = [
                    vehicle_type
                    for vehicle_type in small_vehicle_types
                    if vehicle_type in source_unit.eligible_vehicle_types
                    and vehicle_type in left_neighbor.eligible_vehicle_types
                    and vehicle_type in right_neighbor.eligible_vehicle_types
                ]
                common_big_types = [
                    vehicle_type
                    for vehicle_type in big_vehicle_types
                    if vehicle_type in source_unit.eligible_vehicle_types
                    and vehicle_type in left_neighbor.eligible_vehicle_types
                    and vehicle_type in right_neighbor.eligible_vehicle_types
                ]
                triple_unit_ids = (source_unit_id, left_neighbor_id, right_neighbor_id)
                triple_singleton_cost = (
                    singleton_cost_by_unit[source_unit_id]
                    + singleton_cost_by_unit[left_neighbor_id]
                    + singleton_cost_by_unit[right_neighbor_id]
                )
                avg_overlap = self._average_time_window_overlap_minutes(triple_unit_ids)
                avg_distance = self._average_customer_distance_km(triple_unit_ids)
                if common_small_types:
                    best_candidate = self._best_route_candidate_for_units(common_small_types, triple_unit_ids)
                    if best_candidate is not None:
                        vehicle_type, ordered_unit_ids, route_eval = best_candidate
                        saving = triple_singleton_cost - route_eval.best_cost
                        residual_flex_small_records.append((saving, avg_overlap, avg_distance, route_eval.best_cost, vehicle_type, ordered_unit_ids))
                if common_big_types:
                    best_candidate = self._best_route_candidate_for_units(common_big_types, triple_unit_ids)
                    if best_candidate is not None:
                        vehicle_type, ordered_unit_ids, route_eval = best_candidate
                        saving = triple_singleton_cost - route_eval.best_cost
                        residual_promotion_records.append((saving, avg_overlap, avg_distance, route_eval.best_cost, vehicle_type, ordered_unit_ids))

            residual_promotion_records.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4], item[5]))
            residual_flex_small_records.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4], item[5]))

            for saving, _, _, _, vehicle_type, ordered_unit_ids in residual_promotion_records[:RESIDUAL_PROMOTION_PER_SOURCE_LIMIT]:
                self._register_candidate_route(route_pool, vehicle_type, ordered_unit_ids, "residual_promotion")
                key = (vehicle_type, ordered_unit_ids)
                residual_saving_by_key[key] = max(residual_saving_by_key.get(key, -np.inf), float(saving))
            for saving, _, _, _, vehicle_type, ordered_unit_ids in residual_flex_small_records[:RESIDUAL_FLEX_SMALL_PER_SOURCE_LIMIT]:
                self._register_candidate_route(route_pool, vehicle_type, ordered_unit_ids, "residual_flex_small")
                key = (vehicle_type, ordered_unit_ids)
                residual_saving_by_key[key] = max(residual_saving_by_key.get(key, -np.inf), float(saving))

        return residual_saving_by_key

    def _augment_route_pool_with_pass3_flex_small_candidates(
        self,
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec],
        pass_result: dict[str, object],
        residual_saving_by_key: dict[tuple[str, tuple[int, ...]], float],
    ) -> dict[tuple[str, tuple[int, ...]], float]:
        selected_columns = [
            pass_result["columns"][idx]
            for idx in pass_result["selected_indices"]
        ]
        singleton_cost_by_unit: dict[int, float] = {}
        for column in pass_result["columns"]:
            if int(column["unit_count"]) != 1:
                continue
            unit_id = int(column["unit_ids"][0])
            if self._unit_is_heavy_big_only(unit_id):
                continue
            singleton_cost_by_unit[unit_id] = min(
                singleton_cost_by_unit.get(unit_id, np.inf),
                float(column["best_cost"]),
            )
        residual_unit_ids = list(singleton_cost_by_unit)
        if not residual_unit_ids:
            return residual_saving_by_key

        ranked_source_unit_ids = self._rank_residual_source_unit_ids(
            source_unit_ids=residual_unit_ids,
            singleton_cost_by_unit=singleton_cost_by_unit,
            source_limit=PASS3_SOURCE_LIMIT,
        )
        small_vehicle_types = ("fuel_1250", "ev_1250", "fuel_1500")
        selected_small_columns = [
            column
            for column in selected_columns
            if not self._column_has_any_role(column, {"rigid_big"})
            and self._column_has_any_role(column, {"singleton", "flex_small", "residual_flex_small", "cluster_flex_small"})
        ]
        if not selected_small_columns:
            return residual_saving_by_key

        for source_unit_id in ranked_source_unit_ids:
            source_unit = self.unit_by_id[source_unit_id]
            scored_cluster_columns: list[tuple[float, float, float, int, tuple[int, ...]]] = []
            for column in selected_small_columns:
                unit_ids = tuple(int(unit_id) for unit_id in column["unit_ids"])
                if source_unit_id in unit_ids:
                    continue
                route_overlap = max(
                    self._time_window_overlap_minutes(source_unit_id, unit_id)
                    for unit_id in unit_ids
                )
                min_distance = min(
                    self._distance_between(source_unit.orig_cust_id, self.unit_by_id[unit_id].orig_cust_id)
                    for unit_id in unit_ids
                )
                scored_cluster_columns.append(
                    (
                        -route_overlap,
                        min_distance,
                        -float(column["candidate_score"]),
                        -len(unit_ids),
                        unit_ids,
                    )
                )
            scored_cluster_columns.sort()
            cluster_unit_ids = {source_unit_id}
            for _, _, _, _, unit_ids in scored_cluster_columns[:PASS3_CLUSTER_ROUTE_LIMIT]:
                cluster_unit_ids.update(unit_ids)
            if len(cluster_unit_ids) <= 1:
                continue
            ranked_partner_ids = sorted(
                [unit_id for unit_id in cluster_unit_ids if unit_id != source_unit_id],
                key=lambda unit_id: (
                    -self._time_window_overlap_minutes(source_unit_id, unit_id),
                    self._distance_between(source_unit.orig_cust_id, self.unit_by_id[unit_id].orig_cust_id),
                    -singleton_cost_by_unit.get(unit_id, 0.0),
                    unit_id,
                ),
            )[:PASS3_CLUSTER_PARTNER_UNIT_LIMIT]
            candidate_neighbor_ids = ranked_partner_ids
            if not candidate_neighbor_ids:
                continue

            pair_records: list[tuple[float, float, float, float, str, tuple[int, ...]]] = []
            triple_records: list[tuple[float, float, float, float, str, tuple[int, ...]]] = []
            quad_records: list[tuple[float, float, float, float, str, tuple[int, ...]]] = []
            partner_best_saving: dict[int, float] = {}

            for neighbor_unit_id in candidate_neighbor_ids:
                neighbor_unit = self.unit_by_id[neighbor_unit_id]
                common_small_types = [
                    vehicle_type
                    for vehicle_type in small_vehicle_types
                    if vehicle_type in source_unit.eligible_vehicle_types and vehicle_type in neighbor_unit.eligible_vehicle_types
                ]
                if not common_small_types:
                    continue
                best_candidate = self._best_route_candidate_for_units(common_small_types, (source_unit_id, neighbor_unit_id))
                if best_candidate is None:
                    continue
                vehicle_type, ordered_unit_ids, route_eval = best_candidate
                saving = singleton_cost_by_unit[source_unit_id] + singleton_cost_by_unit[neighbor_unit_id] - route_eval.best_cost
                avg_overlap = self._average_time_window_overlap_minutes((source_unit_id, neighbor_unit_id))
                avg_distance = self._average_customer_distance_km((source_unit_id, neighbor_unit_id))
                pair_records.append((saving, avg_overlap, avg_distance, route_eval.best_cost, vehicle_type, ordered_unit_ids))
                partner_best_saving[neighbor_unit_id] = max(partner_best_saving.get(neighbor_unit_id, -np.inf), saving)

            top_partner_ids = [
                neighbor_unit_id
                for neighbor_unit_id, _ in sorted(
                    partner_best_saving.items(),
                    key=lambda item: (
                        -item[1],
                        self._distance_between(source_unit.orig_cust_id, self.unit_by_id[item[0]].orig_cust_id),
                        item[0],
                    ),
                )[:PASS3_TRIPLE_PARTNER_LIMIT]
            ]

            for left_neighbor_id, right_neighbor_id in combinations(top_partner_ids, 2):
                left_neighbor = self.unit_by_id[left_neighbor_id]
                right_neighbor = self.unit_by_id[right_neighbor_id]
                if len({source_unit.orig_cust_id, left_neighbor.orig_cust_id, right_neighbor.orig_cust_id}) < 3:
                    continue
                common_small_types = [
                    vehicle_type
                    for vehicle_type in small_vehicle_types
                    if vehicle_type in source_unit.eligible_vehicle_types
                    and vehicle_type in left_neighbor.eligible_vehicle_types
                    and vehicle_type in right_neighbor.eligible_vehicle_types
                ]
                if not common_small_types:
                    continue
                triple_unit_ids = (source_unit_id, left_neighbor_id, right_neighbor_id)
                best_candidate = self._best_route_candidate_for_units(common_small_types[:2], triple_unit_ids)
                if best_candidate is None:
                    continue
                vehicle_type, ordered_unit_ids, route_eval = best_candidate
                saving = (
                    singleton_cost_by_unit[source_unit_id]
                    + singleton_cost_by_unit[left_neighbor_id]
                    + singleton_cost_by_unit[right_neighbor_id]
                    - route_eval.best_cost
                )
                avg_overlap = self._average_time_window_overlap_minutes(triple_unit_ids)
                avg_distance = self._average_customer_distance_km(triple_unit_ids)
                triple_records.append((saving, avg_overlap, avg_distance, route_eval.best_cost, vehicle_type, ordered_unit_ids))

            for first_neighbor_id, second_neighbor_id, third_neighbor_id in combinations(top_partner_ids, 3):
                neighbor_ids = (source_unit_id, first_neighbor_id, second_neighbor_id, third_neighbor_id)
                if len({self.unit_by_id[unit_id].orig_cust_id for unit_id in neighbor_ids}) < 4:
                    continue
                common_small_types = [
                    vehicle_type
                    for vehicle_type in small_vehicle_types
                    if all(vehicle_type in self.unit_by_id[unit_id].eligible_vehicle_types for unit_id in neighbor_ids)
                ]
                if not common_small_types:
                    continue
                best_candidate = self._best_route_candidate_for_units(common_small_types[:2], neighbor_ids)
                if best_candidate is None:
                    continue
                vehicle_type, ordered_unit_ids, route_eval = best_candidate
                saving = sum(singleton_cost_by_unit[unit_id] for unit_id in neighbor_ids) - route_eval.best_cost
                avg_overlap = self._average_time_window_overlap_minutes(neighbor_ids)
                avg_distance = self._average_customer_distance_km(neighbor_ids)
                quad_records.append((saving, avg_overlap, avg_distance, route_eval.best_cost, vehicle_type, ordered_unit_ids))

            pair_records.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4], item[5]))
            triple_records.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4], item[5]))
            quad_records.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4], item[5]))

            for record_set, limit in (
                (pair_records, PASS3_FLEX_SMALL_PAIR_PER_SOURCE_LIMIT),
                (triple_records, PASS3_FLEX_SMALL_TRIPLE_PER_SOURCE_LIMIT),
                (quad_records, PASS3_FLEX_SMALL_QUAD_PER_SOURCE_LIMIT),
            ):
                for saving, _, _, _, vehicle_type, ordered_unit_ids in record_set[:limit]:
                    self._register_candidate_route(route_pool, vehicle_type, ordered_unit_ids, "cluster_flex_small")
                    key = (vehicle_type, ordered_unit_ids)
                    residual_saving_by_key[key] = max(residual_saving_by_key.get(key, -np.inf), float(saving))

        return residual_saving_by_key

    def _build_route_pool_columns(
        self,
        seed_routes: list[TypedRoute],
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec] | None = None,
        max_columns: int = ROUTE_POOL_MAX_COLUMNS,
        promotion_limit: int = ROUTE_POOL_PROMOTION_LIMIT,
        flex_small_limit: int = ROUTE_POOL_FLEX_SMALL_LIMIT,
        residual_promotion_limit: int = ROUTE_POOL_RESIDUAL_PROMOTION_LIMIT,
        residual_flex_small_limit: int = ROUTE_POOL_RESIDUAL_FLEX_SMALL_LIMIT,
        residual_saving_by_key: dict[tuple[str, tuple[int, ...]], float] | None = None,
    ) -> list[dict[str, object]]:
        raw_pool = route_pool if route_pool is not None else self._generate_route_pool(seed_routes)
        residual_saving_by_key = residual_saving_by_key or {}
        current_route_by_unit = {
            int(unit_id): route
            for route in seed_routes
            for unit_id in route.unit_ids
        }
        current_route_cost_by_key = {
            self._route_key(route): float(self.evaluate_route(route).best_cost)
            for route in seed_routes
        }
        singleton_cost_by_unit: dict[int, float] = {}
        for spec in raw_pool.values():
            if len(spec.route.unit_ids) != 1:
                continue
            unit_id = spec.route.unit_ids[0]
            singleton_cost_by_unit[unit_id] = min(singleton_cost_by_unit.get(unit_id, np.inf), spec.route_eval.best_cost)

        grouped_columns: dict[tuple[str, frozenset[int]], list[dict[str, object]]] = defaultdict(list)
        for key, spec in raw_pool.items():
            roles = set(spec.roles)
            if "seed" in roles:
                roles.add("current")
            (
                promotion_like_big_route_flag,
                piggyback_big_route_flag,
                blocking_big_flexible_route_flag,
                blocking_big_flexible_unit_count,
                big_route_flag,
                big_flexible_unit_count,
                heavy_big_only_unit_count,
            ) = self._route_big_structure_metrics(spec.route)
            big_flexible_route_flag = int(big_flexible_unit_count > 0)
            singleton_cover_cost = float(
                sum(singleton_cost_by_unit.get(unit_id, spec.route_eval.best_cost) for unit_id in spec.route.unit_ids)
            )
            current_cover_keys = {
                self._route_key(current_route_by_unit[int(unit_id)])
                for unit_id in spec.route.unit_ids
            }
            current_cover_cost = float(sum(current_route_cost_by_key[route_key] for route_key in current_cover_keys))
            current_cost_saving = current_cover_cost - float(spec.route_eval.best_cost)
            avg_time_window_overlap_min = self._average_time_window_overlap_minutes(spec.route.unit_ids)
            avg_customer_distance_km = self._average_customer_distance_km(spec.route.unit_ids)
            candidate_family = "support"
            if piggyback_big_route_flag:
                candidate_family = "piggyback_big"
            elif promotion_like_big_route_flag:
                candidate_family = "promotion_like_big"
            elif (
                roles & {"flex_small", "residual_flex_small", "cluster_flex_small"}
                and 2 <= len(spec.route.unit_ids) <= 4
            ):
                candidate_family = "flex_small"
            elif roles & {"seed", "current"}:
                candidate_family = "seed_current"
            grouped_columns[(spec.route.vehicle_type, frozenset(spec.route.unit_ids))].append(
                {
                    "key": key,
                    "vehicle_type": spec.route.vehicle_type,
                    "unit_ids": spec.route.unit_ids,
                    "unit_set": frozenset(spec.route.unit_ids),
                    "unit_count": int(len(spec.route.unit_ids)),
                    "best_cost": float(spec.route_eval.best_cost),
                    "single_stop_flag": int(len(spec.route.unit_ids) == 1),
                    "big_route_flag": big_route_flag,
                    "big_flexible_route_flag": int(big_flexible_route_flag),
                    "big_flexible_unit_count": int(big_flexible_unit_count),
                    "mixed_big_route_flag": int(big_flexible_unit_count > 0),
                    "piggyback_big_route_flag": int(piggyback_big_route_flag),
                    "promotion_like_big_route_flag": int(promotion_like_big_route_flag),
                    "blocking_big_flexible_route_flag": int(blocking_big_flexible_route_flag),
                    "blocking_big_flexible_unit_count": int(blocking_big_flexible_unit_count),
                    "bad_big_flexible_route_flag": int(blocking_big_flexible_route_flag),
                    "bad_big_flexible_unit_count": int(blocking_big_flexible_unit_count),
                    "heavy_big_only_unit_count": int(heavy_big_only_unit_count),
                    "roles": tuple(sorted(roles)),
                    "reference_cost": singleton_cover_cost,
                    "promotion_saving": singleton_cover_cost - float(spec.route_eval.best_cost),
                    "current_cover_cost": current_cover_cost,
                    "current_cover_route_count": int(len(current_cover_keys)),
                    "current_cost_saving": current_cost_saving,
                    "saving_vs_pass1_singletons": residual_saving_by_key.get(key),
                    "avg_time_window_overlap_min": avg_time_window_overlap_min,
                    "avg_customer_distance_km": avg_customer_distance_km,
                    "candidate_family": candidate_family,
                    "pool_pass": self._column_pool_pass(roles),
                    "route": spec.route,
                    "route_eval": spec.route_eval,
                }
            )

        deduped_columns: list[dict[str, object]] = []
        for _, candidates in grouped_columns.items():
            candidates.sort(
                key=lambda item: (
                    item["best_cost"],
                    -len(item["unit_ids"]),
                    item["unit_ids"],
                )
            )
            best_candidate = dict(candidates[0])
            merged_roles = sorted({role for candidate in candidates for role in candidate["roles"]})
            residual_savings = [candidate["saving_vs_pass1_singletons"] for candidate in candidates if candidate["saving_vs_pass1_singletons"] is not None]
            best_candidate["roles"] = tuple(merged_roles)
            best_candidate["pool_pass"] = self._column_pool_pass(merged_roles)
            best_candidate["saving_vs_pass1_singletons"] = max(residual_savings) if residual_savings else None
            best_candidate["candidate_score"] = self._column_candidate_score(best_candidate)
            deduped_columns.append(best_candidate)

        support_columns: list[dict[str, object]] = []
        flex_small_columns: list[dict[str, object]] = []
        promotion_like_columns: list[dict[str, object]] = []
        piggyback_big_columns: list[dict[str, object]] = []
        overflow_columns: list[dict[str, object]] = []
        for column in deduped_columns:
            roles = set(column["roles"])
            if roles & {"seed", "current", "rigid_big", "singleton"}:
                support_columns.append(column)
            elif column["candidate_family"] == "piggyback_big" and float(column["current_cost_saving"]) > COST_IMPROVEMENT_EPS:
                piggyback_big_columns.append(column)
            elif (
                column["candidate_family"] == "promotion_like_big"
                and float(column["current_cost_saving"]) > COST_IMPROVEMENT_EPS
            ):
                promotion_like_columns.append(column)
            elif (
                column["candidate_family"] == "flex_small"
                and float(column["current_cost_saving"]) > COST_IMPROVEMENT_EPS
            ):
                flex_small_columns.append(column)
            else:
                overflow_columns.append(column)

        flex_small_columns.sort(key=self._column_candidate_sort_key)
        piggyback_big_columns.sort(key=self._column_candidate_sort_key)
        promotion_like_columns.sort(key=self._column_candidate_sort_key)
        overflow_columns.sort(key=self._column_candidate_sort_key)

        selected_columns: list[dict[str, object]] = []
        seen_keys: set[tuple[str, tuple[int, ...]]] = set()

        def add_columns(columns: Iterable[dict[str, object]]) -> None:
            for column in columns:
                if column["key"] in seen_keys:
                    continue
                selected_columns.append(column)
                seen_keys.add(column["key"])

        add_columns(support_columns)
        add_columns(flex_small_columns[:flex_small_limit])
        add_columns(piggyback_big_columns[:ROUTE_POOL_PIGGYBACK_BIG_LIMIT])
        add_columns(promotion_like_columns[:promotion_limit])

        if len(selected_columns) < max_columns:
            remaining_columns = [
                column
                for column in (
                    flex_small_columns[flex_small_limit:]
                    + piggyback_big_columns[ROUTE_POOL_PIGGYBACK_BIG_LIMIT:]
                    + promotion_like_columns[promotion_limit:]
                    + overflow_columns
                )
                if column["key"] not in seen_keys
            ]
            remaining_columns.sort(key=self._column_candidate_sort_key)
            add_columns(remaining_columns[: max(0, max_columns - len(selected_columns))])

        selected_columns.sort(key=lambda item: (item["vehicle_type"], item["unit_ids"]))
        return selected_columns

    def _build_set_partitioning_matrices(
        self,
        route_pool_columns: list[dict[str, object]],
    ) -> dict[str, object]:
        unit_ids = sorted(self.active_unit_ids)
        unit_to_idx = {unit_id: idx for idx, unit_id in enumerate(unit_ids)}
        vehicle_types = [vehicle.vehicle_type for vehicle in self.vehicles]
        vehicle_to_idx = {vehicle_type: idx for idx, vehicle_type in enumerate(vehicle_types)}
        row_indices: list[int] = []
        col_indices: list[int] = []
        data_values: list[int] = []
        vehicle_row_indices: list[int] = []
        vehicle_col_indices: list[int] = []
        vehicle_values: list[int] = []

        for col_idx, column in enumerate(route_pool_columns):
            for unit_id in column["unit_ids"]:
                row_indices.append(unit_to_idx[int(unit_id)])
                col_indices.append(col_idx)
                data_values.append(1)
            vehicle_row_indices.append(vehicle_to_idx[str(column["vehicle_type"])])
            vehicle_col_indices.append(col_idx)
            vehicle_values.append(1)

        cover_matrix = csc_matrix(
            (np.array(data_values, dtype=float), (np.array(row_indices), np.array(col_indices))),
            shape=(len(unit_ids), len(route_pool_columns)),
        )
        vehicle_matrix = csc_matrix(
            (np.array(vehicle_values, dtype=float), (np.array(vehicle_row_indices), np.array(vehicle_col_indices))),
            shape=(len(vehicle_types), len(route_pool_columns)),
        )

        metric_vectors = {
            "bad_big_flexible_route_count": np.array([float(column["bad_big_flexible_route_flag"]) for column in route_pool_columns], dtype=float),
            "bad_big_flexible_unit_count": np.array([float(column["bad_big_flexible_unit_count"]) for column in route_pool_columns], dtype=float),
            "blocking_big_flexible_route_count": np.array([float(column["blocking_big_flexible_route_flag"]) for column in route_pool_columns], dtype=float),
            "blocking_big_flexible_unit_count": np.array([float(column["blocking_big_flexible_unit_count"]) for column in route_pool_columns], dtype=float),
            "piggyback_big_route_count": np.array([float(column["piggyback_big_route_flag"]) for column in route_pool_columns], dtype=float),
            "mixed_big_route_count": np.array([float(column["mixed_big_route_flag"]) for column in route_pool_columns], dtype=float),
            "promotion_like_big_route_count": np.array([float(column["promotion_like_big_route_flag"]) for column in route_pool_columns], dtype=float),
            "big_route_count": np.array([float(column["big_route_flag"]) for column in route_pool_columns], dtype=float),
            "single_stop_count": np.array([float(column["single_stop_flag"]) for column in route_pool_columns], dtype=float),
            "route_count": np.ones(len(route_pool_columns), dtype=float),
            "total_cost": np.array([float(column["best_cost"]) for column in route_pool_columns], dtype=float),
        }

        cover_constraint = LinearConstraint(cover_matrix, np.ones(len(unit_ids)), np.ones(len(unit_ids)))
        vehicle_upper = np.array([float(self.vehicle_by_name[vehicle_type].vehicle_count) for vehicle_type in vehicle_types], dtype=float)
        vehicle_constraint = LinearConstraint(vehicle_matrix, -np.inf * np.ones(len(vehicle_types)), vehicle_upper)

        return {
            "columns": route_pool_columns,
            "unit_ids": unit_ids,
            "vehicle_types": vehicle_types,
            "cover_constraint": cover_constraint,
            "vehicle_constraint": vehicle_constraint,
            "metric_vectors": metric_vectors,
            "bounds": Bounds(np.zeros(len(route_pool_columns)), np.ones(len(route_pool_columns))),
            "integrality": np.ones(len(route_pool_columns), dtype=int),
        }

    @staticmethod
    def _selected_indices_from_vector(x: np.ndarray | None) -> list[int]:
        if x is None:
            return []
        return [idx for idx, value in enumerate(x) if float(value) >= 0.5]

    def _metric_value_from_indices(
        self,
        metric_vector: np.ndarray,
        selected_indices: list[int],
    ) -> float:
        if not selected_indices:
            return 0.0
        return float(metric_vector[np.array(selected_indices, dtype=int)].sum())

    def _build_metric_constraint(
        self,
        metric_vector: np.ndarray,
        target_value: float,
    ) -> LinearConstraint:
        rounded_target = float(round(target_value))
        return LinearConstraint(
            csc_matrix(metric_vector.reshape(1, -1)),
            np.array([rounded_target], dtype=float),
            np.array([rounded_target], dtype=float),
        )

    def _build_metric_upper_bound_constraint(
        self,
        metric_vector: np.ndarray,
        upper_bound: float,
    ) -> LinearConstraint:
        rounded_upper_bound = float(round(upper_bound))
        return LinearConstraint(
            csc_matrix(metric_vector.reshape(1, -1)),
            np.array([-np.inf], dtype=float),
            np.array([rounded_upper_bound], dtype=float),
        )

    def _selected_route_pool_metrics(
        self,
        route_pool_columns: list[dict[str, object]],
        selected_indices: list[int],
    ) -> dict[str, object]:
        selected_columns = [route_pool_columns[idx] for idx in selected_indices]
        return {
            "selected_singleton_count": sum(int(self._column_has_any_role(column, {"singleton"})) for column in selected_columns),
            "selected_flexible_singleton_count": sum(
                int(int(column["unit_count"]) == 1 and not self._column_has_any_role(column, {"rigid_big"}))
                for column in selected_columns
            ),
            "selected_rigid_big_singleton_count": sum(
                int(int(column["unit_count"]) == 1 and self._column_has_any_role(column, {"rigid_big"}))
                for column in selected_columns
            ),
            "selected_promotion_count": sum(int(self._column_has_any_role(column, {"promotion", "residual_promotion"})) for column in selected_columns),
            "selected_promotion_like_count": sum(int(column["promotion_like_big_route_flag"]) for column in selected_columns),
            "selected_piggyback_big_count": sum(int(column["piggyback_big_route_flag"]) for column in selected_columns),
            "selected_flex_small_count": sum(int(self._column_has_any_role(column, {"flex_small", "residual_flex_small", "cluster_flex_small"})) for column in selected_columns),
            "selected_flex_small_count_by_size": self._flex_small_count_by_size(selected_columns),
            "selected_mixed_big_route_count": sum(int(column["mixed_big_route_flag"]) for column in selected_columns),
            "selected_blocking_big_flexible_route_count": sum(int(column["blocking_big_flexible_route_flag"]) for column in selected_columns),
            "selected_blocking_big_flexible_unit_count": sum(int(column["blocking_big_flexible_unit_count"]) for column in selected_columns),
            "selected_bad_big_flexible_route_count": sum(int(column["bad_big_flexible_route_flag"]) for column in selected_columns),
            "selected_bad_big_flexible_unit_count": sum(int(column["bad_big_flexible_unit_count"]) for column in selected_columns),
            "big_route_count": sum(int(column["big_route_flag"]) for column in selected_columns),
        }

    def _milp_phase_status(
        self,
        phase_name: str,
        stage: str,
        result_status: str,
        fallback_used: bool,
        selected_indices: list[int],
        objective_value: float | None,
    ) -> dict[str, object]:
        return {
            "phase": phase_name,
            "stage": stage,
            "status": result_status,
            "fallback_used": int(fallback_used),
            "selected_route_count": len(selected_indices),
            "objective_value": None if objective_value is None else float(objective_value),
        }

    def _solve_milp_phase(
        self,
        model_bundle: dict[str, object],
        phase_name: str,
        stage: str,
        objective_vector: np.ndarray,
        fixed_constraints: list[LinearConstraint],
        fallback_objective_vector: np.ndarray | None = None,
    ) -> tuple[list[int] | None, dict[str, object]]:
        constraints = [
            model_bundle["cover_constraint"],
            model_bundle["vehicle_constraint"],
            *fixed_constraints,
        ]
        options = {
            "time_limit": MILP_PHASE_TIME_LIMIT_SEC,
            "disp": False,
            "mip_rel_gap": MILP_REL_GAP,
        }
        result = milp(
            c=objective_vector,
            integrality=model_bundle["integrality"],
            bounds=model_bundle["bounds"],
            constraints=constraints,
            options=options,
        )
        if bool(result.success) and result.x is not None:
            selected_indices = self._selected_indices_from_vector(result.x)
            return selected_indices, self._milp_phase_status(
                phase_name=phase_name,
                stage=stage,
                result_status="optimal",
                fallback_used=False,
                selected_indices=selected_indices,
                objective_value=float(result.fun),
            )

        if fallback_objective_vector is not None:
            fallback_objective = fallback_objective_vector
        elif stage == "unlock":
            fallback_objective = (
                1e6 * model_bundle["metric_vectors"]["bad_big_flexible_route_count"]
                + 1e4 * model_bundle["metric_vectors"]["bad_big_flexible_unit_count"]
                + 1e2 * model_bundle["metric_vectors"]["big_route_count"]
                + model_bundle["metric_vectors"]["total_cost"]
            )
        else:
            fallback_objective = (
                1e6 * model_bundle["metric_vectors"]["single_stop_count"]
                + 1e4 * model_bundle["metric_vectors"]["route_count"]
                + 1e1 * model_bundle["metric_vectors"]["big_route_count"]
                + model_bundle["metric_vectors"]["total_cost"]
            )
        fallback_result = milp(
            c=fallback_objective,
            integrality=model_bundle["integrality"],
            bounds=model_bundle["bounds"],
            constraints=constraints,
            options=options,
        )
        if fallback_result.x is None:
            return None, self._milp_phase_status(
                phase_name=phase_name,
                stage=stage,
                result_status="failed",
                fallback_used=True,
                selected_indices=[],
                objective_value=None,
            )
        selected_indices = self._selected_indices_from_vector(fallback_result.x)
        status = "fallback_success" if bool(fallback_result.success) else f"fallback_status_{getattr(fallback_result, 'status', 'unknown')}"
        return selected_indices, self._milp_phase_status(
            phase_name=phase_name,
            stage=stage,
            result_status=status,
            fallback_used=True,
            selected_indices=selected_indices,
            objective_value=float(getattr(fallback_result, "fun", np.nan)),
        )

    def _global_solution_selection_key(
        self,
        routes: list[TypedRoute],
        solution_eval: SolutionEvaluation,
    ) -> tuple[float, int, int, int, float, float]:
        return (
            round(solution_eval.total_cost, 6),
            abs(solution_eval.split_customer_count - solution_eval.mandatory_split_customer_count),
            solution_eval.route_count,
            solution_eval.single_stop_route_count,
            round(solution_eval.total_late_min, 6),
            round(solution_eval.total_carbon_kg, 6),
        )

    def _solve_global_milp_pass(
        self,
        route_pool_columns: list[dict[str, object]],
        pass_label: str,
    ) -> dict[str, object]:
        model_bundle = self._build_set_partitioning_matrices(route_pool_columns)
        selected_indices, phase_status = self._solve_milp_phase(
            model_bundle=model_bundle,
            phase_name="cost_first_total_cost",
            stage="cost_first",
            objective_vector=model_bundle["metric_vectors"]["total_cost"],
            fixed_constraints=[],
            fallback_objective_vector=(
                model_bundle["metric_vectors"]["total_cost"]
                + 1e-3 * model_bundle["metric_vectors"]["route_count"]
                + 1e-4 * model_bundle["metric_vectors"]["single_stop_count"]
            ),
        )
        phase_status["pass_label"] = pass_label
        phase_status["big_route_bound_mode"] = None
        if selected_indices is None:
            return {
                "status": "failed",
                "phase_statuses": [phase_status],
                "columns": route_pool_columns,
                "selected_routes": None,
                "selected_indices": [],
                "phase_metric_targets": {},
                "big_route_bound_mode": None,
            }

        selected_role_metrics = self._selected_route_pool_metrics(route_pool_columns, selected_indices)
        phase_metric_targets = {
            "total_cost": self._metric_value_from_indices(model_bundle["metric_vectors"]["total_cost"], selected_indices),
            "route_count": self._metric_value_from_indices(model_bundle["metric_vectors"]["route_count"], selected_indices),
            "single_stop_count": self._metric_value_from_indices(model_bundle["metric_vectors"]["single_stop_count"], selected_indices),
            "big_route_count": self._metric_value_from_indices(model_bundle["metric_vectors"]["big_route_count"], selected_indices),
            "mixed_big_route_count": self._metric_value_from_indices(model_bundle["metric_vectors"]["mixed_big_route_count"], selected_indices),
            "piggyback_big_route_count": self._metric_value_from_indices(model_bundle["metric_vectors"]["piggyback_big_route_count"], selected_indices),
            "promotion_like_big_route_count": self._metric_value_from_indices(
                model_bundle["metric_vectors"]["promotion_like_big_route_count"],
                selected_indices,
            ),
            "blocking_big_flexible_route_count": self._metric_value_from_indices(
                model_bundle["metric_vectors"]["blocking_big_flexible_route_count"],
                selected_indices,
            ),
            "blocking_big_flexible_unit_count": self._metric_value_from_indices(
                model_bundle["metric_vectors"]["blocking_big_flexible_unit_count"],
                selected_indices,
            ),
            "selected_piggyback_big_count": float(selected_role_metrics["selected_piggyback_big_count"]),
            "selected_flex_small_count": float(selected_role_metrics["selected_flex_small_count"]),
        }
        return {
            "status": "ok",
            "phase_statuses": [phase_status],
            "columns": route_pool_columns,
            "selected_routes": [route_pool_columns[idx]["route"] for idx in selected_indices],
            "selected_indices": selected_indices,
            "phase_metric_targets": phase_metric_targets,
            "big_route_bound_mode": None,
        }

    def _global_reoptimize_with_milp(
        self,
        baseline_routes: list[TypedRoute],
        baseline_solution_eval: SolutionEvaluation,
    ) -> dict[str, object]:
        pass1_columns = self._build_route_pool_columns(baseline_routes)
        pass1_result = self._solve_global_milp_pass(
            route_pool_columns=pass1_columns,
            pass_label="pass1",
        )

        pass2_result: dict[str, object] | None = None
        pass3_result: dict[str, object] | None = None
        pass1_routes: list[TypedRoute] | None = None
        if pass1_result["selected_routes"] is not None:
            pass1_routes = [self._two_opt_route(route) for route in pass1_result["selected_routes"]]
            pass1_eval = self.evaluate_solution(pass1_routes)
            if pass1_eval is not None:
                pass2_route_pool = self._generate_route_pool(pass1_routes)
                residual_saving_by_key = self._augment_route_pool_with_residual_candidates(pass2_route_pool, pass1_routes)
                residual_saving_by_key = self._augment_route_pool_with_pass3_flex_small_candidates(
                    pass2_route_pool,
                    pass1_result,
                    residual_saving_by_key,
                )
                pass2_columns = self._build_route_pool_columns(
                    pass1_routes,
                    route_pool=pass2_route_pool,
                    max_columns=ROUTE_POOL_PASS2_MAX_COLUMNS,
                    residual_saving_by_key=residual_saving_by_key,
                )
                pass2_result = self._solve_global_milp_pass(
                    route_pool_columns=pass2_columns,
                    pass_label="pass2",
                )

        final_columns = (
            pass2_result["columns"]
            if pass2_result is not None
            else pass1_result["columns"]
        )
        return {
            "status": "ok" if pass1_result["status"] == "ok" else "failed",
            "pass1": pass1_result,
            "pass2": pass2_result,
            "pass3": pass3_result,
            "columns": final_columns,
        }

    def _evaluate_global_pass_result(
        self,
        pass_result: dict[str, object] | None,
    ) -> dict[str, object]:
        summary = {
            "routes": None,
            "solution_eval": None,
            "route_counts": None,
            "routes_with_flexible_units_on_big": None,
            "flexible_units_on_big_routes": None,
            "mixed_big_route_count": None,
            "mixed_big_flexible_unit_count": None,
            "piggyback_big_count": None,
            "promotion_like_big_count": None,
            "blocking_big_flexible_route_count": None,
            "blocking_big_flexible_unit_count": None,
            "bad_big_flexible_route_count": None,
            "bad_big_flexible_unit_count": None,
            "current_single_pairs_feasible": None,
            "current_single_pairs_inventory_feasible": None,
            "selected_singleton_count": None,
            "selected_flexible_singleton_count": None,
            "selected_rigid_big_singleton_count": None,
            "selected_promotion_count": None,
            "selected_promotion_like_count": None,
            "selected_piggyback_big_count": None,
            "selected_flex_small_count": None,
            "selected_flex_small_count_by_size": {},
            "selected_mixed_big_route_count": None,
            "selected_blocking_big_flexible_route_count": None,
            "selected_blocking_big_flexible_unit_count": None,
            "selected_bad_big_flexible_route_count": None,
            "selected_bad_big_flexible_unit_count": None,
            "big_route_count": None,
            "selected_route_keys": set(),
            "big_route_bound_mode": None,
        }
        if pass_result is None or pass_result["selected_routes"] is None:
            return summary
        candidate_routes = [self._two_opt_route(route) for route in pass_result["selected_routes"]]
        candidate_eval = self.evaluate_solution(candidate_routes)
        if candidate_eval is None:
            return summary
        route_counts = self._route_counts(candidate_routes)
        big_route_diagnostics = self._solution_big_route_diagnostics(candidate_routes)
        current_single_pairs_feasible, current_single_pairs_inventory_feasible = self._current_single_pair_inventory_counts(candidate_routes)
        selected_role_metrics = self._selected_route_pool_metrics(pass_result["columns"], pass_result["selected_indices"])
        return {
            "routes": candidate_routes,
            "solution_eval": candidate_eval,
            "route_counts": route_counts,
            "routes_with_flexible_units_on_big": big_route_diagnostics["mixed_big_route_count"],
            "flexible_units_on_big_routes": big_route_diagnostics["mixed_big_flexible_unit_count"],
            "mixed_big_route_count": big_route_diagnostics["mixed_big_route_count"],
            "mixed_big_flexible_unit_count": big_route_diagnostics["mixed_big_flexible_unit_count"],
            "piggyback_big_count": big_route_diagnostics["piggyback_big_count"],
            "promotion_like_big_count": big_route_diagnostics["promotion_like_big_count"],
            "blocking_big_flexible_route_count": big_route_diagnostics["blocking_big_flexible_count"],
            "blocking_big_flexible_unit_count": big_route_diagnostics["blocking_big_flexible_unit_count"],
            "bad_big_flexible_route_count": big_route_diagnostics["blocking_big_flexible_count"],
            "bad_big_flexible_unit_count": big_route_diagnostics["blocking_big_flexible_unit_count"],
            "current_single_pairs_feasible": current_single_pairs_feasible,
            "current_single_pairs_inventory_feasible": current_single_pairs_inventory_feasible,
            "selected_singleton_count": selected_role_metrics["selected_singleton_count"],
            "selected_flexible_singleton_count": selected_role_metrics["selected_flexible_singleton_count"],
            "selected_rigid_big_singleton_count": selected_role_metrics["selected_rigid_big_singleton_count"],
            "selected_promotion_count": selected_role_metrics["selected_promotion_count"],
            "selected_promotion_like_count": selected_role_metrics["selected_promotion_like_count"],
            "selected_piggyback_big_count": selected_role_metrics["selected_piggyback_big_count"],
            "selected_flex_small_count": selected_role_metrics["selected_flex_small_count"],
            "selected_flex_small_count_by_size": selected_role_metrics["selected_flex_small_count_by_size"],
            "selected_mixed_big_route_count": selected_role_metrics["selected_mixed_big_route_count"],
            "selected_blocking_big_flexible_route_count": selected_role_metrics["selected_blocking_big_flexible_route_count"],
            "selected_blocking_big_flexible_unit_count": selected_role_metrics["selected_blocking_big_flexible_unit_count"],
            "selected_bad_big_flexible_route_count": selected_role_metrics["selected_bad_big_flexible_route_count"],
            "selected_bad_big_flexible_unit_count": selected_role_metrics["selected_bad_big_flexible_unit_count"],
            "big_route_count": selected_role_metrics["big_route_count"],
            "selected_route_keys": {
                (pass_result["columns"][idx]["vehicle_type"], pass_result["columns"][idx]["unit_ids"])
                for idx in pass_result["selected_indices"]
            },
            "big_route_bound_mode": pass_result["big_route_bound_mode"],
        }

    def _search_cluster_cover(
        self,
        cluster_routes: list[TypedRoute],
        current_routes: list[TypedRoute],
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec],
        phase: str,
        allowed_roles: set[str],
        max_cover_routes: int,
    ) -> list[TypedRoute] | None:
        cluster_unit_set = frozenset(unit_id for route in cluster_routes for unit_id in route.unit_ids)
        current_metric_tuple = self._aggregate_route_metric_tuple(cluster_routes)
        base_route_counts = self._route_counts(current_routes)
        for route in cluster_routes:
            base_route_counts[route.vehicle_type] -= 1
            if base_route_counts[route.vehicle_type] <= 0:
                base_route_counts.pop(route.vehicle_type, None)

        relevant_specs = [
            spec
            for spec in route_pool.values()
            if spec.roles & allowed_roles and set(spec.route.unit_ids).issubset(cluster_unit_set)
        ]
        if not relevant_specs:
            return None

        # Always allow the current route shapes inside the cluster search.
        local_pool = dict(route_pool)
        for route in cluster_routes:
            self._register_candidate_route(local_pool, route.vehicle_type, route.unit_ids, "current")
        relevant_specs = [
            spec
            for spec in local_pool.values()
            if spec.roles & (allowed_roles | {"current"}) and set(spec.route.unit_ids).issubset(cluster_unit_set)
        ]
        relevant_specs.sort(
            key=lambda spec: (
                -len(spec.route.unit_ids),
                0 if "promotion" in spec.roles else 1 if "rigid_big" in spec.roles else 2,
                spec.route_eval.best_cost,
                spec.route.vehicle_type,
                spec.route.unit_ids,
            )
        )
        relevant_specs = relevant_specs[:40]
        specs_by_unit: dict[int, list[int]] = defaultdict(list)
        for idx, spec in enumerate(relevant_specs):
            for unit_id in spec.route.unit_ids:
                specs_by_unit[unit_id].append(idx)

        best_cover_routes: list[TypedRoute] | None = None
        best_metric_key: tuple[object, ...] | None = None

        def metric_key(metric_tuple: tuple[int, int, int, int, int, float]) -> tuple[object, ...]:
            if phase == "release":
                return self._release_metric_key(metric_tuple)
            return self._promotion_metric_key(metric_tuple)

        current_metric_key = metric_key(current_metric_tuple)

        def dfs(
            uncovered_units: frozenset[int],
            selected_specs: list[CandidateRouteSpec],
            selected_metric_tuple: tuple[int, int, int, int, int, float],
            projected_counts: Counter[str],
        ) -> None:
            nonlocal best_cover_routes, best_metric_key
            if len(selected_specs) > max_cover_routes:
                return
            if not uncovered_units:
                candidate_key = metric_key(selected_metric_tuple)
                if candidate_key < current_metric_key and (best_metric_key is None or candidate_key < best_metric_key):
                    best_metric_key = candidate_key
                    best_cover_routes = [spec.route for spec in selected_specs]
                return
            pivot_unit_id = min(uncovered_units, key=lambda unit_id: len(specs_by_unit.get(unit_id, [])))
            for spec_idx in specs_by_unit.get(pivot_unit_id, []):
                spec = relevant_specs[spec_idx]
                spec_unit_set = set(spec.route.unit_ids)
                if not spec_unit_set.issubset(uncovered_units):
                    continue
                projected_count = projected_counts[spec.route.vehicle_type] + 1
                if projected_count > self.vehicle_by_name[spec.route.vehicle_type].vehicle_count:
                    continue
                new_counts = projected_counts.copy()
                new_counts[spec.route.vehicle_type] = projected_count
                route_metric = self._route_metric_tuple(spec.route, spec.route_eval)
                new_metric_tuple = (
                    selected_metric_tuple[0] + route_metric[0],
                    selected_metric_tuple[1] + route_metric[1],
                    selected_metric_tuple[2] + route_metric[2],
                    selected_metric_tuple[3] + route_metric[3],
                    selected_metric_tuple[4] + route_metric[4],
                    selected_metric_tuple[5] + route_metric[5],
                )
                dfs(
                    frozenset(uncovered_units - spec_unit_set),
                    selected_specs + [spec],
                    new_metric_tuple,
                    new_counts,
                )

        dfs(
            uncovered_units=cluster_unit_set,
            selected_specs=[],
            selected_metric_tuple=(0, 0, 0, 0, 0, 0.0),
            projected_counts=base_route_counts,
        )
        return best_cover_routes

    def _apply_cluster_rebuild(
        self,
        current_routes: list[TypedRoute],
        cluster_indices: tuple[int, ...],
        replacement_routes: list[TypedRoute],
    ) -> list[TypedRoute]:
        rebuilt_routes = [
            route
            for idx, route in enumerate(current_routes)
            if idx not in set(cluster_indices)
        ]
        rebuilt_routes.extend(replacement_routes)
        return rebuilt_routes

    def _try_release_with_route_pool(
        self,
        current_routes: list[TypedRoute],
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec],
    ) -> tuple[bool, list[TypedRoute], dict[str, object] | None]:
        prioritized_indices = sorted(
            range(len(current_routes)),
            key=lambda idx: (
                -self._route_big_flexible_metrics(current_routes[idx])[1],
                -len(current_routes[idx].unit_ids),
                idx,
            ),
        )
        for route_idx in prioritized_indices:
            route = current_routes[route_idx]
            flexible_metric = self._route_big_flexible_metrics(route)
            if flexible_metric[1] == 0:
                continue
            replacement_routes = self._search_cluster_cover(
                cluster_routes=[route],
                current_routes=current_routes,
                route_pool=route_pool,
                phase="release",
                allowed_roles={"seed", "singleton", "flex_small", "rigid_big", "current"},
                max_cover_routes=2,
            )
            if replacement_routes is None:
                continue
            candidate_routes = self._apply_cluster_rebuild(current_routes, (route_idx,), replacement_routes)
            if self.evaluate_solution(candidate_routes) is None:
                continue
            return True, candidate_routes, {
                "source_state": "diagnostic",
                "candidate_type": "release",
                "cluster_route_ids": str((route_idx + 1,)),
                "cluster_unit_ids": ",".join(str(unit_id) for unit_id in route.unit_ids),
                "old_route_count": 1,
                "new_route_count": len(replacement_routes),
                "old_flexible_big_routes": flexible_metric[0],
                "new_flexible_big_routes": sum(self._route_big_flexible_metrics(item)[0] for item in replacement_routes),
                "old_flexible_big_units": flexible_metric[1],
                "new_flexible_big_units": sum(self._route_big_flexible_metrics(item)[1] for item in replacement_routes),
                "accepted": 1,
            }
        return False, current_routes, None

    def _promotion_cluster_candidates(
        self,
        current_routes: list[TypedRoute],
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec],
    ) -> list[tuple[int, ...]]:
        unit_to_route_idx: dict[int, int] = {}
        for route_idx, route in enumerate(current_routes):
            for unit_id in route.unit_ids:
                unit_to_route_idx[unit_id] = route_idx
        clusters: list[tuple[int, ...]] = []
        seen_clusters: set[tuple[int, ...]] = set()
        for spec in route_pool.values():
            if "promotion" not in spec.roles:
                continue
            if len(spec.route.unit_ids) not in {2, 3}:
                continue
            cluster_indices = tuple(sorted({unit_to_route_idx[unit_id] for unit_id in spec.route.unit_ids}))
            if len(cluster_indices) <= 1 or len(cluster_indices) > 3:
                continue
            cluster_routes = [current_routes[idx] for idx in cluster_indices]
            if any(route.vehicle_type in {"fuel_3000", "ev_3000"} for route in cluster_routes):
                continue
            if any(len(route.unit_ids) > 2 for route in cluster_routes):
                continue
            if cluster_indices in seen_clusters:
                continue
            seen_clusters.add(cluster_indices)
            clusters.append(cluster_indices)
        clusters.sort(
            key=lambda cluster_indices: (
                -len(cluster_indices),
                -sum(int(len(current_routes[idx].unit_ids) == 1) for idx in cluster_indices),
                cluster_indices,
            )
        )
        return clusters

    def _try_promotion_with_route_pool(
        self,
        current_routes: list[TypedRoute],
        route_pool: dict[tuple[str, tuple[int, ...]], CandidateRouteSpec],
    ) -> tuple[bool, list[TypedRoute], dict[str, object] | None]:
        for cluster_indices in self._promotion_cluster_candidates(current_routes, route_pool):
            cluster_routes = [current_routes[idx] for idx in cluster_indices]
            old_single_stop_count = sum(int(len(route.unit_ids) == 1) for route in cluster_routes)
            replacement_routes = self._search_cluster_cover(
                cluster_routes=cluster_routes,
                current_routes=current_routes,
                route_pool=route_pool,
                phase="promotion",
                allowed_roles={"seed", "singleton", "flex_small", "promotion", "current"},
                max_cover_routes=len(cluster_routes),
            )
            if replacement_routes is None:
                continue
            new_single_stop_count = sum(int(len(route.unit_ids) == 1) for route in replacement_routes)
            if len(replacement_routes) >= len(cluster_routes) and new_single_stop_count >= old_single_stop_count:
                continue
            candidate_routes = self._apply_cluster_rebuild(current_routes, cluster_indices, replacement_routes)
            if self.evaluate_solution(candidate_routes) is None:
                continue
            return True, candidate_routes, {
                "source_state": "diagnostic",
                "candidate_type": "promotion",
                "cluster_route_ids": str(tuple(idx + 1 for idx in cluster_indices)),
                "cluster_unit_ids": ",".join(str(unit_id) for route in cluster_routes for unit_id in route.unit_ids),
                "old_route_count": len(cluster_routes),
                "new_route_count": len(replacement_routes),
                "old_single_stop_count": old_single_stop_count,
                "new_single_stop_count": new_single_stop_count,
                "accepted": 1,
            }
        return False, current_routes, None

    def _route_pool_solution_key(
        self,
        routes: list[TypedRoute],
        solution_eval: SolutionEvaluation,
    ) -> tuple[int, int, int, int, float]:
        routes_with_flexible_on_big, flexible_units_on_big = self._final_solution_structure_metrics(routes)
        return (
            routes_with_flexible_on_big,
            flexible_units_on_big,
            solution_eval.single_stop_route_count,
            solution_eval.route_count,
            round(solution_eval.total_cost, 6),
        )

    def _route_pool_optimize_solution(
        self,
        solution: list[TypedRoute],
    ) -> tuple[list[TypedRoute], dict[str, object]]:
        current_routes = [self._two_opt_route(route) for route in solution]
        route_pool = self._generate_route_pool(current_routes)
        diagnostics: list[dict[str, object]] = []
        stats = {
            "route_pool_release_success_count": 0,
            "route_pool_promotion_success_count": 0,
        }

        release_improved = True
        release_loops = 0
        while release_improved and release_loops < 12:
            release_improved = False
            changed, current_routes, diag_row = self._try_release_with_route_pool(current_routes, route_pool)
            if changed:
                release_improved = True
                release_loops += 1
                stats["route_pool_release_success_count"] += 1
                if diag_row is not None:
                    diagnostics.append(diag_row)
                for route in current_routes:
                    self._register_candidate_route(route_pool, route.vehicle_type, route.unit_ids, "current")
                if self._fuel_3000_free_count(self._route_counts(current_routes)) >= FUEL_3000_SEARCH_RESERVE:
                    break

        promotion_improved = True
        promotion_loops = 0
        while promotion_improved and promotion_loops < 12:
            promotion_improved = False
            changed, current_routes, diag_row = self._try_promotion_with_route_pool(current_routes, route_pool)
            if changed:
                promotion_improved = True
                promotion_loops += 1
                stats["route_pool_promotion_success_count"] += 1
                if diag_row is not None:
                    diagnostics.append(diag_row)
                for route in current_routes:
                    self._register_candidate_route(route_pool, route.vehicle_type, route.unit_ids, "current")

        return current_routes, {
            **stats,
            "route_pool_candidate_count": len(route_pool),
            "route_pool_role_counts": self._route_pool_role_counts(route_pool),
            "route_pool_diagnostics_rows": diagnostics[:SINGLE_MERGE_SAMPLE_LIMIT],
        }

    def _current_single_pair_inventory_counts(self, routes: list[TypedRoute]) -> tuple[int, int]:
        route_counts = self._route_counts(routes)
        single_routes = [
            route
            for route in routes
            if len(route.unit_ids) == 1 and route.vehicle_type in {"fuel_1500", "fuel_1250", "ev_1250"}
        ]
        all_feasible = 0
        inventory_feasible = 0
        for left_route, right_route in combinations(single_routes, 2):
            left_unit = self.unit_by_id[left_route.unit_ids[0]]
            right_unit = self.unit_by_id[right_route.unit_ids[0]]
            if left_unit.orig_cust_id == right_unit.orig_cust_id:
                continue
            shared_vehicle_types = set(left_unit.eligible_vehicle_types) & set(right_unit.eligible_vehicle_types)
            if not ({"fuel_3000", "ev_3000"} & shared_vehicle_types):
                continue
            pair_is_feasible = False
            pair_is_inventory_feasible = False
            for vehicle_type in ("fuel_3000", "ev_3000"):
                if vehicle_type not in shared_vehicle_types:
                    continue
                projected_count = (
                    route_counts.get(vehicle_type, 0)
                    + 1
                    - int(left_route.vehicle_type == vehicle_type)
                    - int(right_route.vehicle_type == vehicle_type)
                )
                inventory_ok = projected_count <= self.vehicle_by_name[vehicle_type].vehicle_count
                for ordered_unit_ids in ((left_route.unit_ids[0], right_route.unit_ids[0]), (right_route.unit_ids[0], left_route.unit_ids[0])):
                    candidate_eval = self.evaluate_route(TypedRoute(vehicle_type=vehicle_type, unit_ids=ordered_unit_ids))
                    if not candidate_eval.feasible:
                        continue
                    pair_is_feasible = True
                    if inventory_ok:
                        pair_is_inventory_feasible = True
                    break
            all_feasible += int(pair_is_feasible)
            inventory_feasible += int(pair_is_inventory_feasible)
        return all_feasible, inventory_feasible

    def _two_opt_route(self, route: TypedRoute) -> TypedRoute:
        if len(route.unit_ids) < 4:
            return route
        best_route = route
        best_cost = self.evaluate_route(route).best_cost
        improved = True
        while improved:
            improved = False
            for left in range(1, len(best_route.unit_ids) - 2):
                for right in range(left + 1, len(best_route.unit_ids) - 1):
                    candidate_unit_ids = (
                        best_route.unit_ids[:left]
                        + tuple(reversed(best_route.unit_ids[left : right + 1]))
                        + best_route.unit_ids[right + 1 :]
                    )
                    candidate_route = TypedRoute(best_route.vehicle_type, candidate_unit_ids)
                    candidate_eval = self.evaluate_route(candidate_route)
                    if candidate_eval.feasible and candidate_eval.best_cost + 1e-6 < best_cost:
                        best_route = candidate_route
                        best_cost = candidate_eval.best_cost
                        improved = True
                        break
                if improved:
                    break
        return best_route

    def _try_route_merge(self, routes: list[TypedRoute]) -> tuple[bool, list[TypedRoute]]:
        base_eval = self.evaluate_solution(routes)
        if base_eval is None:
            return False, routes
        route_counts = Counter(route.vehicle_type for route in routes)
        route_evals = [self.evaluate_route(route) for route in routes]
        prioritized_indices = sorted(
            range(len(routes)),
            key=lambda idx: (
                0 if len(routes[idx].unit_ids) == 1 else 1 if len(routes[idx].unit_ids) == 2 else 2,
                len(routes[idx].unit_ids),
                idx,
            ),
        )[:MERGE_ROUTE_LIMIT]
        pair_count = 0
        for left_order, left_idx in enumerate(prioritized_indices):
            left_route = routes[left_idx]
            if len(left_route.unit_ids) > 2:
                continue
            left_customers = self._route_customers(left_route)
            for right_idx in prioritized_indices[left_order + 1 :]:
                pair_count += 1
                if pair_count > MERGE_PAIR_LIMIT:
                    return False, routes
                right_route = routes[right_idx]
                if len(right_route.unit_ids) > 2:
                    continue
                right_customers = self._route_customers(right_route)
                if left_customers & right_customers:
                    continue
                common_vehicle_types = [left_route.vehicle_type] if left_route.vehicle_type == right_route.vehicle_type else []
                current_units = [self.unit_by_id[unit_id] for unit_id in (*left_route.unit_ids, *right_route.unit_ids)]
                shared_types = set(current_units[0].eligible_vehicle_types)
                for unit in current_units[1:]:
                    shared_types &= set(unit.eligible_vehicle_types)
                for vehicle_type in sorted(shared_types, key=lambda name: self.vehicle_size_rank[name]):
                    if vehicle_type not in common_vehicle_types:
                        common_vehicle_types.append(vehicle_type)
                sequences = {
                    left_route.unit_ids + right_route.unit_ids,
                    right_route.unit_ids + left_route.unit_ids,
                    tuple(reversed(left_route.unit_ids)) + right_route.unit_ids,
                    left_route.unit_ids + tuple(reversed(right_route.unit_ids)),
                    tuple(reversed(right_route.unit_ids)) + left_route.unit_ids,
                    right_route.unit_ids + tuple(reversed(left_route.unit_ids)),
                }
                kept_routes = [route for idx, route in enumerate(routes) if idx not in {left_idx, right_idx}]
                base_pair_cost = route_evals[left_idx].best_cost + route_evals[right_idx].best_cost
                for vehicle_type in common_vehicle_types:
                    projected_count = (
                        route_counts[vehicle_type]
                        + 1
                        - int(left_route.vehicle_type == vehicle_type)
                        - int(right_route.vehicle_type == vehicle_type)
                    )
                    if projected_count > self.vehicle_by_name[vehicle_type].vehicle_count:
                        continue
                    for unit_ids in sequences:
                        candidate_route = TypedRoute(vehicle_type=vehicle_type, unit_ids=unit_ids)
                        route_eval = self.evaluate_route(candidate_route)
                        if not route_eval.feasible:
                            continue
                        if route_eval.best_cost <= base_pair_cost + ROUTE_MERGE_COST_ALLOWANCE:
                            return True, kept_routes + [candidate_route]
        return False, routes

    def _try_route_type_change(self, routes: list[TypedRoute]) -> tuple[bool, list[TypedRoute]]:
        route_counts = Counter(route.vehicle_type for route in routes)
        for idx, route in enumerate(routes):
            current_eval = self.evaluate_route(route)
            current_units = [self.unit_by_id[unit_id] for unit_id in route.unit_ids]
            common_vehicle_types = set(current_units[0].eligible_vehicle_types)
            for unit in current_units[1:]:
                common_vehicle_types &= set(unit.eligible_vehicle_types)
            candidate_vehicle_types = [
                vehicle_type
                for vehicle_type in common_vehicle_types
                if vehicle_type != route.vehicle_type
            ]
            for vehicle_type in sorted(candidate_vehicle_types, key=lambda name: self.vehicle_size_rank[name]):
                candidate_route = TypedRoute(vehicle_type, route.unit_ids)
                candidate_eval = self.evaluate_route(candidate_route)
                if candidate_eval.feasible:
                    projected_count = route_counts[vehicle_type] + 1
                    if route.vehicle_type == vehicle_type:
                        projected_count -= 1
                    if projected_count > self.vehicle_by_name[vehicle_type].vehicle_count:
                        continue
                    if candidate_eval.best_cost <= current_eval.best_cost + ROUTE_TYPE_CHANGE_COST_ALLOWANCE:
                        updated = list(routes)
                        updated[idx] = candidate_route
                        return True, updated
        return False, routes

    def _try_relocate(self, routes: list[TypedRoute]) -> tuple[bool, list[TypedRoute]]:
        base_eval = self.evaluate_solution(routes)
        if base_eval is None:
            return False, routes
        base_cost = base_eval.total_cost
        route_evals = [self.evaluate_route(route) for route in routes]
        prioritized_source_indices = sorted(
            range(len(routes)),
            key=lambda idx: (
                0 if len(routes[idx].unit_ids) == 1 else 1 if len(routes[idx].unit_ids) == 2 else 2,
                len(routes[idx].unit_ids),
                idx,
            ),
        )[:RELOCATE_ROUTE_LIMIT]
        for source_idx in prioritized_source_indices:
            source_route = routes[source_idx]
            for position, unit_id in enumerate(source_route.unit_ids):
                unit = self.unit_by_id[unit_id]
                source_remainder = source_route.unit_ids[:position] + source_route.unit_ids[position + 1 :]
                source_routes = list(routes)
                new_source_eval = None
                if source_remainder:
                    new_source_route = TypedRoute(source_route.vehicle_type, source_remainder)
                    new_source_eval = self.evaluate_route(new_source_route)
                    if not new_source_eval.feasible:
                        continue
                    source_routes[source_idx] = new_source_route
                else:
                    source_routes.pop(source_idx)
                prioritized_target_indices = sorted(
                    range(len(source_routes)),
                    key=lambda idx: (
                        0 if len(source_routes[idx].unit_ids) == 1 else 1 if len(source_routes[idx].unit_ids) == 2 else 2,
                        len(source_routes[idx].unit_ids),
                        idx,
                    ),
                )
                for target_idx in prioritized_target_indices:
                    target_route = source_routes[target_idx]
                    if target_route.vehicle_type not in unit.eligible_vehicle_types:
                        continue
                    if unit.orig_cust_id in self._route_customers(target_route):
                        continue
                    for insert_pos in range(len(target_route.unit_ids) + 1):
                        target_unit_ids = target_route.unit_ids[:insert_pos] + (unit_id,) + target_route.unit_ids[insert_pos:]
                        candidate_routes = list(source_routes)
                        new_target_route = TypedRoute(target_route.vehicle_type, target_unit_ids)
                        new_target_eval = self.evaluate_route(new_target_route)
                        if not new_target_eval.feasible:
                            continue
                        candidate_routes[target_idx] = new_target_route
                        local_cost = base_cost - route_evals[source_idx].best_cost - self.evaluate_route(target_route).best_cost
                        if new_source_eval is not None:
                            local_cost += new_source_eval.best_cost
                        local_cost += new_target_eval.best_cost
                        if local_cost <= base_cost + (0.0 if source_remainder else RELOCATE_REMOVE_COST_ALLOWANCE):
                            return True, candidate_routes
        return False, routes

    def _try_swap(self, routes: list[TypedRoute]) -> tuple[bool, list[TypedRoute]]:
        base_eval = self.evaluate_solution(routes)
        if base_eval is None:
            return False, routes
        base_cost = base_eval.total_cost
        for left_idx in range(len(routes)):
            left_route = routes[left_idx]
            left_customers = self._route_customers(left_route)
            for right_idx in range(left_idx + 1, len(routes)):
                right_route = routes[right_idx]
                right_customers = self._route_customers(right_route)
                for left_pos, left_unit_id in enumerate(left_route.unit_ids):
                    left_unit = self.unit_by_id[left_unit_id]
                    for right_pos, right_unit_id in enumerate(right_route.unit_ids):
                        right_unit = self.unit_by_id[right_unit_id]
                        if left_route.vehicle_type not in right_unit.eligible_vehicle_types:
                            continue
                        if right_route.vehicle_type not in left_unit.eligible_vehicle_types:
                            continue
                        if right_unit.orig_cust_id in left_customers - {left_unit.orig_cust_id}:
                            continue
                        if left_unit.orig_cust_id in right_customers - {right_unit.orig_cust_id}:
                            continue
                        left_units = list(left_route.unit_ids)
                        right_units = list(right_route.unit_ids)
                        left_units[left_pos], right_units[right_pos] = right_unit_id, left_unit_id
                        candidate_routes = list(routes)
                        candidate_routes[left_idx] = TypedRoute(left_route.vehicle_type, tuple(left_units))
                        candidate_routes[right_idx] = TypedRoute(right_route.vehicle_type, tuple(right_units))
                        candidate_eval = self.evaluate_solution(candidate_routes)
                        if candidate_eval is not None and candidate_eval.total_cost + 1e-6 < base_cost:
                            return True, candidate_routes
        return False, routes

    def improve_solution(self, solution: list[TypedRoute]) -> tuple[list[TypedRoute], dict[str, int]]:
        routes = [self._two_opt_route(route) for route in solution]
        stats = {
            "route_merge_success_count": 0,
            "relocate_success_count": 0,
            "route_type_change_success_count": 0,
        }
        changed, routes = self._try_route_merge(routes)
        if changed:
            stats["route_merge_success_count"] += 1
        changed, routes = self._try_route_type_change(routes)
        if changed:
            stats["route_type_change_success_count"] += 1
        changed, routes = self._try_relocate(routes)
        if changed:
            stats["relocate_success_count"] += 1
        changed, routes = self._try_route_type_change(routes)
        if changed:
            stats["route_type_change_success_count"] += 1
        return [route for route in routes if route.unit_ids], stats

    def _compute_remove_count(self, generation: int, rng: random.Random) -> int:
        cauchy_draw = rng.random()
        cauchy_term = math.tan(math.pi * (cauchy_draw - 0.5))
        anneal = 1.0 - generation / max(self.max_generations, 1)
        remove_count = int(round(BASE_REMOVE_COUNT + 4.0 * cauchy_term * anneal))
        return int(max(REMOVE_MIN, min(REMOVE_MAX, remove_count)))

    def _mutate_particle(
        self,
        solution: list[TypedRoute],
        solution_eval: SolutionEvaluation,
        generation: int,
        rng: random.Random,
    ) -> tuple[list[TypedRoute] | None, SolutionEvaluation | None, str]:
        q_remove = self._compute_remove_count(generation, rng)
        draw = rng.random()
        if draw < 0.2:
            operator_name = "random_remove"
            partial_routes, removed = self._random_remove(solution, q_remove, rng)
        elif draw < 0.4:
            operator_name = "worst_cost_remove"
            partial_routes, removed = self._worst_cost_remove(solution_eval, q_remove, rng)
        elif draw < 0.6:
            operator_name = "late_route_remove"
            partial_routes, removed = self._late_route_remove(solution_eval, q_remove)
        elif draw < 0.8:
            operator_name = "typed_route_merge_remove"
            partial_routes, removed = self._typed_route_merge_remove(solution_eval, q_remove)
        else:
            operator_name = "mandatory_split_cluster_remove"
            partial_routes, removed = self._mandatory_split_cluster_remove(solution_eval, q_remove, rng)
        repaired = self.repair_solution(partial_routes, removed)
        if repaired is None:
            return None, None, operator_name
        improved, _ = self.improve_solution(repaired)
        evaluated = self.evaluate_solution(improved)
        return improved, evaluated, operator_name

    def _solve_single_configuration(self) -> tuple[SolutionEvaluation, list[TypedRoute], dict[str, object]]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        best_solution_eval: SolutionEvaluation | None = None
        best_solution_routes: list[TypedRoute] | None = None
        best_seed = None
        run_records: list[dict[str, object]] = []
        best_operator_stats = {
            "route_merge_success_count": 0,
            "relocate_success_count": 0,
            "route_type_change_success_count": 0,
        }
        search_start = time.perf_counter()

        for seed in self.seed_list:
            rng = random.Random(seed)
            seed_best_routes: list[TypedRoute] | None = None
            seed_best_eval: SolutionEvaluation | None = None
            seed_best_stats = {
                "route_merge_success_count": 0,
                "relocate_success_count": 0,
                "route_type_change_success_count": 0,
            }
            for _ in range(self.particle_count):
                candidate = self._build_initial_solution(rng)
                candidate, candidate_stats = self.improve_solution(candidate)
                evaluated = self.evaluate_solution(candidate)
                if evaluated is None:
                    continue
                if seed_best_eval is None or self._solution_rank_key(evaluated) < self._solution_rank_key(seed_best_eval):
                    seed_best_routes = candidate
                    seed_best_eval = evaluated
                    seed_best_stats = dict(candidate_stats)

            if seed_best_eval is None or seed_best_routes is None:
                continue
            seed_incumbent_eval = seed_best_eval
            if best_solution_eval is None or self._solution_rank_key(seed_best_eval) < self._solution_rank_key(best_solution_eval):
                best_solution_eval = seed_best_eval
                best_solution_routes = seed_best_routes
                best_seed = seed
                best_operator_stats = dict(seed_best_stats)

            for _ in range(max(self.max_generations - 1, 0)):
                refined_routes, refined_stats = self.improve_solution(seed_best_routes)
                refined_eval = self.evaluate_solution(refined_routes)
                if refined_eval is None or self._solution_rank_key(refined_eval) >= self._solution_rank_key(seed_incumbent_eval):
                    break
                seed_best_routes = refined_routes
                seed_incumbent_eval = refined_eval
                for key, value in refined_stats.items():
                    seed_best_stats[key] += value
                if best_solution_eval is None or self._solution_rank_key(refined_eval) < self._solution_rank_key(best_solution_eval):
                    best_solution_eval = refined_eval
                    best_solution_routes = refined_routes
                    best_seed = seed
                    best_operator_stats = dict(seed_best_stats)

            run_records.append(
                {
                    "seed": seed,
                    "best_cost": seed_incumbent_eval.total_cost,
                    "route_count": seed_incumbent_eval.route_count,
                    "used_vehicle_count": seed_incumbent_eval.used_vehicle_count,
                    "split_customer_count": seed_incumbent_eval.split_customer_count,
                    "single_stop_route_count": seed_incumbent_eval.single_stop_route_count,
                    "late_positive_stops": seed_incumbent_eval.late_positive_stops,
                    "latest_return_min": seed_incumbent_eval.latest_return_min,
                }
            )

        if best_solution_eval is None or best_solution_routes is None:
            raise RuntimeError("Solver failed to produce a feasible solution")

        baseline_solution_eval = best_solution_eval
        baseline_solution_routes = list(best_solution_routes)
        baseline_route_counts = self._route_counts(baseline_solution_routes)
        baseline_big_route_diagnostics = self._solution_big_route_diagnostics(baseline_solution_routes)
        baseline_bad_big_flexible_route_count = baseline_big_route_diagnostics["blocking_big_flexible_count"]
        baseline_bad_big_flexible_unit_count = baseline_big_route_diagnostics["blocking_big_flexible_unit_count"]
        baseline_current_single_pairs_feasible, baseline_current_single_pairs_inventory_feasible = self._current_single_pair_inventory_counts(
            baseline_solution_routes
        )

        global_result = self._global_reoptimize_with_milp(baseline_solution_routes, baseline_solution_eval)
        pass1_result = global_result["pass1"]
        pass2_result = global_result["pass2"]
        pass3_result = global_result["pass3"]
        pass1_summary = self._evaluate_global_pass_result(pass1_result)
        pass2_summary = self._evaluate_global_pass_result(pass2_result)
        pass3_summary = self._evaluate_global_pass_result(pass3_result)

        available_global_summaries = [
            ("pass1", pass1_summary),
            ("pass2", pass2_summary),
            ("pass3", pass3_summary),
        ]
        feasible_global_summaries = [
            (label, summary)
            for label, summary in available_global_summaries
            if summary["solution_eval"] is not None
        ]
        if feasible_global_summaries:
            active_global_label, active_global_summary = min(
                feasible_global_summaries,
                key=lambda item: self._global_solution_selection_key(
                    item[1]["routes"],
                    item[1]["solution_eval"],
                ),
            )
        else:
            active_global_summary = pass1_summary
            active_global_label = "pass1"

        best_solution_routes = baseline_solution_routes
        best_solution_eval = baseline_solution_eval
        final_solution_source = "baseline"
        global_validation_status = "cost_first_model_failed"
        cost_first_improved = False
        if active_global_summary["solution_eval"] is not None:
            candidate_eval = active_global_summary["solution_eval"]
            split_guard_ok = (
                candidate_eval.split_customer_count == candidate_eval.mandatory_split_customer_count
            )
            cost_improved = (
                split_guard_ok
                and candidate_eval.total_cost + COST_IMPROVEMENT_EPS < baseline_solution_eval.total_cost
            )
            if not split_guard_ok:
                global_validation_status = "cost_first_split_guard_failed"
            elif cost_improved:
                global_validation_status = "cost_first_improved"
            else:
                global_validation_status = "cost_first_no_improvement"
            if cost_improved and (
                self._global_solution_selection_key(
                    active_global_summary["routes"],
                    candidate_eval,
                )
                < self._global_solution_selection_key(
                    baseline_solution_routes,
                    baseline_solution_eval,
                )
            ):
                best_solution_routes = active_global_summary["routes"]
                best_solution_eval = candidate_eval
                final_solution_source = active_global_label
                cost_first_improved = True

        elapsed_sec = time.perf_counter() - search_start
        final_route_counts = self._route_counts(best_solution_routes)
        final_big_route_diagnostics = self._solution_big_route_diagnostics(best_solution_routes)
        final_bad_big_flexible_route_count = final_big_route_diagnostics["blocking_big_flexible_count"]
        final_bad_big_flexible_unit_count = final_big_route_diagnostics["blocking_big_flexible_unit_count"]
        final_current_single_pairs_feasible, final_current_single_pairs_inventory_feasible = self._current_single_pair_inventory_counts(best_solution_routes)
        selected_route_pool_role_counts = Counter[str]()
        for column in global_result["columns"]:
            for role in column["roles"]:
                selected_route_pool_role_counts[role] += 1
        promotion_like_candidate_count = sum(int(column["promotion_like_big_route_flag"]) for column in global_result["columns"])
        positive_saving_promotion_like_candidate_count = sum(
            int(
                column["promotion_like_big_route_flag"]
                and float(column["current_cost_saving"]) > COST_IMPROVEMENT_EPS
            )
            for column in global_result["columns"]
        )
        piggyback_big_candidate_count = sum(int(column["piggyback_big_route_flag"]) for column in global_result["columns"])
        positive_saving_piggyback_big_candidate_count = sum(
            int(
                column["piggyback_big_route_flag"]
                and float(column["current_cost_saving"]) > COST_IMPROVEMENT_EPS
            )
            for column in global_result["columns"]
        )
        baseline_route_key_set = {self._route_key(route) for route in baseline_solution_routes}
        pass1_selected_route_keys = pass1_summary["selected_route_keys"]
        pass2_selected_route_keys = pass2_summary["selected_route_keys"]
        pass3_selected_route_keys = pass3_summary["selected_route_keys"]
        active_global_selected_route_keys = active_global_summary["selected_route_keys"]
        flex_small_candidate_count_by_size = self._flex_small_count_by_size(global_result["columns"])
        selected_flex_small_count_by_size = active_global_summary["selected_flex_small_count_by_size"] or {}
        active_column_limit = (
            ROUTE_POOL_PASS2_MAX_COLUMNS
            if pass3_result is not None or pass2_result is not None
            else ROUTE_POOL_MAX_COLUMNS
        )
        candidate_pool_cap_binding_flag = int(len(global_result["columns"]) >= 0.95 * active_column_limit)
        route_pool_summary_rows = [
            {
                "route_key": f"{column['vehicle_type']}|{','.join(str(unit_id) for unit_id in column['unit_ids'])}",
                "vehicle_type": column["vehicle_type"],
                "unit_ids": ",".join(str(unit_id) for unit_id in column["unit_ids"]),
                "unit_count": int(column["unit_count"]),
                "best_cost": round(float(column["best_cost"]), 6),
                "current_cover_cost": round(float(column["current_cover_cost"]), 6),
                "current_cover_route_count": int(column["current_cover_route_count"]),
                "current_cost_saving": round(float(column["current_cost_saving"]), 6),
                "single_stop_flag": int(column["single_stop_flag"]),
                "big_route_flag": int(column["big_route_flag"]),
                "big_flexible_route_flag": int(column["big_flexible_route_flag"]),
                "big_flexible_unit_count": int(column["big_flexible_unit_count"]),
                "mixed_big_route_flag": int(column["mixed_big_route_flag"]),
                "piggyback_big_route_flag": int(column["piggyback_big_route_flag"]),
                "promotion_like_big_route_flag": int(column["promotion_like_big_route_flag"]),
                "blocking_big_flexible_route_flag": int(column["blocking_big_flexible_route_flag"]),
                "blocking_big_flexible_unit_count": int(column["blocking_big_flexible_unit_count"]),
                "bad_big_flexible_route_flag": int(column["bad_big_flexible_route_flag"]),
                "bad_big_flexible_unit_count": int(column["bad_big_flexible_unit_count"]),
                "avg_time_window_overlap_min": round(float(column["avg_time_window_overlap_min"]), 6),
                "candidate_score": round(float(column["candidate_score"]), 6),
                "candidate_family": column["candidate_family"],
                "pool_pass": column["pool_pass"],
                "roles": ",".join(column["roles"]),
                "saving_vs_pass1_singletons": None
                if column["saving_vs_pass1_singletons"] is None
                else round(float(column["saving_vs_pass1_singletons"]), 6),
                "selected_in_baseline": int((column["vehicle_type"], column["unit_ids"]) in baseline_route_key_set),
                "selected_in_global_pass1": int((column["vehicle_type"], column["unit_ids"]) in pass1_selected_route_keys),
                "selected_in_global_pass2": int((column["vehicle_type"], column["unit_ids"]) in pass2_selected_route_keys),
                "selected_in_global_pass3": int((column["vehicle_type"], column["unit_ids"]) in pass3_selected_route_keys),
                "selected_in_global": int((column["vehicle_type"], column["unit_ids"]) in active_global_selected_route_keys),
            }
            for column in global_result["columns"]
        ]
        global_phase_statuses = [
            *(pass1_result["phase_statuses"] if pass1_result is not None else []),
            *(pass2_result["phase_statuses"] if pass2_result is not None else []),
            *(pass3_result["phase_statuses"] if pass3_result is not None else []),
        ]
        active_global_eval = active_global_summary["solution_eval"]
        active_global_route_counts = active_global_summary["route_counts"]
        metadata = {
            "best_seed": best_seed,
            "elapsed_sec": elapsed_sec,
            "run_records": run_records,
            "packing_strategy": self.packing_strategy,
            "route_pool_iteration_count": COST_FIRST_ROUTE_POOL_ITERATIONS,
            "cost_first_improved": int(cost_first_improved),
            "service_unit_count": len(self.service_units),
            "route_cache_size": len(self.route_cache._cache),
            "mandatory_split_customer_count": self.service_unit_summary["mandatory_split_customer_count"],
            "heavy_big_only_count": self.service_unit_summary["mandatory_heavy_big_only_count"],
            "total_heavy_big_only_count": self.service_unit_summary["heavy_big_only_count"],
            "normal_heavy_big_only_count": self.service_unit_summary["normal_heavy_big_only_count"],
            "heavy_big_only_capacity": self.service_unit_summary["heavy_big_only_capacity"],
            "big_vehicle_inventory": self.service_unit_summary["big_vehicle_inventory"],
            "big_vehicle_reserve": self.service_unit_summary["big_vehicle_reserve"],
            "fuel_3000_used_count": final_route_counts.get("fuel_3000", 0),
            "fuel_3000_free_count": self._fuel_3000_free_count(final_route_counts),
            "single_single_merge_feasible_pair_count": final_current_single_pairs_feasible,
            "single_single_merge_inventory_blocked_pair_count": max(
                final_current_single_pairs_feasible - final_current_single_pairs_inventory_feasible,
                0,
            ),
            "reserve_repair_success_count": 0,
            "batch_merge_success_count": 0,
            "pre_merge_single_stop_route_count": baseline_solution_eval.single_stop_route_count,
            "post_merge_single_stop_route_count": best_solution_eval.single_stop_route_count,
            "merge_diagnostics_rows": global_phase_statuses,
            "final_routes_with_flexible_units_on_big": final_big_route_diagnostics["mixed_big_route_count"],
            "final_flexible_units_on_big_routes": final_big_route_diagnostics["mixed_big_flexible_unit_count"],
            "final_piggyback_big_count": final_big_route_diagnostics["piggyback_big_count"],
            "final_promotion_like_big_count": final_big_route_diagnostics["promotion_like_big_count"],
            "final_blocking_big_flexible_count": final_big_route_diagnostics["blocking_big_flexible_count"],
            "final_blocking_big_flexible_unit_count": final_big_route_diagnostics["blocking_big_flexible_unit_count"],
            "final_current_single_pairs_inventory_feasible": final_current_single_pairs_inventory_feasible,
            "diagnostic_unlock_success_count": 0,
            "diagnostic_promotion_success_count": 0,
            "route_pool_candidate_count": len(global_result["columns"]),
            "route_pool_role_counts": dict(selected_route_pool_role_counts),
            "promotion_like_candidate_count": promotion_like_candidate_count,
            "positive_saving_promotion_like_candidate_count": positive_saving_promotion_like_candidate_count,
            "piggyback_big_candidate_count": piggyback_big_candidate_count,
            "positive_saving_piggyback_big_candidate_count": positive_saving_piggyback_big_candidate_count,
            "candidate_pool_cap_binding_flag": candidate_pool_cap_binding_flag,
            "flex_small_candidate_count_by_size": flex_small_candidate_count_by_size,
            "selected_flex_small_count_by_size": selected_flex_small_count_by_size,
            "baseline_route_count": baseline_solution_eval.route_count,
            "baseline_single_stop_route_count": baseline_solution_eval.single_stop_route_count,
            "baseline_total_cost": baseline_solution_eval.total_cost,
            "baseline_fuel_3000_used_count": baseline_route_counts.get("fuel_3000", 0),
            "baseline_fuel_3000_free_count": self._fuel_3000_free_count(baseline_route_counts),
            "baseline_routes_with_flexible_units_on_big": baseline_big_route_diagnostics["mixed_big_route_count"],
            "baseline_flexible_units_on_big_routes": baseline_big_route_diagnostics["mixed_big_flexible_unit_count"],
            "baseline_piggyback_big_count": baseline_big_route_diagnostics["piggyback_big_count"],
            "baseline_promotion_like_big_count": baseline_big_route_diagnostics["promotion_like_big_count"],
            "baseline_blocking_big_flexible_count": baseline_big_route_diagnostics["blocking_big_flexible_count"],
            "baseline_blocking_big_flexible_unit_count": baseline_big_route_diagnostics["blocking_big_flexible_unit_count"],
            "baseline_bad_big_flexible_route_count": baseline_bad_big_flexible_route_count,
            "baseline_bad_big_flexible_unit_count": baseline_bad_big_flexible_unit_count,
            "baseline_current_single_pairs_feasible": baseline_current_single_pairs_feasible,
            "baseline_current_single_pairs_inventory_feasible": baseline_current_single_pairs_inventory_feasible,
            "global_model_status": global_result["status"],
            "global_phase_statuses": global_phase_statuses,
            "global_selected_as_final": int(cost_first_improved),
            "global_validation_status": global_validation_status,
            "global_route_pool_candidate_count": len(global_result["columns"]),
            "global_unlock_big_mixed_route_count": None,
            "global_unlock_big_mixed_unit_count": None,
            "global_unlock_bad_big_flexible_route_count": None,
            "global_unlock_bad_big_flexible_unit_count": None,
            "global_unlock_big_route_count": None,
            "global_final_total_cost": None if active_global_eval is None else active_global_eval.total_cost,
            "global_final_route_count": None if active_global_eval is None else active_global_eval.route_count,
            "global_final_single_stop_route_count": None if active_global_eval is None else active_global_eval.single_stop_route_count,
            "global_final_current_single_pairs_feasible": active_global_summary["current_single_pairs_feasible"],
            "global_final_current_single_pairs_inventory_feasible": active_global_summary["current_single_pairs_inventory_feasible"],
            "global_final_routes_with_flexible_units_on_big": active_global_summary["routes_with_flexible_units_on_big"],
            "global_final_flexible_units_on_big_routes": active_global_summary["flexible_units_on_big_routes"],
            "global_final_piggyback_big_count": active_global_summary["piggyback_big_count"],
            "global_final_promotion_like_big_count": active_global_summary["promotion_like_big_count"],
            "global_final_blocking_big_flexible_count": active_global_summary["blocking_big_flexible_route_count"],
            "global_final_blocking_big_flexible_unit_count": active_global_summary["blocking_big_flexible_unit_count"],
            "global_fuel_3000_used_count": None if active_global_route_counts is None else active_global_route_counts.get("fuel_3000", 0),
            "global_fuel_3000_free_count": None if active_global_route_counts is None else self._fuel_3000_free_count(active_global_route_counts),
            "global_big_route_bound_mode": active_global_summary["big_route_bound_mode"],
            "global_pass1_route_count": None if pass1_summary["solution_eval"] is None else pass1_summary["solution_eval"].route_count,
            "global_pass1_single_stop_route_count": None if pass1_summary["solution_eval"] is None else pass1_summary["solution_eval"].single_stop_route_count,
            "global_pass1_selected_singleton_count": pass1_summary["selected_singleton_count"],
            "global_pass1_selected_promotion_count": pass1_summary["selected_promotion_count"],
            "global_pass1_selected_promotion_like_count": pass1_summary["selected_promotion_like_count"],
            "global_pass1_selected_piggyback_big_count": pass1_summary["selected_piggyback_big_count"],
            "global_pass1_big_route_count": pass1_summary["big_route_count"],
            "global_pass1_mixed_big_route_count": pass1_summary["mixed_big_route_count"],
            "global_pass1_blocking_big_flexible_route_count": pass1_summary["blocking_big_flexible_route_count"],
            "global_pass1_blocking_big_flexible_unit_count": pass1_summary["blocking_big_flexible_unit_count"],
            "global_pass1_bad_big_flexible_route_count": pass1_summary["bad_big_flexible_route_count"],
            "global_pass1_bad_big_flexible_unit_count": pass1_summary["bad_big_flexible_unit_count"],
            "global_pass2_route_count": None if pass2_summary["solution_eval"] is None else pass2_summary["solution_eval"].route_count,
            "global_pass2_single_stop_route_count": None if pass2_summary["solution_eval"] is None else pass2_summary["solution_eval"].single_stop_route_count,
            "global_pass2_selected_singleton_count": pass2_summary["selected_singleton_count"],
            "global_pass2_selected_promotion_count": pass2_summary["selected_promotion_count"],
            "global_pass2_selected_promotion_like_count": pass2_summary["selected_promotion_like_count"],
            "global_pass2_selected_piggyback_big_count": pass2_summary["selected_piggyback_big_count"],
            "global_pass2_routes_with_flexible_units_on_big": pass2_summary["routes_with_flexible_units_on_big"],
            "global_pass2_flexible_units_on_big_routes": pass2_summary["flexible_units_on_big_routes"],
            "global_pass2_big_route_count": pass2_summary["big_route_count"],
            "global_pass2_mixed_big_route_count": pass2_summary["mixed_big_route_count"],
            "global_pass2_blocking_big_flexible_route_count": pass2_summary["blocking_big_flexible_route_count"],
            "global_pass2_blocking_big_flexible_unit_count": pass2_summary["blocking_big_flexible_unit_count"],
            "global_pass2_bad_big_flexible_route_count": pass2_summary["bad_big_flexible_route_count"],
            "global_pass2_bad_big_flexible_unit_count": pass2_summary["bad_big_flexible_unit_count"],
            "global_pass3_route_count": None if pass3_summary["solution_eval"] is None else pass3_summary["solution_eval"].route_count,
            "global_pass3_single_stop_route_count": None if pass3_summary["solution_eval"] is None else pass3_summary["solution_eval"].single_stop_route_count,
            "global_pass3_selected_singleton_count": pass3_summary["selected_singleton_count"],
            "global_pass3_selected_flexible_singleton_count": pass3_summary["selected_flexible_singleton_count"],
            "global_pass3_selected_rigid_big_singleton_count": pass3_summary["selected_rigid_big_singleton_count"],
            "global_pass3_selected_flex_small_count": pass3_summary["selected_flex_small_count"],
            "global_pass3_selected_promotion_like_count": pass3_summary["selected_promotion_like_count"],
            "global_pass3_selected_piggyback_big_count": pass3_summary["selected_piggyback_big_count"],
            "global_pass3_bad_big_flexible_route_count": pass3_summary["bad_big_flexible_route_count"],
            "global_pass3_big_route_count": pass3_summary["big_route_count"],
            "split_packing_sensitivity_executed": 0,
            "split_packing_sensitivity_status": "not_run",
            "split_packing_sensitivity_total_cost": None,
            "split_packing_sensitivity_route_count": None,
            "split_packing_sensitivity_reference_total_cost": None,
            "split_packing_sensitivity_reference_route_count": None,
            "route_pool_summary_rows": route_pool_summary_rows,
            "final_solution_source": final_solution_source,
            **best_operator_stats,
        }
        return best_solution_eval, best_solution_routes, metadata

    def solve(self) -> tuple[SolutionEvaluation, dict[str, object]]:
        best_solution_eval, best_solution_routes, metadata = self._solve_single_configuration()
        if (
            self.enable_split_packing_sensitivity
            and metadata["global_selected_as_final"] == 0
        ):
            sensitivity_solver = Question1Solver(
                workspace=self.workspace,
                input_root=self.input_root,
                output_root=self.output_root,
                seed_list=self.seed_list,
                max_generations=self.max_generations,
                particle_count=self.particle_count,
                top_route_candidates=self.top_route_candidates,
                packing_strategy="reduced_big_dependency",
                enable_split_packing_sensitivity=False,
            )
            sensitivity_eval, sensitivity_routes, sensitivity_metadata = sensitivity_solver._solve_single_configuration()
            metadata["split_packing_sensitivity_executed"] = 1
            metadata["split_packing_sensitivity_total_cost"] = sensitivity_eval.total_cost
            metadata["split_packing_sensitivity_route_count"] = sensitivity_eval.route_count
            metadata["split_packing_sensitivity_reference_total_cost"] = best_solution_eval.total_cost
            metadata["split_packing_sensitivity_reference_route_count"] = best_solution_eval.route_count
            if sensitivity_eval.total_cost + COST_IMPROVEMENT_EPS < best_solution_eval.total_cost:
                sensitivity_metadata["split_packing_sensitivity_executed"] = 1
                sensitivity_metadata["split_packing_sensitivity_status"] = "selected_as_final"
                sensitivity_metadata["split_packing_sensitivity_total_cost"] = sensitivity_eval.total_cost
                sensitivity_metadata["split_packing_sensitivity_route_count"] = sensitivity_eval.route_count
                sensitivity_metadata["split_packing_sensitivity_reference_total_cost"] = best_solution_eval.total_cost
                sensitivity_metadata["split_packing_sensitivity_reference_route_count"] = best_solution_eval.route_count
                sensitivity_metadata["global_validation_status"] = "split_packing_sensitivity_improved"
                sensitivity_metadata["final_solution_source"] = f"packing_sensitivity_{sensitivity_metadata['final_solution_source']}"
                best_solution_eval = sensitivity_eval
                best_solution_routes = sensitivity_routes
                metadata = sensitivity_metadata
            else:
                metadata["split_packing_sensitivity_status"] = "no_improvement"

        self._write_outputs(best_solution_eval, best_solution_routes, metadata)
        return best_solution_eval, metadata

    def _write_outputs(
        self,
        solution_eval: SolutionEvaluation,
        solution_routes: list[TypedRoute],
        metadata: dict[str, object],
    ) -> None:
        route_summary_rows: list[dict[str, object]] = []
        stop_rows: list[dict[str, object]] = []
        vehicle_schedule_rows: list[dict[str, object]] = []
        customer_rows: list[dict[str, object]] = []

        for route in sorted(solution_eval.assigned_routes, key=lambda item: item.route_index):
            vehicle = self.vehicle_by_name[route.vehicle_type]
            route_summary_rows.append(
                {
                    "route_id": route.route_index,
                    "vehicle_type": route.vehicle_type,
                    "power_type": route.power_type,
                    "vehicle_instance": route.vehicle_instance,
                    "unit_sequence": ",".join(str(unit_id) for unit_id in route.unit_ids),
                    "customer_sequence": "->".join(["0"] + [str(unit.orig_cust_id) for unit in route.units] + ["0"]),
                    "unit_type_sequence": "->".join(unit.unit_type for unit in route.units),
                    "departure_min": route.departure_min,
                    "departure_hhmm": self._minutes_to_hhmm(route.departure_min),
                    "return_min": route.return_min,
                    "return_hhmm": self._minutes_to_hhmm(route.return_min),
                    "route_cost": round(route.route_cost, 6),
                    "startup_cost": round(route.startup_cost, 6),
                    "energy_cost": round(route.energy_cost, 6),
                    "carbon_cost": round(route.carbon_cost, 6),
                    "waiting_cost": round(route.waiting_cost, 6),
                    "late_cost": round(route.late_cost, 6),
                    "total_wait_min": round(route.total_wait_min, 6),
                    "total_late_min": round(route.total_late_min, 6),
                    "fuel_l": round(route.total_fuel_l, 6),
                    "electricity_kwh": round(route.total_electricity_kwh, 6),
                    "route_distance_km": round(route.route_distance_km, 6),
                    "after_hours_travel_km": round(route.after_hours_travel_km, 6),
                    "after_hours_service_count": route.after_hours_service_count,
                    "after_hours_return_flag": int(route.after_hours_return_flag),
                    "late_positive_stop_count": route.late_positive_stop_count,
                    "max_late_min": round(route.max_late_min, 6),
                    "load_used_kg": round(sum(unit.weight for unit in route.units), 6),
                    "load_used_m3": round(sum(unit.volume for unit in route.units), 6),
                    "weight_utilization": round(sum(unit.weight for unit in route.units) / vehicle.capacity_kg, 6),
                    "volume_utilization": round(sum(unit.volume for unit in route.units) / vehicle.capacity_m3, 6),
                }
            )
            stop_rows.extend(route.stop_rows)
            vehicle_schedule_rows.append(
                {
                    "vehicle_type": route.vehicle_type,
                    "vehicle_instance": route.vehicle_instance,
                    "route_id": route.route_index,
                    "departure_min": route.departure_min,
                    "departure_hhmm": self._minutes_to_hhmm(route.departure_min),
                    "return_min": route.return_min,
                    "return_hhmm": self._minutes_to_hhmm(route.return_min),
                    "route_cost": round(route.route_cost, 6),
                    "after_hours_return_flag": int(route.after_hours_return_flag),
                }
            )
            for unit in route.units:
                customer_rows.append(
                    {
                        "orig_cust_id": unit.orig_cust_id,
                        "route_id": route.route_index,
                        "vehicle_type": route.vehicle_type,
                        "unit_id": unit.unit_id,
                        "unit_type": unit.unit_type,
                        "visit_index": unit.visit_index,
                        "required_visit_count": unit.required_visit_count,
                        "served_weight_kg": round(unit.weight, 6),
                        "served_volume_m3": round(unit.volume, 6),
                    }
                )

        route_summary_df = pd.DataFrame(route_summary_rows).sort_values("route_id")
        stop_schedule_df = pd.DataFrame(stop_rows).sort_values(["route_id", "stop_index"])
        vehicle_schedule_df = pd.DataFrame(vehicle_schedule_rows).sort_values(["vehicle_type", "vehicle_instance", "route_id"])
        customer_aggregate_df = (
            pd.DataFrame(customer_rows)
            .groupby("orig_cust_id", as_index=False)
            .agg(
                route_ids=("route_id", lambda values: ",".join(str(value) for value in sorted(set(values)))),
                route_visit_count=("route_id", "nunique"),
                unit_ids=("unit_id", lambda values: ",".join(str(value) for value in sorted(values))),
                unit_count=("unit_id", "size"),
                unit_type=("unit_type", lambda values: "mandatory_split" if "mandatory_split" in set(values) else "normal"),
                required_visit_count=("required_visit_count", "max"),
                served_weight_kg=("served_weight_kg", "sum"),
                served_volume_m3=("served_volume_m3", "sum"),
            )
            .sort_values("orig_cust_id")
        )

        service_unit_rows = [
            {
                "unit_id": unit.unit_id,
                "orig_cust_id": unit.orig_cust_id,
                "unit_type": unit.unit_type,
                "visit_index": unit.visit_index,
                "required_visit_count": unit.required_visit_count,
                "weight_kg": round(unit.weight, 6),
                "volume_m3": round(unit.volume, 6),
                "tw_start_min": unit.tw_start_min,
                "tw_end_min": unit.tw_end_min,
                "eligible_vehicle_types": ",".join(unit.eligible_vehicle_types),
                "source_order_ids": ",".join(str(order_id) for order_id in unit.source_order_ids),
            }
            for unit in self.service_units
        ]

        route_summary_df.to_csv(self.output_root / "q1_route_summary.csv", index=False, encoding=CSV_ENCODING)
        stop_schedule_df.to_csv(self.output_root / "q1_stop_schedule.csv", index=False, encoding=CSV_ENCODING)
        vehicle_schedule_df.to_csv(self.output_root / "q1_vehicle_schedule.csv", index=False, encoding=CSV_ENCODING)
        customer_aggregate_df.to_csv(self.output_root / "q1_customer_aggregate.csv", index=False, encoding=CSV_ENCODING)
        pd.DataFrame(service_unit_rows).to_csv(self.output_root / "q1_service_units.csv", index=False, encoding=CSV_ENCODING)
        pd.DataFrame(self.split_plan_rows).to_csv(self.output_root / "q1_split_plan.csv", index=False, encoding=CSV_ENCODING)
        pd.DataFrame(metadata["merge_diagnostics_rows"]).to_csv(
            self.output_root / "q1_merge_diagnostics.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
        pd.DataFrame(metadata["route_pool_summary_rows"]).to_csv(
            self.output_root / "q1_route_pool_summary.csv",
            index=False,
            encoding=CSV_ENCODING,
        )

        stale_atom_path = self.output_root / "q1_atoms.csv"
        if stale_atom_path.exists():
            stale_atom_path.unlink()

        cost_summary = {
            "total_cost": solution_eval.total_cost,
            "startup_cost": solution_eval.total_startup_cost,
            "energy_cost": solution_eval.total_energy_cost,
            "carbon_cost": solution_eval.total_carbon_cost,
            "waiting_cost": solution_eval.total_waiting_cost,
            "late_cost": solution_eval.total_late_cost,
            "total_late_min": solution_eval.total_late_min,
            "total_fuel_l": solution_eval.total_fuel_l,
            "total_electricity_kwh": solution_eval.total_electricity_kwh,
            "total_carbon_kg": solution_eval.total_carbon_kg,
            "total_distance_km": solution_eval.total_distance_km,
            "route_count": solution_eval.route_count,
            "used_vehicle_count": solution_eval.used_vehicle_count,
            "split_customer_count": solution_eval.split_customer_count,
            "mandatory_split_customer_count": solution_eval.mandatory_split_customer_count,
            "mandatory_split_visit_count": solution_eval.mandatory_split_visit_count,
            "normal_customer_count": solution_eval.normal_customer_count,
            "single_stop_route_count": solution_eval.single_stop_route_count,
            "two_stop_route_count": solution_eval.two_stop_route_count,
            "three_plus_route_count": solution_eval.three_plus_route_count,
            "heavy_big_only_count": metadata["heavy_big_only_count"],
            "total_heavy_big_only_count": metadata["total_heavy_big_only_count"],
            "normal_heavy_big_only_count": metadata["normal_heavy_big_only_count"],
            "heavy_big_only_capacity": metadata["heavy_big_only_capacity"],
            "big_vehicle_inventory": metadata["big_vehicle_inventory"],
            "big_vehicle_reserve": metadata["big_vehicle_reserve"],
            "fuel_3000_used_count": metadata["fuel_3000_used_count"],
            "fuel_3000_free_count": metadata["fuel_3000_free_count"],
            "single_single_merge_feasible_pair_count": metadata["single_single_merge_feasible_pair_count"],
            "single_single_merge_inventory_blocked_pair_count": metadata["single_single_merge_inventory_blocked_pair_count"],
            "final_current_single_pairs_inventory_feasible": metadata["final_current_single_pairs_inventory_feasible"],
            "final_routes_with_flexible_units_on_big": metadata["final_routes_with_flexible_units_on_big"],
            "final_flexible_units_on_big_routes": metadata["final_flexible_units_on_big_routes"],
            "final_piggyback_big_count": metadata["final_piggyback_big_count"],
            "final_promotion_like_big_count": metadata["final_promotion_like_big_count"],
            "final_blocking_big_flexible_count": metadata["final_blocking_big_flexible_count"],
            "final_blocking_big_flexible_unit_count": metadata["final_blocking_big_flexible_unit_count"],
            "diagnostic_unlock_success_count": metadata["diagnostic_unlock_success_count"],
            "diagnostic_promotion_success_count": metadata["diagnostic_promotion_success_count"],
            "route_pool_candidate_count": metadata["route_pool_candidate_count"],
            "route_pool_role_counts": metadata["route_pool_role_counts"],
            "promotion_like_candidate_count": metadata["promotion_like_candidate_count"],
            "positive_saving_promotion_like_candidate_count": metadata["positive_saving_promotion_like_candidate_count"],
            "piggyback_big_candidate_count": metadata["piggyback_big_candidate_count"],
            "positive_saving_piggyback_big_candidate_count": metadata["positive_saving_piggyback_big_candidate_count"],
            "candidate_pool_cap_binding_flag": metadata["candidate_pool_cap_binding_flag"],
            "flex_small_candidate_count_by_size": metadata["flex_small_candidate_count_by_size"],
            "selected_flex_small_count_by_size": metadata["selected_flex_small_count_by_size"],
            "baseline_route_count": metadata["baseline_route_count"],
            "baseline_single_stop_route_count": metadata["baseline_single_stop_route_count"],
            "baseline_total_cost": metadata["baseline_total_cost"],
            "baseline_fuel_3000_used_count": metadata["baseline_fuel_3000_used_count"],
            "baseline_fuel_3000_free_count": metadata["baseline_fuel_3000_free_count"],
            "baseline_routes_with_flexible_units_on_big": metadata["baseline_routes_with_flexible_units_on_big"],
            "baseline_flexible_units_on_big_routes": metadata["baseline_flexible_units_on_big_routes"],
            "baseline_piggyback_big_count": metadata["baseline_piggyback_big_count"],
            "baseline_promotion_like_big_count": metadata["baseline_promotion_like_big_count"],
            "baseline_blocking_big_flexible_count": metadata["baseline_blocking_big_flexible_count"],
            "baseline_blocking_big_flexible_unit_count": metadata["baseline_blocking_big_flexible_unit_count"],
            "baseline_bad_big_flexible_route_count": metadata["baseline_bad_big_flexible_route_count"],
            "baseline_bad_big_flexible_unit_count": metadata["baseline_bad_big_flexible_unit_count"],
            "baseline_current_single_pairs_feasible": metadata["baseline_current_single_pairs_feasible"],
            "baseline_current_single_pairs_inventory_feasible": metadata["baseline_current_single_pairs_inventory_feasible"],
            "global_model_status": metadata["global_model_status"],
            "global_phase_statuses": metadata["global_phase_statuses"],
            "global_selected_as_final": metadata["global_selected_as_final"],
            "global_validation_status": metadata["global_validation_status"],
            "global_route_pool_candidate_count": metadata["global_route_pool_candidate_count"],
            "global_unlock_big_mixed_route_count": metadata["global_unlock_big_mixed_route_count"],
            "global_unlock_big_mixed_unit_count": metadata["global_unlock_big_mixed_unit_count"],
            "global_unlock_bad_big_flexible_route_count": metadata["global_unlock_bad_big_flexible_route_count"],
            "global_unlock_bad_big_flexible_unit_count": metadata["global_unlock_bad_big_flexible_unit_count"],
            "global_unlock_big_route_count": metadata["global_unlock_big_route_count"],
            "global_big_route_bound_mode": metadata["global_big_route_bound_mode"],
            "global_pass1_route_count": metadata["global_pass1_route_count"],
            "global_pass1_single_stop_route_count": metadata["global_pass1_single_stop_route_count"],
            "global_pass1_selected_singleton_count": metadata["global_pass1_selected_singleton_count"],
            "global_pass1_selected_promotion_count": metadata["global_pass1_selected_promotion_count"],
            "global_pass1_selected_promotion_like_count": metadata["global_pass1_selected_promotion_like_count"],
            "global_pass1_big_route_count": metadata["global_pass1_big_route_count"],
            "global_pass1_bad_big_flexible_route_count": metadata["global_pass1_bad_big_flexible_route_count"],
            "global_pass1_bad_big_flexible_unit_count": metadata["global_pass1_bad_big_flexible_unit_count"],
            "global_pass2_route_count": metadata["global_pass2_route_count"],
            "global_pass2_single_stop_route_count": metadata["global_pass2_single_stop_route_count"],
            "global_pass2_selected_singleton_count": metadata["global_pass2_selected_singleton_count"],
            "global_pass2_selected_promotion_count": metadata["global_pass2_selected_promotion_count"],
            "global_pass2_selected_promotion_like_count": metadata["global_pass2_selected_promotion_like_count"],
            "global_pass2_routes_with_flexible_units_on_big": metadata["global_pass2_routes_with_flexible_units_on_big"],
            "global_pass2_flexible_units_on_big_routes": metadata["global_pass2_flexible_units_on_big_routes"],
            "global_pass2_big_route_count": metadata["global_pass2_big_route_count"],
            "global_pass2_bad_big_flexible_route_count": metadata["global_pass2_bad_big_flexible_route_count"],
            "global_pass2_bad_big_flexible_unit_count": metadata["global_pass2_bad_big_flexible_unit_count"],
            "global_pass3_route_count": metadata["global_pass3_route_count"],
            "global_pass3_single_stop_route_count": metadata["global_pass3_single_stop_route_count"],
            "global_pass3_selected_singleton_count": metadata["global_pass3_selected_singleton_count"],
            "global_pass3_selected_flexible_singleton_count": metadata["global_pass3_selected_flexible_singleton_count"],
            "global_pass3_selected_rigid_big_singleton_count": metadata["global_pass3_selected_rigid_big_singleton_count"],
            "global_pass3_selected_flex_small_count": metadata["global_pass3_selected_flex_small_count"],
            "global_pass3_selected_promotion_like_count": metadata["global_pass3_selected_promotion_like_count"],
            "global_pass3_bad_big_flexible_route_count": metadata["global_pass3_bad_big_flexible_route_count"],
            "global_pass3_big_route_count": metadata["global_pass3_big_route_count"],
            "global_final_total_cost": metadata["global_final_total_cost"],
            "global_final_route_count": metadata["global_final_route_count"],
            "global_final_single_stop_route_count": metadata["global_final_single_stop_route_count"],
            "global_final_current_single_pairs_feasible": metadata["global_final_current_single_pairs_feasible"],
            "global_final_current_single_pairs_inventory_feasible": metadata["global_final_current_single_pairs_inventory_feasible"],
            "global_final_routes_with_flexible_units_on_big": metadata["global_final_routes_with_flexible_units_on_big"],
            "global_final_flexible_units_on_big_routes": metadata["global_final_flexible_units_on_big_routes"],
            "global_final_piggyback_big_count": metadata["global_final_piggyback_big_count"],
            "global_final_promotion_like_big_count": metadata["global_final_promotion_like_big_count"],
            "global_final_blocking_big_flexible_count": metadata["global_final_blocking_big_flexible_count"],
            "global_final_blocking_big_flexible_unit_count": metadata["global_final_blocking_big_flexible_unit_count"],
            "global_fuel_3000_used_count": metadata["global_fuel_3000_used_count"],
            "global_fuel_3000_free_count": metadata["global_fuel_3000_free_count"],
            "global_pass1_selected_piggyback_big_count": metadata["global_pass1_selected_piggyback_big_count"],
            "global_pass1_mixed_big_route_count": metadata["global_pass1_mixed_big_route_count"],
            "global_pass1_blocking_big_flexible_route_count": metadata["global_pass1_blocking_big_flexible_route_count"],
            "global_pass1_blocking_big_flexible_unit_count": metadata["global_pass1_blocking_big_flexible_unit_count"],
            "global_pass2_selected_piggyback_big_count": metadata["global_pass2_selected_piggyback_big_count"],
            "global_pass2_mixed_big_route_count": metadata["global_pass2_mixed_big_route_count"],
            "global_pass2_blocking_big_flexible_route_count": metadata["global_pass2_blocking_big_flexible_route_count"],
            "global_pass2_blocking_big_flexible_unit_count": metadata["global_pass2_blocking_big_flexible_unit_count"],
            "global_pass3_selected_piggyback_big_count": metadata["global_pass3_selected_piggyback_big_count"],
            "reserve_repair_success_count": metadata["reserve_repair_success_count"],
            "batch_merge_success_count": metadata["batch_merge_success_count"],
            "pre_merge_single_stop_route_count": metadata["pre_merge_single_stop_route_count"],
            "post_merge_single_stop_route_count": metadata["post_merge_single_stop_route_count"],
            "late_positive_stops": solution_eval.late_positive_stops,
            "max_late_min": solution_eval.max_late_min,
            "latest_return_min": solution_eval.latest_return_min,
            "after_hours_service_count": solution_eval.after_hours_service_count,
            "after_hours_return_count": solution_eval.after_hours_return_count,
            "after_hours_travel_km": solution_eval.after_hours_travel_km,
            "vehicle_type_usage": solution_eval.vehicle_type_usage,
            "route_merge_success_count": metadata["route_merge_success_count"],
            "relocate_success_count": metadata["relocate_success_count"],
            "route_type_change_success_count": metadata["route_type_change_success_count"],
            "best_seed": metadata["best_seed"],
            "service_unit_count": metadata["service_unit_count"],
            "route_cache_size": metadata["route_cache_size"],
            "elapsed_sec": metadata["elapsed_sec"],
            "packing_strategy": metadata["packing_strategy"],
            "route_pool_iteration_count": metadata["route_pool_iteration_count"],
            "cost_first_improved": metadata["cost_first_improved"],
            "split_packing_sensitivity_executed": metadata["split_packing_sensitivity_executed"],
            "split_packing_sensitivity_status": metadata["split_packing_sensitivity_status"],
            "split_packing_sensitivity_total_cost": metadata["split_packing_sensitivity_total_cost"],
            "split_packing_sensitivity_route_count": metadata["split_packing_sensitivity_route_count"],
            "split_packing_sensitivity_reference_total_cost": metadata["split_packing_sensitivity_reference_total_cost"],
            "split_packing_sensitivity_reference_route_count": metadata["split_packing_sensitivity_reference_route_count"],
            "final_solution_source": metadata["final_solution_source"],
        }
        (self.output_root / "q1_cost_summary.json").write_text(json.dumps(cost_summary, indent=2), encoding="utf-8")

        report_lines = [
            "# Question 1 Solver Report",
            "",
            "## Run Summary",
            f"- Best seed: {metadata['best_seed']}",
            f"- Service unit count: {metadata['service_unit_count']}",
            f"- Packing strategy: {metadata['packing_strategy']}",
            f"- Cost-first improved: {metadata['cost_first_improved']}",
            f"- Route count: {solution_eval.route_count}",
            f"- Used vehicle count: {solution_eval.used_vehicle_count}",
            f"- Total cost: {solution_eval.total_cost:.3f}",
            f"- Startup cost: {solution_eval.total_startup_cost:.3f}",
            f"- Energy cost: {solution_eval.total_energy_cost:.3f}",
            f"- Carbon cost: {solution_eval.total_carbon_cost:.3f}",
            f"- Waiting cost: {solution_eval.total_waiting_cost:.3f}",
            f"- Late cost: {solution_eval.total_late_cost:.3f}",
            f"- Total late minutes: {solution_eval.total_late_min:.3f}",
            f"- Total fuel: {solution_eval.total_fuel_l:.3f} L",
            f"- Total electricity: {solution_eval.total_electricity_kwh:.3f} kWh",
            f"- Total carbon: {solution_eval.total_carbon_kg:.3f} kg",
            f"- Total distance: {solution_eval.total_distance_km:.3f} km",
            f"- Split customers: {solution_eval.split_customer_count}",
            f"- Mandatory split customers: {solution_eval.mandatory_split_customer_count}",
            f"- Mandatory split visits: {solution_eval.mandatory_split_visit_count}",
            f"- Normal customers: {solution_eval.normal_customer_count}",
            f"- Single-stop routes: {solution_eval.single_stop_route_count}",
            f"- Two-stop routes: {solution_eval.two_stop_route_count}",
            f"- Three-plus-stop routes: {solution_eval.three_plus_route_count}",
            f"- Heavy big-only count: {metadata['heavy_big_only_count']}",
            f"- Normal heavy big-only count: {metadata['normal_heavy_big_only_count']}",
            f"- Total heavy big-only count: {metadata['total_heavy_big_only_count']}",
            f"- Heavy big-only capacity: {metadata['heavy_big_only_capacity']}",
            f"- Big-vehicle inventory: {metadata['big_vehicle_inventory']}",
            f"- Big-vehicle reserve: {metadata['big_vehicle_reserve']}",
            f"- Fuel 3000 used count: {metadata['fuel_3000_used_count']}",
            f"- Fuel 3000 free count: {metadata['fuel_3000_free_count']}",
            f"- Single-single merge feasible pair count: {metadata['single_single_merge_feasible_pair_count']}",
            f"- Single-single merge inventory-blocked pair count: {metadata['single_single_merge_inventory_blocked_pair_count']}",
            f"- Final current single-pairs inventory-feasible count: {metadata['final_current_single_pairs_inventory_feasible']}",
            f"- Final routes with flexible units on big: {metadata['final_routes_with_flexible_units_on_big']}",
            f"- Final flexible units on big routes: {metadata['final_flexible_units_on_big_routes']}",
            f"- Final piggyback/promotion-like/blocking big count: {metadata['final_piggyback_big_count']}/{metadata['final_promotion_like_big_count']}/{metadata['final_blocking_big_flexible_count']}",
            f"- Final blocking big flexible unit count: {metadata['final_blocking_big_flexible_unit_count']}",
            f"- Diagnostic unlock success count: {metadata['diagnostic_unlock_success_count']}",
            f"- Diagnostic promotion success count: {metadata['diagnostic_promotion_success_count']}",
            f"- Route-pool candidate count: {metadata['route_pool_candidate_count']}",
            f"- Route-pool role counts: {json.dumps(metadata['route_pool_role_counts'], ensure_ascii=False)}",
            f"- Promotion-like candidate count: {metadata['promotion_like_candidate_count']}",
            f"- Positive-saving promotion-like candidate count: {metadata['positive_saving_promotion_like_candidate_count']}",
            f"- Piggyback-big candidate count: {metadata['piggyback_big_candidate_count']}",
            f"- Positive-saving piggyback-big candidate count: {metadata['positive_saving_piggyback_big_candidate_count']}",
            f"- Candidate-pool cap binding flag: {metadata['candidate_pool_cap_binding_flag']}",
            f"- Flex-small candidate count by size: {json.dumps(metadata['flex_small_candidate_count_by_size'], ensure_ascii=False)}",
            f"- Selected flex-small count by size: {json.dumps(metadata['selected_flex_small_count_by_size'], ensure_ascii=False)}",
            f"- Final solution source: {metadata['final_solution_source']}",
            f"- Baseline route count: {metadata['baseline_route_count']}",
            f"- Baseline single-stop route count: {metadata['baseline_single_stop_route_count']}",
            f"- Baseline total cost: {metadata['baseline_total_cost']:.3f}",
            f"- Baseline fuel 3000 used/free: {metadata['baseline_fuel_3000_used_count']}/{metadata['baseline_fuel_3000_free_count']}",
            f"- Baseline routes with flexible units on big: {metadata['baseline_routes_with_flexible_units_on_big']}",
            f"- Baseline flexible units on big routes: {metadata['baseline_flexible_units_on_big_routes']}",
            f"- Baseline piggyback/promotion-like/blocking big count: {metadata['baseline_piggyback_big_count']}/{metadata['baseline_promotion_like_big_count']}/{metadata['baseline_blocking_big_flexible_count']}",
            f"- Baseline blocking big flexible unit count: {metadata['baseline_blocking_big_flexible_unit_count']}",
            f"- Baseline bad big-flexible route/unit count: {metadata['baseline_bad_big_flexible_route_count']}/{metadata['baseline_bad_big_flexible_unit_count']}",
            f"- Baseline current single-pairs feasible/inventory-feasible: {metadata['baseline_current_single_pairs_feasible']}/{metadata['baseline_current_single_pairs_inventory_feasible']}",
            f"- Global model status: {metadata['global_model_status']}",
            f"- Global selected as final: {metadata['global_selected_as_final']}",
            f"- Global validation status: {metadata['global_validation_status']}",
            f"- Global route-pool candidate count: {metadata['global_route_pool_candidate_count']}",
            f"- Route-pool iterations configured: {metadata['route_pool_iteration_count']}",
            f"- Global pass1 route count: {metadata['global_pass1_route_count']}",
            f"- Global pass1 single-stop route count: {metadata['global_pass1_single_stop_route_count']}",
            f"- Global pass1 selected singleton count: {metadata['global_pass1_selected_singleton_count']}",
            f"- Global pass1 selected promotion count: {metadata['global_pass1_selected_promotion_count']}",
            f"- Global pass1 selected promotion-like/piggyback count: {metadata['global_pass1_selected_promotion_like_count']}/{metadata['global_pass1_selected_piggyback_big_count']}",
            f"- Global pass1 big-route count: {metadata['global_pass1_big_route_count']}",
            f"- Global pass1 mixed/blocking big route count: {metadata['global_pass1_mixed_big_route_count']}/{metadata['global_pass1_blocking_big_flexible_route_count']}",
            f"- Global pass1 bad big-flexible route/unit count: {metadata['global_pass1_bad_big_flexible_route_count']}/{metadata['global_pass1_bad_big_flexible_unit_count']}",
            f"- Global pass2 route count: {metadata['global_pass2_route_count']}",
            f"- Global pass2 single-stop route count: {metadata['global_pass2_single_stop_route_count']}",
            f"- Global pass2 selected singleton count: {metadata['global_pass2_selected_singleton_count']}",
            f"- Global pass2 selected promotion count: {metadata['global_pass2_selected_promotion_count']}",
            f"- Global pass2 selected promotion-like/piggyback count: {metadata['global_pass2_selected_promotion_like_count']}/{metadata['global_pass2_selected_piggyback_big_count']}",
            f"- Global pass2 routes with flexible units on big: {metadata['global_pass2_routes_with_flexible_units_on_big']}",
            f"- Global pass2 flexible units on big routes: {metadata['global_pass2_flexible_units_on_big_routes']}",
            f"- Global pass2 big-route count: {metadata['global_pass2_big_route_count']}",
            f"- Global pass2 mixed/blocking big route count: {metadata['global_pass2_mixed_big_route_count']}/{metadata['global_pass2_blocking_big_flexible_route_count']}",
            f"- Global pass2 bad big-flexible route/unit count: {metadata['global_pass2_bad_big_flexible_route_count']}/{metadata['global_pass2_bad_big_flexible_unit_count']}",
            f"- Global pass3 route count: {metadata['global_pass3_route_count']}",
            f"- Global pass3 single-stop route count: {metadata['global_pass3_single_stop_route_count']}",
            f"- Global pass3 selected singleton/flexible-singleton/rigid-big-singleton count: {metadata['global_pass3_selected_singleton_count']}/{metadata['global_pass3_selected_flexible_singleton_count']}/{metadata['global_pass3_selected_rigid_big_singleton_count']}",
            f"- Global pass3 selected flex-small count: {metadata['global_pass3_selected_flex_small_count']}",
            f"- Global pass3 selected promotion-like/piggyback count: {metadata['global_pass3_selected_promotion_like_count']}/{metadata['global_pass3_selected_piggyback_big_count']}",
            f"- Global pass3 bad big-flexible route count: {metadata['global_pass3_bad_big_flexible_route_count']}",
            f"- Global pass3 big-route count: {metadata['global_pass3_big_route_count']}",
            f"- Global final total cost: {metadata['global_final_total_cost']}",
            f"- Global final route count: {metadata['global_final_route_count']}",
            f"- Global final single-stop route count: {metadata['global_final_single_stop_route_count']}",
            f"- Global final current single-pairs feasible/inventory-feasible: {metadata['global_final_current_single_pairs_feasible']}/{metadata['global_final_current_single_pairs_inventory_feasible']}",
            f"- Global final routes with flexible units on big: {metadata['global_final_routes_with_flexible_units_on_big']}",
            f"- Global final flexible units on big routes: {metadata['global_final_flexible_units_on_big_routes']}",
            f"- Global final piggyback/promotion-like/blocking big count: {metadata['global_final_piggyback_big_count']}/{metadata['global_final_promotion_like_big_count']}/{metadata['global_final_blocking_big_flexible_count']}",
            f"- Global fuel 3000 used/free: {metadata['global_fuel_3000_used_count']}/{metadata['global_fuel_3000_free_count']}",
            f"- Split/packing sensitivity executed/status: {metadata['split_packing_sensitivity_executed']}/{metadata['split_packing_sensitivity_status']}",
            f"- Split/packing sensitivity total cost/route count: {metadata['split_packing_sensitivity_total_cost']}/{metadata['split_packing_sensitivity_route_count']}",
            f"- Reserve repair success count: {metadata['reserve_repair_success_count']}",
            f"- Batch merge success count: {metadata['batch_merge_success_count']}",
            f"- Pre-merge single-stop route count: {metadata['pre_merge_single_stop_route_count']}",
            f"- Post-merge single-stop route count: {metadata['post_merge_single_stop_route_count']}",
            f"- Late-positive stops: {solution_eval.late_positive_stops}",
            f"- Max late: {solution_eval.max_late_min:.3f} min",
            f"- Latest return: {solution_eval.latest_return_min:.3f} min",
            f"- After-hours service count: {solution_eval.after_hours_service_count}",
            f"- After-hours return count: {solution_eval.after_hours_return_count}",
            f"- After-hours travel: {solution_eval.after_hours_travel_km:.3f} km",
            f"- Vehicle type usage: {json.dumps(solution_eval.vehicle_type_usage, ensure_ascii=False)}",
            f"- Route merge successes: {metadata['route_merge_success_count']}",
            f"- Relocate successes: {metadata['relocate_success_count']}",
            f"- Route type change successes: {metadata['route_type_change_success_count']}",
            f"- Elapsed time: {metadata['elapsed_sec']:.2f} s",
            "",
            "## Per-Seed Best",
        ]
        report_lines.extend(
            [
                "",
                "## Cost-First MILP Phases",
                *[
                    "- "
                    + json.dumps(
                        phase_status,
                        ensure_ascii=False,
                    )
                    for phase_status in metadata["global_phase_statuses"]
                ],
            ]
        )
        for item in metadata["run_records"]:
            report_lines.append(
                f"- Seed {item['seed']}: best cost {item['best_cost']:.3f}, routes {item['route_count']}, "
                f"vehicles {item['used_vehicle_count']}, split customers {item['split_customer_count']}, "
                f"single-stop routes {item['single_stop_route_count']}, "
                f"late-positive stops {item['late_positive_stops']}, latest return {item['latest_return_min']:.3f}"
            )
        (self.output_root / "q1_run_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve question 1 with customer-level service units.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--input-root", type=Path, default=Path.cwd() / "preprocess_artifacts")
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "question1_artifacts")
    parser.add_argument("--seed-list", type=str, default="11")
    parser.add_argument("--max-generations", type=int, default=2)
    parser.add_argument("--particle-count", type=int, default=1)
    parser.add_argument("--top-route-candidates", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_list = [int(seed.strip()) for seed in args.seed_list.split(",") if seed.strip()]
    solver = Question1Solver(
        workspace=args.workspace,
        input_root=args.input_root,
        output_root=args.output_root,
        seed_list=seed_list,
        max_generations=args.max_generations,
        particle_count=args.particle_count,
        top_route_candidates=args.top_route_candidates,
    )
    best_solution, metadata = solver.solve()
    print(
        json.dumps(
            {
                "total_cost": best_solution.total_cost,
                "route_count": best_solution.route_count,
                "used_vehicle_count": best_solution.used_vehicle_count,
                "split_customer_count": best_solution.split_customer_count,
                "late_positive_stops": best_solution.late_positive_stops,
                "latest_return_min": best_solution.latest_return_min,
                "best_seed": metadata["best_seed"],
                "elapsed_sec": metadata["elapsed_sec"],
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
