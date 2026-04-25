from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import permutations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import solve_question1 as q1
import solve_question2 as q2
import solve_question2_costfirst_v2 as q2v2


CSV_ENCODING = q1.CSV_ENCODING
EPS = 1e-9
LOCAL_MILP_TIME_LIMIT_SEC = 10
LOCAL_ONBOARD_NEIGHBOR_TRIALS = 200
LOCAL_CONE_EXPANSION_BATCH = 5


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    event_time: int
    event_type: str
    target_customer: int
    node_from: int | None
    node_to: int | None
    new_tw_start: int | None
    new_tw_end: int | None
    weight: float | None
    volume: float | None


@dataclass(frozen=True)
class DynamicPolicyServiceUnit(q2.PolicyServiceUnit):
    business_customer_id: int
    service_node_id: int
    baseline_customer_id: int
    latest_event_id: str | None = None


@dataclass(frozen=True)
class DynamicRoutePlan:
    vehicle_slot: str
    vehicle_type: str
    vehicle_instance: int
    unit_ids: tuple[int, ...]
    planned_departure_min: float
    baseline_route_id: int | None
    source_tag: str


@dataclass
class VehicleDynamicState:
    vehicle_slot: str
    vehicle_type: str
    vehicle_instance: int
    power_type: str
    baseline_route_id: int | None
    status: str
    planned_departure_min: float
    full_return_min: float
    release_node: int
    release_time: float
    current_arc_frozen: str
    fixed_prefix: tuple[int, ...]
    current_frozen_unit_id: int | None
    editable_suffix: tuple[int, ...]
    remaining_unit_ids: tuple[int, ...]
    remaining_total_cost: float
    remaining_total_late_min: float
    realized_cost_to_event: float
    future_stop_rows: list[dict[str, object]]
    full_simulation: dict[str, object]


@dataclass
class Snapshot:
    event_time: float
    route_states: list[VehicleDynamicState]
    route_state_by_slot: dict[str, VehicleDynamicState]
    remaining_stop_rows: list[dict[str, object]]
    unit_to_slot: dict[int, str]
    realized_total_cost: float
    remaining_total_cost: float
    projected_full_day_cost: float
    remaining_total_late_min: float
    projected_full_day_route_count: int
    policy_diagnostics: dict[str, object]
    one_route_one_vehicle_ok: int
    no_mid_arc_change_ok: int
    no_restock_ok: int


DEFAULT_EVENTS = [
    EventRecord(
        event_id="E1",
        event_time=60,
        event_type="new_order",
        target_customer=23,
        node_from=None,
        node_to=23,
        new_tw_start=250,
        new_tw_end=315,
        weight=900.0,
        volume=3.0,
    ),
    EventRecord(
        event_id="E2",
        event_time=360,
        event_type="cancel",
        target_customer=19,
        node_from=19,
        node_to=None,
        new_tw_start=None,
        new_tw_end=None,
        weight=None,
        volume=None,
    ),
    EventRecord(
        event_id="E3",
        event_time=380,
        event_type="tighten_tw",
        target_customer=54,
        node_from=54,
        node_to=54,
        new_tw_start=560,
        new_tw_end=568,
        weight=None,
        volume=None,
    ),
    EventRecord(
        event_id="E4",
        event_time=450,
        event_type="relocate",
        target_customer=71,
        node_from=71,
        node_to=96,
        new_tw_start=679,
        new_tw_end=765,
        weight=None,
        volume=None,
    ),
]


class Question3DynamicSolver(q2v2.Question2CostFirstV2Solver):
    def __init__(
        self,
        workspace: Path,
        input_root: Path,
        baseline_root: Path,
        output_root: Path,
        seed: int,
        top_route_candidates: int,
    ) -> None:
        super().__init__(
            workspace=workspace,
            input_root=input_root,
            output_root=output_root,
            seed_list=[seed],
            max_generations=1,
            particle_count=1,
            top_route_candidates=top_route_candidates,
        )
        self.dynamic_seed = seed
        self.baseline_root = baseline_root
        self.output_root = output_root
        self.events = list(DEFAULT_EVENTS)
        self.event_log_rows: list[dict[str, object]] = []
        self.event_metric_rows: list[dict[str, object]] = []
        baseline_summary = json.loads(
            (self.baseline_root / "q2_hybrid_cost_summary.json").read_text(encoding="utf-8")
        )
        self.q2_baseline_total_cost = float(baseline_summary["total_cost"])
        self.q2_baseline_route_count = int(baseline_summary["route_count"])
        self.dynamic_units: dict[int, DynamicPolicyServiceUnit] = {
            int(unit.unit_id): self._to_dynamic_unit(unit)
            for unit in self.service_units
        }
        self.cancelled_unit_ids: set[int] = set()
        self.orphan_unit_ids: set[int] = set()
        self.next_unit_id = max(self.dynamic_units) + 1 if self.dynamic_units else 0
        self.all_slots_by_type = {
            vehicle.vehicle_type: [
                self._slot_id(vehicle.vehicle_type, instance)
                for instance in range(1, vehicle.vehicle_count + 1)
            ]
            for vehicle in self.vehicles
        }
        self.all_slot_ids = {
            slot_id
            for slot_ids in self.all_slots_by_type.values()
            for slot_id in slot_ids
        }
        self.plan_by_slot: dict[str, DynamicRoutePlan] = {}
        self.unused_slots = set(self.all_slot_ids)
        self._refresh_dynamic_collections()
        self._load_baseline_plan()

    @staticmethod
    def _slot_id(vehicle_type: str, vehicle_instance: int) -> str:
        return f"{vehicle_type}#{vehicle_instance}"

    @staticmethod
    def _slot_sort_key(slot_id: str) -> tuple[str, int]:
        vehicle_type, instance = slot_id.split("#", 1)
        return vehicle_type, int(instance)

    def _minutes_to_hhmm_safe(self, minute_value: float | int | None) -> str:
        if minute_value is None:
            return ""
        return self._minutes_to_hhmm(float(minute_value))

    def _to_dynamic_unit(self, unit: q2.PolicyServiceUnit) -> DynamicPolicyServiceUnit:
        return DynamicPolicyServiceUnit(
            unit_id=int(unit.unit_id),
            orig_cust_id=int(unit.orig_cust_id),
            unit_type=str(unit.unit_type),
            visit_index=int(unit.visit_index),
            weight=float(unit.weight),
            volume=float(unit.volume),
            tw_start_min=int(unit.tw_start_min),
            tw_end_min=int(unit.tw_end_min),
            x_km=float(unit.x_km),
            y_km=float(unit.y_km),
            eligible_vehicle_types=tuple(str(vehicle_type) for vehicle_type in unit.eligible_vehicle_types),
            source_order_ids=tuple(int(order_id) for order_id in unit.source_order_ids),
            required_visit_count=int(unit.required_visit_count),
            in_green_zone=bool(unit.in_green_zone),
            must_use_ev_under_policy=bool(unit.must_use_ev_under_policy),
            fuel_allowed_after_16=bool(unit.fuel_allowed_after_16),
            ev_allowed_flag=bool(unit.ev_allowed_flag),
            fuel_allowed_flag=bool(unit.fuel_allowed_flag),
            ev_tw_start_min=int(unit.ev_tw_start_min),
            ev_tw_end_min=int(unit.ev_tw_end_min),
            fuel_tw_start_min=int(unit.fuel_tw_start_min),
            fuel_tw_end_min=int(unit.fuel_tw_end_min),
            business_customer_id=int(unit.orig_cust_id),
            service_node_id=int(unit.orig_cust_id),
            baseline_customer_id=int(unit.orig_cust_id),
            latest_event_id=None,
        )

    def _refresh_dynamic_collections(self) -> None:
        active_unit_ids = sorted(
            unit_id
            for unit_id in self.dynamic_units
            if unit_id not in self.cancelled_unit_ids
        )
        self.service_units = [self.dynamic_units[unit_id] for unit_id in active_unit_ids]
        self.active_unit_ids = list(active_unit_ids)
        self.unit_by_id = dict(self.dynamic_units)
        self.route_cache = q1.RouteCache()

    def _load_baseline_plan(self) -> None:
        route_summary_path = self.baseline_root / "q2_hybrid_route_summary.csv"
        route_summary_df = pd.read_csv(route_summary_path, encoding=CSV_ENCODING)
        plan_by_slot: dict[str, DynamicRoutePlan] = {}
        for row in route_summary_df.itertuples(index=False):
            slot_id = self._slot_id(str(row.vehicle_type), int(row.vehicle_instance))
            unit_ids = tuple(
                int(unit_id)
                for unit_id in str(row.unit_sequence).split(",")
                if str(unit_id).strip()
            )
            plan_by_slot[slot_id] = DynamicRoutePlan(
                vehicle_slot=slot_id,
                vehicle_type=str(row.vehicle_type),
                vehicle_instance=int(row.vehicle_instance),
                unit_ids=unit_ids,
                planned_departure_min=float(row.departure_min),
                baseline_route_id=int(row.route_id),
                source_tag="baseline",
            )
        self.plan_by_slot = plan_by_slot
        self.unused_slots = set(self.all_slot_ids) - set(self.plan_by_slot)

    def _node_row(self, cust_id: int) -> pd.Series:
        row_df = self.customer_master.loc[self.customer_master["cust_id"] == int(cust_id)]
        if row_df.empty:
            raise KeyError(f"Unknown service node {cust_id}")
        return row_df.iloc[0]

    def _make_dynamic_unit(
        self,
        *,
        unit_id: int,
        business_customer_id: int,
        service_node_id: int,
        weight: float,
        volume: float,
        tw_start_min: int,
        tw_end_min: int,
        source_order_ids: tuple[int, ...],
        unit_type: str,
        visit_index: int,
        required_visit_count: int,
        baseline_customer_id: int,
        latest_event_id: str | None,
    ) -> DynamicPolicyServiceUnit:
        node_row = self._node_row(service_node_id)
        policy = self._policy_for_customer(service_node_id)
        eligible_vehicle_types = self._eligible_vehicle_types_for_load_after_policy(weight, volume, policy)
        if not eligible_vehicle_types:
            raise RuntimeError(
                f"Unable to build a feasible unit for business customer {business_customer_id} at node {service_node_id}"
            )
        (
            ev_allowed_flag,
            fuel_allowed_flag,
            ev_tw_start_min,
            ev_tw_end_min,
            fuel_tw_start_min,
            fuel_tw_end_min,
        ) = self._policy_vehicle_windows(
            policy=policy,
            tw_start_min=tw_start_min,
            tw_end_min=tw_end_min,
            eligible_vehicle_types=eligible_vehicle_types,
            vehicle_by_name=self.vehicle_by_name,
        )
        return DynamicPolicyServiceUnit(
            unit_id=unit_id,
            orig_cust_id=int(service_node_id),
            unit_type=unit_type,
            visit_index=int(visit_index),
            weight=float(weight),
            volume=float(volume),
            tw_start_min=int(tw_start_min),
            tw_end_min=int(tw_end_min),
            x_km=float(node_row.x_km),
            y_km=float(node_row.y_km),
            eligible_vehicle_types=tuple(str(vehicle_type) for vehicle_type in eligible_vehicle_types),
            source_order_ids=tuple(int(order_id) for order_id in source_order_ids),
            required_visit_count=int(required_visit_count),
            in_green_zone=bool(policy["in_green_zone"]),
            must_use_ev_under_policy=bool(policy["must_use_ev_under_policy"]),
            fuel_allowed_after_16=bool(policy["fuel_allowed_after_16"]),
            ev_allowed_flag=bool(ev_allowed_flag),
            fuel_allowed_flag=bool(fuel_allowed_flag),
            ev_tw_start_min=int(ev_tw_start_min),
            ev_tw_end_min=int(ev_tw_end_min),
            fuel_tw_start_min=int(fuel_tw_start_min),
            fuel_tw_end_min=int(fuel_tw_end_min),
            business_customer_id=int(business_customer_id),
            service_node_id=int(service_node_id),
            baseline_customer_id=int(baseline_customer_id),
            latest_event_id=latest_event_id,
        )

    def _replace_unit(
        self,
        unit_id: int,
        *,
        service_node_id: int | None = None,
        tw_start_min: int | None = None,
        tw_end_min: int | None = None,
        latest_event_id: str | None = None,
    ) -> None:
        old_unit = self.dynamic_units[unit_id]
        self.dynamic_units[unit_id] = self._make_dynamic_unit(
            unit_id=unit_id,
            business_customer_id=int(old_unit.business_customer_id),
            service_node_id=int(service_node_id if service_node_id is not None else old_unit.service_node_id),
            weight=float(old_unit.weight),
            volume=float(old_unit.volume),
            tw_start_min=int(tw_start_min if tw_start_min is not None else old_unit.tw_start_min),
            tw_end_min=int(tw_end_min if tw_end_min is not None else old_unit.tw_end_min),
            source_order_ids=tuple(int(order_id) for order_id in old_unit.source_order_ids),
            unit_type=str(old_unit.unit_type),
            visit_index=int(old_unit.visit_index),
            required_visit_count=int(old_unit.required_visit_count),
            baseline_customer_id=int(old_unit.baseline_customer_id),
            latest_event_id=latest_event_id,
        )

    def _route_affinity(self, route: q1.TypedRoute, unit_id: int) -> float:
        unit = self.unit_by_id[unit_id]
        if not route.unit_ids:
            return 0.0
        if unit.orig_cust_id in self.customer_index and all(
            self.unit_by_id[other_id].orig_cust_id in self.customer_index
            for other_id in route.unit_ids
        ):
            return q1.Question1Solver._route_affinity(self, route, unit_id)
        return min(
            self._distance_between(unit.orig_cust_id, self.unit_by_id[other_id].orig_cust_id)
            for other_id in route.unit_ids
        )

    def _travel_leg_metrics(
        self,
        *,
        origin_id: int,
        dest_id: int,
        departure_min: float,
        power_type: str,
        remaining_weight: float,
        capacity_kg: float,
        apply_load_multiplier: bool,
    ) -> dict[str, float]:
        departure_array = np.array([departure_min], dtype=np.float64)
        after_hours_travel, after_hours_fuel, after_hours_electric = self._after_hours_full_values(origin_id, dest_id)
        travel_min = float(
            self._interpolate_metric(
                self.travel_time_lookup,
                origin_id,
                dest_id,
                departure_array,
                after_hours_travel,
            )[0]
        )
        base_fuel = float(
            self._interpolate_metric(
                self.base_fuel_lookup,
                origin_id,
                dest_id,
                departure_array,
                after_hours_fuel,
            )[0]
        )
        base_electric = float(
            self._interpolate_metric(
                self.base_electric_lookup,
                origin_id,
                dest_id,
                departure_array,
                after_hours_electric,
            )[0]
        )
        load_ratio = max(0.0, min(1.0, remaining_weight / capacity_kg)) if apply_load_multiplier else 0.0
        load_multiplier = (
            1.0 + (0.40 if power_type == "fuel" else 0.35) * load_ratio
            if apply_load_multiplier
            else 1.0
        )
        fuel_l = base_fuel * load_multiplier
        electricity_kwh = base_electric * load_multiplier
        if power_type == "fuel":
            energy_value = fuel_l
            energy_cost = fuel_l * q1.FUEL_PRICE
            carbon_cost = fuel_l * q1.FUEL_CARBON_FACTOR * q1.CARBON_COST
        else:
            energy_value = electricity_kwh
            energy_cost = electricity_kwh * q1.ELECTRICITY_PRICE
            carbon_cost = electricity_kwh * q1.ELECTRICITY_CARBON_FACTOR * q1.CARBON_COST
        return {
            "travel_min": float(travel_min),
            "fuel_l": float(fuel_l),
            "electricity_kwh": float(electricity_kwh),
            "energy_value": float(energy_value),
            "energy_cost": float(energy_cost),
            "carbon_cost": float(carbon_cost),
            "distance_km": float(self._distance_between(origin_id, dest_id)),
            "after_hours_distance_km": float(self._scalar_after_hours_distance(origin_id, dest_id, departure_min)),
        }

    def _simulate_partial_route(
        self,
        *,
        vehicle_type: str,
        unit_ids: tuple[int, ...],
        start_node: int,
        start_time: float,
        route_id: int,
        vehicle_instance: int,
        vehicle_slot: str,
        include_startup: bool,
        original_stop_offset: int = 0,
    ) -> dict[str, object]:
        vehicle = self.vehicle_by_name[vehicle_type]
        units = tuple(self.unit_by_id[unit_id] for unit_id in unit_ids)
        total_weight = float(sum(unit.weight for unit in units))
        total_volume = float(sum(unit.volume for unit in units))
        if (
            total_weight > vehicle.capacity_kg + EPS
            or total_volume > vehicle.capacity_m3 + EPS
            or len({unit.orig_cust_id for unit in units}) != len(units)
            or any(vehicle_type not in unit.eligible_vehicle_types for unit in units)
        ):
            return {
                "feasible": False,
                "route_cost": float("inf"),
                "total_late_min": float("inf"),
                "stop_rows": [],
                "stop_details": [],
                "return_arc": {},
                "departure_min": float(start_time),
                "return_min": float("inf"),
            }

        previous_node = int(start_node)
        current_departure = float(start_time)
        remaining_weight = total_weight
        remaining_volume = total_volume
        total_wait = 0.0
        total_late = 0.0
        total_fuel = 0.0
        total_electricity = 0.0
        reference_total_fuel = 0.0
        reference_total_electricity = 0.0
        route_distance = 0.0
        after_hours_travel_km = 0.0
        after_hours_service_count = 0
        late_positive_stop_count = 0
        max_late_min = 0.0
        stop_rows: list[dict[str, object]] = []
        stop_details: list[dict[str, object]] = []

        for offset, unit in enumerate(units, start=1):
            tw_start_min, tw_end_min = self._service_window_for_vehicle(unit, vehicle_type)
            if not np.isfinite(tw_start_min) or not np.isfinite(tw_end_min) or tw_end_min < tw_start_min - EPS:
                return {
                    "feasible": False,
                    "route_cost": float("inf"),
                    "total_late_min": float("inf"),
                    "stop_rows": [],
                    "stop_details": [],
                    "return_arc": {},
                    "departure_min": float(start_time),
                    "return_min": float("inf"),
                }
            leg = self._travel_leg_metrics(
                origin_id=previous_node,
                dest_id=unit.orig_cust_id,
                departure_min=current_departure,
                power_type=vehicle.power_type,
                remaining_weight=remaining_weight,
                capacity_kg=vehicle.capacity_kg,
                apply_load_multiplier=True,
            )
            arrival = current_departure + leg["travel_min"]
            wait = max(float(tw_start_min) - arrival, 0.0)
            service_start = arrival + wait
            late = max(service_start - float(tw_end_min), 0.0)
            service_end = service_start + self.service_time_min
            total_wait += wait
            total_late += late
            reference_total_fuel += leg["fuel_l"]
            reference_total_electricity += leg["electricity_kwh"]
            if vehicle.power_type == "fuel":
                total_fuel += leg["fuel_l"]
            else:
                total_electricity += leg["electricity_kwh"]
            route_distance += leg["distance_km"]
            after_hours_travel_km += leg["after_hours_distance_km"]
            if service_start > q1.DAY_END_MIN + EPS:
                after_hours_service_count += 1
            if late > EPS:
                late_positive_stop_count += 1
                max_late_min = max(max_late_min, late)

            stop_row = {
                "route_id": route_id,
                "vehicle_slot": vehicle_slot,
                "vehicle_type": vehicle_type,
                "power_type": vehicle.power_type,
                "vehicle_instance": vehicle_instance,
                "stop_index": original_stop_offset + offset,
                "unit_id": unit.unit_id,
                "orig_cust_id": unit.orig_cust_id,
                "business_customer_id": unit.business_customer_id,
                "service_node_id": unit.service_node_id,
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
                "tw_start_min": int(unit.tw_start_min),
                "tw_start_hhmm": self._minutes_to_hhmm(unit.tw_start_min),
                "tw_end_min": int(unit.tw_end_min),
                "tw_end_hhmm": self._minutes_to_hhmm(unit.tw_end_min),
                "policy_tw_start_min": round(float(tw_start_min), 1),
                "policy_tw_start_hhmm": self._minutes_to_hhmm(float(tw_start_min)),
                "policy_tw_end_min": round(float(tw_end_min), 1),
                "policy_tw_end_hhmm": self._minutes_to_hhmm(float(tw_end_min)),
                "policy_vehicle_window_type": "fuel" if vehicle.power_type == "fuel" else "ev",
                "in_green_zone": int(unit.in_green_zone),
                "must_use_ev_under_policy": int(unit.must_use_ev_under_policy),
                "fuel_allowed_after_16": int(unit.fuel_allowed_after_16),
                "after_hours_service_flag": int(service_start > q1.DAY_END_MIN + EPS),
                "late_positive_flag": int(late > EPS),
            }
            stop_rows.append(stop_row)
            stop_details.append(
                {
                    **stop_row,
                    "prev_node": previous_node,
                    "arc_departure_min": float(current_departure),
                    "arc_departure_hhmm": self._minutes_to_hhmm(current_departure),
                    "travel_min": float(leg["travel_min"]),
                    "leg_energy_cost": float(leg["energy_cost"]),
                    "leg_carbon_cost": float(leg["carbon_cost"]),
                    "leg_fuel_l": float(leg["fuel_l"]),
                    "leg_electricity_kwh": float(leg["electricity_kwh"]),
                    "leg_distance_km": float(leg["distance_km"]),
                    "leg_after_hours_distance_km": float(leg["after_hours_distance_km"]),
                }
            )
            current_departure = service_end
            previous_node = unit.orig_cust_id
            remaining_weight -= unit.weight
            remaining_volume -= unit.volume

        return_leg = self._travel_leg_metrics(
            origin_id=previous_node,
            dest_id=0,
            departure_min=current_departure,
            power_type=vehicle.power_type,
            remaining_weight=0.0,
            capacity_kg=vehicle.capacity_kg,
            apply_load_multiplier=False,
        )
        return_time = current_departure + return_leg["travel_min"]
        route_distance += return_leg["distance_km"]
        after_hours_travel_km += return_leg["after_hours_distance_km"]
        reference_total_fuel += return_leg["fuel_l"]
        reference_total_electricity += return_leg["electricity_kwh"]
        if vehicle.power_type == "fuel":
            total_fuel += return_leg["fuel_l"]
            energy_cost = total_fuel * q1.FUEL_PRICE
            carbon_cost = total_fuel * q1.FUEL_CARBON_FACTOR * q1.CARBON_COST
        else:
            total_electricity += return_leg["electricity_kwh"]
            energy_cost = total_electricity * q1.ELECTRICITY_PRICE
            carbon_cost = total_electricity * q1.ELECTRICITY_CARBON_FACTOR * q1.CARBON_COST
        waiting_cost = total_wait * q1.WAIT_COST_PER_MIN
        late_cost = total_late * q1.LATE_COST_PER_MIN
        startup_cost = q1.START_COST if include_startup else 0.0
        route_cost = startup_cost + energy_cost + carbon_cost + waiting_cost + late_cost
        return {
            "feasible": True,
            "vehicle_type": vehicle_type,
            "power_type": vehicle.power_type,
            "departure_min": float(start_time),
            "return_min": float(return_time),
            "route_cost": float(route_cost),
            "energy_cost": float(energy_cost),
            "carbon_cost": float(carbon_cost),
            "waiting_cost": float(waiting_cost),
            "late_cost": float(late_cost),
            "startup_cost": float(startup_cost),
            "total_wait_min": float(total_wait),
            "total_late_min": float(total_late),
            "total_fuel_l": float(total_fuel),
            "total_electricity_kwh": float(total_electricity),
            "reference_fuel_l": float(reference_total_fuel),
            "reference_electricity_kwh": float(reference_total_electricity),
            "route_distance_km": float(route_distance),
            "after_hours_travel_km": float(after_hours_travel_km),
            "after_hours_service_count": int(after_hours_service_count),
            "after_hours_return_flag": int(return_time > q1.DAY_END_MIN + EPS),
            "late_positive_stop_count": int(late_positive_stop_count),
            "max_late_min": float(max_late_min),
            "total_weight": float(total_weight),
            "total_volume": float(total_volume),
            "stop_rows": stop_rows,
            "stop_details": stop_details,
            "return_arc": {
                "origin_node": previous_node,
                "dest_node": 0,
                "departure_min": float(current_departure),
                "departure_hhmm": self._minutes_to_hhmm(current_departure),
                "arrival_min": float(return_time),
                "arrival_hhmm": self._minutes_to_hhmm(return_time),
                "travel_min": float(return_leg["travel_min"]),
                "energy_cost": float(return_leg["energy_cost"]),
                "carbon_cost": float(return_leg["carbon_cost"]),
                "fuel_l": float(return_leg["fuel_l"]),
                "electricity_kwh": float(return_leg["electricity_kwh"]),
                "distance_km": float(return_leg["distance_km"]),
                "after_hours_distance_km": float(return_leg["after_hours_distance_km"]),
            },
        }

    def _simulate_route_plan(self, route_plan: DynamicRoutePlan, route_id: int) -> dict[str, object]:
        return self._simulate_partial_route(
            vehicle_type=route_plan.vehicle_type,
            unit_ids=route_plan.unit_ids,
            start_node=0,
            start_time=route_plan.planned_departure_min,
            route_id=route_id,
            vehicle_instance=route_plan.vehicle_instance,
            vehicle_slot=route_plan.vehicle_slot,
            include_startup=True,
            original_stop_offset=0,
        )

    def _distance_covered_since_departure(
        self,
        *,
        origin_id: int,
        dest_id: int,
        departure_min: float,
        query_min: float,
    ) -> float:
        if query_min <= departure_min + EPS:
            return 0.0
        total_distance = float(self._distance_between(origin_id, dest_id))
        if total_distance <= EPS:
            return 0.0
        covered = 0.0
        current_time = float(departure_min)
        end_time = float(query_min)
        while current_time < end_time - EPS and covered < total_distance - EPS:
            if current_time >= q1.DAY_END_MIN:
                interval_min = end_time - current_time
                covered += self.after_hours_speed * interval_min / 60.0
                break
            segment_index = int(np.searchsorted(self.segment_ends, current_time, side="right"))
            if segment_index >= len(self.segment_starts):
                interval_min = end_time - current_time
                covered += self.after_hours_speed * interval_min / 60.0
                break
            interval_end = min(float(self.segment_ends[segment_index]), end_time)
            interval_min = max(interval_end - current_time, 0.0)
            if interval_min <= EPS:
                current_time = interval_end
                continue
            covered += float(self.segment_speeds[segment_index]) * interval_min / 60.0
            current_time = interval_end
        return min(total_distance, max(covered, 0.0))

    def _remaining_leg_fraction(
        self,
        *,
        origin_id: int,
        dest_id: int,
        departure_min: float,
        event_time: float,
    ) -> float:
        total_distance = float(self._distance_between(origin_id, dest_id))
        if total_distance <= EPS:
            return 0.0
        covered = self._distance_covered_since_departure(
            origin_id=origin_id,
            dest_id=dest_id,
            departure_min=departure_min,
            query_min=event_time,
        )
        return max(total_distance - covered, 0.0) / total_distance

    def _route_state_from_full_simulation(
        self,
        *,
        route_plan: DynamicRoutePlan,
        route_id: int,
        full_sim: dict[str, object],
        event_time: float,
    ) -> VehicleDynamicState | None:
        if not full_sim["feasible"]:
            return None
        power_type = str(full_sim["power_type"])
        stop_details = list(full_sim["stop_details"])
        departure_min = float(full_sim["departure_min"])
        if event_time <= departure_min + EPS:
            future_rows = []
            for remaining_index, stop_row in enumerate(stop_details, start=1):
                future_rows.append(
                    {
                        **stop_row,
                        "original_stop_index": int(stop_row["stop_index"]),
                        "remaining_stop_index": remaining_index,
                        "current_frozen_flag": 0,
                        "editable_flag": 1,
                        "status_at_event": "depot_pending",
                    }
                )
            return VehicleDynamicState(
                vehicle_slot=route_plan.vehicle_slot,
                vehicle_type=route_plan.vehicle_type,
                vehicle_instance=route_plan.vehicle_instance,
                power_type=power_type,
                baseline_route_id=route_plan.baseline_route_id,
                status="depot_pending",
                planned_departure_min=departure_min,
                full_return_min=float(full_sim["return_min"]),
                release_node=0,
                release_time=float(event_time),
                current_arc_frozen="",
                fixed_prefix=tuple(),
                current_frozen_unit_id=None,
                editable_suffix=tuple(route_plan.unit_ids),
                remaining_unit_ids=tuple(route_plan.unit_ids),
                remaining_total_cost=float(full_sim["route_cost"]),
                remaining_total_late_min=float(full_sim["total_late_min"]),
                realized_cost_to_event=0.0,
                future_stop_rows=future_rows,
                full_simulation=full_sim,
            )

        fixed_prefix: list[int] = []
        realized_cost = 0.0
        for stop_index, stop_row in enumerate(stop_details):
            stop_service_end = float(stop_row["service_end_min"])
            if event_time >= stop_service_end - EPS:
                fixed_prefix.append(int(stop_row["unit_id"]))
                realized_cost += (
                    float(stop_row["leg_energy_cost"])
                    + float(stop_row["leg_carbon_cost"])
                    + float(stop_row["waiting_min"]) * q1.WAIT_COST_PER_MIN
                    + float(stop_row["late_min"]) * q1.LATE_COST_PER_MIN
                )
                continue
            current_frozen_unit_id = int(stop_row["unit_id"])
            editable_suffix = tuple(int(row["unit_id"]) for row in stop_details[stop_index + 1 :])
            remaining_constant_cost = 0.0
            remaining_constant_late = 0.0
            arrival_min = float(stop_row["arrival_min"])
            service_start_min = float(stop_row["service_start_min"])
            if event_time < arrival_min - EPS:
                elapsed_fraction = 1.0 - self._remaining_leg_fraction(
                    origin_id=int(stop_row["prev_node"]),
                    dest_id=int(stop_row["service_node_id"]),
                    departure_min=float(stop_row["arc_departure_min"]),
                    event_time=event_time,
                )
                realized_cost += (
                    float(stop_row["leg_energy_cost"]) + float(stop_row["leg_carbon_cost"])
                ) * elapsed_fraction
                remaining_fraction = self._remaining_leg_fraction(
                    origin_id=int(stop_row["prev_node"]),
                    dest_id=int(stop_row["service_node_id"]),
                    departure_min=float(stop_row["arc_departure_min"]),
                    event_time=event_time,
                )
                remaining_constant_cost += float(stop_row["leg_energy_cost"]) * remaining_fraction
                remaining_constant_cost += float(stop_row["leg_carbon_cost"]) * remaining_fraction
                if service_start_min > arrival_min + EPS:
                    remaining_constant_cost += (
                        max(service_start_min - max(arrival_min, event_time), 0.0)
                        * q1.WAIT_COST_PER_MIN
                    )
                if event_time < service_start_min - EPS and float(stop_row["late_min"]) > 0.0:
                    remaining_constant_cost += float(stop_row["late_min"]) * q1.LATE_COST_PER_MIN
                    remaining_constant_late += float(stop_row["late_min"])
            elif event_time < service_start_min - EPS:
                realized_cost += float(stop_row["leg_energy_cost"]) + float(stop_row["leg_carbon_cost"])
                realized_cost += (event_time - arrival_min) * q1.WAIT_COST_PER_MIN
                remaining_constant_cost += (service_start_min - event_time) * q1.WAIT_COST_PER_MIN
                if float(stop_row["late_min"]) > 0.0:
                    remaining_constant_cost += float(stop_row["late_min"]) * q1.LATE_COST_PER_MIN
                    remaining_constant_late += float(stop_row["late_min"])
            else:
                realized_cost += float(stop_row["leg_energy_cost"]) + float(stop_row["leg_carbon_cost"])
                realized_cost += float(stop_row["waiting_min"]) * q1.WAIT_COST_PER_MIN
                if event_time > service_start_min + EPS:
                    realized_cost += float(stop_row["late_min"]) * q1.LATE_COST_PER_MIN
            suffix_sim = self._simulate_partial_route(
                vehicle_type=route_plan.vehicle_type,
                unit_ids=editable_suffix,
                start_node=int(stop_row["service_node_id"]),
                start_time=float(stop_row["service_end_min"]),
                route_id=route_id,
                vehicle_instance=route_plan.vehicle_instance,
                vehicle_slot=route_plan.vehicle_slot,
                include_startup=False,
                original_stop_offset=int(stop_row["stop_index"]),
            )
            if not suffix_sim["feasible"]:
                return None
            future_rows = [
                {
                    **stop_row,
                    "original_stop_index": int(stop_row["stop_index"]),
                    "remaining_stop_index": 1,
                    "current_frozen_flag": 1,
                    "editable_flag": 0,
                    "status_at_event": "onboard",
                }
            ]
            for remaining_index, suffix_row in enumerate(suffix_sim["stop_details"], start=2):
                future_rows.append(
                    {
                        **suffix_row,
                        "original_stop_index": int(suffix_row["stop_index"]),
                        "remaining_stop_index": remaining_index,
                        "current_frozen_flag": 0,
                        "editable_flag": 1,
                        "status_at_event": "onboard",
                    }
                )
            return VehicleDynamicState(
                vehicle_slot=route_plan.vehicle_slot,
                vehicle_type=route_plan.vehicle_type,
                vehicle_instance=route_plan.vehicle_instance,
                power_type=power_type,
                baseline_route_id=route_plan.baseline_route_id,
                status="onboard",
                planned_departure_min=departure_min,
                full_return_min=float(suffix_sim["return_min"]),
                release_node=int(stop_row["service_node_id"]),
                release_time=float(stop_row["service_end_min"]),
                current_arc_frozen=f"{int(stop_row['prev_node'])}->{int(stop_row['service_node_id'])}",
                fixed_prefix=tuple(fixed_prefix),
                current_frozen_unit_id=current_frozen_unit_id,
                editable_suffix=editable_suffix,
                remaining_unit_ids=(current_frozen_unit_id, *editable_suffix),
                remaining_total_cost=float(remaining_constant_cost + float(suffix_sim["route_cost"])),
                remaining_total_late_min=float(remaining_constant_late + float(suffix_sim["total_late_min"])),
                realized_cost_to_event=float(realized_cost + q1.START_COST),
                future_stop_rows=future_rows,
                full_simulation=full_sim,
            )

        return_arc = dict(full_sim["return_arc"])
        if event_time < float(return_arc["arrival_min"]) - EPS:
            realized_fraction = 1.0 - self._remaining_leg_fraction(
                origin_id=int(return_arc["origin_node"]),
                dest_id=0,
                departure_min=float(return_arc["departure_min"]),
                event_time=event_time,
            )
            remaining_fraction = self._remaining_leg_fraction(
                origin_id=int(return_arc["origin_node"]),
                dest_id=0,
                departure_min=float(return_arc["departure_min"]),
                event_time=event_time,
            )
            realized_cost = float(full_sim["route_cost"]) - (
                float(return_arc["energy_cost"]) + float(return_arc["carbon_cost"])
            ) * remaining_fraction
            remaining_cost = (
                float(return_arc["energy_cost"]) + float(return_arc["carbon_cost"])
            ) * remaining_fraction
            return VehicleDynamicState(
                vehicle_slot=route_plan.vehicle_slot,
                vehicle_type=route_plan.vehicle_type,
                vehicle_instance=route_plan.vehicle_instance,
                power_type=power_type,
                baseline_route_id=route_plan.baseline_route_id,
                status="returning",
                planned_departure_min=departure_min,
                full_return_min=float(return_arc["arrival_min"]),
                release_node=0,
                release_time=float(return_arc["arrival_min"]),
                current_arc_frozen=f"{int(return_arc['origin_node'])}->0",
                fixed_prefix=tuple(int(stop_row["unit_id"]) for stop_row in stop_details),
                current_frozen_unit_id=None,
                editable_suffix=tuple(),
                remaining_unit_ids=tuple(),
                remaining_total_cost=float(remaining_cost),
                remaining_total_late_min=0.0,
                realized_cost_to_event=float(realized_cost),
                future_stop_rows=[],
                full_simulation=full_sim,
            )
        return None

    def _policy_diagnostics_from_rows(self, stop_rows: Iterable[dict[str, object]]) -> dict[str, object]:
        violation_keys: set[tuple[str, int]] = set()
        route_ids: set[int] = set()
        mandatory_non_ev_customers: set[int] = set()
        fuel_route_green_zone_pre16_visit_count = 0
        fuel_route_green_zone_post16_visit_count = 0
        ev_route_green_zone_visit_count = 0
        for stop_row in stop_rows:
            route_id = int(stop_row["route_id"])
            service_start_min = float(stop_row["service_start_min"])
            power_type = str(stop_row["power_type"])
            in_green_zone = bool(int(stop_row["in_green_zone"]))
            must_use_ev = bool(int(stop_row["must_use_ev_under_policy"]))
            business_customer_id = int(stop_row["business_customer_id"])
            key = (str(stop_row["vehicle_slot"]), int(stop_row["remaining_stop_index"]))
            if not in_green_zone:
                continue
            if power_type == "fuel":
                if service_start_min < self.policy_ban_end_min - EPS:
                    fuel_route_green_zone_pre16_visit_count += 1
                    violation_keys.add(key)
                    route_ids.add(route_id)
                else:
                    fuel_route_green_zone_post16_visit_count += 1
                if must_use_ev:
                    mandatory_non_ev_customers.add(business_customer_id)
                    violation_keys.add(key)
                    route_ids.add(route_id)
            else:
                ev_route_green_zone_visit_count += 1
        return {
            "mandatory_ev_served_by_non_ev_count": int(len(mandatory_non_ev_customers)),
            "mandatory_ev_served_by_non_ev_customer_ids": sorted(mandatory_non_ev_customers),
            "fuel_route_green_zone_pre16_visit_count": int(fuel_route_green_zone_pre16_visit_count),
            "fuel_route_green_zone_post16_visit_count": int(fuel_route_green_zone_post16_visit_count),
            "ev_route_green_zone_visit_count": int(ev_route_green_zone_visit_count),
            "policy_violation_count": int(len(violation_keys)),
            "policy_violation_route_ids": sorted(route_ids),
        }

    def _build_snapshot(
        self,
        *,
        event_time: float,
        plan_by_slot: dict[str, DynamicRoutePlan] | None = None,
    ) -> Snapshot:
        plan_by_slot = self.plan_by_slot if plan_by_slot is None else plan_by_slot
        route_states: list[VehicleDynamicState] = []
        realized_total_cost = 0.0
        for route_id, slot_id in enumerate(sorted(plan_by_slot, key=self._slot_sort_key), start=1):
            route_plan = plan_by_slot[slot_id]
            full_sim = self._simulate_route_plan(route_plan, route_id)
            state = self._route_state_from_full_simulation(
                route_plan=route_plan,
                route_id=route_id,
                full_sim=full_sim,
                event_time=event_time,
            )
            if state is not None:
                route_states.append(state)
                realized_total_cost += float(state.realized_cost_to_event)
            elif full_sim["feasible"]:
                realized_total_cost += float(full_sim["route_cost"])
        active_states = sorted(
            route_states,
            key=lambda item: (item.vehicle_type, item.vehicle_instance, item.vehicle_slot),
        )
        route_id_by_slot = {
            state.vehicle_slot: route_index
            for route_index, state in enumerate(active_states, start=1)
        }
        remaining_stop_rows: list[dict[str, object]] = []
        unit_to_slot: dict[int, str] = {}
        for state in active_states:
            route_id = route_id_by_slot[state.vehicle_slot]
            for stop_row in state.future_stop_rows:
                enriched = dict(stop_row)
                enriched["route_id"] = route_id
                remaining_stop_rows.append(enriched)
                unit_to_slot[int(enriched["unit_id"])] = state.vehicle_slot
        remaining_stop_rows.sort(key=lambda row: (int(row["route_id"]), int(row["remaining_stop_index"])))
        policy_diagnostics = self._policy_diagnostics_from_rows(remaining_stop_rows)
        return Snapshot(
            event_time=float(event_time),
            route_states=active_states,
            route_state_by_slot={state.vehicle_slot: state for state in active_states},
            remaining_stop_rows=remaining_stop_rows,
            unit_to_slot=unit_to_slot,
            realized_total_cost=float(realized_total_cost),
            remaining_total_cost=float(sum(state.remaining_total_cost for state in active_states)),
            projected_full_day_cost=float(realized_total_cost + sum(state.remaining_total_cost for state in active_states)),
            remaining_total_late_min=float(sum(state.remaining_total_late_min for state in active_states)),
            projected_full_day_route_count=int(len(plan_by_slot)),
            policy_diagnostics=policy_diagnostics,
            one_route_one_vehicle_ok=int(len({state.vehicle_slot for state in active_states}) == len(active_states)),
            no_mid_arc_change_ok=1,
            no_restock_ok=1,
        )

    def _event_window_end(self, event: EventRecord) -> int:
        if event.new_tw_end is not None:
            return int(event.new_tw_end)
        return int(event.event_time)

    def _future_target_unit_ids(self, snapshot: Snapshot, target_customer: int) -> list[int]:
        return sorted(
            unit_id
            for unit_id, slot_id in snapshot.unit_to_slot.items()
            if int(self.unit_by_id[unit_id].business_customer_id) == int(target_customer)
            and snapshot.route_state_by_slot[slot_id].status in {"depot_pending", "onboard"}
        )

    def _route_plan_by_slot_copy(
        self,
        plan_by_slot: dict[str, DynamicRoutePlan] | None = None,
    ) -> dict[str, DynamicRoutePlan]:
        source = self.plan_by_slot if plan_by_slot is None else plan_by_slot
        return {slot_id: replace(route_plan) for slot_id, route_plan in source.items()}

    def _apply_cancel_to_plan(
        self,
        *,
        target_unit_ids: Iterable[int],
        plan_by_slot: dict[str, DynamicRoutePlan],
        snapshot_before: Snapshot,
    ) -> None:
        target_unit_set = set(int(unit_id) for unit_id in target_unit_ids)
        for slot_id, route_plan in list(plan_by_slot.items()):
            state = snapshot_before.route_state_by_slot.get(slot_id)
            if state is None or state.status != "depot_pending":
                continue
            kept_unit_ids = tuple(unit_id for unit_id in route_plan.unit_ids if unit_id not in target_unit_set)
            if len(kept_unit_ids) == len(route_plan.unit_ids):
                continue
            if kept_unit_ids:
                plan_by_slot[slot_id] = replace(route_plan, unit_ids=kept_unit_ids, source_tag="event_cancel")
            else:
                plan_by_slot.pop(slot_id, None)

    def _event_feasibility(
        self,
        *,
        event: EventRecord,
        snapshot: Snapshot,
    ) -> dict[str, object]:
        active_target_rows = [
            row
            for row in snapshot.remaining_stop_rows
            if int(row["business_customer_id"]) == int(event.target_customer)
        ]
        if event.event_type == "new_order":
            missing = int(len(active_target_rows) == 0)
            return {
                "satisfied": missing == 0,
                "violation_score": float(missing),
                "violation_count": missing,
                "failure_reason": "" if missing == 0 else "new order is not assigned to any remaining route",
            }
        if event.event_type == "cancel":
            remaining = len(active_target_rows)
            return {
                "satisfied": remaining == 0,
                "violation_score": float(remaining),
                "violation_count": int(remaining),
                "failure_reason": "" if remaining == 0 else "cancelled customer still appears in remaining routes",
            }
        if event.event_type == "tighten_tw":
            violating_rows = [
                row
                for row in active_target_rows
                if float(row["service_start_min"]) > float(event.new_tw_end) + EPS
                or float(row["service_start_min"]) < float(event.new_tw_start) - EPS
            ]
            late_overrun = sum(
                max(float(row["service_start_min"]) - float(event.new_tw_end), 0.0)
                for row in violating_rows
            )
            return {
                "satisfied": len(violating_rows) == 0,
                "violation_score": float(len(violating_rows)) + late_overrun,
                "violation_count": int(len(violating_rows)),
                "failure_reason": (
                    ""
                    if not violating_rows
                    else "tightened time-window customer still has late remaining visits"
                ),
            }
        if event.event_type == "relocate":
            wrong_node_rows = [
                row
                for row in active_target_rows
                if int(row["service_node_id"]) != int(event.node_to)
            ]
            return {
                "satisfied": len(wrong_node_rows) == 0,
                "violation_score": float(len(wrong_node_rows)),
                "violation_count": int(len(wrong_node_rows)),
                "failure_reason": (
                    ""
                    if not wrong_node_rows
                    else "relocated customer still points to the old service node"
                ),
            }
        return {
            "satisfied": True,
            "violation_score": 0.0,
            "violation_count": 0,
            "failure_reason": "",
        }

    def _evaluate_plan_for_event(
        self,
        *,
        event: EventRecord,
        event_time: float,
        plan_by_slot: dict[str, DynamicRoutePlan],
    ) -> tuple[Snapshot, dict[str, object], tuple[int, int, float, float]]:
        snapshot = self._build_snapshot(event_time=event_time, plan_by_slot=plan_by_slot)
        feasibility = self._event_feasibility(event=event, snapshot=snapshot)
        rank_key = (
            int(snapshot.policy_diagnostics["policy_violation_count"]),
            int(not feasibility["satisfied"]),
            round(float(feasibility["violation_score"]), 6),
            round(float(snapshot.remaining_total_cost), 6),
        )
        return snapshot, feasibility, rank_key

    def _modified_vehicle_count(
        self,
        before_plan: dict[str, DynamicRoutePlan],
        after_plan: dict[str, DynamicRoutePlan],
    ) -> int:
        all_slots = set(before_plan) | set(after_plan)
        return int(
            sum(
                int(
                    (
                        before_plan.get(slot_id).unit_ids if slot_id in before_plan else tuple()
                    )
                    != (
                        after_plan.get(slot_id).unit_ids if slot_id in after_plan else tuple()
                    )
                )
                for slot_id in all_slots
            )
        )

    def _switched_depot_unit_count(
        self,
        *,
        before_snapshot: Snapshot,
        after_snapshot: Snapshot,
        tracked_unit_ids: Iterable[int],
    ) -> int:
        count = 0
        for unit_id in tracked_unit_ids:
            before_slot = before_snapshot.unit_to_slot.get(int(unit_id))
            after_slot = after_snapshot.unit_to_slot.get(int(unit_id))
            if before_slot is None or after_slot is None:
                continue
            if before_slot != after_slot:
                count += 1
        return int(count)

    def _simulate_quick_route_after_event(
        self,
        *,
        route_plan: DynamicRoutePlan | None,
        route_id: int,
        event_time: float,
    ) -> VehicleDynamicState | None:
        if route_plan is None:
            return None
        full_sim = self._simulate_route_plan(route_plan, route_id)
        return self._route_state_from_full_simulation(
            route_plan=route_plan,
            route_id=route_id,
            full_sim=full_sim,
            event_time=event_time,
        )

    def _stop_policy_signature(self, stop_row: dict[str, object]) -> tuple[int, int]:
        fuel_green_zone_pre16 = int(
            str(stop_row["power_type"]) == "fuel"
            and bool(int(stop_row["in_green_zone"]))
            and float(stop_row["service_start_min"]) < self.policy_ban_end_min - EPS
        )
        mandatory_non_ev = int(
            str(stop_row["power_type"]) == "fuel"
            and bool(int(stop_row["must_use_ev_under_policy"]))
        )
        return fuel_green_zone_pre16, mandatory_non_ev

    def _dissipation_time_from_states(
        self,
        *,
        old_state: VehicleDynamicState | None,
        new_state: VehicleDynamicState | None,
        event_time: float,
    ) -> float:
        if old_state is None or new_state is None:
            return float(event_time)
        old_rows_by_unit = {
            int(row["unit_id"]): row
            for row in old_state.future_stop_rows
        }
        new_rows = list(new_state.future_stop_rows)
        common_unit_ids = [
            int(row["unit_id"])
            for row in new_rows
            if int(row["unit_id"]) in old_rows_by_unit
        ]
        if not common_unit_ids:
            return float(event_time)
        for start_pos, unit_id in enumerate(common_unit_ids):
            stable = True
            for tail_unit_id in common_unit_ids[start_pos:]:
                new_row = next(row for row in new_rows if int(row["unit_id"]) == tail_unit_id)
                old_row = old_rows_by_unit[tail_unit_id]
                if abs(float(new_row["service_start_min"]) - float(old_row["service_start_min"])) > 1.0 + EPS:
                    stable = False
                    break
                if int(float(new_row["late_min"]) > EPS) != int(float(old_row["late_min"]) > EPS):
                    stable = False
                    break
                if self._stop_policy_signature(new_row) != self._stop_policy_signature(old_row):
                    stable = False
                    break
            if stable:
                stable_row = next(row for row in new_rows if int(row["unit_id"]) == unit_id)
                return float(stable_row["service_end_min"])
        return max(float(old_state.full_return_min), float(new_state.full_return_min))

    def _adaptive_t0(
        self,
        *,
        event: EventRecord,
        event_time: float,
        direct_slots: Iterable[str],
        snapshot_before: Snapshot,
        modified_plan_by_slot: dict[str, DynamicRoutePlan],
    ) -> float:
        t0 = float(self._event_window_end(event))
        for route_id, slot_id in enumerate(sorted(set(direct_slots), key=self._slot_sort_key), start=1):
            old_state = snapshot_before.route_state_by_slot.get(slot_id)
            new_state = self._simulate_quick_route_after_event(
                route_plan=modified_plan_by_slot.get(slot_id),
                route_id=route_id,
                event_time=event_time,
            )
            t0 = max(
                t0,
                self._dissipation_time_from_states(
                    old_state=old_state,
                    new_state=new_state,
                    event_time=event_time,
                ),
            )
        return float(t0)

    def _suffix_candidate_rank(
        self,
        *,
        event: EventRecord,
        current_frozen_row: dict[str, object],
        suffix_sim: dict[str, object],
    ) -> tuple[int, float, float]:
        future_rows = [
            {
                **current_frozen_row,
                "remaining_stop_index": 1,
                "current_frozen_flag": 1,
                "editable_flag": 0,
            }
        ]
        for remaining_index, stop_row in enumerate(suffix_sim["stop_details"], start=2):
            future_rows.append(
                {
                    **stop_row,
                    "remaining_stop_index": remaining_index,
                    "current_frozen_flag": 0,
                    "editable_flag": 1,
                }
            )
        policy_violations = self._policy_diagnostics_from_rows(future_rows)["policy_violation_count"]
        if event.event_type == "tighten_tw":
            target_overrun = sum(
                max(float(row["service_start_min"]) - float(event.new_tw_end), 0.0)
                for row in future_rows
                if int(row["business_customer_id"]) == int(event.target_customer)
            )
        else:
            target_overrun = 0.0
        return (
            int(policy_violations),
            round(float(target_overrun), 6),
            round(float(suffix_sim["route_cost"]), 6),
        )

    def _generate_onboard_suffix_candidates(
        self,
        suffix_unit_ids: tuple[int, ...],
        rng: random.Random,
    ) -> list[tuple[int, ...]]:
        if len(suffix_unit_ids) <= 1:
            return [tuple(suffix_unit_ids)]
        if len(suffix_unit_ids) <= 5:
            ordered_candidates = sorted(set(permutations(suffix_unit_ids)))
            return [tuple(candidate) for candidate in ordered_candidates]
        current = list(suffix_unit_ids)
        candidates = {tuple(current)}
        for _ in range(LOCAL_ONBOARD_NEIGHBOR_TRIALS):
            neighbor = list(current)
            if rng.random() < 0.5:
                left_idx, right_idx = sorted(rng.sample(range(len(neighbor)), 2))
                unit_id = neighbor.pop(left_idx)
                if right_idx > left_idx:
                    right_idx -= 1
                neighbor.insert(right_idx + 1, unit_id)
            else:
                left_idx, right_idx = rng.sample(range(len(neighbor)), 2)
                neighbor[left_idx], neighbor[right_idx] = neighbor[right_idx], neighbor[left_idx]
            candidates.add(tuple(neighbor))
            current = neighbor
        return sorted(candidates)

    def _run_onboard_local_reopt(
        self,
        *,
        event: EventRecord,
        event_time: float,
        plan_by_slot: dict[str, DynamicRoutePlan],
        snapshot: Snapshot,
        affected_onboard_slots: Iterable[str],
    ) -> dict[str, DynamicRoutePlan]:
        updated_plan = self._route_plan_by_slot_copy(plan_by_slot)
        for slot_id in sorted(set(affected_onboard_slots), key=self._slot_sort_key):
            state = snapshot.route_state_by_slot.get(slot_id)
            if state is None or state.status != "onboard" or not state.editable_suffix:
                continue
            route_plan = updated_plan[slot_id]
            current_row = next(
                row for row in state.future_stop_rows
                if int(row["remaining_stop_index"]) == 1
            )
            rng = random.Random(
                (hash((event.event_id, slot_id, tuple(state.editable_suffix))) & 0xFFFFFFFF)
            )
            best_suffix = tuple(state.editable_suffix)
            best_suffix_sim = self._simulate_partial_route(
                vehicle_type=route_plan.vehicle_type,
                unit_ids=best_suffix,
                start_node=int(current_row["service_node_id"]),
                start_time=float(current_row["service_end_min"]),
                route_id=1,
                vehicle_instance=route_plan.vehicle_instance,
                vehicle_slot=slot_id,
                include_startup=False,
                original_stop_offset=int(current_row["original_stop_index"]),
            )
            if not best_suffix_sim["feasible"]:
                continue
            best_rank = self._suffix_candidate_rank(
                event=event,
                current_frozen_row=current_row,
                suffix_sim=best_suffix_sim,
            )
            for candidate_suffix in self._generate_onboard_suffix_candidates(tuple(state.editable_suffix), rng):
                suffix_sim = self._simulate_partial_route(
                    vehicle_type=route_plan.vehicle_type,
                    unit_ids=candidate_suffix,
                    start_node=int(current_row["service_node_id"]),
                    start_time=float(current_row["service_end_min"]),
                    route_id=1,
                    vehicle_instance=route_plan.vehicle_instance,
                    vehicle_slot=slot_id,
                    include_startup=False,
                    original_stop_offset=int(current_row["original_stop_index"]),
                )
                if not suffix_sim["feasible"]:
                    continue
                candidate_rank = self._suffix_candidate_rank(
                    event=event,
                    current_frozen_row=current_row,
                    suffix_sim=suffix_sim,
                )
                if candidate_rank < best_rank:
                    best_rank = candidate_rank
                    best_suffix = tuple(candidate_suffix)
                    best_suffix_sim = suffix_sim
            prefix_with_current = (*state.fixed_prefix, int(state.current_frozen_unit_id))
            updated_plan[slot_id] = replace(
                route_plan,
                unit_ids=(*prefix_with_current, *best_suffix),
                source_tag=f"onboard_{event.event_id.lower()}",
            )
        return updated_plan

    def _local_solver_for_depot(
        self,
        *,
        active_unit_ids: Iterable[int],
        available_counts: dict[str, int],
        earliest_departure_min: float,
    ) -> "Question3DynamicSolver":
        local_solver = object.__new__(Question3DynamicSolver)
        local_solver.__dict__ = dict(self.__dict__)
        local_solver.route_cache = q1.RouteCache()
        local_solver.dynamic_units = dict(self.dynamic_units)
        local_solver.service_units = [self.dynamic_units[unit_id] for unit_id in sorted(set(active_unit_ids))]
        local_solver.unit_by_id = dict(local_solver.dynamic_units)
        local_solver.active_unit_ids = sorted(set(active_unit_ids))
        local_solver.vehicles = [
            replace(vehicle, vehicle_count=int(available_counts.get(vehicle.vehicle_type, 0)))
            for vehicle in self.vehicles
        ]
        local_solver.vehicle_by_name = {
            vehicle.vehicle_type: vehicle
            for vehicle in local_solver.vehicles
        }
        lower_bound = int(math.ceil(earliest_departure_min))
        full_grid = np.arange(q1.DAY_END_MIN + 1, dtype=np.float64)
        full_grid[:lower_bound] = np.nan
        local_solver.route_start_grid = full_grid
        return local_solver

    def _evaluate_route_with_lower_bound(
        self,
        *,
        solver: "Question3DynamicSolver",
        route: q1.TypedRoute,
        lower_bound: float,
    ) -> q1.RouteEvaluation:
        original_grid = solver.route_start_grid
        original_cache = solver.route_cache
        full_grid = np.arange(q1.DAY_END_MIN + 1, dtype=np.float64)
        full_grid[: int(math.ceil(lower_bound))] = np.nan
        solver.route_start_grid = full_grid
        solver.route_cache = q1.RouteCache()
        try:
            return solver.evaluate_route(route)
        finally:
            solver.route_start_grid = original_grid
            solver.route_cache = original_cache

    def _singleton_seed_route(
        self,
        *,
        local_solver: "Question3DynamicSolver",
        unit_id: int,
        available_counts: dict[str, int],
    ) -> q1.TypedRoute | None:
        unit = local_solver.unit_by_id[unit_id]
        best_route: q1.TypedRoute | None = None
        best_cost = float("inf")
        for vehicle_type in unit.eligible_vehicle_types:
            if int(available_counts.get(vehicle_type, 0)) <= 0:
                continue
            candidate_route = q1.TypedRoute(vehicle_type=vehicle_type, unit_ids=(unit_id,))
            candidate_eval = local_solver.evaluate_route(candidate_route)
            if candidate_eval.feasible and float(candidate_eval.best_cost) < best_cost:
                best_cost = float(candidate_eval.best_cost)
                best_route = candidate_route
        if best_route is not None:
            return best_route
        for vehicle_type in unit.eligible_vehicle_types:
            candidate_route = q1.TypedRoute(vehicle_type=vehicle_type, unit_ids=(unit_id,))
            candidate_eval = local_solver.evaluate_route(candidate_route)
            if candidate_eval.feasible and float(candidate_eval.best_cost) < best_cost:
                best_cost = float(candidate_eval.best_cost)
                best_route = candidate_route
        return best_route

    def _assign_local_routes_to_slots(
        self,
        *,
        selected_routes: list[q1.TypedRoute],
        available_slots: Iterable[str],
        previous_plan: dict[str, DynamicRoutePlan],
        route_eval_by_key: dict[tuple[str, tuple[int, ...]], q1.RouteEvaluation],
        source_tag: str,
    ) -> tuple[dict[str, DynamicRoutePlan], set[str]]:
        slot_pool_by_type: dict[str, list[str]] = defaultdict(list)
        for slot_id in available_slots:
            vehicle_type, _ = slot_id.split("#", 1)
            slot_pool_by_type[vehicle_type].append(slot_id)
        for vehicle_type in slot_pool_by_type:
            slot_pool_by_type[vehicle_type].sort(key=self._slot_sort_key)
        assignments: dict[str, DynamicRoutePlan] = {}
        remaining_slots = {
            vehicle_type: list(slot_ids)
            for vehicle_type, slot_ids in slot_pool_by_type.items()
        }
        for route in sorted(selected_routes, key=lambda item: (-len(item.unit_ids), item.vehicle_type, item.unit_ids)):
            candidates = remaining_slots.get(route.vehicle_type, [])
            if not candidates:
                raise RuntimeError(f"No available slot remains for route {route}")
            def score(slot_id: str) -> tuple[int, int, int, int]:
                previous = previous_plan.get(slot_id)
                overlap = 0
                exact = 1
                occupied = 1
                if previous is not None:
                    overlap = len(set(previous.unit_ids) & set(route.unit_ids))
                    exact = 0 if previous.unit_ids == route.unit_ids else 1
                    occupied = 0
                _, instance = slot_id.split("#", 1)
                return (
                    exact,
                    -overlap,
                    occupied,
                    int(instance),
                )
            selected_slot = min(candidates, key=score)
            remaining_slots[route.vehicle_type].remove(selected_slot)
            previous = previous_plan.get(selected_slot)
            vehicle_type, instance = selected_slot.split("#", 1)
            route_eval = route_eval_by_key[(route.vehicle_type, tuple(route.unit_ids))]
            assignments[selected_slot] = DynamicRoutePlan(
                vehicle_slot=selected_slot,
                vehicle_type=vehicle_type,
                vehicle_instance=int(instance),
                unit_ids=tuple(int(unit_id) for unit_id in route.unit_ids),
                planned_departure_min=float(route_eval.best_start),
                baseline_route_id=None if previous is None else previous.baseline_route_id,
                source_tag=source_tag,
            )
        unassigned_slots = {
            slot_id
            for slot_ids in remaining_slots.values()
            for slot_id in slot_ids
        }
        return assignments, unassigned_slots

    def _run_depot_local_reopt(
        self,
        *,
        event: EventRecord,
        event_time: float,
        plan_by_slot: dict[str, DynamicRoutePlan],
        unused_slots: set[str],
        included_slots: set[str],
    ) -> tuple[dict[str, DynamicRoutePlan], set[str], bool]:
        local_unit_ids: set[int] = set(self.orphan_unit_ids)
        current_routes: list[q1.TypedRoute] = []
        for slot_id in sorted(included_slots, key=self._slot_sort_key):
            route_plan = plan_by_slot.get(slot_id)
            if route_plan is None:
                continue
            local_unit_ids.update(int(unit_id) for unit_id in route_plan.unit_ids)
            current_routes.append(
                q1.TypedRoute(
                    vehicle_type=route_plan.vehicle_type,
                    unit_ids=tuple(int(unit_id) for unit_id in route_plan.unit_ids),
                )
            )
        available_slot_ids = set(included_slots) | set(unused_slots)
        available_counts = Counter[str]()
        for slot_id in available_slot_ids:
            vehicle_type, _ = slot_id.split("#", 1)
            available_counts[vehicle_type] += 1
        if not local_unit_ids:
            updated_plan = {
                slot_id: route_plan
                for slot_id, route_plan in plan_by_slot.items()
                if slot_id not in included_slots
            }
            updated_unused = (set(unused_slots) | set(included_slots))
            return updated_plan, updated_unused, True

        local_solver = self._local_solver_for_depot(
            active_unit_ids=local_unit_ids,
            available_counts=dict(available_counts),
            earliest_departure_min=event_time,
        )
        seed_routes = list(current_routes)
        covered_unit_ids = {
            int(unit_id)
            for route in current_routes
            for unit_id in route.unit_ids
        }
        missing_unit_ids = sorted(local_unit_ids - covered_unit_ids)
        for unit_id in missing_unit_ids:
            singleton_route = self._singleton_seed_route(
                local_solver=local_solver,
                unit_id=unit_id,
                available_counts=dict(available_counts),
            )
            if singleton_route is not None:
                seed_routes.append(singleton_route)

        selected_routes: list[q1.TypedRoute] | None = None
        original_time_limit = q1.MILP_PHASE_TIME_LIMIT_SEC
        q1.MILP_PHASE_TIME_LIMIT_SEC = LOCAL_MILP_TIME_LIMIT_SEC
        try:
            route_pool_columns = local_solver._build_route_pool_columns(seed_routes=seed_routes)
            pass_result = local_solver._solve_global_milp_pass(
                route_pool_columns=route_pool_columns,
                pass_label=f"q3_{event.event_id.lower()}",
            )
            if pass_result["selected_routes"] is not None:
                selected_routes = [
                    local_solver._two_opt_route(route)
                    for route in pass_result["selected_routes"]
                ]
                if local_solver.evaluate_solution(selected_routes) is None:
                    selected_routes = None
        finally:
            q1.MILP_PHASE_TIME_LIMIT_SEC = original_time_limit

        if selected_routes is None:
            greedy_routes = local_solver.repair_solution([], sorted(local_unit_ids, key=local_solver._unit_priority_key))
            if greedy_routes is not None and local_solver.evaluate_solution(greedy_routes) is not None:
                selected_routes = [local_solver._two_opt_route(route) for route in greedy_routes]
        if selected_routes is None:
            return plan_by_slot, set(unused_slots), False

        unit_departure_lb: dict[int, float] = {
            int(unit_id): float(event_time)
            for unit_id in local_unit_ids
        }
        for slot_id in included_slots:
            route_plan = plan_by_slot.get(slot_id)
            if route_plan is None:
                continue
            for unit_id in route_plan.unit_ids:
                unit_departure_lb[int(unit_id)] = float(route_plan.planned_departure_min)
        route_eval_by_key = {
            (route.vehicle_type, tuple(route.unit_ids)): self._evaluate_route_with_lower_bound(
                solver=local_solver,
                route=route,
                lower_bound=min(unit_departure_lb[int(unit_id)] for unit_id in route.unit_ids),
            )
            for route in selected_routes
        }
        if any(not route_eval.feasible for route_eval in route_eval_by_key.values()):
            return plan_by_slot, set(unused_slots), False
        previous_plan = {
            slot_id: route_plan
            for slot_id, route_plan in plan_by_slot.items()
            if slot_id in available_slot_ids
        }
        assigned_routes, leftover_slots = self._assign_local_routes_to_slots(
            selected_routes=selected_routes,
            available_slots=available_slot_ids,
            previous_plan=previous_plan,
            route_eval_by_key=route_eval_by_key,
            source_tag=f"depot_{event.event_id.lower()}",
        )
        updated_plan = {
            slot_id: route_plan
            for slot_id, route_plan in plan_by_slot.items()
            if slot_id not in included_slots
        }
        updated_plan.update(assigned_routes)
        updated_unused = (set(unused_slots) | set(included_slots)) - set(assigned_routes)
        updated_unused |= leftover_slots
        return updated_plan, updated_unused, True

    def _apply_event_modifications(
        self,
        *,
        event: EventRecord,
        snapshot_before: Snapshot,
        plan_by_slot: dict[str, DynamicRoutePlan],
    ) -> tuple[dict[str, DynamicRoutePlan], set[int], set[int], set[int], str]:
        affected_onboard_slots: set[str] = set()
        affected_depot_slots: set[str] = set()
        blocked_current_target_units: set[int] = set()
        failure_reason = ""
        if event.event_type == "new_order":
            new_unit_id = self.next_unit_id
            self.next_unit_id += 1
            self.dynamic_units[new_unit_id] = self._make_dynamic_unit(
                unit_id=new_unit_id,
                business_customer_id=event.target_customer,
                service_node_id=int(event.node_to),
                weight=float(event.weight),
                volume=float(event.volume),
                tw_start_min=int(event.new_tw_start),
                tw_end_min=int(event.new_tw_end),
                source_order_ids=(-100000 - new_unit_id,),
                unit_type="normal",
                visit_index=1,
                required_visit_count=1,
                baseline_customer_id=event.target_customer,
                latest_event_id=event.event_id,
            )
            self.orphan_unit_ids.add(new_unit_id)
            self._refresh_dynamic_collections()
            return plan_by_slot, affected_onboard_slots, affected_depot_slots, {new_unit_id}, failure_reason

        target_unit_ids = self._future_target_unit_ids(snapshot_before, event.target_customer)
        if event.event_type == "cancel":
            depot_unit_ids = [
                unit_id
                for unit_id in target_unit_ids
                if snapshot_before.route_state_by_slot[snapshot_before.unit_to_slot[unit_id]].status == "depot_pending"
            ]
            direct_slots = {
                snapshot_before.unit_to_slot[unit_id]
                for unit_id in depot_unit_ids
            }
            self._apply_cancel_to_plan(
                target_unit_ids=depot_unit_ids,
                plan_by_slot=plan_by_slot,
                snapshot_before=snapshot_before,
            )
            self.cancelled_unit_ids.update(int(unit_id) for unit_id in depot_unit_ids)
            self.orphan_unit_ids.difference_update(int(unit_id) for unit_id in depot_unit_ids)
            self._refresh_dynamic_collections()
            return plan_by_slot, affected_onboard_slots, set(direct_slots), set(depot_unit_ids), failure_reason

        relevant_unit_ids: set[int] = set()
        for unit_id in target_unit_ids:
            slot_id = snapshot_before.unit_to_slot[unit_id]
            state = snapshot_before.route_state_by_slot[slot_id]
            if state.current_frozen_unit_id == unit_id and state.status == "onboard":
                blocked_current_target_units.add(int(unit_id))
                continue
            if state.status == "onboard":
                affected_onboard_slots.add(slot_id)
            else:
                affected_depot_slots.add(slot_id)
            relevant_unit_ids.add(int(unit_id))
        if blocked_current_target_units:
            failure_reason = "some affected units are already the frozen current target and cannot be changed mid-arc"
        for unit_id in relevant_unit_ids:
            if event.event_type == "tighten_tw":
                self._replace_unit(
                    unit_id,
                    tw_start_min=int(event.new_tw_start),
                    tw_end_min=int(event.new_tw_end),
                    latest_event_id=event.event_id,
                )
            elif event.event_type == "relocate":
                self._replace_unit(
                    unit_id,
                    service_node_id=int(event.node_to),
                    tw_start_min=int(event.new_tw_start),
                    tw_end_min=int(event.new_tw_end),
                    latest_event_id=event.event_id,
                )
        self._refresh_dynamic_collections()
        return plan_by_slot, affected_onboard_slots, affected_depot_slots, relevant_unit_ids, failure_reason

    def _process_event(self, event: EventRecord, cumulative_unmet_events: int) -> tuple[Snapshot, bool]:
        snapshot_before = self._build_snapshot(event_time=event.event_time)
        before_plan = self._route_plan_by_slot_copy(self.plan_by_slot)
        mod_plan = self._route_plan_by_slot_copy(self.plan_by_slot)
        (
            mod_plan,
            affected_onboard_slots,
            affected_depot_slots,
            tracked_unit_ids,
            modification_failure_reason,
        ) = self._apply_event_modifications(
            event=event,
            snapshot_before=snapshot_before,
            plan_by_slot=mod_plan,
        )
        t0_min = self._adaptive_t0(
            event=event,
            event_time=event.event_time,
            direct_slots=set(affected_onboard_slots) | set(affected_depot_slots),
            snapshot_before=snapshot_before,
            modified_plan_by_slot=mod_plan,
        )
        onboard_optimized_plan = self._run_onboard_local_reopt(
            event=event,
            event_time=event.event_time,
            plan_by_slot=mod_plan,
            snapshot=self._build_snapshot(event_time=event.event_time, plan_by_slot=mod_plan),
            affected_onboard_slots=affected_onboard_slots,
        )
        snapshot_after_onboard = self._build_snapshot(
            event_time=event.event_time,
            plan_by_slot=onboard_optimized_plan,
        )
        unlaunched_slot_records = []
        for state in snapshot_after_onboard.route_states:
            if state.status != "depot_pending":
                continue
            first_service_start = (
                float(state.future_stop_rows[0]["service_start_min"])
                if state.future_stop_rows
                else float("inf")
            )
            unlaunched_slot_records.append((state.vehicle_slot, first_service_start))
        unlaunched_slot_records.sort(key=lambda item: (item[1], self._slot_sort_key(item[0])))
        direct_depot_slot_set = set(affected_depot_slots)
        included_slots = {
            slot_id
            for slot_id, first_service_start in unlaunched_slot_records
            if first_service_start <= t0_min + EPS
        } | direct_depot_slot_set
        expansion_candidates = [
            slot_id
            for slot_id, _ in unlaunched_slot_records
            if slot_id not in included_slots
        ]

        best_snapshot: Snapshot | None = None
        best_plan: dict[str, DynamicRoutePlan] | None = None
        best_unused_slots: set[str] | None = None
        best_feasibility: dict[str, object] | None = None
        best_rank: tuple[int, int, float, float] | None = None
        expansion_rounds = 0
        success = False
        final_failure_reason = modification_failure_reason
        base_unused_slots = set(self.unused_slots)
        while True:
            expansion_rounds += 1
            candidate_plan, candidate_unused_slots, local_ok = self._run_depot_local_reopt(
                event=event,
                event_time=event.event_time,
                plan_by_slot=onboard_optimized_plan,
                unused_slots=base_unused_slots,
                included_slots=set(included_slots),
            )
            snapshot_candidate, feasibility_candidate, rank_key = self._evaluate_plan_for_event(
                event=event,
                event_time=event.event_time,
                plan_by_slot=candidate_plan,
            )
            if (
                best_rank is None
                or rank_key < best_rank
            ):
                best_rank = rank_key
                best_snapshot = snapshot_candidate
                best_plan = candidate_plan
                best_unused_slots = candidate_unused_slots
                best_feasibility = feasibility_candidate
                final_failure_reason = (
                    modification_failure_reason
                    or feasibility_candidate["failure_reason"]
                    or ("" if local_ok else "local depot reoptimization failed")
                )
            if (
                local_ok
                and snapshot_candidate.policy_diagnostics["policy_violation_count"] == 0
                and snapshot_candidate.one_route_one_vehicle_ok == 1
                and snapshot_candidate.no_mid_arc_change_ok == 1
                and snapshot_candidate.no_restock_ok == 1
                and bool(feasibility_candidate["satisfied"])
            ):
                success = True
                best_snapshot = snapshot_candidate
                best_plan = candidate_plan
                best_unused_slots = candidate_unused_slots
                best_feasibility = feasibility_candidate
                break
            if not expansion_candidates:
                break
            next_batch = expansion_candidates[:LOCAL_CONE_EXPANSION_BATCH]
            expansion_candidates = expansion_candidates[LOCAL_CONE_EXPANSION_BATCH:]
            included_slots.update(next_batch)

        if best_snapshot is None or best_plan is None or best_unused_slots is None or best_feasibility is None:
            raise RuntimeError(f"Unable to produce any candidate plan for {event.event_id}")

        self.plan_by_slot = best_plan
        self.unused_slots = set(best_unused_slots)
        modified_vehicle_count = self._modified_vehicle_count(before_plan, self.plan_by_slot)
        switched_depot_unit_count = self._switched_depot_unit_count(
            before_snapshot=snapshot_before,
            after_snapshot=best_snapshot,
            tracked_unit_ids=tracked_unit_ids,
        )
        unmet_event_count = cumulative_unmet_events + int(not success)
        self.event_log_rows.append(
            {
                "event_id": event.event_id,
                "event_time_min": event.event_time,
                "event_time_hhmm": self._minutes_to_hhmm_safe(event.event_time),
                "event_type": event.event_type,
                "target_customer": event.target_customer,
                "node_from": event.node_from,
                "node_to": event.node_to,
                "new_tw_start": event.new_tw_start,
                "new_tw_end": event.new_tw_end,
                "weight_kg": event.weight,
                "volume_m3": event.volume,
                "direct_onboard_route_count": len(affected_onboard_slots),
                "direct_depot_route_count": len(affected_depot_slots),
                "initial_t0_min": round(t0_min, 3),
                "initial_t0_hhmm": self._minutes_to_hhmm_safe(t0_min),
                "expansion_round_count": expansion_rounds,
                "included_unlaunched_route_count": len(included_slots),
                "status": "satisfied" if success else "partial_failure",
                "failure_reason": "" if success else final_failure_reason,
            }
        )
        self.event_metric_rows.append(
            {
                "event_id": event.event_id,
                "event_time_min": event.event_time,
                "event_time_hhmm": self._minutes_to_hhmm_safe(event.event_time),
                "t0_min": round(t0_min, 3),
                "t0_hhmm": self._minutes_to_hhmm_safe(t0_min),
                "realized_cost_to_event": round(best_snapshot.realized_total_cost, 6),
                "remaining_total_cost": round(best_snapshot.remaining_total_cost, 6),
                "projected_full_day_cost": round(best_snapshot.projected_full_day_cost, 6),
                "cost_change_vs_q2_baseline": round(
                    best_snapshot.projected_full_day_cost - self.q2_baseline_total_cost,
                    6,
                ),
                "remaining_total_late_min": round(best_snapshot.remaining_total_late_min, 6),
                "projected_full_day_route_count": int(best_snapshot.projected_full_day_route_count),
                "route_count_change_vs_q2_baseline": int(
                    best_snapshot.projected_full_day_route_count - self.q2_baseline_route_count
                ),
                "unmet_event_count": int(unmet_event_count),
                "modified_vehicle_count": int(modified_vehicle_count),
                "switched_depot_unit_count": int(switched_depot_unit_count),
                "policy_violation_count": int(best_snapshot.policy_diagnostics["policy_violation_count"]),
                "mandatory_ev_served_by_non_ev_count": int(
                    best_snapshot.policy_diagnostics["mandatory_ev_served_by_non_ev_count"]
                ),
                "fuel_route_green_zone_pre16_visit_count": int(
                    best_snapshot.policy_diagnostics["fuel_route_green_zone_pre16_visit_count"]
                ),
                "one_route_one_vehicle_ok": int(best_snapshot.one_route_one_vehicle_ok),
                "no_mid_arc_change_ok": int(best_snapshot.no_mid_arc_change_ok),
                "no_restock_ok": int(best_snapshot.no_restock_ok),
            }
        )
        return best_snapshot, success

    def _service_units_output_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for unit_id in sorted(
            unit_id
            for unit_id in self.dynamic_units
            if unit_id not in self.cancelled_unit_ids
        ):
            unit = self.dynamic_units[unit_id]
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "business_customer_id": unit.business_customer_id,
                    "service_node_id": unit.service_node_id,
                    "baseline_customer_id": unit.baseline_customer_id,
                    "unit_type": unit.unit_type,
                    "visit_index": unit.visit_index,
                    "required_visit_count": unit.required_visit_count,
                    "weight_kg": round(unit.weight, 6),
                    "volume_m3": round(unit.volume, 6),
                    "tw_start_min": unit.tw_start_min,
                    "tw_start_hhmm": self._minutes_to_hhmm_safe(unit.tw_start_min),
                    "tw_end_min": unit.tw_end_min,
                    "tw_end_hhmm": self._minutes_to_hhmm_safe(unit.tw_end_min),
                    "eligible_vehicle_types": ",".join(unit.eligible_vehicle_types),
                    "source_order_ids": ",".join(str(order_id) for order_id in unit.source_order_ids),
                    "in_green_zone": int(unit.in_green_zone),
                    "must_use_ev_under_policy": int(unit.must_use_ev_under_policy),
                    "fuel_allowed_after_16": int(unit.fuel_allowed_after_16),
                    "latest_event_id": unit.latest_event_id or "",
                }
            )
        return rows

    def _route_summary_rows(self, snapshot: Snapshot) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        sorted_states = snapshot.route_states
        route_id_by_slot = {
            state.vehicle_slot: route_index
            for route_index, state in enumerate(sorted_states, start=1)
        }
        for state in sorted_states:
            route_id = route_id_by_slot[state.vehicle_slot]
            remaining_business_sequence = "->".join(
                str(self.unit_by_id[unit_id].business_customer_id)
                for unit_id in state.remaining_unit_ids
            )
            remaining_service_sequence = "->".join(
                str(self.unit_by_id[unit_id].service_node_id)
                for unit_id in state.remaining_unit_ids
            )
            first_remaining_service_min = (
                float(state.future_stop_rows[0]["service_start_min"])
                if state.future_stop_rows
                else None
            )
            rows.append(
                {
                    "route_id": route_id,
                    "vehicle_slot": state.vehicle_slot,
                    "vehicle_type": state.vehicle_type,
                    "power_type": state.power_type,
                    "vehicle_instance": state.vehicle_instance,
                    "baseline_route_id": state.baseline_route_id,
                    "status_at_event": state.status,
                    "planned_departure_min": round(state.planned_departure_min, 3),
                    "planned_departure_hhmm": self._minutes_to_hhmm_safe(state.planned_departure_min),
                    "release_node": state.release_node,
                    "release_time_min": round(state.release_time, 3),
                    "release_time_hhmm": self._minutes_to_hhmm_safe(state.release_time),
                    "fixed_prefix_unit_sequence": ",".join(str(unit_id) for unit_id in state.fixed_prefix),
                    "current_frozen_unit_id": state.current_frozen_unit_id,
                    "editable_suffix_unit_sequence": ",".join(str(unit_id) for unit_id in state.editable_suffix),
                    "remaining_unit_sequence": ",".join(str(unit_id) for unit_id in state.remaining_unit_ids),
                    "remaining_business_customer_sequence": remaining_business_sequence,
                    "remaining_service_node_sequence": remaining_service_sequence,
                    "remaining_stop_count": len(state.future_stop_rows),
                    "first_remaining_service_start_min": None if first_remaining_service_min is None else round(first_remaining_service_min, 3),
                    "first_remaining_service_start_hhmm": "" if first_remaining_service_min is None else self._minutes_to_hhmm_safe(first_remaining_service_min),
                    "planned_return_min": round(state.full_return_min, 3),
                    "planned_return_hhmm": self._minutes_to_hhmm_safe(state.full_return_min),
                    "remaining_route_cost": round(state.remaining_total_cost, 6),
                    "remaining_total_late_min": round(state.remaining_total_late_min, 6),
                    "current_arc_frozen": state.current_arc_frozen,
                }
            )
        return rows

    def _vehicle_state_rows(self, snapshot: Snapshot) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for state in snapshot.route_states:
            rows.append(
                {
                    "event_time_min": round(snapshot.event_time, 3),
                    "event_time_hhmm": self._minutes_to_hhmm_safe(snapshot.event_time),
                    "vehicle_slot": state.vehicle_slot,
                    "vehicle_type": state.vehicle_type,
                    "vehicle_instance": state.vehicle_instance,
                    "power_type": state.power_type,
                    "baseline_route_id": state.baseline_route_id,
                    "status": state.status,
                    "planned_departure_min": round(state.planned_departure_min, 3),
                    "planned_departure_hhmm": self._minutes_to_hhmm_safe(state.planned_departure_min),
                    "release_node": state.release_node,
                    "release_time_min": round(state.release_time, 3),
                    "release_time_hhmm": self._minutes_to_hhmm_safe(state.release_time),
                    "fixed_prefix_unit_sequence": ",".join(str(unit_id) for unit_id in state.fixed_prefix),
                    "current_frozen_unit_id": state.current_frozen_unit_id,
                    "editable_suffix_unit_sequence": ",".join(str(unit_id) for unit_id in state.editable_suffix),
                    "remaining_unit_sequence": ",".join(str(unit_id) for unit_id in state.remaining_unit_ids),
                    "remaining_cost": round(state.remaining_total_cost, 6),
                    "remaining_total_late_min": round(state.remaining_total_late_min, 6),
                    "current_arc_frozen": state.current_arc_frozen,
                }
            )
        return rows

    def _write_outputs(
        self,
        *,
        final_snapshot: Snapshot,
        cumulative_unmet_events: int,
    ) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.event_log_rows).to_csv(
            self.output_root / "q3_event_log.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
        pd.DataFrame(self.event_metric_rows).to_csv(
            self.output_root / "q3_event_metrics.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
        pd.DataFrame(self._route_summary_rows(final_snapshot)).to_csv(
            self.output_root / "q3_dynamic_route_summary.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
        pd.DataFrame(final_snapshot.remaining_stop_rows).to_csv(
            self.output_root / "q3_dynamic_stop_schedule.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
        pd.DataFrame(self._vehicle_state_rows(final_snapshot)).to_csv(
            self.output_root / "q3_dynamic_vehicle_state.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
        pd.DataFrame(self._service_units_output_rows()).to_csv(
            self.output_root / "q3_dynamic_service_units.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
        summary_payload = {
            "baseline_root": str(self.baseline_root),
            "seed": self.dynamic_seed,
            "q2_baseline_total_cost": self.q2_baseline_total_cost,
            "q2_baseline_route_count": self.q2_baseline_route_count,
            "event_count": len(self.events),
            "processed_event_ids": [event.event_id for event in self.events],
            "final_event_time_min": round(final_snapshot.event_time, 3),
            "final_event_time_hhmm": self._minutes_to_hhmm_safe(final_snapshot.event_time),
            "realized_cost_to_event": final_snapshot.realized_total_cost,
            "remaining_total_cost": final_snapshot.remaining_total_cost,
            "projected_full_day_cost": final_snapshot.projected_full_day_cost,
            "cost_change_vs_q2_baseline": final_snapshot.projected_full_day_cost - self.q2_baseline_total_cost,
            "remaining_total_late_min": final_snapshot.remaining_total_late_min,
            "projected_full_day_route_count": final_snapshot.projected_full_day_route_count,
            "route_count_change_vs_q2_baseline": final_snapshot.projected_full_day_route_count - self.q2_baseline_route_count,
            "unmet_event_count": int(cumulative_unmet_events),
            "policy_violation_count": int(final_snapshot.policy_diagnostics["policy_violation_count"]),
            "mandatory_ev_served_by_non_ev_count": int(
                final_snapshot.policy_diagnostics["mandatory_ev_served_by_non_ev_count"]
            ),
            "fuel_route_green_zone_pre16_visit_count": int(
                final_snapshot.policy_diagnostics["fuel_route_green_zone_pre16_visit_count"]
            ),
            "one_route_one_vehicle_ok": int(final_snapshot.one_route_one_vehicle_ok),
            "no_mid_arc_change_ok": int(final_snapshot.no_mid_arc_change_ok),
            "no_restock_ok": int(final_snapshot.no_restock_ok),
            "active_remaining_route_count": len(final_snapshot.route_states),
            "unused_depot_slots_by_type": {
                vehicle_type: sum(
                    1 for slot_id in self.unused_slots
                    if slot_id.startswith(f"{vehicle_type}#")
                )
                for vehicle_type in sorted(self.vehicle_by_name)
            },
        }
        (self.output_root / "q3_dynamic_summary.json").write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def run(self) -> dict[str, object]:
        cumulative_unmet_events = 0
        latest_snapshot = self._build_snapshot(event_time=0.0)
        for event in self.events:
            latest_snapshot, success = self._process_event(event, cumulative_unmet_events)
            cumulative_unmet_events += int(not success)
        self._write_outputs(
            final_snapshot=latest_snapshot,
            cumulative_unmet_events=cumulative_unmet_events,
        )
        return {
            "realized_cost_to_event": latest_snapshot.realized_total_cost,
            "remaining_total_cost": latest_snapshot.remaining_total_cost,
            "projected_full_day_cost": latest_snapshot.projected_full_day_cost,
            "cost_change_vs_q2_baseline": latest_snapshot.projected_full_day_cost - self.q2_baseline_total_cost,
            "remaining_total_late_min": latest_snapshot.remaining_total_late_min,
            "projected_full_day_route_count": latest_snapshot.projected_full_day_route_count,
            "route_count_change_vs_q2_baseline": latest_snapshot.projected_full_day_route_count - self.q2_baseline_route_count,
            "unmet_event_count": cumulative_unmet_events,
            "policy_violation_count": int(latest_snapshot.policy_diagnostics["policy_violation_count"]),
            "mandatory_ev_served_by_non_ev_count": int(
                latest_snapshot.policy_diagnostics["mandatory_ev_served_by_non_ev_count"]
            ),
            "fuel_route_green_zone_pre16_visit_count": int(
                latest_snapshot.policy_diagnostics["fuel_route_green_zone_pre16_visit_count"]
            ),
            "active_remaining_route_count": len(latest_snapshot.route_states),
        }


def parse_args() -> argparse.Namespace:
    workspace = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Question 3 dynamic rolling-horizon solver based on Q2 s11."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=workspace,
        help="Project workspace root.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=workspace / "preprocess_artifacts",
        help="Preprocess artifact root.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=workspace / "question2_artifacts_hybrid_standard_s11",
        help="Baseline Q2 s11 artifact root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workspace / "question3_artifacts_dynamic_s11",
        help="Q3 output root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Seed label for reporting.",
    )
    parser.add_argument(
        "--top-route-candidates",
        type=int,
        default=4,
        help="Top route candidates for local greedy repair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solver = Question3DynamicSolver(
        workspace=args.workspace,
        input_root=args.input_root,
        baseline_root=args.baseline_root,
        output_root=args.output_root,
        seed=args.seed,
        top_route_candidates=args.top_route_candidates,
    )
    result = solver.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
