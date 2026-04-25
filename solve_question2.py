from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import solve_question1 as q1


POLICY_GREEN_ZONE_BASIS = "attachment_geometry_radius_10km"


@dataclass(frozen=True)
class PolicyServiceUnit(q1.ServiceUnit):
    in_green_zone: bool
    must_use_ev_under_policy: bool
    fuel_allowed_after_16: bool
    ev_allowed_flag: bool
    fuel_allowed_flag: bool
    ev_tw_start_min: int
    ev_tw_end_min: int
    fuel_tw_start_min: int
    fuel_tw_end_min: int


class Question2Solver(q1.Question1Solver):
    def __init__(
        self,
        workspace: Path,
        input_root: Path,
        output_root: Path,
        seed_list: list[int],
        max_generations: int,
        particle_count: int,
        top_route_candidates: int,
    ) -> None:
        self._policy_customer_df_cache: pd.DataFrame | None = None
        self._policy_customer_lookup_cache: dict[int, dict[str, object]] | None = None
        super().__init__(
            workspace=workspace,
            input_root=input_root,
            output_root=output_root,
            seed_list=seed_list,
            max_generations=max_generations,
            particle_count=particle_count,
            top_route_candidates=top_route_candidates,
            enable_split_packing_sensitivity=False,
        )
        config = json.loads((self.input_root / "preprocess_config.json").read_text(encoding="utf-8"))
        self.policy_ban_start_min = int(config.get("policy_ban_start_min", 0))
        self.policy_ban_end_min = int(config.get("policy_ban_end_min", 480))
        self.policy_feasibility_df = pd.read_csv(self.input_root / "tables" / "policy_feasibility.csv")
        self.ev_policy_summary_df = pd.read_csv(self.input_root / "tables" / "ev_policy_summary.csv")

    def _policy_customer_df(self) -> pd.DataFrame:
        if self._policy_customer_df_cache is None:
            policy_fields = [
                "cust_id",
                "in_green_zone",
                "must_use_ev_under_policy",
                "fuel_allowed_after_16",
                "fuel_partial_overlap_ban",
                "fuel_service_window_start_min",
                "fuel_service_window_end_min",
                "tw_start_min",
                "tw_end_min",
            ]
            self._policy_customer_df_cache = self.customer_master[policy_fields].copy()
        return self._policy_customer_df_cache

    def _policy_customer_lookup(self) -> dict[int, dict[str, object]]:
        if self._policy_customer_lookup_cache is None:
            df = self._policy_customer_df()
            lookup: dict[int, dict[str, object]] = {}
            for row in df.itertuples(index=False):
                lookup[int(row.cust_id)] = {
                    "in_green_zone": bool(row.in_green_zone),
                    "must_use_ev_under_policy": bool(row.must_use_ev_under_policy),
                    "fuel_allowed_after_16": bool(row.fuel_allowed_after_16),
                    "fuel_partial_overlap_ban": bool(row.fuel_partial_overlap_ban),
                    "tw_start_min": int(row.tw_start_min),
                    "tw_end_min": int(row.tw_end_min),
                    "fuel_service_window_start_min": (
                        int(round(float(row.fuel_service_window_start_min)))
                        if pd.notna(row.fuel_service_window_start_min)
                        else -1
                    ),
                    "fuel_service_window_end_min": (
                        int(round(float(row.fuel_service_window_end_min)))
                        if pd.notna(row.fuel_service_window_end_min)
                        else -1
                    ),
                }
            self._policy_customer_lookup_cache = lookup
        return self._policy_customer_lookup_cache

    def _policy_for_customer(self, cust_id: int) -> dict[str, object]:
        return self._policy_customer_lookup()[int(cust_id)]

    def _vehicle_types_after_policy(
        self,
        eligible_vehicle_types: Iterable[str],
        must_use_ev_under_policy: bool,
    ) -> tuple[str, ...]:
        if not must_use_ev_under_policy:
            return tuple(eligible_vehicle_types)
        return tuple(
            vehicle_type
            for vehicle_type in eligible_vehicle_types
            if self.vehicle_by_name[vehicle_type].power_type == "ev"
        )

    def _eligible_vehicle_types_for_load_after_policy(
        self,
        weight: float,
        volume: float,
        policy: dict[str, object],
    ) -> tuple[str, ...]:
        base_eligible = self._eligible_vehicle_types_for_load(weight, volume)
        return self._vehicle_types_after_policy(base_eligible, bool(policy["must_use_ev_under_policy"]))

    @staticmethod
    def _is_ev3000_only_policy_unit(eligible_vehicle_types: tuple[str, ...]) -> bool:
        return eligible_vehicle_types == ("ev_3000",)

    def _policy_pack_score(
        self,
        bins: list[dict[str, object]],
        policy: dict[str, object],
    ) -> tuple[float, float, float]:
        non_empty = [bin_state for bin_state in bins if bin_state["fragments"]]
        eligible_sum = sum(len(bin_state["eligible_vehicle_types"]) for bin_state in non_empty)
        slack = sum(
            (q1.MAX_SINGLE_WEIGHT - float(bin_state["weight"])) / q1.MAX_SINGLE_WEIGHT
            + (q1.MAX_SINGLE_VOLUME - float(bin_state["volume"])) / q1.MAX_SINGLE_VOLUME
            for bin_state in non_empty
        )
        ev3000_only_count = sum(
            1
            for bin_state in non_empty
            if bool(policy["must_use_ev_under_policy"])
            and self._is_ev3000_only_policy_unit(tuple(bin_state["eligible_vehicle_types"]))
        )
        return float(ev3000_only_count), -float(eligible_sum), float(slack)

    def _policy_greedy_pack_items(
        self,
        cust_id: int,
        fragments: list[q1.OrderFragment],
        bin_count: int,
        rng: random.Random,
        policy: dict[str, object],
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
                max(fragment.weight / q1.MAX_SINGLE_WEIGHT, fragment.volume / q1.MAX_SINGLE_VOLUME),
                fragment.weight,
                fragment.volume,
                rng.random(),
            ),
            reverse=True,
        )
        for fragment in order:
            options: list[tuple[tuple[bool, float, bool, float, float], int, tuple[str, ...]]] = []
            for idx, bin_state in enumerate(bins):
                new_weight = float(bin_state["weight"]) + fragment.weight
                new_volume = float(bin_state["volume"]) + fragment.volume
                eligible = self._eligible_vehicle_types_for_load_after_policy(new_weight, new_volume, policy)
                if not eligible:
                    continue
                slack_weight = (q1.MAX_SINGLE_WEIGHT - new_weight) / q1.MAX_SINGLE_WEIGHT
                slack_volume = (q1.MAX_SINGLE_VOLUME - new_volume) / q1.MAX_SINGLE_VOLUME
                ev3000_only = self._is_ev3000_only_policy_unit(eligible)
                score = (
                    ev3000_only,
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

    def _policy_pack_customer_fragments_for_count(
        self,
        cust_id: int,
        fragments: list[q1.OrderFragment],
        bin_count: int,
        policy: dict[str, object],
    ) -> list[dict[str, object]] | None:
        best_bins: list[dict[str, object]] | None = None
        best_score: tuple[float, float, float] | None = None
        for attempt in range(q1.PACKING_ATTEMPTS):
            rng = random.Random(cust_id * 1000 + bin_count * 100 + attempt)
            candidate = self._policy_greedy_pack_items(cust_id, fragments, bin_count, rng, policy)
            if candidate is None:
                continue
            score = self._policy_pack_score(candidate, policy)
            if best_bins is None or score < best_score:
                best_bins = candidate
                best_score = score
        return best_bins

    @staticmethod
    def _policy_vehicle_windows(
        policy: dict[str, object],
        tw_start_min: int,
        tw_end_min: int,
        eligible_vehicle_types: tuple[str, ...],
        vehicle_by_name: dict[str, q1.VehicleType],
    ) -> tuple[bool, bool, int, int, int, int]:
        ev_allowed_flag = any(vehicle_by_name[vehicle_type].power_type == "ev" for vehicle_type in eligible_vehicle_types)
        fuel_candidate_flag = any(
            vehicle_by_name[vehicle_type].power_type == "fuel" for vehicle_type in eligible_vehicle_types
        )
        if bool(policy["must_use_ev_under_policy"]):
            fuel_allowed_flag = False
            fuel_tw_start_min = -1
            fuel_tw_end_min = -1
        elif bool(policy["fuel_allowed_after_16"]):
            fuel_allowed_flag = fuel_candidate_flag
            fuel_tw_start_min = int(policy["fuel_service_window_start_min"])
            fuel_tw_end_min = int(policy["fuel_service_window_end_min"])
        else:
            fuel_allowed_flag = fuel_candidate_flag
            fuel_tw_start_min = int(tw_start_min)
            fuel_tw_end_min = int(tw_end_min)
        return (
            ev_allowed_flag,
            fuel_allowed_flag,
            int(tw_start_min),
            int(tw_end_min),
            fuel_tw_start_min,
            fuel_tw_end_min,
        )

    def _build_service_units(self) -> tuple[list[PolicyServiceUnit], list[dict[str, object]]]:
        policy_lookup = self._policy_customer_lookup()
        policy_units: list[PolicyServiceUnit] = []
        split_rows: list[dict[str, object]] = []
        unit_id = 0
        mandatory_customers: list[dict[str, object]] = []
        normal_heavy_big_only_count = 0
        normal_ev3000_only_count = 0
        for row in self.active_customer_df.itertuples(index=False):
            cust_id = int(row.cust_id)
            policy = policy_lookup[cust_id]
            total_weight = float(row.total_weight)
            total_volume = float(row.total_volume)
            eligible_vehicle_types = self._eligible_vehicle_types_for_load_after_policy(total_weight, total_volume, policy)
            customer_orders = self.orders.loc[self.orders["cust_id"] == cust_id].copy()
            source_order_ids = tuple(sorted(customer_orders["order_id"].astype(int).tolist()))
            if eligible_vehicle_types:
                (
                    ev_allowed_flag,
                    fuel_allowed_flag,
                    ev_tw_start_min,
                    ev_tw_end_min,
                    fuel_tw_start_min,
                    fuel_tw_end_min,
                ) = self._policy_vehicle_windows(
                    policy=policy,
                    tw_start_min=int(row.tw_start_min),
                    tw_end_min=int(row.tw_end_min),
                    eligible_vehicle_types=eligible_vehicle_types,
                    vehicle_by_name=self.vehicle_by_name,
                )
                policy_units.append(
                    PolicyServiceUnit(
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
                        in_green_zone=bool(policy["in_green_zone"]),
                        must_use_ev_under_policy=bool(policy["must_use_ev_under_policy"]),
                        fuel_allowed_after_16=bool(policy["fuel_allowed_after_16"]),
                        ev_allowed_flag=ev_allowed_flag,
                        fuel_allowed_flag=fuel_allowed_flag,
                        ev_tw_start_min=ev_tw_start_min,
                        ev_tw_end_min=ev_tw_end_min,
                        fuel_tw_start_min=fuel_tw_start_min,
                        fuel_tw_end_min=fuel_tw_end_min,
                    )
                )
                if self._is_heavy_big_only_unit(total_weight, total_volume, eligible_vehicle_types):
                    normal_heavy_big_only_count += 1
                if bool(policy["must_use_ev_under_policy"]) and self._is_ev3000_only_policy_unit(eligible_vehicle_types):
                    normal_ev3000_only_count += 1
                unit_id += 1
                continue

            fragments: list[q1.OrderFragment] = []
            for order in customer_orders.itertuples(index=False):
                fragments.extend(
                    self._split_order_into_fragments(
                        order_id=int(order.order_id),
                        weight=float(order.weight),
                        volume=float(order.volume),
                    )
                )
            lower_bound = max(
                math.ceil(total_weight / q1.MAX_SINGLE_WEIGHT),
                math.ceil(total_volume / q1.MAX_SINGLE_VOLUME),
            )
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
                packed_bins = self._policy_pack_customer_fragments_for_count(cust_id, fragments, bin_count, policy)
                if packed_bins is None:
                    continue
                heavy_big_only_count = 0
                big_only_count = 0
                eligible_sum = 0
                ev3000_only_count = 0
                for bin_state in packed_bins:
                    eligible = tuple(bin_state["eligible_vehicle_types"])
                    eligible_sum += len(eligible)
                    if self._is_big_only_unit(eligible):
                        big_only_count += 1
                    if self._is_heavy_big_only_unit(float(bin_state["weight"]), float(bin_state["volume"]), eligible):
                        heavy_big_only_count += 1
                    if bool(policy["must_use_ev_under_policy"]) and self._is_ev3000_only_policy_unit(eligible):
                        ev3000_only_count += 1
                candidate_packings.append(
                    {
                        "visit_count": len(packed_bins),
                        "heavy_big_only_count": heavy_big_only_count,
                        "big_only_count": big_only_count,
                        "eligible_sum": eligible_sum,
                        "ev3000_only_count": ev3000_only_count,
                        "bins": packed_bins,
                    }
                )
            if not candidate_packings:
                raise RuntimeError(f"Unable to build policy-aware packing candidates for mandatory split customer {cust_id}")
            candidate_packings.sort(
                key=lambda item: (
                    int(item["visit_count"]),
                    int(item["ev3000_only_count"]),
                    int(item["heavy_big_only_count"]),
                    int(item["big_only_count"]),
                    -int(item["eligible_sum"]),
                )
            )
            mandatory_customers.append(
                {
                    "cust_id": cust_id,
                    "row": row,
                    "policy": policy,
                    "candidates": candidate_packings,
                    "selected_index": 0,
                    "source_order_ids": source_order_ids,
                }
            )

        big_vehicle_inventory = sum(vehicle.vehicle_count for vehicle in self.vehicles if vehicle.capacity_kg >= 3000.0)
        heavy_big_only_capacity = max(big_vehicle_inventory - q1.BIG_VEHICLE_RESERVE - normal_heavy_big_only_count, 0)
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

        ev3000_capacity = max(self.vehicle_by_name["ev_3000"].vehicle_count - normal_ev3000_only_count, 0)
        current_ev3000_only = sum(
            int(customer["candidates"][customer["selected_index"]]["ev3000_only_count"])
            for customer in mandatory_customers
            if bool(customer["policy"]["must_use_ev_under_policy"])
        )
        while current_ev3000_only > ev3000_capacity:
            best_customer_idx = None
            best_candidate_idx = None
            best_score = None
            for customer_idx, customer in enumerate(mandatory_customers):
                if not bool(customer["policy"]["must_use_ev_under_policy"]):
                    continue
                current_candidate = customer["candidates"][customer["selected_index"]]
                for candidate_idx in range(customer["selected_index"] + 1, len(customer["candidates"])):
                    next_candidate = customer["candidates"][candidate_idx]
                    ev_reduction = int(current_candidate["ev3000_only_count"]) - int(next_candidate["ev3000_only_count"])
                    extra_visits = int(next_candidate["visit_count"]) - int(current_candidate["visit_count"])
                    if ev_reduction <= 0 or extra_visits <= 0:
                        continue
                    score = (
                        -(ev_reduction / extra_visits),
                        extra_visits,
                        int(next_candidate["ev3000_only_count"]),
                        int(next_candidate["visit_count"]),
                        int(next_candidate["heavy_big_only_count"]),
                        -int(next_candidate["eligible_sum"]),
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_customer_idx = customer_idx
                        best_candidate_idx = candidate_idx
            if best_customer_idx is None or best_candidate_idx is None:
                raise RuntimeError(
                    "Question 2 is infeasible under current one-vehicle-one-route assumptions: "
                    "policy-aware mandatory EV packing still exceeds ev_3000 inventory."
                )
            current_candidate = mandatory_customers[best_customer_idx]["candidates"][mandatory_customers[best_customer_idx]["selected_index"]]
            next_candidate = mandatory_customers[best_customer_idx]["candidates"][best_candidate_idx]
            current_ev3000_only += int(next_candidate["ev3000_only_count"]) - int(current_candidate["ev3000_only_count"])
            mandatory_customers[best_customer_idx]["selected_index"] = best_candidate_idx

        mandatory_ev3000_only_count = sum(
            int(customer["candidates"][customer["selected_index"]]["ev3000_only_count"])
            for customer in mandatory_customers
            if bool(customer["policy"]["must_use_ev_under_policy"])
        )
        current_heavy_big_only = sum(
            int(customer["candidates"][customer["selected_index"]]["heavy_big_only_count"])
            for customer in mandatory_customers
        )
        for customer in mandatory_customers:
            row = customer["row"]
            policy = customer["policy"]
            cust_id = int(row.cust_id)
            selected = customer["candidates"][customer["selected_index"]]
            packed_bins = selected["bins"]
            required_visit_count = len(packed_bins)
            for visit_index, bin_state in enumerate(packed_bins, start=1):
                unit_weight = float(bin_state["weight"])
                unit_volume = float(bin_state["volume"])
                eligible = tuple(bin_state["eligible_vehicle_types"])
                order_ids = tuple(sorted(set(int(order_id) for order_id in bin_state["order_ids"])))
                (
                    ev_allowed_flag,
                    fuel_allowed_flag,
                    ev_tw_start_min,
                    ev_tw_end_min,
                    fuel_tw_start_min,
                    fuel_tw_end_min,
                ) = self._policy_vehicle_windows(
                    policy=policy,
                    tw_start_min=int(row.tw_start_min),
                    tw_end_min=int(row.tw_end_min),
                    eligible_vehicle_types=eligible,
                    vehicle_by_name=self.vehicle_by_name,
                )
                policy_units.append(
                    PolicyServiceUnit(
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
                        in_green_zone=bool(policy["in_green_zone"]),
                        must_use_ev_under_policy=bool(policy["must_use_ev_under_policy"]),
                        fuel_allowed_after_16=bool(policy["fuel_allowed_after_16"]),
                        ev_allowed_flag=ev_allowed_flag,
                        fuel_allowed_flag=fuel_allowed_flag,
                        ev_tw_start_min=ev_tw_start_min,
                        ev_tw_end_min=ev_tw_end_min,
                        fuel_tw_start_min=fuel_tw_start_min,
                        fuel_tw_end_min=fuel_tw_end_min,
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
                        "in_green_zone": int(policy["in_green_zone"]),
                        "must_use_ev_under_policy": int(policy["must_use_ev_under_policy"]),
                        "fuel_allowed_after_16": int(policy["fuel_allowed_after_16"]),
                    }
                )
                unit_id += 1
        self.service_unit_summary = {
            "mandatory_split_customer_count": len(mandatory_customers),
            "normal_heavy_big_only_count": normal_heavy_big_only_count,
            "mandatory_heavy_big_only_count": current_heavy_big_only,
            "heavy_big_only_count": normal_heavy_big_only_count + current_heavy_big_only,
            "big_vehicle_inventory": big_vehicle_inventory,
            "big_vehicle_reserve": q1.BIG_VEHICLE_RESERVE,
            "heavy_big_only_capacity": heavy_big_only_capacity,
            "normal_ev3000_only_count": normal_ev3000_only_count,
            "mandatory_ev3000_only_count": mandatory_ev3000_only_count,
            "ev3000_only_count": normal_ev3000_only_count + mandatory_ev3000_only_count,
            "ev3000_only_capacity": ev3000_capacity,
        }
        return policy_units, split_rows

    def _unit_priority_key(self, unit_id: int) -> tuple[int, int, int, int, int, float, int, int]:
        unit = self.unit_by_id[unit_id]
        return (
            0 if getattr(unit, "must_use_ev_under_policy", False) else 1,
            0 if unit.eligible_vehicle_types == ("ev_3000",) else 1,
            0 if getattr(unit, "fuel_allowed_after_16", False) else 1,
            *super()._unit_priority_key(unit_id),
        )

    def _service_window_for_vehicle(self, unit: PolicyServiceUnit, vehicle_type: str) -> tuple[float, float]:
        power_type = self.vehicle_by_name[vehicle_type].power_type
        if power_type == "fuel":
            if not unit.fuel_allowed_flag or unit.fuel_tw_start_min < 0 or unit.fuel_tw_end_min < unit.fuel_tw_start_min:
                return np.inf, -np.inf
            return float(unit.fuel_tw_start_min), float(unit.fuel_tw_end_min)
        if not unit.ev_allowed_flag:
            return np.inf, -np.inf
        return float(unit.ev_tw_start_min), float(unit.ev_tw_end_min)

    def _time_window_overlap_minutes(self, left_unit_id: int, right_unit_id: int) -> float:
        left_unit = self.unit_by_id[left_unit_id]
        right_unit = self.unit_by_id[right_unit_id]
        shared_vehicle_types = set(left_unit.eligible_vehicle_types) & set(right_unit.eligible_vehicle_types)
        if not shared_vehicle_types:
            return 0.0
        best_overlap = 0.0
        for vehicle_type in shared_vehicle_types:
            left_start, left_end = self._service_window_for_vehicle(left_unit, vehicle_type)
            right_start, right_end = self._service_window_for_vehicle(right_unit, vehicle_type)
            overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
            best_overlap = max(best_overlap, overlap)
        return float(best_overlap)

    def evaluate_route(self, route: q1.TypedRoute) -> q1.RouteEvaluation:
        cache_key = self._route_key(route)
        cached = self.route_cache.get(cache_key)
        if cached is not None:
            return cached

        vehicle = self.vehicle_by_name[route.vehicle_type]
        units = tuple(self.unit_by_id[unit_id] for unit_id in route.unit_ids)
        total_weight = float(sum(unit.weight for unit in units))
        total_volume = float(sum(unit.volume for unit in units))
        customer_ids = [unit.orig_cust_id for unit in units]
        if (
            total_weight > vehicle.capacity_kg + 1e-6
            or total_volume > vehicle.capacity_m3 + 1e-6
            or len(set(customer_ids)) != len(customer_ids)
            or any(route.vehicle_type not in unit.eligible_vehicle_types for unit in units)
        ):
            evaluation = q1.RouteEvaluation(
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
            tw_start_min, tw_end_min = self._service_window_for_vehicle(unit, route.vehicle_type)
            if not np.isfinite(tw_start_min) or not np.isfinite(tw_end_min) or tw_end_min < tw_start_min - 1e-9:
                evaluation = q1.RouteEvaluation(
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
            wait = np.maximum(tw_start_min - arrival, 0.0)
            service_start = arrival + wait
            late = np.maximum(service_start - tw_end_min, 0.0)
            service_end = service_start + self.service_time_min

            valid_mask &= np.isfinite(travel) & np.isfinite(base_energy) & np.isfinite(service_end)
            total_wait += np.where(valid_mask, wait, 0.0)
            total_late += np.where(valid_mask, late, 0.0)
            total_energy += np.where(valid_mask, base_energy * load_multiplier, 0.0)

            departure_vec = service_end
            previous_node = unit.orig_cust_id
            remaining_weight -= unit.weight

        after_hours_travel, after_hours_fuel, after_hours_electric = self._after_hours_full_values(previous_node, 0)
        travel_back = self._interpolate_metric(
            self.travel_time_lookup,
            previous_node,
            0,
            departure_vec,
            after_hours_travel,
        )
        base_energy_lookup = self.base_fuel_lookup if vehicle.power_type == "fuel" else self.base_electric_lookup
        base_energy_back = self._interpolate_metric(
            base_energy_lookup,
            previous_node,
            0,
            departure_vec,
            after_hours_fuel if vehicle.power_type == "fuel" else after_hours_electric,
        )
        return_vec = departure_vec + travel_back
        total_energy += np.where(np.isfinite(base_energy_back), base_energy_back, 0.0)
        valid_mask &= np.isfinite(travel_back) & np.isfinite(return_vec)

        if not np.any(valid_mask):
            evaluation = q1.RouteEvaluation(
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

        startup_cost = q1.START_COST
        energy_unit_cost = q1.FUEL_PRICE if vehicle.power_type == "fuel" else q1.ELECTRICITY_PRICE
        carbon_factor = q1.FUEL_CARBON_FACTOR if vehicle.power_type == "fuel" else q1.ELECTRICITY_CARBON_FACTOR
        energy_cost = total_energy * energy_unit_cost
        carbon_cost = total_energy * carbon_factor * q1.CARBON_COST
        total_cost = (
            startup_cost
            + energy_cost
            + carbon_cost
            + total_wait * q1.WAIT_COST_PER_MIN
            + total_late * q1.LATE_COST_PER_MIN
        )
        total_cost = np.where(valid_mask, total_cost, np.inf)
        best_idx = int(np.argmin(total_cost))
        evaluation = q1.RouteEvaluation(
            vehicle_type=route.vehicle_type,
            unit_ids=route.unit_ids,
            feasible=np.isfinite(total_cost[best_idx]),
            total_weight=total_weight,
            total_volume=total_volume,
            best_cost=float(total_cost[best_idx]),
            best_start=int(best_idx),
            best_return=float(return_vec[best_idx]) if np.isfinite(return_vec[best_idx]) else np.inf,
        )
        self.route_cache.set(cache_key, evaluation)
        return evaluation

    def _simulate_route_scalar(self, route: q1.TypedRoute, route_index: int, vehicle_instance: int) -> q1.AssignedRoute:
        vehicle = self.vehicle_by_name[route.vehicle_type]
        evaluation = self.evaluate_route(route)
        if not evaluation.feasible:
            raise ValueError(f"Cannot simulate infeasible route {route}")

        units = tuple(self.unit_by_id[unit_id] for unit_id in route.unit_ids)
        current_departure = float(evaluation.best_start)
        previous_node = 0
        remaining_weight = float(sum(unit.weight for unit in units))
        remaining_volume = float(sum(unit.volume for unit in units))
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

        for stop_index, unit in enumerate(units, start=1):
            tw_start_min, tw_end_min = self._service_window_for_vehicle(unit, route.vehicle_type)
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
            wait = max(tw_start_min - arrival, 0.0)
            service_start = arrival + wait
            late = max(service_start - tw_end_min, 0.0)
            service_end = service_start + self.service_time_min
            load_ratio = max(0.0, min(1.0, remaining_weight / vehicle.capacity_kg))
            load_multiplier = 1.0 + (0.40 if vehicle.power_type == "fuel" else 0.35) * load_ratio
            fuel = base_fuel * load_multiplier
            electricity = base_electric * load_multiplier
            total_wait += wait
            total_late += late
            reference_total_fuel += fuel
            reference_total_electricity += electricity
            if vehicle.power_type == "fuel":
                total_fuel += fuel
            else:
                total_electricity += electricity
            route_distance += self._distance_between(previous_node, unit.orig_cust_id)
            after_hours_travel_km += self._scalar_after_hours_distance(previous_node, unit.orig_cust_id, current_departure)
            if service_start > q1.DAY_END_MIN + 1e-9:
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
                    "tw_start_min": int(unit.tw_start_min),
                    "tw_start_hhmm": self._minutes_to_hhmm(unit.tw_start_min),
                    "tw_end_min": int(unit.tw_end_min),
                    "tw_end_hhmm": self._minutes_to_hhmm(unit.tw_end_min),
                    "policy_tw_start_min": round(tw_start_min, 1),
                    "policy_tw_start_hhmm": self._minutes_to_hhmm(tw_start_min),
                    "policy_tw_end_min": round(tw_end_min, 1),
                    "policy_tw_end_hhmm": self._minutes_to_hhmm(tw_end_min),
                    "policy_vehicle_window_type": "fuel" if vehicle.power_type == "fuel" else "ev",
                    "in_green_zone": int(unit.in_green_zone),
                    "must_use_ev_under_policy": int(unit.must_use_ev_under_policy),
                    "fuel_allowed_after_16": int(unit.fuel_allowed_after_16),
                    "after_hours_service_flag": int(service_start > q1.DAY_END_MIN + 1e-9),
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
        return_time = current_departure + travel_back
        route_distance += self._distance_between(previous_node, 0)
        after_hours_travel_km += self._scalar_after_hours_distance(previous_node, 0, current_departure)
        after_hours_return_flag = return_time > q1.DAY_END_MIN + 1e-9

        reference_total_fuel += base_fuel_back
        reference_total_electricity += base_electric_back
        if vehicle.power_type == "fuel":
            total_fuel += base_fuel_back
            energy_cost = total_fuel * q1.FUEL_PRICE
            carbon_cost = total_fuel * q1.FUEL_CARBON_FACTOR * q1.CARBON_COST
        else:
            total_electricity += base_electric_back
            energy_cost = total_electricity * q1.ELECTRICITY_PRICE
            carbon_cost = total_electricity * q1.ELECTRICITY_CARBON_FACTOR * q1.CARBON_COST
        waiting_cost = total_wait * q1.WAIT_COST_PER_MIN
        late_cost = total_late * q1.LATE_COST_PER_MIN
        route_cost = q1.START_COST + energy_cost + carbon_cost + waiting_cost + late_cost

        return q1.AssignedRoute(
            route_index=route_index,
            vehicle_type=route.vehicle_type,
            power_type=vehicle.power_type,
            vehicle_instance=vehicle_instance,
            unit_ids=route.unit_ids,
            departure_min=float(evaluation.best_start),
            return_min=return_time,
            route_cost=route_cost,
            energy_cost=energy_cost,
            carbon_cost=carbon_cost,
            waiting_cost=waiting_cost,
            late_cost=late_cost,
            startup_cost=q1.START_COST,
            total_wait_min=total_wait,
            total_late_min=total_late,
            total_fuel_l=total_fuel,
            total_electricity_kwh=total_electricity,
            reference_fuel_l=reference_total_fuel,
            reference_electricity_kwh=reference_total_electricity,
            route_distance_km=route_distance,
            after_hours_travel_km=after_hours_travel_km,
            after_hours_service_count=after_hours_service_count,
            after_hours_return_flag=after_hours_return_flag,
            late_positive_stop_count=late_positive_stop_count,
            max_late_min=max_late_min,
            units=units,
            stop_rows=stop_rows,
        )

    def _policy_solution_diagnostics(self, solution_eval: q1.SolutionEvaluation) -> dict[str, object]:
        mandatory_ev_served_by_non_ev: set[int] = set()
        policy_violation_route_ids: set[int] = set()
        violation_stop_keys: set[tuple[int, int]] = set()
        fuel_route_green_zone_pre16_visit_count = 0
        fuel_route_green_zone_post16_visit_count = 0
        ev_route_green_zone_visit_count = 0

        for assigned_route in solution_eval.assigned_routes:
            for stop_row in assigned_route.stop_rows:
                cust_id = int(stop_row["orig_cust_id"])
                policy = self._policy_for_customer(cust_id)
                if not bool(policy["in_green_zone"]):
                    continue
                stop_key = (int(assigned_route.route_index), int(stop_row["stop_index"]))
                service_start_min = float(stop_row["service_start_min"])
                if assigned_route.power_type == "fuel":
                    if service_start_min < self.policy_ban_end_min - 1e-9:
                        fuel_route_green_zone_pre16_visit_count += 1
                        violation_stop_keys.add(stop_key)
                        policy_violation_route_ids.add(int(assigned_route.route_index))
                    else:
                        fuel_route_green_zone_post16_visit_count += 1
                    if bool(policy["must_use_ev_under_policy"]):
                        mandatory_ev_served_by_non_ev.add(cust_id)
                        violation_stop_keys.add(stop_key)
                        policy_violation_route_ids.add(int(assigned_route.route_index))
                else:
                    ev_route_green_zone_visit_count += 1

        active_green_zone_customer_count = int(self.active_customer_df["in_green_zone"].sum())
        total_green_zone_customer_count = int(self.customer_master["in_green_zone"].sum())
        mandatory_ev_customer_count = int(self.customer_master["must_use_ev_under_policy"].sum())
        mandatory_ev_active_customer_count = int(self.active_customer_df["must_use_ev_under_policy"].sum())
        return {
            "green_zone_basis": POLICY_GREEN_ZONE_BASIS,
            "green_zone_customer_count_used": active_green_zone_customer_count,
            "green_zone_customer_count_total_geometry": total_green_zone_customer_count,
            "problem_statement_green_zone_count": 30,
            "mandatory_ev_customer_count": mandatory_ev_customer_count,
            "mandatory_ev_active_customer_count": mandatory_ev_active_customer_count,
            "mandatory_ev_served_by_non_ev_count": int(len(mandatory_ev_served_by_non_ev)),
            "mandatory_ev_served_by_non_ev_customer_ids": sorted(int(cust_id) for cust_id in mandatory_ev_served_by_non_ev),
            "fuel_route_green_zone_pre16_visit_count": int(fuel_route_green_zone_pre16_visit_count),
            "fuel_route_green_zone_post16_visit_count": int(fuel_route_green_zone_post16_visit_count),
            "ev_route_green_zone_visit_count": int(ev_route_green_zone_visit_count),
            "policy_violation_count": int(len(violation_stop_keys)),
            "policy_violation_route_ids": sorted(int(route_id) for route_id in policy_violation_route_ids),
        }

    def _solve_single_configuration(self) -> tuple[q1.SolutionEvaluation, list[q1.TypedRoute], dict[str, object]]:
        solution_eval, solution_routes, metadata = super()._solve_single_configuration()
        policy_diagnostics = self._policy_solution_diagnostics(solution_eval)
        metadata.update(policy_diagnostics)
        return solution_eval, solution_routes, metadata

    @staticmethod
    def _read_optional_csv(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, encoding=q1.CSV_ENCODING)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _write_outputs(
        self,
        solution_eval: q1.SolutionEvaluation,
        solution_routes: list[q1.TypedRoute],
        metadata: dict[str, object],
    ) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        super()._write_outputs(solution_eval, solution_routes, metadata)

        q1_route_summary_path = self.output_root / "q1_route_summary.csv"
        q1_stop_schedule_path = self.output_root / "q1_stop_schedule.csv"
        q1_vehicle_schedule_path = self.output_root / "q1_vehicle_schedule.csv"
        q1_customer_aggregate_path = self.output_root / "q1_customer_aggregate.csv"
        q1_service_units_path = self.output_root / "q1_service_units.csv"
        q1_split_plan_path = self.output_root / "q1_split_plan.csv"
        q1_cost_summary_path = self.output_root / "q1_cost_summary.json"
        q1_run_report_path = self.output_root / "q1_run_report.md"
        q1_merge_diagnostics_path = self.output_root / "q1_merge_diagnostics.csv"
        q1_route_pool_summary_path = self.output_root / "q1_route_pool_summary.csv"

        route_summary_df = self._read_optional_csv(q1_route_summary_path)
        stop_schedule_df = self._read_optional_csv(q1_stop_schedule_path)
        vehicle_schedule_df = self._read_optional_csv(q1_vehicle_schedule_path)
        customer_aggregate_df = self._read_optional_csv(q1_customer_aggregate_path)
        service_units_df = self._read_optional_csv(q1_service_units_path)
        split_plan_df = self._read_optional_csv(q1_split_plan_path)
        merge_diagnostics_df = self._read_optional_csv(q1_merge_diagnostics_path)
        route_pool_summary_df = self._read_optional_csv(q1_route_pool_summary_path)
        cost_summary = json.loads(q1_cost_summary_path.read_text(encoding="utf-8"))
        report_lines = q1_run_report_path.read_text(encoding="utf-8").splitlines()
        if report_lines and report_lines[0].startswith("# "):
            report_lines[0] = "# Question 2 Solver Report"

        policy_df = self._policy_customer_df()
        policy_customer_columns = [
            "cust_id",
            "in_green_zone",
            "must_use_ev_under_policy",
            "fuel_allowed_after_16",
        ]

        if not stop_schedule_df.empty:
            missing_stop_policy_columns = [
                column for column in policy_customer_columns[1:] if column not in stop_schedule_df.columns
            ]
            if missing_stop_policy_columns:
                stop_schedule_df = stop_schedule_df.merge(
                    policy_df[["cust_id", *missing_stop_policy_columns]],
                    left_on="orig_cust_id",
                    right_on="cust_id",
                    how="left",
                ).drop(columns=["cust_id"], errors="ignore")

        if not route_summary_df.empty and not stop_schedule_df.empty:
            route_policy_df = (
                stop_schedule_df.assign(
                    fuel_green_zone_pre16_violation=lambda df: (
                        df["in_green_zone"].astype(bool)
                        & df["vehicle_type"].astype(str).str.startswith("fuel")
                        & (pd.to_numeric(df["service_start_min"], errors="coerce") < self.policy_ban_end_min - 1e-9)
                    ).astype(int),
                    green_zone_visit=lambda df: df["in_green_zone"].astype(bool).astype(int),
                )
                .groupby("route_id", as_index=False)
                .agg(
                    green_zone_visit_count=("green_zone_visit", "sum"),
                    fuel_green_zone_pre16_violation_count=("fuel_green_zone_pre16_violation", "sum"),
                )
            )
            route_summary_df = route_summary_df.merge(route_policy_df, on="route_id", how="left").fillna(
                {"green_zone_visit_count": 0, "fuel_green_zone_pre16_violation_count": 0}
            )

        if not customer_aggregate_df.empty:
            stop_customer_agg = (
                stop_schedule_df.groupby("orig_cust_id", as_index=False)
                .agg(
                    served_vehicle_types=("vehicle_type", lambda values: ",".join(sorted(set(str(value) for value in values)))),
                    first_service_start_min=("service_start_min", "min"),
                )
            )
            customer_aggregate_df = customer_aggregate_df.merge(
                stop_customer_agg,
                on="orig_cust_id",
                how="left",
            )
            customer_aggregate_df = customer_aggregate_df.merge(
                policy_df[policy_customer_columns],
                left_on="orig_cust_id",
                right_on="cust_id",
                how="left",
            ).drop(columns=["cust_id"], errors="ignore")
            customer_aggregate_df["served_vehicle_types"] = customer_aggregate_df["served_vehicle_types"].fillna("")
            customer_aggregate_df["served_by_ev"] = customer_aggregate_df["served_vehicle_types"].str.contains("ev").astype(int)
            customer_aggregate_df["served_by_fuel"] = customer_aggregate_df["served_vehicle_types"].str.contains("fuel").astype(int)

        if not service_units_df.empty:
            policy_unit_rows = [
                {
                    "unit_id": unit.unit_id,
                    "in_green_zone": int(unit.in_green_zone),
                    "must_use_ev_under_policy": int(unit.must_use_ev_under_policy),
                    "fuel_allowed_after_16": int(unit.fuel_allowed_after_16),
                    "ev_tw_start_min": unit.ev_tw_start_min,
                    "ev_tw_end_min": unit.ev_tw_end_min,
                    "fuel_tw_start_min": unit.fuel_tw_start_min,
                    "fuel_tw_end_min": unit.fuel_tw_end_min,
                }
                for unit in self.service_units
            ]
            service_units_df = service_units_df.merge(pd.DataFrame(policy_unit_rows), on="unit_id", how="left")

        if not split_plan_df.empty:
            missing_split_policy_columns = [
                column for column in policy_customer_columns[1:] if column not in split_plan_df.columns
            ]
            if missing_split_policy_columns:
                split_plan_df = split_plan_df.merge(
                    policy_df[["cust_id", *missing_split_policy_columns]],
                    on="cust_id",
                    how="left",
                )

        cost_summary.update(
            {
                "green_zone_basis": metadata["green_zone_basis"],
                "green_zone_customer_count_used": metadata["green_zone_customer_count_used"],
                "green_zone_customer_count_total_geometry": metadata["green_zone_customer_count_total_geometry"],
                "problem_statement_green_zone_count": metadata["problem_statement_green_zone_count"],
                "mandatory_ev_customer_count": metadata["mandatory_ev_customer_count"],
                "mandatory_ev_active_customer_count": metadata["mandatory_ev_active_customer_count"],
                "mandatory_ev_served_by_non_ev_count": metadata["mandatory_ev_served_by_non_ev_count"],
                "mandatory_ev_served_by_non_ev_customer_ids": metadata["mandatory_ev_served_by_non_ev_customer_ids"],
                "fuel_route_green_zone_pre16_visit_count": metadata["fuel_route_green_zone_pre16_visit_count"],
                "fuel_route_green_zone_post16_visit_count": metadata["fuel_route_green_zone_post16_visit_count"],
                "ev_route_green_zone_visit_count": metadata["ev_route_green_zone_visit_count"],
                "policy_violation_count": metadata["policy_violation_count"],
                "policy_violation_route_ids": metadata["policy_violation_route_ids"],
                "requested_seed_list": metadata["seed_list"],
                "requested_particle_count": metadata["particle_count"],
                "requested_max_generations": metadata["max_generations"],
                "requested_top_route_candidates": metadata["top_route_candidates"],
            }
        )

        q1_baseline_summary_path = self.workspace / "question1_artifacts" / "q1_cost_summary.json"
        if q1_baseline_summary_path.exists():
            q1_baseline_summary = json.loads(q1_baseline_summary_path.read_text(encoding="utf-8"))
            q1_ev_usage_baseline = int(
                sum(
                    int(count)
                    for vehicle_type, count in q1_baseline_summary.get("vehicle_type_usage", {}).items()
                    if str(vehicle_type).startswith("ev")
                )
            )
            cost_summary["q1_ev_usage_baseline"] = q1_ev_usage_baseline
        q2_ev_usage_total = int(
            sum(
                int(count)
                for vehicle_type, count in solution_eval.vehicle_type_usage.items()
                if str(vehicle_type).startswith("ev")
            )
        )
        cost_summary["q2_ev_usage_total"] = q2_ev_usage_total
        if not customer_aggregate_df.empty:
            cost_summary["ordinary_customers_served_by_ev"] = int(
                (
                    (customer_aggregate_df["must_use_ev_under_policy"] == 0)
                    & (customer_aggregate_df["served_by_ev"] == 1)
                ).sum()
            )
            cost_summary["must_use_ev_customers_served_by_ev_only"] = int(
                (
                    (customer_aggregate_df["must_use_ev_under_policy"] == 1)
                    & (customer_aggregate_df["served_by_ev"] == 1)
                    & (customer_aggregate_df["served_by_fuel"] == 0)
                ).sum()
            )

        ev_policy_summary_record = (
            self.ev_policy_summary_df.iloc[0].to_dict()
            if not self.ev_policy_summary_df.empty
            else {}
        )
        policy_summary_payload = {
            **ev_policy_summary_record,
            "green_zone_basis": metadata["green_zone_basis"],
            "green_zone_customer_count_used": metadata["green_zone_customer_count_used"],
            "green_zone_customer_count_total_geometry": metadata["green_zone_customer_count_total_geometry"],
            "problem_statement_green_zone_count": metadata["problem_statement_green_zone_count"],
            "mandatory_ev_customer_count": metadata["mandatory_ev_customer_count"],
            "mandatory_ev_active_customer_count": metadata["mandatory_ev_active_customer_count"],
            "mandatory_ev_served_by_non_ev_count": metadata["mandatory_ev_served_by_non_ev_count"],
            "mandatory_ev_served_by_non_ev_customer_ids": metadata["mandatory_ev_served_by_non_ev_customer_ids"],
            "fuel_route_green_zone_pre16_visit_count": metadata["fuel_route_green_zone_pre16_visit_count"],
            "fuel_route_green_zone_post16_visit_count": metadata["fuel_route_green_zone_post16_visit_count"],
            "ev_route_green_zone_visit_count": metadata["ev_route_green_zone_visit_count"],
            "policy_violation_count": metadata["policy_violation_count"],
            "policy_violation_route_ids": metadata["policy_violation_route_ids"],
            "final_total_cost": solution_eval.total_cost,
            "final_route_count": solution_eval.route_count,
        }

        report_lines.extend(
            [
                "",
                "## Question 2 Policy Summary",
                f"- Applied budget signature: {metadata['applied_budget_signature']}",
                f"- Requested seed list / particles / generations / top candidates: "
                f"{metadata['seed_list']} / {metadata['particle_count']} / {metadata['max_generations']} / {metadata['top_route_candidates']}",
                f"- Run record count / mutation attempts / accepted / elapsed sec: "
                f"{metadata['run_record_count']} / {metadata['total_mutation_attempt_count']} / "
                f"{metadata['total_accepted_mutation_count']} / {metadata['elapsed_sec']:.2f}",
                "- Output cache only stores arc lookup data; final solutions are recomputed each run.",
                f"- Green-zone basis: {metadata['green_zone_basis']}",
                f"- Green-zone active customers used: {metadata['green_zone_customer_count_used']}",
                f"- Mandatory-EV customers (all / active): {metadata['mandatory_ev_customer_count']} / {metadata['mandatory_ev_active_customer_count']}",
                f"- Fuel visits inside green zone before 16:00: {metadata['fuel_route_green_zone_pre16_visit_count']}",
                f"- Fuel visits inside green zone after 16:00: {metadata['fuel_route_green_zone_post16_visit_count']}",
                f"- EV visits inside green zone: {metadata['ev_route_green_zone_visit_count']}",
                f"- Mandatory-EV customers served by non-EV: {metadata['mandatory_ev_served_by_non_ev_count']}",
                f"- Policy violation count / route ids: {metadata['policy_violation_count']} / {metadata['policy_violation_route_ids']}",
                f"- Q1 EV usage baseline / Q2 EV usage total: {cost_summary.get('q1_ev_usage_baseline')} / {cost_summary['q2_ev_usage_total']}",
                f"- Ordinary customers served by EV / must-use-EV customers served by EV only: "
                f"{cost_summary.get('ordinary_customers_served_by_ev')} / {cost_summary.get('must_use_ev_customers_served_by_ev_only')}",
            ]
        )

        route_summary_df.to_csv(self.output_root / "q2_route_summary.csv", index=False, encoding=q1.CSV_ENCODING)
        stop_schedule_df.to_csv(self.output_root / "q2_stop_schedule.csv", index=False, encoding=q1.CSV_ENCODING)
        vehicle_schedule_df.to_csv(self.output_root / "q2_vehicle_schedule.csv", index=False, encoding=q1.CSV_ENCODING)
        customer_aggregate_df.to_csv(self.output_root / "q2_customer_aggregate.csv", index=False, encoding=q1.CSV_ENCODING)
        service_units_df.to_csv(self.output_root / "q2_service_units.csv", index=False, encoding=q1.CSV_ENCODING)
        split_plan_df.to_csv(self.output_root / "q2_split_plan.csv", index=False, encoding=q1.CSV_ENCODING)
        merge_diagnostics_df.to_csv(self.output_root / "q2_merge_diagnostics.csv", index=False, encoding=q1.CSV_ENCODING)
        route_pool_summary_df.to_csv(self.output_root / "q2_route_pool_summary.csv", index=False, encoding=q1.CSV_ENCODING)
        (self.output_root / "q2_cost_summary.json").write_text(
            json.dumps(cost_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.output_root / "q2_run_report.md").write_text("\n".join(report_lines), encoding="utf-8")
        (self.output_root / "q2_policy_summary.json").write_text(
            json.dumps(policy_summary_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for source_name in (
            "q1_route_summary.csv",
            "q1_stop_schedule.csv",
            "q1_vehicle_schedule.csv",
            "q1_customer_aggregate.csv",
            "q1_cost_summary.json",
            "q1_run_report.md",
            "q1_service_units.csv",
            "q1_split_plan.csv",
            "q1_merge_diagnostics.csv",
            "q1_route_pool_summary.csv",
        ):
            source_path = self.output_root / source_name
            if source_path.exists():
                source_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve question 2 with policy-aware service windows.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--input-root", type=Path, default=Path.cwd() / "preprocess_artifacts")
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "question2_artifacts")
    parser.add_argument("--seed-list", type=str, default="11")
    parser.add_argument("--max-generations", type=int, default=2)
    parser.add_argument("--particle-count", type=int, default=1)
    parser.add_argument("--top-route-candidates", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_list = [int(seed.strip()) for seed in args.seed_list.split(",") if seed.strip()]
    solver = Question2Solver(
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
                "split_customer_count": best_solution.split_customer_count,
                "late_positive_stops": best_solution.late_positive_stops,
                "policy_violation_count": metadata["policy_violation_count"],
                "mandatory_ev_served_by_non_ev_count": metadata["mandatory_ev_served_by_non_ev_count"],
                "green_zone_customer_count_used": metadata["green_zone_customer_count_used"],
                "seed_count": metadata["seed_count"],
                "particle_count": metadata["particle_count"],
                "max_generations": metadata["max_generations"],
                "run_record_count": metadata["run_record_count"],
                "total_mutation_attempt_count": metadata["total_mutation_attempt_count"],
                "total_accepted_mutation_count": metadata["total_accepted_mutation_count"],
                "elapsed_sec": metadata["elapsed_sec"],
                "applied_budget_signature": metadata["applied_budget_signature"],
                "output_root": str(args.output_root),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
