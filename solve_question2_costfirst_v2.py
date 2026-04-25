from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

import solve_question1 as q1
import solve_question2 as q2


Q2_V2_EV_SOURCE_LIMIT = 32
Q2_V2_LATE_FUEL_SOURCE_LIMIT = 32
Q2_V2_NEIGHBOR_LIMIT = 10
Q2_V2_PAIR_PARTNER_LIMIT = 6
Q2_V2_TRIPLE_PARTNER_LIMIT = 4


class Question2CostFirstV2Solver(q2.Question2Solver):
    def _policy_pack_score(
        self,
        bins: list[dict[str, object]],
        policy: dict[str, object],
    ) -> tuple[float, float, float]:
        return q2.Question2Solver._policy_pack_score(self, bins, policy)

    def _policy_greedy_pack_items(
        self,
        cust_id: int,
        fragments: list[q1.OrderFragment],
        bin_count: int,
        rng,
        policy: dict[str, object],
    ) -> list[dict[str, object]] | None:
        return q2.Question2Solver._policy_greedy_pack_items(self, cust_id, fragments, bin_count, rng, policy)

    def _unit_priority_key(self, unit_id: int) -> tuple[int, int, int, int, int, int, float, int, int]:
        unit = self.unit_by_id[unit_id]
        return (
            0 if getattr(unit, "must_use_ev_under_policy", False) else 1,
            0 if unit.eligible_vehicle_types == ("ev_3000",) else 1,
            0 if getattr(unit, "must_use_ev_under_policy", False) and "ev_1250" in unit.eligible_vehicle_types else 1,
            0 if getattr(unit, "fuel_allowed_after_16", False) else 1,
            0 if getattr(unit, "in_green_zone", False) else 1,
            *super()._unit_priority_key(unit_id),
        )

    def _q2_choice_rank_key(
        self,
        route_counts: Counter[str],
        current_vehicle_type: str | None,
        candidate_vehicle_type: str,
        unit_id: int,
        candidate_route: q1.TypedRoute,
        delta_cost: float,
    ) -> tuple[int, int, int, int, int, int, float, tuple[float, float, int]]:
        unit = self.unit_by_id[unit_id]
        power_type = self.vehicle_by_name[candidate_vehicle_type].power_type
        uses_ev = power_type == "ev"
        ordinary_with_fuel_option = (not unit.must_use_ev_under_policy) and unit.fuel_allowed_flag
        late_fuel_cluster_count = sum(
            int(getattr(self.unit_by_id[candidate_unit_id], "fuel_allowed_after_16", False))
            for candidate_unit_id in candidate_route.unit_ids
        )
        return (
            1 if ordinary_with_fuel_option and uses_ev else 0,
            1 if ordinary_with_fuel_option and candidate_vehicle_type == "ev_3000" else 0,
            0 if unit.fuel_allowed_after_16 and power_type == "fuel" else 1 if unit.fuel_allowed_after_16 else 0,
            0 if unit.fuel_allowed_after_16 and power_type == "fuel" and late_fuel_cluster_count >= 2 else 1,
            0 if unit.must_use_ev_under_policy and candidate_vehicle_type == "ev_1250" else 1 if unit.must_use_ev_under_policy else 0,
            0 if unit.must_use_ev_under_policy and uses_ev else 1 if unit.must_use_ev_under_policy else 0,
            round(float(delta_cost), 6),
            self.vehicle_size_rank[candidate_vehicle_type],
        )

    def _choose_best_new_route(self, unit_id: int, route_counts: Counter[str]) -> tuple[q1.TypedRoute, q1.RouteEvaluation] | None:
        unit = self.unit_by_id[unit_id]
        options: list[tuple[tuple[int, int, int, int, int, int, float, tuple[float, float, int]], q1.TypedRoute, q1.RouteEvaluation]] = []
        for vehicle_type in unit.eligible_vehicle_types:
            if route_counts[vehicle_type] >= self.vehicle_by_name[vehicle_type].vehicle_count:
                continue
            candidate_route = q1.TypedRoute(vehicle_type=vehicle_type, unit_ids=(unit_id,))
            candidate_eval = self.evaluate_route(candidate_route)
            if not candidate_eval.feasible:
                continue
            rank_key = self._q2_choice_rank_key(
                route_counts=route_counts,
                current_vehicle_type=None,
                candidate_vehicle_type=vehicle_type,
                unit_id=unit_id,
                candidate_route=candidate_route,
                delta_cost=candidate_eval.best_cost,
            )
            options.append((rank_key, candidate_route, candidate_eval))
        if not options:
            return None
        options.sort(key=lambda item: item[0])
        _, best_route, best_eval = options[0]
        return best_route, best_eval

    def _insert_unit_best(self, routes: list[q1.TypedRoute], unit_id: int) -> list[q1.TypedRoute] | None:
        route_counts = self._route_counts(routes)
        best_rank = None
        best_routes: list[q1.TypedRoute] | None = None
        candidate_groups = [self._candidate_route_indices_for_unit(routes, unit_id) if routes else []]
        if routes:
            fallback_indices = [idx for idx in range(len(routes)) if idx not in set(candidate_groups[0])]
            candidate_groups.append(fallback_indices)
        for candidate_indices in candidate_groups:
            for route_idx in candidate_indices:
                route = routes[route_idx]
                if route.vehicle_type not in self.unit_by_id[unit_id].eligible_vehicle_types:
                    continue
                if self.unit_by_id[unit_id].orig_cust_id in self._route_customers(route):
                    continue
                base_eval = self.evaluate_route(route)
                if not base_eval.feasible:
                    continue
                for position in range(len(route.unit_ids) + 1):
                    new_unit_ids = route.unit_ids[:position] + (unit_id,) + route.unit_ids[position:]
                    candidate_route = q1.TypedRoute(vehicle_type=route.vehicle_type, unit_ids=new_unit_ids)
                    candidate_eval = self.evaluate_route(candidate_route)
                    if not candidate_eval.feasible:
                        continue
                    rank_key = self._q2_choice_rank_key(
                        route_counts=route_counts,
                        current_vehicle_type=route.vehicle_type,
                        candidate_vehicle_type=route.vehicle_type,
                        unit_id=unit_id,
                        candidate_route=candidate_route,
                        delta_cost=candidate_eval.best_cost - base_eval.best_cost,
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
            new_route_rank = self._q2_choice_rank_key(
                route_counts=route_counts,
                current_vehicle_type=None,
                candidate_vehicle_type=new_route.vehicle_type,
                unit_id=unit_id,
                candidate_route=new_route,
                delta_cost=new_eval.best_cost,
            )
            if best_rank is None or new_route_rank < best_rank:
                best_routes = list(routes) + [new_route]
        return best_routes

    def _register_candidate_route_with_roles(
        self,
        route_pool: dict[tuple[str, tuple[int, ...]], q1.CandidateRouteSpec],
        vehicle_type: str,
        unit_ids: tuple[int, ...],
        base_role: str,
        q2_role: str,
    ) -> None:
        self._register_candidate_route(route_pool, vehicle_type, unit_ids, base_role)
        key = (vehicle_type, tuple(unit_ids))
        if key in route_pool:
            route_pool[key].roles.add(q2_role)

    def _augment_route_pool_with_q2_targeted_candidates(
        self,
        route_pool: dict[tuple[str, tuple[int, ...]], q1.CandidateRouteSpec],
    ) -> None:
        units_by_customer: dict[int, list[int]] = defaultdict(list)
        for unit in self.service_units:
            units_by_customer[unit.orig_cust_id].append(unit.unit_id)

        singleton_cost_by_unit: dict[int, float] = {}
        for unit_id in self.active_unit_ids:
            singleton_cost_by_unit[unit_id] = min(
                (
                    self.evaluate_route(q1.TypedRoute(vehicle_type, (unit_id,))).best_cost
                    for vehicle_type in self.unit_by_id[unit_id].eligible_vehicle_types
                    if self.evaluate_route(q1.TypedRoute(vehicle_type, (unit_id,))).feasible
                ),
                default=float("inf"),
            )

        ev_target_unit_ids = [
            unit_id
            for unit_id in self.active_unit_ids
            if getattr(self.unit_by_id[unit_id], "must_use_ev_under_policy", False)
        ]
        ranked_ev_sources = self._rank_residual_source_unit_ids(
            ev_target_unit_ids,
            singleton_cost_by_unit,
            Q2_V2_EV_SOURCE_LIMIT,
        )
        for source_unit_id in ranked_ev_sources:
            neighbor_unit_ids = self._residual_candidate_neighbor_ids(
                source_unit_id,
                ev_target_unit_ids,
                units_by_customer,
                singleton_cost_by_unit,
                spatial_limit=q1.RESIDUAL_SPATIAL_NEIGHBOR_LIMIT,
                tw_limit=q1.RESIDUAL_TW_NEIGHBOR_LIMIT,
                neighbor_limit=Q2_V2_NEIGHBOR_LIMIT,
            )
            good_partners: list[int] = []
            for neighbor_unit_id in neighbor_unit_ids:
                if self.unit_by_id[neighbor_unit_id].orig_cust_id == self.unit_by_id[source_unit_id].orig_cust_id:
                    continue
                shared_vehicle_types = tuple(
                    vehicle_type
                    for vehicle_type in ("ev_1250", "ev_3000")
                    if vehicle_type in self.unit_by_id[source_unit_id].eligible_vehicle_types
                    and vehicle_type in self.unit_by_id[neighbor_unit_id].eligible_vehicle_types
                )
                if not shared_vehicle_types:
                    continue
                best_candidate = self._best_route_candidate_for_units(shared_vehicle_types, (source_unit_id, neighbor_unit_id))
                if best_candidate is None:
                    continue
                vehicle_type, ordered_unit_ids, _ = best_candidate
                self._register_candidate_route_with_roles(
                    route_pool,
                    vehicle_type,
                    ordered_unit_ids,
                    "flex_small",
                    "ev_flex_small_q2",
                )
                good_partners.append(neighbor_unit_id)
            for left_id, right_id in combinations(good_partners[:Q2_V2_TRIPLE_PARTNER_LIMIT], 2):
                shared_vehicle_types = tuple(
                    vehicle_type
                    for vehicle_type in ("ev_1250", "ev_3000")
                    if vehicle_type in self.unit_by_id[source_unit_id].eligible_vehicle_types
                    and vehicle_type in self.unit_by_id[left_id].eligible_vehicle_types
                    and vehicle_type in self.unit_by_id[right_id].eligible_vehicle_types
                )
                if not shared_vehicle_types:
                    continue
                best_candidate = self._best_route_candidate_for_units(shared_vehicle_types, (source_unit_id, left_id, right_id))
                if best_candidate is None:
                    continue
                vehicle_type, ordered_unit_ids, _ = best_candidate
                self._register_candidate_route_with_roles(
                    route_pool,
                    vehicle_type,
                    ordered_unit_ids,
                    "flex_small",
                    "ev_flex_small_q2",
                )

        late_fuel_unit_ids = [
            unit_id
            for unit_id in self.active_unit_ids
            if getattr(self.unit_by_id[unit_id], "fuel_allowed_after_16", False)
            and getattr(self.unit_by_id[unit_id], "fuel_allowed_flag", False)
            and not getattr(self.unit_by_id[unit_id], "must_use_ev_under_policy", False)
        ]
        ranked_late_sources = self._rank_residual_source_unit_ids(
            late_fuel_unit_ids,
            singleton_cost_by_unit,
            Q2_V2_LATE_FUEL_SOURCE_LIMIT,
        )
        for source_unit_id in ranked_late_sources:
            neighbor_unit_ids = self._residual_candidate_neighbor_ids(
                source_unit_id,
                late_fuel_unit_ids,
                units_by_customer,
                singleton_cost_by_unit,
                spatial_limit=q1.RESIDUAL_SPATIAL_NEIGHBOR_LIMIT,
                tw_limit=q1.RESIDUAL_TW_NEIGHBOR_LIMIT,
                neighbor_limit=Q2_V2_NEIGHBOR_LIMIT,
            )
            good_partners: list[int] = []
            for neighbor_unit_id in neighbor_unit_ids:
                if self.unit_by_id[neighbor_unit_id].orig_cust_id == self.unit_by_id[source_unit_id].orig_cust_id:
                    continue
                shared_vehicle_types = tuple(
                    vehicle_type
                    for vehicle_type in ("fuel_1250", "fuel_1500", "fuel_3000")
                    if vehicle_type in self.unit_by_id[source_unit_id].eligible_vehicle_types
                    and vehicle_type in self.unit_by_id[neighbor_unit_id].eligible_vehicle_types
                )
                if not shared_vehicle_types:
                    continue
                best_candidate = self._best_route_candidate_for_units(shared_vehicle_types, (source_unit_id, neighbor_unit_id))
                if best_candidate is None:
                    continue
                vehicle_type, ordered_unit_ids, _ = best_candidate
                base_role = "promotion" if vehicle_type == "fuel_3000" else "flex_small"
                self._register_candidate_route_with_roles(
                    route_pool,
                    vehicle_type,
                    ordered_unit_ids,
                    base_role,
                    "late_fuel_cluster_q2",
                )
                good_partners.append(neighbor_unit_id)
            for left_id, right_id in combinations(good_partners[:Q2_V2_TRIPLE_PARTNER_LIMIT], 2):
                shared_vehicle_types = tuple(
                    vehicle_type
                    for vehicle_type in ("fuel_1250", "fuel_1500", "fuel_3000")
                    if vehicle_type in self.unit_by_id[source_unit_id].eligible_vehicle_types
                    and vehicle_type in self.unit_by_id[left_id].eligible_vehicle_types
                    and vehicle_type in self.unit_by_id[right_id].eligible_vehicle_types
                )
                if not shared_vehicle_types:
                    continue
                best_candidate = self._best_route_candidate_for_units(shared_vehicle_types, (source_unit_id, left_id, right_id))
                if best_candidate is None:
                    continue
                vehicle_type, ordered_unit_ids, _ = best_candidate
                base_role = "promotion" if vehicle_type == "fuel_3000" else "flex_small"
                self._register_candidate_route_with_roles(
                    route_pool,
                    vehicle_type,
                    ordered_unit_ids,
                    base_role,
                    "late_fuel_cluster_q2",
                )

        for spec in route_pool.values():
            if "piggyback_big" in spec.roles:
                spec.roles.add("policy_piggyback_q2")

    def _generate_route_pool(
        self,
        seed_routes: list[q1.TypedRoute],
    ) -> dict[tuple[str, tuple[int, ...]], q1.CandidateRouteSpec]:
        route_pool = super()._generate_route_pool(seed_routes)
        self._augment_route_pool_with_q2_targeted_candidates(route_pool)
        return route_pool

    def _q2_column_policy_profile(self, column: dict[str, object]) -> dict[str, int]:
        units = [self.unit_by_id[int(unit_id)] for unit_id in column["unit_ids"]]
        ordinary_with_fuel_option = sum(
            int((not unit.must_use_ev_under_policy) and unit.fuel_allowed_flag)
            for unit in units
        )
        must_ev_count = sum(int(unit.must_use_ev_under_policy) for unit in units)
        late_fuel_count = sum(int(unit.fuel_allowed_after_16 and unit.fuel_allowed_flag) for unit in units)
        return {
            "must_ev_count": must_ev_count,
            "late_fuel_count": late_fuel_count,
            "ordinary_with_fuel_option": ordinary_with_fuel_option,
        }

    def _column_effective_saving(self, column: dict[str, object]) -> float:
        base_saving = super()._column_effective_saving(column)
        roles = set(column["roles"])
        policy_profile = self._q2_column_policy_profile(column)
        power_type = self.vehicle_by_name[str(column["vehicle_type"])].power_type
        adjusted_saving = base_saving
        if "ev_flex_small_q2" in roles:
            adjusted_saving += 18.0 * policy_profile["must_ev_count"]
            if str(column["vehicle_type"]) == "ev_1250":
                adjusted_saving += 8.0 * policy_profile["must_ev_count"]
        if "late_fuel_cluster_q2" in roles and power_type == "fuel":
            adjusted_saving += 15.0 * policy_profile["late_fuel_count"]
        if "policy_piggyback_q2" in roles:
            adjusted_saving += 4.0
        if power_type == "ev":
            adjusted_saving -= 12.0 * policy_profile["ordinary_with_fuel_option"]
            if str(column["vehicle_type"]) == "ev_3000":
                adjusted_saving -= 6.0 * policy_profile["ordinary_with_fuel_option"]
        if power_type == "fuel" and policy_profile["late_fuel_count"] > 0:
            adjusted_saving += 4.0 * policy_profile["late_fuel_count"]
        return float(adjusted_saving)

    def _column_candidate_sort_key(self, column: dict[str, object]) -> tuple[float, float, float, int, float, str, tuple[int, ...]]:
        return (
            -float(self._column_effective_saving(column)),
            -float(column["avg_time_window_overlap_min"]),
            float(column["avg_customer_distance_km"]),
            -int(column["unit_count"]),
            float(column["best_cost"]),
            str(column["vehicle_type"]),
            tuple(int(unit_id) for unit_id in column["unit_ids"]),
        )

    def _write_outputs(
        self,
        solution_eval: q1.SolutionEvaluation,
        solution_routes: list[q1.TypedRoute],
        metadata: dict[str, object],
    ) -> None:
        super()._write_outputs(solution_eval, solution_routes, metadata)
        q2_cost_summary_path = self.output_root / "q2_cost_summary.json"
        q2_run_report_path = self.output_root / "q2_run_report.md"
        q2_customer_aggregate_path = self.output_root / "q2_customer_aggregate.csv"

        cost_summary = json.loads(q2_cost_summary_path.read_text(encoding="utf-8"))
        report_lines = q2_run_report_path.read_text(encoding="utf-8").splitlines()
        customer_df = pd.read_csv(q2_customer_aggregate_path)

        baseline_summary_path = self.workspace / "question2_artifacts" / "q2_cost_summary.json"
        if baseline_summary_path.exists() and baseline_summary_path.resolve() != q2_cost_summary_path.resolve():
            baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
            cost_summary["q2_baseline_total_cost"] = baseline_summary.get("total_cost")
            cost_summary["q2_baseline_route_count"] = baseline_summary.get("route_count")
            cost_summary["q2_baseline_single_stop_route_count"] = baseline_summary.get("single_stop_route_count")
            if baseline_summary.get("total_cost") is not None:
                cost_summary["cost_improvement_vs_q2_baseline"] = float(baseline_summary["total_cost"]) - float(solution_eval.total_cost)

        ev_usage_by_policy_class = {
            "must_use_ev_customers_served_by_ev_only": int(
                (
                    (customer_df["must_use_ev_under_policy"] == 1)
                    & (customer_df["served_by_ev"] == 1)
                    & (customer_df["served_by_fuel"] == 0)
                ).sum()
            ),
            "fuel_after_16_customers_served_by_fuel": int(
                (
                    (customer_df["fuel_allowed_after_16"] == 1)
                    & (customer_df["served_by_fuel"] == 1)
                ).sum()
            ),
            "ordinary_customers_served_by_ev": int(
                (
                    (customer_df["must_use_ev_under_policy"] == 0)
                    & (customer_df["fuel_allowed_after_16"] == 0)
                    & (customer_df["served_by_ev"] == 1)
                ).sum()
            ),
        }
        cost_summary["ev_usage_by_policy_class"] = ev_usage_by_policy_class
        cost_summary["fuel_after_16_service_customer_count"] = ev_usage_by_policy_class["fuel_after_16_customers_served_by_fuel"]

        if report_lines and report_lines[0].startswith("# "):
            report_lines[0] = "# Question 2 Cost-First v2 Report"
        report_lines.extend(
            [
                "",
                "## Q2 v2 Summary",
                f"- EV usage by policy class: {json.dumps(ev_usage_by_policy_class, ensure_ascii=False)}",
                f"- Fuel-after-16 customers served by fuel: {cost_summary['fuel_after_16_service_customer_count']}",
                f"- Cost improvement vs Q2 baseline: {cost_summary.get('cost_improvement_vs_q2_baseline')}",
            ]
        )

        q2_cost_summary_path.write_text(json.dumps(cost_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        q2_run_report_path.write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve question 2 with policy-aware cost-first v2 enhancements.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--input-root", type=Path, default=Path.cwd() / "preprocess_artifacts")
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "question2_artifacts_costfirst_v2")
    parser.add_argument("--seed-list", type=str, default="11")
    parser.add_argument("--max-generations", type=int, default=2)
    parser.add_argument("--particle-count", type=int, default=1)
    parser.add_argument("--top-route-candidates", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_list = [int(seed.strip()) for seed in args.seed_list.split(",") if seed.strip()]
    solver = Question2CostFirstV2Solver(
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
