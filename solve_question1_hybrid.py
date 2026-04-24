from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

import solve_question1 as base


CSV_ENCODING = base.CSV_ENCODING
HYBRID_TAU0 = 1.0
HYBRID_RHO = 0.15
HYBRID_ALPHA = 1.0
HYBRID_PBEST_BONUS = 1.15
HYBRID_GBEST_BONUS = 1.25
HYBRID_CAUCHY_GAMMA0 = 0.15
HYBRID_CAUCHY_DECAY = 0.9
HYBRID_DESTROY_RATIO_MIN = 0.05
HYBRID_DESTROY_RATIO_MAX = 0.25
HYBRID_TEMPERATURE_MIN = 0.2
HYBRID_TEMPERATURE_MAX = 2.0
HYBRID_PROBABILITY_SHORTLIST = 6
HYBRID_ELITE_COUNT = 2
HYBRID_ARCHIVE_SIZE = 6
BASELINE_REFERENCE_PARTICLES = 1
BASELINE_REFERENCE_GENERATIONS = 2
DEFAULT_MINIMAL_PARTICLES = 4
DEFAULT_MINIMAL_GENERATIONS = 3
DEFAULT_STANDARD_PARTICLES = 8
DEFAULT_STANDARD_GENERATIONS = 6
OUTER_NO_IMPROVEMENT_STOP = 2
OUTER_OPERATOR_NAMES = (
    "random_remove",
    "worst_cost_remove",
    "late_route_remove",
    "typed_route_merge_remove",
    "mandatory_split_cluster_remove",
)


@dataclass
class HybridParticle:
    particle_id: int
    current_routes: list[base.TypedRoute]
    current_eval: base.SolutionEvaluation
    personal_best_routes: list[base.TypedRoute]
    personal_best_eval: base.SolutionEvaluation
    destroy_ratio: float
    construction_temperature: float
    operator_weights: dict[str, float]
    cauchy_scale: float


@dataclass
class HybridInnerResult:
    routes: list[base.TypedRoute]
    solution_eval: base.SolutionEvaluation
    selected_label: str
    phase_statuses: list[dict[str, object]]
    route_pool_summary_rows: list[dict[str, object]]
    route_pool_candidate_count: int
    route_pool_role_counts: dict[str, int]
    summary_by_label: dict[str, dict[str, object]]


class HybridQuestion1Solver(base.Question1Solver):
    def __init__(
        self,
        workspace: Path,
        input_root: Path,
        output_root: Path,
        baseline_summary_path: Path,
        seed: int,
        particles: int,
        generations: int,
        top_route_candidates: int,
        mode: str,
    ) -> None:
        super().__init__(
            workspace=workspace,
            input_root=input_root,
            output_root=output_root,
            seed_list=[seed],
            max_generations=generations,
            particle_count=particles,
            top_route_candidates=top_route_candidates,
            enable_split_packing_sensitivity=False,
        )
        self.hybrid_seed = seed
        self.hybrid_mode = mode
        self.baseline_summary_path = baseline_summary_path
        self.outer_search_trace_rows: list[dict[str, object]] = []
        self.pheromone_edge: dict[tuple[int, int, str], float] = {}
        self.pheromone_start: dict[tuple[int, str], float] = {}
        self.elite_archive: list[tuple[list[base.TypedRoute], base.SolutionEvaluation, str]] = []
        self.operator_history = Counter[str]()
        self.baseline_summary = self._load_baseline_summary()

    def _load_baseline_summary(self) -> dict[str, object]:
        if self.baseline_summary_path.exists():
            return json.loads(self.baseline_summary_path.read_text(encoding="utf-8"))
        return {}

    def _clone_routes(self, routes: Iterable[base.TypedRoute]) -> list[base.TypedRoute]:
        return [base.TypedRoute(route.vehicle_type, tuple(route.unit_ids)) for route in routes]

    def _vehicle_family(self, vehicle_type: str) -> str:
        return "big" if vehicle_type in {"fuel_3000", "ev_3000"} else "small"

    def _extract_route_feature_sets(
        self,
        route: base.TypedRoute,
    ) -> tuple[set[tuple[int, str]], set[tuple[int, int, str]]]:
        start_features: set[tuple[int, str]] = set()
        edge_features: set[tuple[int, int, str]] = set()
        if not route.unit_ids:
            return start_features, edge_features
        vehicle_family = self._vehicle_family(route.vehicle_type)
        start_features.add((route.unit_ids[0], vehicle_family))
        for left_unit_id, right_unit_id in zip(route.unit_ids, route.unit_ids[1:]):
            edge_features.add((left_unit_id, right_unit_id, vehicle_family))
        return start_features, edge_features

    def _extract_solution_feature_sets(
        self,
        routes: Iterable[base.TypedRoute],
    ) -> tuple[set[tuple[int, str]], set[tuple[int, int, str]]]:
        start_features: set[tuple[int, str]] = set()
        edge_features: set[tuple[int, int, str]] = set()
        for route in routes:
            route_starts, route_edges = self._extract_route_feature_sets(route)
            start_features |= route_starts
            edge_features |= route_edges
        return start_features, edge_features

    def _route_tau_average(self, route: base.TypedRoute) -> float:
        start_features, edge_features = self._extract_route_feature_sets(route)
        tau_values = [
            self.pheromone_start.get(feature, HYBRID_TAU0)
            for feature in start_features
        ] + [
            self.pheromone_edge.get(feature, HYBRID_TAU0)
            for feature in edge_features
        ]
        if not tau_values:
            return HYBRID_TAU0
        return float(sum(tau_values) / len(tau_values))

    def _guide_bonus(
        self,
        route: base.TypedRoute,
        pbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
        gbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
    ) -> float:
        route_starts, route_edges = self._extract_route_feature_sets(route)
        bonus = 1.0
        if pbest_features is not None:
            if (route_starts & pbest_features[0]) or (route_edges & pbest_features[1]):
                bonus *= HYBRID_PBEST_BONUS
        if gbest_features is not None:
            if (route_starts & gbest_features[0]) or (route_edges & gbest_features[1]):
                bonus *= HYBRID_GBEST_BONUS
        return bonus

    def _rank_key_penalty(
        self,
        rank_key: tuple[int, int, float, tuple[float, float, int]],
    ) -> float:
        reserve_penalty = 250.0 * float(rank_key[0])
        mixed_big_penalty = 120.0 * float(rank_key[1])
        vehicle_penalty = 0.01 * float(rank_key[3][0]) + 0.1 * float(rank_key[3][1]) + 5.0 * float(rank_key[3][2])
        return reserve_penalty + mixed_big_penalty + vehicle_penalty

    def _weighted_route_choice(
        self,
        options: list[dict[str, object]],
        rng: random.Random,
    ) -> list[base.TypedRoute] | None:
        if not options:
            return None
        total_weight = float(sum(float(option["weight"]) for option in options))
        if total_weight <= 0.0 or not math.isfinite(total_weight):
            best_option = min(options, key=lambda item: (float(item["generalized_delta_cost"]), -float(item["tau_avg"])))
            return self._clone_routes(best_option["routes"])
        draw = rng.random() * total_weight
        cumulative = 0.0
        for option in options:
            cumulative += float(option["weight"])
            if draw <= cumulative:
                return self._clone_routes(option["routes"])
        return self._clone_routes(options[-1]["routes"])

    def _build_weighted_options_for_unit(
        self,
        routes: list[base.TypedRoute],
        unit_id: int,
        temperature: float,
        pbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
        gbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
    ) -> list[dict[str, object]]:
        unit = self.unit_by_id[unit_id]
        route_counts = self._route_counts(routes)
        unit_is_heavy_big_only = self._is_heavy_big_only_unit(unit.weight, unit.volume, unit.eligible_vehicle_types)
        options: list[dict[str, object]] = []
        candidate_groups = [self._candidate_route_indices_for_unit(routes, unit_id) if routes else []]
        if routes:
            primary = set(candidate_groups[0])
            fallback_indices = [idx for idx in range(len(routes)) if idx not in primary]
            candidate_groups.append(fallback_indices)
        for candidate_indices in candidate_groups:
            group_options: list[dict[str, object]] = []
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
                    candidate_route = base.TypedRoute(vehicle_type=route.vehicle_type, unit_ids=new_unit_ids)
                    candidate_eval = self.evaluate_route(candidate_route)
                    if not candidate_eval.feasible:
                        continue
                    delta_cost = float(candidate_eval.best_cost - base_eval.best_cost)
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
                        delta_cost=delta_cost,
                    )
                    updated_routes = self._clone_routes(routes)
                    updated_routes[route_idx] = candidate_route
                    tau_avg = self._route_tau_average(candidate_route)
                    guide_bonus = self._guide_bonus(candidate_route, pbest_features, gbest_features)
                    generalized_delta_cost = max(delta_cost, 0.0) + self._rank_key_penalty(rank_key)
                    weight = (max(tau_avg, 1e-9) ** HYBRID_ALPHA) * math.exp(
                        -generalized_delta_cost / max(temperature, 1e-9)
                    ) * guide_bonus
                    group_options.append(
                        {
                            "routes": updated_routes,
                            "weight": float(max(weight, 1e-12)),
                            "generalized_delta_cost": float(generalized_delta_cost),
                            "tau_avg": float(tau_avg),
                        }
                    )
            if group_options:
                group_options.sort(key=lambda item: (float(item["generalized_delta_cost"]), -float(item["tau_avg"])))
                options.extend(group_options[:HYBRID_PROBABILITY_SHORTLIST])
                break

        new_route_options: list[dict[str, object]] = []
        for vehicle_type in unit.eligible_vehicle_types:
            if route_counts[vehicle_type] >= self.vehicle_by_name[vehicle_type].vehicle_count:
                continue
            candidate_route = base.TypedRoute(vehicle_type=vehicle_type, unit_ids=(unit_id,))
            candidate_eval = self.evaluate_route(candidate_route)
            if not candidate_eval.feasible:
                continue
            rank_key = self._choice_rank_key(
                route_counts=route_counts,
                current_vehicle_type=None,
                candidate_vehicle_type=vehicle_type,
                unit_is_heavy_big_only=unit_is_heavy_big_only,
                mixes_flexible_into_big=(vehicle_type == "fuel_3000" and not unit_is_heavy_big_only),
                delta_cost=float(candidate_eval.best_cost),
            )
            tau_avg = self._route_tau_average(candidate_route)
            guide_bonus = self._guide_bonus(candidate_route, pbest_features, gbest_features)
            generalized_delta_cost = float(candidate_eval.best_cost) + self._rank_key_penalty(rank_key)
            weight = (max(tau_avg, 1e-9) ** HYBRID_ALPHA) * math.exp(
                -generalized_delta_cost / max(temperature, 1e-9)
            ) * guide_bonus
            new_route_options.append(
                {
                    "routes": self._clone_routes(routes) + [candidate_route],
                    "weight": float(max(weight, 1e-12)),
                    "generalized_delta_cost": float(generalized_delta_cost),
                    "tau_avg": float(tau_avg),
                }
            )
        if new_route_options:
            new_route_options.sort(key=lambda item: (float(item["generalized_delta_cost"]), -float(item["tau_avg"])))
            options.extend(new_route_options[: max(1, HYBRID_PROBABILITY_SHORTLIST // 2)])

        options.sort(key=lambda item: (float(item["generalized_delta_cost"]), -float(item["tau_avg"])))
        return options[:HYBRID_PROBABILITY_SHORTLIST]

    def _insert_unit_probabilistic(
        self,
        routes: list[base.TypedRoute],
        unit_id: int,
        rng: random.Random,
        temperature: float,
        pbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
        gbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
    ) -> list[base.TypedRoute] | None:
        options = self._build_weighted_options_for_unit(
            routes=routes,
            unit_id=unit_id,
            temperature=temperature,
            pbest_features=pbest_features,
            gbest_features=gbest_features,
        )
        weighted_choice = self._weighted_route_choice(options, rng)
        if weighted_choice is not None:
            return weighted_choice
        deterministic = self._insert_unit_best(routes, unit_id)
        if deterministic is None:
            return None
        return self._clone_routes(deterministic)

    def _build_probabilistic_initial_solution(
        self,
        rng: random.Random,
        temperature: float,
        pbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
        gbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
    ) -> list[base.TypedRoute]:
        ordered_units = sorted(
            self.active_unit_ids,
            key=lambda unit_id: (*self._unit_priority_key(unit_id), rng.random()),
        )
        routes: list[base.TypedRoute] = []
        for unit_id in ordered_units:
            updated_routes = self._insert_unit_probabilistic(
                routes=routes,
                unit_id=unit_id,
                rng=rng,
                temperature=temperature,
                pbest_features=pbest_features,
                gbest_features=gbest_features,
            )
            if updated_routes is None:
                raise RuntimeError(f"Unable to probabilistically insert service unit {unit_id}")
            routes = updated_routes
        return routes

    def _repair_solution_probabilistic(
        self,
        partial_routes: list[base.TypedRoute],
        removed_units: list[int],
        rng: random.Random,
        temperature: float,
        pbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
        gbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
    ) -> list[base.TypedRoute] | None:
        routes = self._clone_routes(partial_routes)
        for unit_id in sorted(set(removed_units), key=self._unit_priority_key):
            updated_routes = self._insert_unit_probabilistic(
                routes=routes,
                unit_id=unit_id,
                rng=rng,
                temperature=temperature,
                pbest_features=pbest_features,
                gbest_features=gbest_features,
            )
            if updated_routes is None:
                return None
            routes = updated_routes
        return routes

    def _cauchy(self, rng: random.Random, scale: float) -> float:
        draw = min(max(rng.random(), 1e-9), 1.0 - 1e-9)
        return math.tan(math.pi * (draw - 0.5)) * scale

    def _normalize_operator_weights(self, operator_weights: dict[str, float]) -> dict[str, float]:
        normalized = {name: max(0.05, float(weight)) for name, weight in operator_weights.items()}
        total = float(sum(normalized.values()))
        return {name: value / total for name, value in normalized.items()}

    def _mutate_particle_parameters(
        self,
        particle: HybridParticle,
        generation: int,
        rng: random.Random,
    ) -> None:
        gamma_t = particle.cauchy_scale * (HYBRID_CAUCHY_DECAY ** generation)
        delta = self._cauchy(rng, gamma_t)
        particle.destroy_ratio = min(
            HYBRID_DESTROY_RATIO_MAX,
            max(HYBRID_DESTROY_RATIO_MIN, particle.destroy_ratio + 0.08 * delta),
        )
        particle.construction_temperature = min(
            HYBRID_TEMPERATURE_MAX,
            max(HYBRID_TEMPERATURE_MIN, particle.construction_temperature * math.exp(0.35 * delta)),
        )
        updated_weights = {}
        for operator_name, base_weight in particle.operator_weights.items():
            updated_weights[operator_name] = max(0.05, base_weight * math.exp(self._cauchy(rng, gamma_t)))
        particle.operator_weights = self._normalize_operator_weights(updated_weights)

    def _choose_source_solution(
        self,
        particle: HybridParticle,
        global_best: HybridInnerResult,
        rng: random.Random,
    ) -> tuple[str, list[base.TypedRoute], base.SolutionEvaluation]:
        draw = rng.random()
        if draw < 0.4:
            return "current", self._clone_routes(particle.current_routes), particle.current_eval
        if draw < 0.7:
            return "personal_best", self._clone_routes(particle.personal_best_routes), particle.personal_best_eval
        return "global_best", self._clone_routes(global_best.routes), global_best.solution_eval

    def _choose_destroy_operator(
        self,
        operator_weights: dict[str, float],
        rng: random.Random,
    ) -> str:
        total_weight = float(sum(operator_weights.values()))
        draw = rng.random() * total_weight
        cumulative = 0.0
        for operator_name in OUTER_OPERATOR_NAMES:
            cumulative += float(operator_weights.get(operator_name, 0.0))
            if draw <= cumulative:
                return operator_name
        return OUTER_OPERATOR_NAMES[-1]

    def _destroy_solution(
        self,
        routes: list[base.TypedRoute],
        solution_eval: base.SolutionEvaluation,
        destroy_ratio: float,
        operator_weights: dict[str, float],
        rng: random.Random,
    ) -> tuple[list[base.TypedRoute], list[int], str]:
        flat_units = self._flatten_solution(routes)
        q_remove = int(round(len(flat_units) * destroy_ratio))
        q_remove = max(base.REMOVE_MIN, min(base.REMOVE_MAX, q_remove))
        q_remove = max(1, min(q_remove, len(flat_units)))
        operator_name = self._choose_destroy_operator(operator_weights, rng)
        self.operator_history[operator_name] += 1
        if operator_name == "random_remove":
            return (*self._random_remove(routes, q_remove, rng), operator_name)
        if operator_name == "worst_cost_remove":
            return (*self._worst_cost_remove(solution_eval, q_remove, rng), operator_name)
        if operator_name == "late_route_remove":
            return (*self._late_route_remove(solution_eval, q_remove), operator_name)
        if operator_name == "typed_route_merge_remove":
            return (*self._typed_route_merge_remove(solution_eval, q_remove), operator_name)
        return (*self._mandatory_split_cluster_remove(solution_eval, q_remove, rng), operator_name)

    def _passes_hard_guards(
        self,
        solution_routes: list[base.TypedRoute],
        solution_eval: base.SolutionEvaluation | None,
    ) -> bool:
        if solution_eval is None:
            return False
        if solution_eval.split_customer_count != solution_eval.mandatory_split_customer_count:
            return False
        customer_route_map: dict[int, set[int]] = defaultdict(set)
        mandatory_split_customers: set[int] = set()
        for route in solution_eval.assigned_routes:
            for unit in route.units:
                customer_route_map[unit.orig_cust_id].add(route.route_index)
                if unit.unit_type == "mandatory_split":
                    mandatory_split_customers.add(unit.orig_cust_id)
        for customer_id, route_ids in customer_route_map.items():
            if customer_id not in mandatory_split_customers and len(route_ids) > 1:
                return False
        covered_unit_ids = {unit_id for route in solution_routes for unit_id in route.unit_ids}
        return len(covered_unit_ids) == len(self.active_unit_ids)

    def _outer_solution_key(
        self,
        solution_eval: base.SolutionEvaluation,
    ) -> tuple[float, int, float, int, int]:
        return (
            round(solution_eval.total_cost, 6),
            int(solution_eval.late_positive_stops),
            round(solution_eval.max_late_min, 6),
            int(solution_eval.route_count),
            int(solution_eval.single_stop_route_count),
        )

    def _is_outer_better(
        self,
        candidate_eval: base.SolutionEvaluation,
        reference_eval: base.SolutionEvaluation | None,
    ) -> bool:
        return reference_eval is None or self._outer_solution_key(candidate_eval) < self._outer_solution_key(reference_eval)

    def _build_route_pool_summary_rows(
        self,
        global_result: dict[str, object],
        summary_by_label: dict[str, dict[str, object]],
        active_label: str,
    ) -> list[dict[str, object]]:
        pass1_keys = summary_by_label.get("pass1", {}).get("selected_route_keys", set()) or set()
        pass2_keys = summary_by_label.get("pass2", {}).get("selected_route_keys", set()) or set()
        pass3_keys = summary_by_label.get("pass3", {}).get("selected_route_keys", set()) or set()
        active_keys = summary_by_label.get(active_label, {}).get("selected_route_keys", set()) or set()
        rows: list[dict[str, object]] = []
        for column in global_result["columns"]:
            route_key = (column["vehicle_type"], column["unit_ids"])
            rows.append(
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
                    "mixed_big_route_flag": int(column["mixed_big_route_flag"]),
                    "piggyback_big_route_flag": int(column["piggyback_big_route_flag"]),
                    "promotion_like_big_route_flag": int(column["promotion_like_big_route_flag"]),
                    "blocking_big_flexible_route_flag": int(column["blocking_big_flexible_route_flag"]),
                    "blocking_big_flexible_unit_count": int(column["blocking_big_flexible_unit_count"]),
                    "avg_time_window_overlap_min": round(float(column["avg_time_window_overlap_min"]), 6),
                    "candidate_score": round(float(column["candidate_score"]), 6),
                    "candidate_family": column["candidate_family"],
                    "pool_pass": column["pool_pass"],
                    "roles": ",".join(column["roles"]),
                    "selected_in_pass1": int(route_key in pass1_keys),
                    "selected_in_pass2": int(route_key in pass2_keys),
                    "selected_in_pass3": int(route_key in pass3_keys),
                    "selected_in_final": int(route_key in active_keys),
                }
            )
        return rows

    def _inner_refine_solution(
        self,
        routes: list[base.TypedRoute],
    ) -> HybridInnerResult | None:
        improved_routes, _ = self.improve_solution(self._clone_routes(routes))
        candidate_eval = self.evaluate_solution(improved_routes)
        if not self._passes_hard_guards(improved_routes, candidate_eval):
            return None

        global_result = self._global_reoptimize_with_milp(improved_routes, candidate_eval)
        summary_by_label = {
            "candidate": {
                "routes": improved_routes,
                "solution_eval": candidate_eval,
                "selected_route_keys": {self._route_key(route) for route in improved_routes},
            },
            "pass1": self._evaluate_global_pass_result(global_result["pass1"]),
            "pass2": self._evaluate_global_pass_result(global_result["pass2"]),
            "pass3": self._evaluate_global_pass_result(global_result["pass3"]),
        }
        feasible_candidates: list[tuple[str, list[base.TypedRoute], base.SolutionEvaluation]] = []
        for label, summary in summary_by_label.items():
            summary_routes = summary["routes"]
            summary_eval = summary["solution_eval"]
            if summary_routes is None or summary_eval is None:
                continue
            if self._passes_hard_guards(summary_routes, summary_eval):
                feasible_candidates.append((label, summary_routes, summary_eval))
        if not feasible_candidates:
            return None
        best_label, best_routes, best_eval = min(
            feasible_candidates,
            key=lambda item: self._global_solution_selection_key(item[1], item[2]),
        )
        route_pool_summary_rows = self._build_route_pool_summary_rows(
            global_result=global_result,
            summary_by_label=summary_by_label,
            active_label=best_label,
        )
        route_pool_role_counts = Counter[str]()
        for column in global_result["columns"]:
            for role in column["roles"]:
                route_pool_role_counts[role] += 1
        phase_statuses = []
        for pass_label in ("pass1", "pass2", "pass3"):
            pass_result = global_result[pass_label]
            if pass_result is not None:
                phase_statuses.extend(pass_result["phase_statuses"])
        return HybridInnerResult(
            routes=self._clone_routes(best_routes),
            solution_eval=best_eval,
            selected_label=best_label,
            phase_statuses=phase_statuses,
            route_pool_summary_rows=route_pool_summary_rows,
            route_pool_candidate_count=len(global_result["columns"]),
            route_pool_role_counts=dict(route_pool_role_counts),
            summary_by_label=summary_by_label,
        )

    def _update_elite_archive(
        self,
        routes: list[base.TypedRoute],
        solution_eval: base.SolutionEvaluation,
        source_label: str,
    ) -> None:
        route_signature = tuple(sorted(self._route_key(route) for route in routes))
        retained: list[tuple[list[base.TypedRoute], base.SolutionEvaluation, str]] = []
        seen_signatures = {route_signature}
        retained.append((self._clone_routes(routes), solution_eval, source_label))
        for archive_routes, archive_eval, archive_label in self.elite_archive:
            archive_signature = tuple(sorted(self._route_key(route) for route in archive_routes))
            if archive_signature in seen_signatures:
                continue
            retained.append((self._clone_routes(archive_routes), archive_eval, archive_label))
            seen_signatures.add(archive_signature)
        retained.sort(key=lambda item: self._outer_solution_key(item[1]))
        self.elite_archive = retained[:HYBRID_ARCHIVE_SIZE]

    def _evaporate_pheromones(self) -> None:
        self.pheromone_edge = {
            key: value * (1.0 - HYBRID_RHO)
            for key, value in self.pheromone_edge.items()
            if value * (1.0 - HYBRID_RHO) > 1e-8
        }
        self.pheromone_start = {
            key: value * (1.0 - HYBRID_RHO)
            for key, value in self.pheromone_start.items()
            if value * (1.0 - HYBRID_RHO) > 1e-8
        }

    def _deposit_solution_pheromone(
        self,
        routes: list[base.TypedRoute],
        solution_eval: base.SolutionEvaluation,
        multiplier: float = 1.0,
    ) -> None:
        delta_tau = multiplier * (1000.0 / max(solution_eval.total_cost, 1e-6))
        for route in routes:
            route_starts, route_edges = self._extract_route_feature_sets(route)
            for start_feature in route_starts:
                self.pheromone_start[start_feature] = self.pheromone_start.get(start_feature, HYBRID_TAU0) + delta_tau
            for edge_feature in route_edges:
                self.pheromone_edge[edge_feature] = self.pheromone_edge.get(edge_feature, HYBRID_TAU0) + delta_tau

    def _update_pheromones(
        self,
        particles: list[HybridParticle],
        global_best: HybridInnerResult,
    ) -> None:
        self._evaporate_pheromones()
        elite_candidates = sorted(
            [(particle.personal_best_routes, particle.personal_best_eval) for particle in particles],
            key=lambda item: self._outer_solution_key(item[1]),
        )[:HYBRID_ELITE_COUNT]
        for routes, solution_eval in elite_candidates:
            self._deposit_solution_pheromone(routes, solution_eval, multiplier=1.0)
        self._deposit_solution_pheromone(global_best.routes, global_best.solution_eval, multiplier=1.5)

    def _run_baseline_reference(self) -> tuple[base.SolutionEvaluation, list[base.TypedRoute], dict[str, object]]:
        baseline_solver = base.Question1Solver(
            workspace=self.workspace,
            input_root=self.input_root,
            output_root=self.output_root,
            seed_list=[self.hybrid_seed],
            max_generations=BASELINE_REFERENCE_GENERATIONS,
            particle_count=BASELINE_REFERENCE_PARTICLES,
            top_route_candidates=self.top_route_candidates,
            enable_split_packing_sensitivity=False,
        )
        return baseline_solver._solve_single_configuration()

    def _build_hybrid_run_record(
        self,
        final_eval: base.SolutionEvaluation,
    ) -> list[dict[str, object]]:
        return [
            {
                "seed": self.hybrid_seed,
                "best_cost": final_eval.total_cost,
                "route_count": final_eval.route_count,
                "used_vehicle_count": final_eval.used_vehicle_count,
                "split_customer_count": final_eval.split_customer_count,
                "single_stop_route_count": final_eval.single_stop_route_count,
                "late_positive_stops": final_eval.late_positive_stops,
                "latest_return_min": final_eval.latest_return_min,
                "initial_feasible_particle_count": self.particle_count,
                "mutation_attempt_count": sum(
                    1 for row in self.outer_search_trace_rows if row["generation"] >= 1
                ),
                "accepted_mutation_count": sum(
                    1 for row in self.outer_search_trace_rows if int(row["accepted_current"]) == 1
                ),
                "best_update_count": sum(
                    1 for row in self.outer_search_trace_rows if int(row["accepted_pbest"]) == 1
                ),
                "operator_usage": dict(self.operator_history),
            }
        ]

    def _build_hybrid_metadata(
        self,
        final_inner: HybridInnerResult,
        baseline_reference_eval: base.SolutionEvaluation,
        baseline_reference_routes: list[base.TypedRoute],
        baseline_reference_metadata: dict[str, object],
        elapsed_sec: float,
        generations_completed: int,
    ) -> dict[str, object]:
        final_routes = final_inner.routes
        final_eval = final_inner.solution_eval
        final_route_counts = self._route_counts(final_routes)
        final_big_diagnostics = self._solution_big_route_diagnostics(final_routes)
        final_pairs_feasible, final_pairs_inventory_feasible = self._current_single_pair_inventory_counts(final_routes)
        baseline_summary = self.baseline_summary
        cost_improved_vs_baseline = (
            float(final_eval.total_cost) + base.COST_IMPROVEMENT_EPS
            < float(baseline_summary.get("total_cost", baseline_reference_eval.total_cost))
        )
        metadata = dict(baseline_reference_metadata)
        metadata.update(
            {
                "best_seed": self.hybrid_seed,
                "elapsed_sec": elapsed_sec,
                "run_records": self._build_hybrid_run_record(final_eval),
                "seed_list": [self.hybrid_seed],
                "seed_count": 1,
                "particle_count": self.particle_count,
                "max_generations": self.max_generations,
                "top_route_candidates": self.top_route_candidates,
                "run_record_count": 1,
                "total_mutation_attempt_count": sum(
                    1 for row in self.outer_search_trace_rows if row["generation"] >= 1
                ),
                "total_accepted_mutation_count": sum(
                    1 for row in self.outer_search_trace_rows if int(row["accepted_current"]) == 1
                ),
                "total_best_update_count": sum(
                    1 for row in self.outer_search_trace_rows if int(row["accepted_pbest"]) == 1
                ),
                "packing_strategy": self.packing_strategy,
                "route_pool_iteration_count": base.COST_FIRST_ROUTE_POOL_ITERATIONS,
                "cost_first_improved": int(cost_improved_vs_baseline),
                "service_unit_count": len(self.service_units),
                "route_cache_size": len(self.route_cache._cache),
                "fuel_3000_used_count": final_route_counts.get("fuel_3000", 0),
                "fuel_3000_free_count": self._fuel_3000_free_count(final_route_counts),
                "single_single_merge_feasible_pair_count": final_pairs_feasible,
                "single_single_merge_inventory_blocked_pair_count": max(
                    final_pairs_feasible - final_pairs_inventory_feasible,
                    0,
                ),
                "final_current_single_pairs_inventory_feasible": final_pairs_inventory_feasible,
                "merge_diagnostics_rows": final_inner.phase_statuses,
                "route_pool_summary_rows": final_inner.route_pool_summary_rows,
                "final_routes_with_flexible_units_on_big": final_big_diagnostics["mixed_big_route_count"],
                "final_flexible_units_on_big_routes": final_big_diagnostics["mixed_big_flexible_unit_count"],
                "final_piggyback_big_count": final_big_diagnostics["piggyback_big_count"],
                "final_promotion_like_big_count": final_big_diagnostics["promotion_like_big_count"],
                "final_blocking_big_flexible_count": final_big_diagnostics["blocking_big_flexible_count"],
                "final_blocking_big_flexible_unit_count": final_big_diagnostics["blocking_big_flexible_unit_count"],
                "route_pool_candidate_count": final_inner.route_pool_candidate_count,
                "route_pool_role_counts": final_inner.route_pool_role_counts,
                "baseline_route_count": int(baseline_summary.get("route_count", baseline_reference_eval.route_count)),
                "baseline_single_stop_route_count": int(
                    baseline_summary.get("single_stop_route_count", baseline_reference_eval.single_stop_route_count)
                ),
                "baseline_total_cost": float(baseline_summary.get("total_cost", baseline_reference_eval.total_cost)),
                "baseline_fuel_3000_used_count": int(
                    baseline_summary.get(
                        "fuel_3000_used_count",
                        self._route_counts(baseline_reference_routes).get("fuel_3000", 0),
                    )
                ),
                "baseline_fuel_3000_free_count": int(
                    baseline_summary.get(
                        "fuel_3000_free_count",
                        self._fuel_3000_free_count(self._route_counts(baseline_reference_routes)),
                    )
                ),
                "baseline_routes_with_flexible_units_on_big": int(
                    baseline_summary.get(
                        "final_routes_with_flexible_units_on_big",
                        baseline_reference_metadata["final_routes_with_flexible_units_on_big"],
                    )
                ),
                "baseline_flexible_units_on_big_routes": int(
                    baseline_summary.get(
                        "final_flexible_units_on_big_routes",
                        baseline_reference_metadata["final_flexible_units_on_big_routes"],
                    )
                ),
                "baseline_piggyback_big_count": int(
                    baseline_summary.get(
                        "final_piggyback_big_count",
                        baseline_reference_metadata["final_piggyback_big_count"],
                    )
                ),
                "baseline_promotion_like_big_count": int(
                    baseline_summary.get(
                        "final_promotion_like_big_count",
                        baseline_reference_metadata["final_promotion_like_big_count"],
                    )
                ),
                "baseline_blocking_big_flexible_count": int(
                    baseline_summary.get(
                        "final_blocking_big_flexible_count",
                        baseline_reference_metadata["final_blocking_big_flexible_count"],
                    )
                ),
                "baseline_blocking_big_flexible_unit_count": int(
                    baseline_summary.get(
                        "final_blocking_big_flexible_unit_count",
                        baseline_reference_metadata["final_blocking_big_flexible_unit_count"],
                    )
                ),
                "global_model_status": "hybrid_ok",
                "global_phase_statuses": final_inner.phase_statuses,
                "global_selected_as_final": 1,
                "global_validation_status": "hybrid_cost_improved" if cost_improved_vs_baseline else "hybrid_no_improvement",
                "global_route_pool_candidate_count": final_inner.route_pool_candidate_count,
                "global_final_total_cost": final_eval.total_cost,
                "global_final_route_count": final_eval.route_count,
                "global_final_single_stop_route_count": final_eval.single_stop_route_count,
                "global_final_current_single_pairs_feasible": final_pairs_feasible,
                "global_final_current_single_pairs_inventory_feasible": final_pairs_inventory_feasible,
                "global_final_routes_with_flexible_units_on_big": final_big_diagnostics["mixed_big_route_count"],
                "global_final_flexible_units_on_big_routes": final_big_diagnostics["mixed_big_flexible_unit_count"],
                "global_final_piggyback_big_count": final_big_diagnostics["piggyback_big_count"],
                "global_final_promotion_like_big_count": final_big_diagnostics["promotion_like_big_count"],
                "global_final_blocking_big_flexible_count": final_big_diagnostics["blocking_big_flexible_count"],
                "global_final_blocking_big_flexible_unit_count": final_big_diagnostics["blocking_big_flexible_unit_count"],
                "global_fuel_3000_used_count": final_route_counts.get("fuel_3000", 0),
                "global_fuel_3000_free_count": self._fuel_3000_free_count(final_route_counts),
                "split_packing_sensitivity_executed": 0,
                "split_packing_sensitivity_status": "not_run_hybrid",
                "split_packing_sensitivity_total_cost": None,
                "split_packing_sensitivity_route_count": None,
                "split_packing_sensitivity_reference_total_cost": None,
                "split_packing_sensitivity_reference_route_count": None,
                "final_solution_source": f"hybrid_{final_inner.selected_label}",
                "hybrid_outer_seed": self.hybrid_seed,
                "hybrid_mode": self.hybrid_mode,
                "hybrid_generations_completed": generations_completed,
                "hybrid_outer_trace_count": len(self.outer_search_trace_rows),
                "hybrid_pheromone_edge_count": len(self.pheromone_edge),
                "hybrid_pheromone_start_count": len(self.pheromone_start),
                "hybrid_elite_archive_size": len(self.elite_archive),
                "hybrid_baseline_reference_total_cost": baseline_reference_eval.total_cost,
                "hybrid_baseline_reference_route_count": baseline_reference_eval.route_count,
                "hybrid_baseline_reference_single_stop_route_count": baseline_reference_eval.single_stop_route_count,
            }
        )
        return metadata

    def _rename_base_outputs(self) -> None:
        rename_map = {
            "q1_route_summary.csv": "q1_hybrid_route_summary.csv",
            "q1_stop_schedule.csv": "q1_hybrid_stop_schedule.csv",
            "q1_vehicle_schedule.csv": "q1_hybrid_vehicle_schedule.csv",
            "q1_customer_aggregate.csv": "q1_hybrid_customer_aggregate.csv",
            "q1_cost_summary.json": "q1_hybrid_cost_summary.json",
            "q1_run_report.md": "q1_hybrid_run_report.md",
            "q1_service_units.csv": "q1_hybrid_service_units.csv",
            "q1_split_plan.csv": "q1_hybrid_split_plan.csv",
            "q1_route_pool_summary.csv": "q1_hybrid_route_pool_summary.csv",
            "q1_merge_diagnostics.csv": "q1_hybrid_merge_diagnostics.csv",
        }
        for source_name, target_name in rename_map.items():
            source_path = self.output_root / source_name
            if source_path.exists():
                target_path = self.output_root / target_name
                if target_path.exists():
                    target_path.unlink()
                source_path.rename(target_path)

    def _write_outer_search_trace(self) -> None:
        trace_df = pd.DataFrame(self.outer_search_trace_rows)
        trace_df.to_csv(
            self.output_root / "q1_hybrid_outer_search_trace.csv",
            index=False,
            encoding=CSV_ENCODING,
        )

    def _write_pheromone_top_edges(self) -> None:
        rows: list[dict[str, object]] = []
        for (unit_id, vehicle_family), tau_value in self.pheromone_start.items():
            rows.append(
                {
                    "feature_type": "start",
                    "left_unit_id": unit_id,
                    "right_unit_id": None,
                    "vehicle_family": vehicle_family,
                    "tau": tau_value,
                }
            )
        for (left_unit_id, right_unit_id, vehicle_family), tau_value in self.pheromone_edge.items():
            rows.append(
                {
                    "feature_type": "edge",
                    "left_unit_id": left_unit_id,
                    "right_unit_id": right_unit_id,
                    "vehicle_family": vehicle_family,
                    "tau": tau_value,
                }
            )
        rows.sort(key=lambda item: (-float(item["tau"]), str(item["feature_type"]), int(item["left_unit_id"])))
        pd.DataFrame(rows[:200]).to_csv(
            self.output_root / "q1_hybrid_pheromone_top_edges.csv",
            index=False,
            encoding=CSV_ENCODING,
        )

    def _write_compare_to_baseline(
        self,
        final_eval: base.SolutionEvaluation,
        metadata: dict[str, object],
    ) -> None:
        baseline_summary = self.baseline_summary
        compare_payload = {
            "baseline_total_cost": float(baseline_summary.get("total_cost", metadata["baseline_total_cost"])),
            "hybrid_total_cost": float(final_eval.total_cost),
            "baseline_route_count": int(baseline_summary.get("route_count", metadata["baseline_route_count"])),
            "hybrid_route_count": int(final_eval.route_count),
            "baseline_single_stop_route_count": int(
                baseline_summary.get("single_stop_route_count", metadata["baseline_single_stop_route_count"])
            ),
            "hybrid_single_stop_route_count": int(final_eval.single_stop_route_count),
            "baseline_late_positive_stops": int(
                baseline_summary.get("late_positive_stops", baseline_summary.get("late_positive_stop_count", 0))
            ),
            "hybrid_late_positive_stops": int(final_eval.late_positive_stops),
            "baseline_max_late_min": float(baseline_summary.get("max_late_min", 0.0)),
            "hybrid_max_late_min": float(final_eval.max_late_min),
            "baseline_vehicle_type_usage": baseline_summary.get("vehicle_type_usage", {}),
            "hybrid_vehicle_type_usage": final_eval.vehicle_type_usage,
            "baseline_final_solution_source": baseline_summary.get("final_solution_source", "baseline"),
            "hybrid_final_solution_source": metadata["final_solution_source"],
        }
        (self.output_root / "q1_hybrid_compare_to_baseline.json").write_text(
            json.dumps(compare_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _augment_hybrid_cost_summary_and_report(
        self,
        final_eval: base.SolutionEvaluation,
        metadata: dict[str, object],
    ) -> None:
        cost_summary_path = self.output_root / "q1_hybrid_cost_summary.json"
        report_path = self.output_root / "q1_hybrid_run_report.md"
        if cost_summary_path.exists():
            cost_summary = json.loads(cost_summary_path.read_text(encoding="utf-8"))
            cost_summary.update(
                {
                    "hybrid_outer_seed": metadata["hybrid_outer_seed"],
                    "hybrid_mode": metadata["hybrid_mode"],
                    "hybrid_generations_completed": metadata["hybrid_generations_completed"],
                    "hybrid_outer_trace_count": metadata["hybrid_outer_trace_count"],
                    "hybrid_pheromone_edge_count": metadata["hybrid_pheromone_edge_count"],
                    "hybrid_pheromone_start_count": metadata["hybrid_pheromone_start_count"],
                    "hybrid_elite_archive_size": metadata["hybrid_elite_archive_size"],
                    "hybrid_baseline_reference_total_cost": metadata["hybrid_baseline_reference_total_cost"],
                    "hybrid_baseline_reference_route_count": metadata["hybrid_baseline_reference_route_count"],
                    "hybrid_baseline_reference_single_stop_route_count": metadata["hybrid_baseline_reference_single_stop_route_count"],
                }
            )
            cost_summary_path.write_text(json.dumps(cost_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if report_path.exists():
            report_lines = report_path.read_text(encoding="utf-8").splitlines()
            report_lines.extend(
                [
                    "",
                    "## Hybrid Outer Search",
                    f"- Hybrid mode: {metadata['hybrid_mode']}",
                    f"- Hybrid outer seed: {metadata['hybrid_outer_seed']}",
                    f"- Generations completed: {metadata['hybrid_generations_completed']}",
                    f"- Outer trace rows: {metadata['hybrid_outer_trace_count']}",
                    f"- Pheromone edge/start count: {metadata['hybrid_pheromone_edge_count']}/{metadata['hybrid_pheromone_start_count']}",
                    f"- Elite archive size: {metadata['hybrid_elite_archive_size']}",
                    f"- Baseline reference total cost/route count/single-stop: "
                    f"{metadata['hybrid_baseline_reference_total_cost']:.3f}/"
                    f"{metadata['hybrid_baseline_reference_route_count']}/"
                    f"{metadata['hybrid_baseline_reference_single_stop_route_count']}",
                    f"- Hybrid final total cost/route count/single-stop: "
                    f"{final_eval.total_cost:.3f}/{final_eval.route_count}/{final_eval.single_stop_route_count}",
                ]
            )
            report_path.write_text("\n".join(report_lines), encoding="utf-8")

    def _write_hybrid_outputs(
        self,
        final_eval: base.SolutionEvaluation,
        final_routes: list[base.TypedRoute],
        metadata: dict[str, object],
    ) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        super()._write_outputs(final_eval, final_routes, metadata)
        self._rename_base_outputs()
        self._write_outer_search_trace()
        self._write_pheromone_top_edges()
        self._write_compare_to_baseline(final_eval, metadata)
        self._augment_hybrid_cost_summary_and_report(final_eval, metadata)

    def solve(self) -> tuple[base.SolutionEvaluation, dict[str, object]]:
        search_start = time.perf_counter()
        baseline_reference_eval, baseline_reference_routes, baseline_reference_metadata = self._run_baseline_reference()
        baseline_inner = HybridInnerResult(
            routes=self._clone_routes(baseline_reference_routes),
            solution_eval=baseline_reference_eval,
            selected_label="baseline_reference",
            phase_statuses=list(baseline_reference_metadata.get("global_phase_statuses", [])),
            route_pool_summary_rows=list(baseline_reference_metadata.get("route_pool_summary_rows", [])),
            route_pool_candidate_count=int(baseline_reference_metadata.get("route_pool_candidate_count", 0)),
            route_pool_role_counts=dict(baseline_reference_metadata.get("route_pool_role_counts", {})),
            summary_by_label={},
        )
        global_best = baseline_inner
        self._update_elite_archive(global_best.routes, global_best.solution_eval, global_best.selected_label)
        self._deposit_solution_pheromone(global_best.routes, global_best.solution_eval, multiplier=1.0)

        master_rng = random.Random(self.hybrid_seed)
        particles: list[HybridParticle] = []
        for particle_id in range(self.particle_count):
            particle_rng = random.Random(master_rng.randint(1, 10**9))
            gbest_features = self._extract_solution_feature_sets(global_best.routes)
            candidate_routes = self._build_probabilistic_initial_solution(
                rng=particle_rng,
                temperature=1.0,
                pbest_features=None,
                gbest_features=gbest_features,
            )
            inner_result = self._inner_refine_solution(candidate_routes)
            if inner_result is None:
                inner_result = baseline_inner
            particle = HybridParticle(
                particle_id=particle_id,
                current_routes=self._clone_routes(inner_result.routes),
                current_eval=inner_result.solution_eval,
                personal_best_routes=self._clone_routes(inner_result.routes),
                personal_best_eval=inner_result.solution_eval,
                destroy_ratio=0.12 + 0.02 * particle_id,
                construction_temperature=1.0,
                operator_weights=self._normalize_operator_weights({name: 1.0 for name in OUTER_OPERATOR_NAMES}),
                cauchy_scale=HYBRID_CAUCHY_GAMMA0,
            )
            particles.append(particle)
            if self._is_outer_better(inner_result.solution_eval, global_best.solution_eval):
                global_best = inner_result
            self._update_elite_archive(inner_result.routes, inner_result.solution_eval, f"init_particle_{particle_id}")
            self.outer_search_trace_rows.append(
                {
                    "generation": 0,
                    "particle_id": particle_id,
                    "source_mode": "construct",
                    "operator_name": "construct",
                    "destroy_ratio": round(particle.destroy_ratio, 6),
                    "construction_temperature": round(particle.construction_temperature, 6),
                    "candidate_total_cost": round(inner_result.solution_eval.total_cost, 6),
                    "candidate_route_count": inner_result.solution_eval.route_count,
                    "candidate_single_stop_route_count": inner_result.solution_eval.single_stop_route_count,
                    "candidate_label": inner_result.selected_label,
                    "accepted_current": 1,
                    "accepted_pbest": 1,
                    "accepted_gbest": int(inner_result.solution_eval is global_best.solution_eval),
                }
            )

        generations_completed = 0
        stagnant_generations = 0
        for generation in range(1, self.max_generations + 1):
            generation_improved = False
            for particle in particles:
                particle_rng = random.Random(master_rng.randint(1, 10**9))
                self._mutate_particle_parameters(particle, generation, particle_rng)
                source_mode, source_routes, source_eval = self._choose_source_solution(particle, global_best, particle_rng)
                partial_routes, removed_units, operator_name = self._destroy_solution(
                    routes=source_routes,
                    solution_eval=source_eval,
                    destroy_ratio=particle.destroy_ratio,
                    operator_weights=particle.operator_weights,
                    rng=particle_rng,
                )
                pbest_features = self._extract_solution_feature_sets(particle.personal_best_routes)
                gbest_features = self._extract_solution_feature_sets(global_best.routes)
                repaired_routes = self._repair_solution_probabilistic(
                    partial_routes=partial_routes,
                    removed_units=removed_units,
                    rng=particle_rng,
                    temperature=particle.construction_temperature,
                    pbest_features=pbest_features,
                    gbest_features=gbest_features,
                )
                accepted_current = 0
                accepted_pbest = 0
                accepted_gbest = 0
                candidate_total_cost = None
                candidate_route_count = None
                candidate_single_stop_route_count = None
                candidate_label = None
                if repaired_routes is not None:
                    inner_result = self._inner_refine_solution(repaired_routes)
                    if inner_result is not None:
                        candidate_total_cost = round(inner_result.solution_eval.total_cost, 6)
                        candidate_route_count = inner_result.solution_eval.route_count
                        candidate_single_stop_route_count = inner_result.solution_eval.single_stop_route_count
                        candidate_label = inner_result.selected_label
                        if self._is_outer_better(inner_result.solution_eval, particle.current_eval):
                            particle.current_routes = self._clone_routes(inner_result.routes)
                            particle.current_eval = inner_result.solution_eval
                            accepted_current = 1
                        if self._is_outer_better(inner_result.solution_eval, particle.personal_best_eval):
                            particle.personal_best_routes = self._clone_routes(inner_result.routes)
                            particle.personal_best_eval = inner_result.solution_eval
                            accepted_pbest = 1
                            generation_improved = True
                            self._update_elite_archive(
                                inner_result.routes,
                                inner_result.solution_eval,
                                f"particle_{particle.particle_id}_gen_{generation}",
                            )
                        if self._is_outer_better(inner_result.solution_eval, global_best.solution_eval):
                            global_best = inner_result
                            accepted_gbest = 1
                            generation_improved = True
                            self._update_elite_archive(
                                inner_result.routes,
                                inner_result.solution_eval,
                                f"global_best_gen_{generation}",
                            )
                self.outer_search_trace_rows.append(
                    {
                        "generation": generation,
                        "particle_id": particle.particle_id,
                        "source_mode": source_mode,
                        "operator_name": operator_name,
                        "destroy_ratio": round(particle.destroy_ratio, 6),
                        "construction_temperature": round(particle.construction_temperature, 6),
                        "candidate_total_cost": candidate_total_cost,
                        "candidate_route_count": candidate_route_count,
                        "candidate_single_stop_route_count": candidate_single_stop_route_count,
                        "candidate_label": candidate_label,
                        "accepted_current": accepted_current,
                        "accepted_pbest": accepted_pbest,
                        "accepted_gbest": accepted_gbest,
                    }
                )
            generations_completed = generation
            self._update_pheromones(particles, global_best)
            if generation_improved:
                stagnant_generations = 0
            else:
                stagnant_generations += 1
                if stagnant_generations >= OUTER_NO_IMPROVEMENT_STOP:
                    break

        elapsed_sec = time.perf_counter() - search_start
        final_metadata = self._build_hybrid_metadata(
            final_inner=global_best,
            baseline_reference_eval=baseline_reference_eval,
            baseline_reference_routes=baseline_reference_routes,
            baseline_reference_metadata=baseline_reference_metadata,
            elapsed_sec=elapsed_sec,
            generations_completed=generations_completed,
        )
        self._write_hybrid_outputs(global_best.solution_eval, global_best.routes, final_metadata)
        return global_best.solution_eval, final_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve question 1 with a hybrid outer-search + cost-first inner optimizer.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--input-root", type=Path, default=Path.cwd() / "preprocess_artifacts")
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "question1_artifacts_hybrid")
    parser.add_argument("--baseline-summary", type=Path, default=Path.cwd() / "question1_artifacts" / "q1_cost_summary.json")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--top-route-candidates", type=int, default=8)
    parser.add_argument("--mode", choices=["minimal", "standard"], default="minimal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    particles = args.particles
    generations = args.generations
    if particles is None:
        particles = DEFAULT_MINIMAL_PARTICLES if args.mode == "minimal" else DEFAULT_STANDARD_PARTICLES
    if generations is None:
        generations = DEFAULT_MINIMAL_GENERATIONS if args.mode == "minimal" else DEFAULT_STANDARD_GENERATIONS

    solver = HybridQuestion1Solver(
        workspace=args.workspace,
        input_root=args.input_root,
        output_root=args.output_root,
        baseline_summary_path=args.baseline_summary,
        seed=args.seed,
        particles=particles,
        generations=generations,
        top_route_candidates=args.top_route_candidates,
        mode=args.mode,
    )
    best_solution, metadata = solver.solve()
    print(
        json.dumps(
            {
                "total_cost": best_solution.total_cost,
                "route_count": best_solution.route_count,
                "single_stop_route_count": best_solution.single_stop_route_count,
                "split_customer_count": best_solution.split_customer_count,
                "late_positive_stops": best_solution.late_positive_stops,
                "max_late_min": best_solution.max_late_min,
                "hybrid_outer_seed": metadata["hybrid_outer_seed"],
                "hybrid_mode": metadata["hybrid_mode"],
                "hybrid_generations_completed": metadata["hybrid_generations_completed"],
                "output_root": str(args.output_root),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
