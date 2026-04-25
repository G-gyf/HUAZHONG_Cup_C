from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import solve_question1 as q1
import solve_question1_hybrid as hybrid
import solve_question2 as q2
import solve_question2_costfirst_v2 as q2v2


CSV_ENCODING = q1.CSV_ENCODING
DEFAULT_MINIMAL_PARTICLES = hybrid.DEFAULT_MINIMAL_PARTICLES
DEFAULT_MINIMAL_GENERATIONS = hybrid.DEFAULT_MINIMAL_GENERATIONS


class HybridQuestion2Solver(hybrid.HybridQuestion1Solver, q2v2.Question2CostFirstV2Solver):
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
        q2.Question2Solver.__init__(
            self,
            workspace=workspace,
            input_root=input_root,
            output_root=output_root,
            seed_list=[seed],
            max_generations=generations,
            particle_count=particles,
            top_route_candidates=top_route_candidates,
        )
        self.hybrid_seed = seed
        self.hybrid_mode = mode
        self.baseline_summary_path = baseline_summary_path
        self.outer_search_trace_rows: list[dict[str, object]] = []
        self.pheromone_edge: dict[tuple[int, int, str], float] = {}
        self.pheromone_start: dict[tuple[int, str], float] = {}
        self.elite_archive: list[tuple[list[q1.TypedRoute], q1.SolutionEvaluation, str]] = []
        self.observed_results: list[hybrid.HybridInnerResult] = []
        self.operator_history = q1.Counter[str]()
        self.cluster_remove_attempt_count = 0
        self.cluster_remove_accepted_count = 0
        self.baseline_summary = self._load_baseline_summary()

    def _budget_signature(self) -> str:
        return (
            f"seeds={self.hybrid_seed}|"
            f"particles={self.particle_count}|"
            f"generations={self.max_generations}|"
            f"top={self.top_route_candidates}"
        )

    def _rank_key_penalty(
        self,
        rank_key: tuple[int, int, int, int, int, int, float, tuple[float, float, int]],
    ) -> float:
        ordinary_ev_penalty = 220.0 * float(rank_key[0])
        ordinary_big_ev_penalty = 140.0 * float(rank_key[1])
        late_fuel_penalty = 180.0 * float(rank_key[2])
        late_cluster_penalty = 90.0 * float(rank_key[3])
        must_ev_small_penalty = 60.0 * float(rank_key[4])
        must_ev_non_ev_penalty = 400.0 * float(rank_key[5])
        vehicle_penalty = 0.01 * float(rank_key[7][0]) + 0.1 * float(rank_key[7][1]) + 5.0 * float(rank_key[7][2])
        return (
            ordinary_ev_penalty
            + ordinary_big_ev_penalty
            + late_fuel_penalty
            + late_cluster_penalty
            + must_ev_small_penalty
            + must_ev_non_ev_penalty
            + vehicle_penalty
        )

    def _build_weighted_options_for_unit(
        self,
        routes: list[q1.TypedRoute],
        unit_id: int,
        temperature: float,
        pbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
        gbest_features: tuple[set[tuple[int, str]], set[tuple[int, int, str]]] | None,
    ) -> list[dict[str, object]]:
        unit = self.unit_by_id[unit_id]
        route_counts = self._route_counts(routes)
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
                    candidate_route = q1.TypedRoute(vehicle_type=route.vehicle_type, unit_ids=new_unit_ids)
                    candidate_eval = self.evaluate_route(candidate_route)
                    if not candidate_eval.feasible:
                        continue
                    delta_cost = float(candidate_eval.best_cost - base_eval.best_cost)
                    rank_key = self._q2_choice_rank_key(
                        route_counts=route_counts,
                        current_vehicle_type=route.vehicle_type,
                        candidate_vehicle_type=route.vehicle_type,
                        unit_id=unit_id,
                        candidate_route=candidate_route,
                        delta_cost=delta_cost,
                    )
                    updated_routes = self._clone_routes(routes)
                    updated_routes[route_idx] = candidate_route
                    tau_avg = self._route_tau_average(candidate_route)
                    guide_bonus = self._guide_bonus(candidate_route, pbest_features, gbest_features)
                    generalized_delta_cost = max(delta_cost, 0.0) + self._rank_key_penalty(rank_key)
                    weight = (max(tau_avg, 1e-9) ** hybrid.HYBRID_ALPHA) * math.exp(
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
                options.extend(group_options[: hybrid.HYBRID_PROBABILITY_SHORTLIST])
                break

        new_route_options: list[dict[str, object]] = []
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
                delta_cost=float(candidate_eval.best_cost),
            )
            tau_avg = self._route_tau_average(candidate_route)
            guide_bonus = self._guide_bonus(candidate_route, pbest_features, gbest_features)
            generalized_delta_cost = float(candidate_eval.best_cost) + self._rank_key_penalty(rank_key)
            weight = (max(tau_avg, 1e-9) ** hybrid.HYBRID_ALPHA) * math.exp(
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
            options.extend(new_route_options[: max(1, hybrid.HYBRID_PROBABILITY_SHORTLIST // 2)])

        options.sort(key=lambda item: (float(item["generalized_delta_cost"]), -float(item["tau_avg"])))
        return options[: hybrid.HYBRID_PROBABILITY_SHORTLIST]

    def _passes_hard_guards(
        self,
        solution_routes: list[q1.TypedRoute],
        solution_eval: q1.SolutionEvaluation | None,
    ) -> bool:
        if not hybrid.HybridQuestion1Solver._passes_hard_guards(self, solution_routes, solution_eval):
            return False
        policy_diagnostics = self._policy_solution_diagnostics(solution_eval)
        return (
            int(policy_diagnostics["policy_violation_count"]) == 0
            and int(policy_diagnostics["mandatory_ev_served_by_non_ev_count"]) == 0
            and int(policy_diagnostics["fuel_route_green_zone_pre16_visit_count"]) == 0
        )

    def _run_baseline_reference(self) -> tuple[q1.SolutionEvaluation, list[q1.TypedRoute], dict[str, object]]:
        baseline_solver = q2v2.Question2CostFirstV2Solver(
            workspace=self.workspace,
            input_root=self.input_root,
            output_root=self.output_root,
            seed_list=[self.hybrid_seed],
            max_generations=hybrid.BASELINE_REFERENCE_GENERATIONS,
            particle_count=hybrid.BASELINE_REFERENCE_PARTICLES,
            top_route_candidates=self.top_route_candidates,
        )
        return baseline_solver._solve_single_configuration()

    def _result_specific_metadata(
        self,
        metadata: dict[str, object],
        inner_result: hybrid.HybridInnerResult,
        route_pool_rows: list[dict[str, object]],
        final_source_name: str,
    ) -> dict[str, object]:
        result_metadata = hybrid.HybridQuestion1Solver._result_specific_metadata(
            self,
            metadata=metadata,
            inner_result=inner_result,
            route_pool_rows=route_pool_rows,
            final_source_name=final_source_name,
        )
        result_metadata.update(self._policy_solution_diagnostics(inner_result.solution_eval))
        result_metadata["applied_budget_signature"] = self._budget_signature()
        return result_metadata

    def _build_hybrid_metadata(
        self,
        cost_best: hybrid.HybridInnerResult,
        balanced_best: hybrid.HybridInnerResult,
        baseline_reference_eval: q1.SolutionEvaluation,
        baseline_reference_routes: list[q1.TypedRoute],
        baseline_reference_metadata: dict[str, object],
        elapsed_sec: float,
        generations_completed: int,
    ) -> dict[str, object]:
        metadata = hybrid.HybridQuestion1Solver._build_hybrid_metadata(
            self,
            cost_best=cost_best,
            balanced_best=balanced_best,
            baseline_reference_eval=baseline_reference_eval,
            baseline_reference_routes=baseline_reference_routes,
            baseline_reference_metadata=baseline_reference_metadata,
            elapsed_sec=elapsed_sec,
            generations_completed=generations_completed,
        )
        metadata.update(self._policy_solution_diagnostics(cost_best.solution_eval))
        metadata["applied_budget_signature"] = self._budget_signature()
        finished_at = datetime.now().astimezone()
        started_at = finished_at - timedelta(seconds=float(elapsed_sec))
        metadata["run_started_at"] = started_at.isoformat(timespec="seconds")
        metadata["run_finished_at"] = finished_at.isoformat(timespec="seconds")
        return metadata

    def _rename_base_outputs(self) -> None:
        rename_map = {
            "q2_route_summary.csv": "q2_hybrid_route_summary.csv",
            "q2_stop_schedule.csv": "q2_hybrid_stop_schedule.csv",
            "q2_vehicle_schedule.csv": "q2_hybrid_vehicle_schedule.csv",
            "q2_customer_aggregate.csv": "q2_hybrid_customer_aggregate.csv",
            "q2_cost_summary.json": "q2_hybrid_cost_summary.json",
            "q2_run_report.md": "q2_hybrid_run_report.md",
            "q2_service_units.csv": "q2_hybrid_service_units.csv",
            "q2_split_plan.csv": "q2_hybrid_split_plan.csv",
            "q2_route_pool_summary.csv": "q2_hybrid_route_pool_summary.csv",
            "q2_merge_diagnostics.csv": "q2_hybrid_merge_diagnostics.csv",
        }
        for source_name, target_name in rename_map.items():
            source_path = self.output_root / source_name
            if source_path.exists():
                target_path = self.output_root / target_name
                if target_path.exists():
                    target_path.unlink()
                source_path.rename(target_path)
        policy_summary_path = self.output_root / "q2_policy_summary.json"
        if policy_summary_path.exists():
            policy_summary_path.unlink()

    def _write_outer_search_trace(self) -> None:
        pd.DataFrame(self.outer_search_trace_rows).to_csv(
            self.output_root / "q2_hybrid_outer_search_trace.csv",
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
            self.output_root / "q2_hybrid_pheromone_top_edges.csv",
            index=False,
            encoding=CSV_ENCODING,
        )

    def _write_compare_to_baseline(
        self,
        cost_best: hybrid.HybridInnerResult,
        balanced_best: hybrid.HybridInnerResult,
        metadata: dict[str, object],
    ) -> None:
        baseline_summary = self.baseline_summary
        baseline_total_cost = float(baseline_summary.get("total_cost", metadata["baseline_total_cost"]))
        compare_payload = {
            "baseline_label": "question2_artifacts_costfirst_v2",
            "baseline_total_cost": baseline_total_cost,
            "baseline_route_count": int(baseline_summary.get("route_count", metadata["baseline_route_count"])),
            "baseline_single_stop_route_count": int(
                baseline_summary.get("single_stop_route_count", metadata["baseline_single_stop_route_count"])
            ),
            "baseline_late_positive_stops": int(baseline_summary.get("late_positive_stops", 0)),
            "baseline_max_late_min": float(baseline_summary.get("max_late_min", 0.0)),
            "baseline_policy_violation_count": int(baseline_summary.get("policy_violation_count", 0)),
            "baseline_mandatory_ev_served_by_non_ev_count": int(
                baseline_summary.get("mandatory_ev_served_by_non_ev_count", 0)
            ),
            "baseline_fuel_route_green_zone_pre16_visit_count": int(
                baseline_summary.get("fuel_route_green_zone_pre16_visit_count", 0)
            ),
            "cost_best_total_cost": float(cost_best.solution_eval.total_cost),
            "cost_best_route_count": int(cost_best.solution_eval.route_count),
            "cost_best_single_stop_route_count": int(cost_best.solution_eval.single_stop_route_count),
            "cost_best_late_positive_stops": int(cost_best.solution_eval.late_positive_stops),
            "cost_best_max_late_min": float(cost_best.solution_eval.max_late_min),
            "cost_best_policy_violation_count": int(metadata["policy_violation_count"]),
            "cost_best_mandatory_ev_served_by_non_ev_count": int(metadata["mandatory_ev_served_by_non_ev_count"]),
            "cost_best_fuel_route_green_zone_pre16_visit_count": int(
                metadata["fuel_route_green_zone_pre16_visit_count"]
            ),
            "cost_best_final_solution_source": metadata["final_solution_source"],
            "cost_improvement_vs_baseline": float(baseline_total_cost - float(cost_best.solution_eval.total_cost)),
            "balanced_best_total_cost": float(balanced_best.solution_eval.total_cost),
            "balanced_best_route_count": int(balanced_best.solution_eval.route_count),
            "balanced_best_single_stop_route_count": int(balanced_best.solution_eval.single_stop_route_count),
            "balanced_best_late_positive_stops": int(balanced_best.solution_eval.late_positive_stops),
            "balanced_best_max_late_min": float(balanced_best.solution_eval.max_late_min),
            "balanced_best_final_solution_source": metadata["balanced_final_solution_source"],
        }
        (self.output_root / "q2_hybrid_compare_to_baseline.json").write_text(
            json.dumps(compare_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _normalize_hybrid_bundle_outputs(self, output_root: Path) -> None:
        cost_summary_path = output_root / "q2_hybrid_cost_summary.json"
        if cost_summary_path.exists():
            cost_summary = json.loads(cost_summary_path.read_text(encoding="utf-8"))
            for field_name in (
                "q2_baseline_total_cost",
                "q2_baseline_route_count",
                "q2_baseline_single_stop_route_count",
                "cost_improvement_vs_q2_baseline",
            ):
                cost_summary.pop(field_name, None)
            cost_summary["applied_budget_signature"] = self._budget_signature()
            cost_summary_path.write_text(json.dumps(cost_summary, indent=2, ensure_ascii=False), encoding="utf-8")

        report_path = output_root / "q2_hybrid_run_report.md"
        if report_path.exists():
            report_lines = report_path.read_text(encoding="utf-8").splitlines()
            if report_lines and report_lines[0].startswith("# "):
                report_lines[0] = (
                    "# Question 2 Hybrid Balanced Report"
                    if output_root.name == "balanced"
                    else "# Question 2 Hybrid Report"
                )
            rewritten_lines: list[str] = []
            baseline_total_cost = self.baseline_summary.get("total_cost")
            for line in report_lines:
                if line.strip() == "## Q2 v2 Summary":
                    rewritten_lines.append("## Q2 Hybrid Inner Summary")
                    continue
                if line.startswith("- Cost improvement vs Q2 baseline:"):
                    improvement = None
                    if baseline_total_cost is not None and cost_summary_path.exists():
                        improvement = float(baseline_total_cost) - float(
                            json.loads(cost_summary_path.read_text(encoding="utf-8"))["total_cost"]
                        )
                    rewritten_lines.append(
                        f"- Cost improvement vs Q2 cost-first v2 baseline: {improvement}"
                    )
                    continue
                rewritten_lines.append(line)
            report_path.write_text("\n".join(rewritten_lines), encoding="utf-8")

    def _augment_hybrid_cost_summary_and_report(
        self,
        cost_best: hybrid.HybridInnerResult,
        balanced_best: hybrid.HybridInnerResult,
        metadata: dict[str, object],
    ) -> None:
        cost_summary_path = self.output_root / "q2_hybrid_cost_summary.json"
        report_path = self.output_root / "q2_hybrid_run_report.md"
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
                    "hybrid_baseline_reference_single_stop_route_count": metadata[
                        "hybrid_baseline_reference_single_stop_route_count"
                    ],
                    "cost_best_total_cost": metadata["cost_best_total_cost"],
                    "cost_best_route_count": metadata["cost_best_route_count"],
                    "cost_best_single_stop_route_count": metadata["cost_best_single_stop_route_count"],
                    "cost_best_late_positive_stops": metadata["cost_best_late_positive_stops"],
                    "cost_best_max_late_min": metadata["cost_best_max_late_min"],
                    "balanced_best_total_cost": metadata["balanced_best_total_cost"],
                    "balanced_best_route_count": metadata["balanced_best_route_count"],
                    "balanced_best_single_stop_route_count": metadata["balanced_best_single_stop_route_count"],
                    "balanced_best_late_positive_stops": metadata["balanced_best_late_positive_stops"],
                    "balanced_best_max_late_min": metadata["balanced_best_max_late_min"],
                    "archive_injected_route_count": metadata["archive_injected_route_count"],
                    "archive_selected_column_count": metadata["archive_selected_column_count"],
                    "pheromone_bonus_selected_column_count": metadata["pheromone_bonus_selected_column_count"],
                    "cluster_remove_attempt_count": metadata["cluster_remove_attempt_count"],
                    "cluster_remove_accepted_count": metadata["cluster_remove_accepted_count"],
                    "balanced_final_solution_source": metadata["balanced_final_solution_source"],
                    "applied_budget_signature": self._budget_signature(),
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
                    f"- Pheromone edge/start count: "
                    f"{metadata['hybrid_pheromone_edge_count']}/{metadata['hybrid_pheromone_start_count']}",
                    f"- Elite archive size: {metadata['hybrid_elite_archive_size']}",
                    f"- Cost best total cost/route count/single-stop/late+/maxlate: "
                    f"{cost_best.solution_eval.total_cost:.3f}/{cost_best.solution_eval.route_count}/"
                    f"{cost_best.solution_eval.single_stop_route_count}/{cost_best.solution_eval.late_positive_stops}/"
                    f"{cost_best.solution_eval.max_late_min:.3f}",
                    f"- Balanced best total cost/route count/single-stop/late+/maxlate: "
                    f"{balanced_best.solution_eval.total_cost:.3f}/{balanced_best.solution_eval.route_count}/"
                    f"{balanced_best.solution_eval.single_stop_route_count}/"
                    f"{balanced_best.solution_eval.late_positive_stops}/"
                    f"{balanced_best.solution_eval.max_late_min:.3f}",
                    f"- Archive injected route count: {metadata['archive_injected_route_count']}",
                    f"- Archive/pheromone selected column count: "
                    f"{metadata['archive_selected_column_count']}/{metadata['pheromone_bonus_selected_column_count']}",
                    f"- Cluster remove attempts/accepted: "
                    f"{metadata['cluster_remove_attempt_count']}/{metadata['cluster_remove_accepted_count']}",
                ]
            )
            report_path.write_text("\n".join(report_lines), encoding="utf-8")

    def _write_single_hybrid_bundle(
        self,
        output_root: Path,
        inner_result: hybrid.HybridInnerResult,
        metadata: dict[str, object],
    ) -> None:
        original_output_root = self.output_root
        try:
            self.output_root = output_root
            self.output_root.mkdir(parents=True, exist_ok=True)
            q2v2.Question2CostFirstV2Solver._write_outputs(self, inner_result.solution_eval, inner_result.routes, metadata)
            self._rename_base_outputs()
            self._normalize_hybrid_bundle_outputs(self.output_root)
        finally:
            self.output_root = original_output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve question 2 with a hybrid outer search and Q2 v2 inner optimizer.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--input-root", type=Path, default=Path.cwd() / "preprocess_artifacts")
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "question2_artifacts_hybrid_minimal")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path.cwd() / "question2_artifacts_costfirst_v2" / "q2_cost_summary.json",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--particles", type=int, default=DEFAULT_MINIMAL_PARTICLES)
    parser.add_argument("--generations", type=int, default=DEFAULT_MINIMAL_GENERATIONS)
    parser.add_argument("--top-route-candidates", type=int, default=8)
    parser.add_argument("--mode", choices=["minimal"], default="minimal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solver = HybridQuestion2Solver(
        workspace=args.workspace,
        input_root=args.input_root,
        output_root=args.output_root,
        baseline_summary_path=args.baseline_summary,
        seed=args.seed,
        particles=args.particles,
        generations=args.generations,
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
                "mandatory_split_customer_count": best_solution.mandatory_split_customer_count,
                "late_positive_stops": best_solution.late_positive_stops,
                "max_late_min": best_solution.max_late_min,
                "policy_violation_count": metadata["policy_violation_count"],
                "mandatory_ev_served_by_non_ev_count": metadata["mandatory_ev_served_by_non_ev_count"],
                "fuel_route_green_zone_pre16_visit_count": metadata["fuel_route_green_zone_pre16_visit_count"],
                "balanced_best_total_cost": metadata["balanced_best_total_cost"],
                "balanced_best_late_positive_stops": metadata["balanced_best_late_positive_stops"],
                "balanced_best_max_late_min": metadata["balanced_best_max_late_min"],
                "hybrid_outer_seed": metadata["hybrid_outer_seed"],
                "hybrid_mode": metadata["hybrid_mode"],
                "hybrid_generations_completed": metadata["hybrid_generations_completed"],
                "applied_budget_signature": metadata["applied_budget_signature"],
                "output_root": str(args.output_root),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
