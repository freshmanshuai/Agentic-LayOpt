import copy
import math
import random
import re
import time

from tool.logistics.facility_logistics import fixed_flow_endpoints
from tool.logistics.relocation_effort import RELOCATION_EFFORT_OBJECTIVE, relocation_effort, validate_relocation_parameters
from tool.logistics.agv_transport_cost import AGV_TRANSPORT_OBJECTIVE, agv_parameters, agv_transport_cost, evaluate_agv_cost, agv_access_required, agv_access_issues
from tool.logistics.transfer_workcell import align_transfer_cells, transfer_cell_violations, transfer_cells, workcell_regions


GEOMETRY_EPSILON_M = 1e-9
DEFAULT_SOLVER_NAME = "genetic_algorithm"
SOLVER_ALIASES = {
    "genetic_algorithm": "genetic_algorithm",
    "genetic": "genetic_algorithm",
    "ga": "genetic_algorithm",
}
WEIGHTED_FLOW_DISTANCE_OBJECTIVE = "minimize_weighted_flow_distance"
HIGH_FLOW_DISTANCE_EXCESS_OBJECTIVE = "minimize_high_flow_distance_excess"
PARETO_OBJECTIVE_ORDER = (
    WEIGHTED_FLOW_DISTANCE_OBJECTIVE,
    HIGH_FLOW_DISTANCE_EXCESS_OBJECTIVE,
)


class SolverAdapter:
    solver_name = "solver"

    def solve(self, flp_ir, *, seed_layout=None, time_limit_s=None):
        raise NotImplementedError


class GeneticAlgorithmSolver(SolverAdapter):
    solver_name = "genetic_algorithm"
    method = "real_coded_genetic_algorithm"

    def __init__(
        self,
        *,
        population_size=48,
        generations=100,
        seed=0,
        crossover_rate=0.85,
        mutation_rate=0.2,
        elite_fraction=0.1,
        tournament_size=3,
        mutation_step_m=None,
        penalty_weight=1_000_000.0,
        stagnation_generations=30,
    ):
        self.population_size = max(4, int(population_size))
        self.generations = max(1, int(generations))
        self.seed = int(seed)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elite_fraction = float(elite_fraction)
        self.tournament_size = max(2, int(tournament_size))
        self.mutation_step_m = mutation_step_m
        self.penalty_weight = float(penalty_weight)
        self.stagnation_generations = max(1, int(stagnation_generations))

    def solve(self, flp_ir, *, seed_layout=None, time_limit_s=None):
        start = time.perf_counter()
        instances = list(flp_ir.get("equipment_instances") or [])
        if not instances:
            empty_layout = _empty_ga_layout(flp_ir)
            return _solver_result(
                self.solver_name,
                "INFEASIBLE",
                empty_layout,
                flp_ir,
                start,
                method=self.method,
                penalty_weight=self.penalty_weight,
                extra_fields={"seed": self.seed, "termination_reason": "empty_problem"},
            )

        rng = random.Random(self.seed)
        site = site_dimensions(flp_ir)
        mutation_step = self.mutation_step_m or max(
            0.25,
            min(site["width_m"], site["depth_m"]) * 0.1,
        )
        population = _initial_genetic_population(
            flp_ir,
            instances,
            seed_layout,
            site,
            self.population_size,
            rng,
        )
        elite_count = max(1, min(self.population_size - 2, int(round(self.population_size * self.elite_fraction))))
        best_chromosome = None
        best_energy = math.inf
        initial_best_energy = None
        generations_completed = 0
        evaluated_individuals = 0
        crossover_count = 0
        mutation_count = 0
        stagnant_generations = 0
        termination_reason = "generation_limit"

        for generation in range(self.generations):
            if time_limit_s is not None and generation > 0 and time.perf_counter() - start >= float(time_limit_s):
                termination_reason = "time_limit"
                break

            evaluated = []
            for chromosome in population:
                layout = _layout_from_genetic_chromosome(flp_ir, instances, chromosome, site)
                energy = genetic_energy(layout, flp_ir, penalty_weight=self.penalty_weight)
                evaluated.append((energy, chromosome))
            evaluated_individuals += len(evaluated)
            evaluated.sort(key=lambda item: item[0])
            generation_best_energy, generation_best = evaluated[0]
            if initial_best_energy is None:
                initial_best_energy = generation_best_energy
            if generation_best_energy < best_energy - GEOMETRY_EPSILON_M:
                best_energy = generation_best_energy
                best_chromosome = _copy_chromosome(generation_best)
                stagnant_generations = 0
            else:
                stagnant_generations += 1
            generations_completed = generation + 1

            if stagnant_generations >= self.stagnation_generations:
                termination_reason = "stagnation"
                break
            if generation + 1 >= self.generations:
                break

            next_population = [
                _copy_chromosome(chromosome)
                for _energy, chromosome in evaluated[:elite_count]
            ]
            while len(next_population) < self.population_size:
                parent_a = _tournament_select(evaluated, self.tournament_size, rng)
                parent_b = _tournament_select(evaluated, self.tournament_size, rng)
                if rng.random() < self.crossover_rate:
                    child_a, child_b = _uniform_crossover(parent_a, parent_b, rng)
                    crossover_count += 1
                else:
                    child_a, child_b = _copy_chromosome(parent_a), _copy_chromosome(parent_b)
                child_a, mutations_a = _mutate_genetic_chromosome(
                    child_a,
                    instances,
                    flp_ir,
                    site,
                    mutation_step,
                    self.mutation_rate,
                    rng,
                )
                child_b, mutations_b = _mutate_genetic_chromosome(
                    child_b,
                    instances,
                    flp_ir,
                    site,
                    mutation_step,
                    self.mutation_rate,
                    rng,
                )
                mutation_count += mutations_a + mutations_b
                next_population.append(child_a)
                if len(next_population) < self.population_size:
                    next_population.append(child_b)
            population = next_population

        if best_chromosome is None:
            best_chromosome = _copy_chromosome(population[0])
            best_layout = _layout_from_genetic_chromosome(flp_ir, instances, best_chromosome, site)
            best_energy = genetic_energy(best_layout, flp_ir, penalty_weight=self.penalty_weight)
            initial_best_energy = best_energy
        else:
            best_layout = _layout_from_genetic_chromosome(flp_ir, instances, best_chromosome, site)
        orientation_report = None
        if (flp_ir.get("task_meta") or {}).get("task_type") in {"generate_feasible_layout", "optimize_layout_from_scratch"}:
            from tool.layout.aisle_facing import improve_aisle_facing
            best_layout, orientation_report = improve_aisle_facing(best_layout, flp_ir)
            best_energy = genetic_energy(best_layout, flp_ir, penalty_weight=self.penalty_weight)
        violations = layout_violations(best_layout, flp_ir)
        return _solver_result(
            self.solver_name,
            "FEASIBLE" if not violations else "INFEASIBLE",
            best_layout,
            flp_ir,
            start,
            method=self.method,
            penalty_weight=self.penalty_weight,
            extra_fields={
                "orientation_preference": orientation_report,
                "energy_value": best_energy,
                "initial_energy": initial_best_energy,
                "best_energy": best_energy,
                "best_fitness": genetic_fitness(best_energy),
                "population_size": self.population_size,
                "generations_completed": generations_completed,
                "evaluated_individuals": evaluated_individuals,
                "elite_count": elite_count,
                "crossover_count": crossover_count,
                "mutation_count": mutation_count,
                "stagnant_generations": stagnant_generations,
                "termination_reason": termination_reason,
                "seed": self.seed,
            },
        )


class ParetoGeneticAlgorithmSolver(GeneticAlgorithmSolver):
    """A feasible-first, real-coded GA with nondominated survivor selection."""

    method = "real_coded_genetic_algorithm_nondominated_sorting"

    def solve(self, flp_ir, *, seed_layout=None, time_limit_s=None):
        parameters = _pareto_objective_parameters(flp_ir)
        if parameters is None:
            if _contains_high_flow_distance_excess_objective(flp_ir) or _contains_relocation_objective(flp_ir):
                raise ValueError(_unsupported_pareto_objective_message())
            return super().solve(
                flp_ir,
                seed_layout=seed_layout,
                time_limit_s=time_limit_s,
            )

        start = time.perf_counter()
        instances = list(flp_ir.get("equipment_instances") or [])
        if not instances:
            empty_layout = _empty_ga_layout(flp_ir)
            return _solver_result(
                self.solver_name,
                "INFEASIBLE",
                empty_layout,
                flp_ir,
                start,
                method=self.method,
                penalty_weight=self.penalty_weight,
                extra_fields=_pareto_report_fields(
                    objective_vector(empty_layout, flp_ir),
                    [],
                    None,
                    [],
                    self,
                    seed_layout=seed_layout,
                    seed_layout_feasible=False,
                    seed_objective_values=None,
                    generations_completed=0,
                    evaluated_individuals=0,
                    crossover_count=0,
                    mutation_count=0,
                    termination_reason="empty_problem",
                ),
            )

        rng = random.Random(self.seed)
        site = site_dimensions(flp_ir)
        mutation_step = self.mutation_step_m or max(
            0.25,
            min(site["width_m"], site["depth_m"]) * 0.1,
        )
        population = _initial_genetic_population(
            flp_ir,
            instances,
            seed_layout,
            site,
            self.population_size,
            rng,
        )
        evaluated = _evaluate_pareto_population(
            population,
            flp_ir,
            instances,
            site,
            generation=1,
            evaluation_offset=0,
        )
        evaluated_individuals = len(evaluated)
        crossover_count = 0
        mutation_count = 0
        generations_completed = 0
        termination_reason = "generation_limit"
        pareto_history = []
        seed_layout_feasible = bool(seed_layout) and not layout_violations(
            seed_layout,
            flp_ir,
        )
        seed_objective_values = (
            objective_vector(seed_layout, flp_ir)
            if seed_layout_feasible
            else None
        )

        for generation in range(self.generations):
            if (
                generation > 0
                and time_limit_s is not None
                and time.perf_counter() - start >= float(time_limit_s)
            ):
                termination_reason = "time_limit"
                break

            ranked_population, fronts = _rank_pareto_population(evaluated)
            front = fronts[0] if fronts else []
            pareto_history.append(
                _pareto_history_entry(
                    generation + 1,
                    ranked_population,
                    front,
                    evaluated_individuals,
                )
            )
            generations_completed = generation + 1
            if generation + 1 >= self.generations:
                break

            offspring = []
            while len(offspring) < self.population_size:
                parent_a = _pareto_tournament_select(
                    ranked_population,
                    self.tournament_size,
                    rng,
                )
                parent_b = _pareto_tournament_select(
                    ranked_population,
                    self.tournament_size,
                    rng,
                )
                if rng.random() < self.crossover_rate:
                    child_a, child_b = _uniform_crossover(
                        parent_a["chromosome"],
                        parent_b["chromosome"],
                        rng,
                    )
                    crossover_count += 1
                else:
                    child_a = _copy_chromosome(parent_a["chromosome"])
                    child_b = _copy_chromosome(parent_b["chromosome"])
                child_a, mutations_a = _mutate_genetic_chromosome(
                    child_a,
                    instances,
                    flp_ir,
                    site,
                    mutation_step,
                    self.mutation_rate,
                    rng,
                )
                child_b, mutations_b = _mutate_genetic_chromosome(
                    child_b,
                    instances,
                    flp_ir,
                    site,
                    mutation_step,
                    self.mutation_rate,
                    rng,
                )
                mutation_count += mutations_a + mutations_b
                offspring.append(child_a)
                if len(offspring) < self.population_size:
                    offspring.append(child_b)

            child_records = _evaluate_pareto_population(
                offspring,
                flp_ir,
                instances,
                site,
                generation=generation + 2,
                evaluation_offset=evaluated_individuals,
            )
            evaluated_individuals += len(child_records)
            evaluated = _select_pareto_survivors(
                ranked_population + child_records,
                self.population_size,
            )

        orientation_reports = []
        if (flp_ir.get("task_meta") or {}).get("task_type") == "optimize_existing_layout":
            from tool.layout.aisle_facing import improve_aisle_facing
            for record in evaluated:
                if record.get("feasible"):
                    record["layout"], facing = improve_aisle_facing(record["layout"], flp_ir,
                        objectives=lambda candidate: objective_vector(candidate, flp_ir))
                    record["objective_values"] = objective_vector(record["layout"], flp_ir)
                    orientation_reports.append(dict(evaluation_index=record["evaluation_index"], **facing))
        # Re-rank the measured, feasible poses before choosing the final front.
        ranked_population, fronts = _rank_pareto_population(evaluated)
        final_front = fronts[0] if fronts else []
        pareto_front = _pareto_candidate_payloads(final_front)
        selected, selection_policy = _select_balanced_pareto_candidate(
            pareto_front,
        )
        if selected is not None:
            layout = copy.deepcopy(selected["layout"])
            layout["generation_policy"] = "pareto_genetic_algorithm"
            status = "FEASIBLE"
            objective_values = copy.deepcopy(selected["objective_values"])
            selected_candidate_id = selected["candidate_id"]
        else:
            fallback = _best_pareto_infeasible_candidate(ranked_population)
            layout = copy.deepcopy(fallback["layout"])
            layout["generation_policy"] = "pareto_genetic_algorithm"
            status = "INFEASIBLE"
            objective_values = copy.deepcopy(fallback["objective_values"])
            selected_candidate_id = None

        result = _solver_result(
            self.solver_name,
            status,
            layout,
            flp_ir,
            start,
            method=self.method,
            penalty_weight=self.penalty_weight,
            extra_fields=_pareto_report_fields(
                objective_values,
                pareto_front,
                selected_candidate_id,
                pareto_history,
                self,
                seed_layout=seed_layout,
                seed_layout_feasible=seed_layout_feasible,
                seed_objective_values=seed_objective_values,
                generations_completed=generations_completed,
                evaluated_individuals=evaluated_individuals,
                crossover_count=crossover_count,
                mutation_count=mutation_count,
                termination_reason=termination_reason,
                selection_policy=selection_policy,
            ),
        )
        result["solver_report"]["orientation_refinements"] = orientation_reports
        if selected_candidate_id is not None:
            selected_candidate = next(
                (
                    item
                    for item in result["solver_report"]["pareto_front"]
                    if item.get("candidate_id") == selected_candidate_id
                ),
                None,
            )
            if selected_candidate is not None:
                selected_candidate["layout"] = copy.deepcopy(result["layout"])
        return result


def normalize_solver_name(value):
    if value is None or not str(value).strip():
        return None
    key = str(value).strip().lower().replace(" ", "_")
    if key not in SOLVER_ALIASES:
        supported = ", ".join(sorted(set(SOLVER_ALIASES.values())))
        raise ValueError(f"Unsupported solver '{value}'. Supported solvers: {supported}")
    return SOLVER_ALIASES[key]


def solver_catalog():
    return {
        "default_solver": DEFAULT_SOLVER_NAME,
        "solvers": [
            {
                "name": "genetic_algorithm",
                "label": "Genetic Algorithm / 遗传算法",
                "available": True,
            },
        ],
    }


def select_solver(flp_ir=None, requested_solver=None, solver_options=None):
    solver_name = normalize_solver_name(requested_solver) or DEFAULT_SOLVER_NAME
    if solver_name != DEFAULT_SOLVER_NAME:
        raise ValueError("Only the genetic_algorithm solver is supported.")
    pareto_parameters = _pareto_objective_parameters(flp_ir)
    if _contains_high_flow_distance_excess_objective(flp_ir) or _contains_relocation_objective(flp_ir):
        if pareto_parameters is None:
            raise ValueError(_unsupported_pareto_objective_message())
        return ParetoGeneticAlgorithmSolver(**dict(solver_options or {}))
    return GeneticAlgorithmSolver(**dict(solver_options or {}))


def annotate_solver_selection(report, requested_solver=None, selection_source=None):
    normalized = normalize_solver_name(requested_solver) or DEFAULT_SOLVER_NAME
    source = selection_source or ("user" if requested_solver else "default")
    report["solver_requested"] = normalized
    report["solver_used"] = report.get("solver") or normalized
    report["selection_source"] = source
    report["default_fallback_used"] = source != "user"
    return report


def solve_canonical_feasibility(
    flp_ir,
    canonical_spec,
    *,
    seed_layout=None,
    reference_layout=None,
    movable_instance_ids=None,
    preserve_instance_ids=None,
    scope="global",
    optimize_objective=False,
    objective_upper_bound=None,
    stop_after_first_solution=False,
    time_limit_s=5.0,
    scale=1000,
    solver_options=None,
):
    """Run a GA-only canonical feasibility or quality search.

    Protected placements never enter the chromosome.  Every evaluated
    chromosome is merged with those authoritative placements before hard
    penalties and the process-flow objective are computed.
    """

    del scale  # Retained only for call compatibility with older receipts.
    spec = canonical_spec or {}
    all_instances = [
        item
        for item in (flp_ir or {}).get("equipment_instances") or []
        if item.get("instance_id")
    ]
    instances_by_id = {
        item["instance_id"]: item for item in all_instances
    }
    preserve = set(
        preserve_instance_ids or spec.get("locked_instance_ids") or []
    )
    requested_movable = (
        movable_instance_ids
        or spec.get("movable_instance_ids")
        or list(instances_by_id)
    )
    movable = [
        instance_id
        for instance_id in requested_movable
        if instance_id in instances_by_id and instance_id not in preserve
    ]
    seed = copy.deepcopy(seed_layout or reference_layout or {})
    pareto_objectives = _pareto_objective_parameters(flp_ir)
    if optimize_objective and pareto_objectives is not None:
        if seed and not layout_violations(seed, flp_ir):
            # A selected Pareto candidate has already been balanced without a
            # user-invented scalar blend. A later scalar-only quality pass may
            # validate it, but must not replace it.
            seed["generation_policy"] = (
                f"canonical_ga_{scope}_preserved_pareto_seed"
            )
            seed["solver_used"] = DEFAULT_SOLVER_NAME
            return seed
        # If a Pareto candidate needs repair, use this path only for hard
        # feasibility. There is no scalar substitute for the two objectives.
        optimize_objective = False
        objective_upper_bound = None
    if not movable:
        return seed

    movable_instances = [
        copy.deepcopy(instances_by_id[instance_id])
        for instance_id in movable
    ]
    for instance in movable_instances:
        constraint = (spec.get("instance_constraints") or {}).get(
            instance["instance_id"]
        ) or {}
        allowed = constraint.get("allowed_orientations_deg")
        if allowed:
            instance["allowed_orientations_deg"] = list(allowed)

    site = site_dimensions(flp_ir)
    site["canonical_constraint_spec"] = spec
    boundary = spec.get("boundary") or {}
    if boundary:
        site.update(
            min_x_m=float(boundary["min_x_m"]),
            max_x_m=float(boundary["max_x_m"]),
            min_z_m=float(boundary["min_z_m"]),
            max_z_m=float(boundary["max_z_m"]),
            center_x_m=(
                float(boundary["min_x_m"])
                + float(boundary["max_x_m"])
            )
            / 2.0,
            center_z_m=(
                float(boundary["min_z_m"])
                + float(boundary["max_z_m"])
            )
            / 2.0,
            width_m=float(boundary["max_x_m"])
            - float(boundary["min_x_m"]),
            depth_m=float(boundary["max_z_m"])
            - float(boundary["min_z_m"]),
        )

    options = dict(solver_options or {})
    rng = random.Random(int(options.get("seed", 0)))
    population_size = max(
        4,
        int(
            options.get(
                "population_size",
                max(16, min(64, len(movable_instances) * 8)),
            )
        ),
    )
    generations = max(1, int(options.get("generations", 80)))
    stagnation_generations = max(
        1,
        int(options.get("stagnation_generations", 24)),
    )
    mutation_step = max(
        0.1,
        min(float(site["width_m"]), float(site["depth_m"])) * 0.05,
    )
    base_state = _initial_genetic_state(
        flp_ir,
        movable_instances,
        seed,
        site,
        rng,
    )
    population = [_encode_genetic_state(base_state, movable_instances)]
    for guided_state in _flow_guided_genetic_states(
        base_state,
        movable_instances,
        flp_ir,
        site,
    ):
        if len(population) >= population_size:
            break
        population.append(
            _encode_genetic_state(guided_state, movable_instances)
        )
    while len(population) < population_size:
        population.append(
            _encode_genetic_state(
                _random_genetic_state(movable_instances, site, rng),
                movable_instances,
            )
        )

    start = time.perf_counter()
    best = None
    best_energy = math.inf
    stagnant = 0
    elite_count = max(2, population_size // 8)
    for generation in range(generations):
        if (
            generation > 0
            and time_limit_s is not None
            and time.perf_counter() - start >= float(time_limit_s)
        ):
            break
        evaluated = []
        for chromosome in population:
            partial = _layout_from_genetic_chromosome(
                flp_ir,
                movable_instances,
                chromosome,
                site,
            )
            candidate = _merge_canonical_ga_layout(seed, partial, site)
            energy, acceptable = _canonical_ga_energy(
                candidate,
                flp_ir,
                seed,
                movable,
                optimize_objective=optimize_objective,
                objective_upper_bound=objective_upper_bound,
            )
            evaluated.append((energy, chromosome, candidate, acceptable))
        evaluated.sort(key=lambda item: item[0])
        generation_best = evaluated[0]
        if generation_best[0] < best_energy - GEOMETRY_EPSILON_M:
            best_energy = generation_best[0]
            best = copy.deepcopy(generation_best[2])
            stagnant = 0
        else:
            stagnant += 1
        acceptable = next(
            (item for item in evaluated if item[3]),
            None,
        )
        if stop_after_first_solution and acceptable is not None:
            best = copy.deepcopy(acceptable[2])
            break
        if stagnant >= stagnation_generations:
            break

        next_population = [
            _copy_chromosome(item[1])
            for item in evaluated[:elite_count]
        ]
        ranked = [(item[0], item[1]) for item in evaluated]
        while len(next_population) < population_size:
            parent_a = _tournament_select(ranked, 3, rng)
            parent_b = _tournament_select(ranked, 3, rng)
            child_a, child_b = _uniform_crossover(
                parent_a,
                parent_b,
                rng,
            )
            child_a, _ = _mutate_genetic_chromosome(
                child_a,
                movable_instances,
                flp_ir,
                site,
                mutation_step,
                0.25,
                rng,
            )
            child_b, _ = _mutate_genetic_chromosome(
                child_b,
                movable_instances,
                flp_ir,
                site,
                mutation_step,
                0.25,
                rng,
            )
            next_population.append(child_a)
            if len(next_population) < population_size:
                next_population.append(child_b)
        population = next_population

    if best is None:
        return None
    _energy, acceptable = _canonical_ga_energy(
        best,
        flp_ir,
        seed,
        movable,
        optimize_objective=optimize_objective,
        objective_upper_bound=objective_upper_bound,
    )
    if not acceptable:
        return None
    best["generation_policy"] = f"canonical_ga_{scope}"
    best["solver_used"] = DEFAULT_SOLVER_NAME
    return best


def _merge_canonical_ga_layout(seed_layout, partial_layout, site):
    result = copy.deepcopy(seed_layout or {})
    result.setdefault("layout_id", "ROUGH_LAYOUT_001")
    result.setdefault("layout_type", "rough_design")
    current_site = layout_site(site)
    prior_site = result.get("site") or {}
    # Keep caller metadata that the current canonical site does not define,
    # but never let an inherited site snapshot replace rebuilt geometry.
    authoritative_site = {
        key: copy.deepcopy(value)
        for key, value in prior_site.items()
        if key not in current_site
    }
    authoritative_site.update(current_site)
    result["site"] = authoritative_site
    result.setdefault("placements", [])
    by_id = {
        item.get("instance_id"): index
        for index, item in enumerate(result["placements"])
        if item.get("instance_id")
    }
    for placement in partial_layout.get("placements") or []:
        instance_id = placement.get("instance_id")
        if instance_id in by_id:
            result["placements"][by_id[instance_id]] = copy.deepcopy(
                placement
            )
        else:
            result["placements"].append(copy.deepcopy(placement))
    return result


def _canonical_ga_energy(
    layout,
    flp_ir,
    seed_layout,
    movable_instance_ids,
    *,
    optimize_objective,
    objective_upper_bound,
):
    violations = layout_violations(layout, flp_ir)
    hard_penalty = constraint_penalty(layout, flp_ir)
    objective = objective_value(layout, flp_ir)
    movement = _layout_movement(
        layout,
        seed_layout,
        movable_instance_ids,
    )
    strict_objective = (
        objective_upper_bound is None
        or objective < float(objective_upper_bound) - GEOMETRY_EPSILON_M
    )
    acceptable = not violations and (
        not optimize_objective or strict_objective
    )
    energy = (
        len(violations) * 1_000_000_000_000.0
        + hard_penalty * 1_000_000_000.0
    )
    if optimize_objective:
        energy += objective * 1_000_000.0
        if not strict_objective and objective_upper_bound is not None:
            energy += 100_000_000_000.0 + max(
                0.0,
                objective - float(objective_upper_bound),
            ) * 1_000_000.0
    energy += movement
    return energy, acceptable


def _layout_movement(layout, reference_layout, instance_ids):
    current = {
        item.get("instance_id"): item
        for item in (layout or {}).get("placements") or []
    }
    reference = {
        item.get("instance_id"): item
        for item in (reference_layout or {}).get("placements") or []
    }
    total = 0.0
    for instance_id in instance_ids:
        left = current.get(instance_id)
        right = reference.get(instance_id)
        if not left or not right:
            continue
        left_position = left.get("position") or left.get("pose") or {}
        right_position = right.get("position") or right.get("pose") or {}
        total += abs(
            float(left_position.get("x_m") or 0.0)
            - float(right_position.get("x_m") or 0.0)
        )
        total += abs(
            float(left_position.get("z_m") or 0.0)
            - float(right_position.get("z_m") or 0.0)
        )
    return total


def _solver_result(
    solver_name,
    status,
    layout,
    flp_ir,
    start,
    *,
    method=None,
    bound=None,
    gap=None,
    penalty_weight=1_000_000.0,
    extra_fields=None,
):
    violations = layout_violations(layout, flp_ir)
    penalty = constraint_penalty(layout, flp_ir)
    objective = objective_value(layout, flp_ir)
    if status in {"FEASIBLE", "OPTIMAL"} and violations:
        status = "INFEASIBLE"
    layout["solver_used"] = solver_name
    report = {
        "solver": solver_name,
        "solver_used": solver_name,
        "method": method or solver_name,
        "status": status,
        "objective_value": objective,
        "energy_value": objective + float(penalty_weight) * penalty,
        "constraint_penalty": penalty,
        "penalty_value": float(penalty_weight) * penalty,
        "bound": bound,
        "gap": gap,
        "runtime_s": time.perf_counter() - start,
        "violations": violations,
        "violation_count": len(violations),
    }
    report.update(extra_fields or {})
    return {
        "layout": layout,
        "solver_used": solver_name,
        "solver_report": report,
    }


def _empty_ga_layout(flp_ir):
    site = site_dimensions(flp_ir)
    return {
        "layout_id": "ROUGH_LAYOUT_001",
        "layout_type": "rough_design",
        "site": layout_site(site),
        "placements": [],
        "generation_policy": "genetic_algorithm",
    }


def site_dimensions(flp_ir):
    site_ref = (flp_ir or {}).get("site_ref") or {}
    site_snapshot = site_ref.get("site_snapshot") or {}
    if not site_snapshot:
        raise ValueError(
            "GeneticAlgorithmSolver requires an authoritative site_snapshot."
        )
    from tool.layout.site_constraint_compiler import (
        CanonicalConstraintCompiler,
        canonical_site_from_spec,
    )

    spec = CanonicalConstraintCompiler().compile(
        site_snapshot,
        flp_ir=flp_ir,
    )
    site = canonical_site_from_spec(spec)
    site["layout_topology"] = site_snapshot.get("layout_topology")
    return site


def layout_site(site):
    result = {
        "site_id": site["site_id"],
        "width_m": site["width_m"],
        "depth_m": site["depth_m"],
        "dimensions_m": {"width_m": site["width_m"], "height_m": 0.0, "depth_m": site["depth_m"]},
        "layout_topology": site.get("layout_topology"),
        "site_features": list(site.get("site_features") or []),
        "fixed_equipment_instances": list(
            site.get("embedded_equipment_instances") or []
        ),
        "non_placeable_zones": non_placeable_zones(site),
        "placement_regions": placement_regions(site),
        "workstation_bays": workstation_bay_geometry(site),
    }
    for key in (
        "center_x_m",
        "center_z_m",
        "min_x_m",
        "max_x_m",
        "min_z_m",
        "max_z_m",
    ):
        if key in site:
            result[key] = site[key]
    if site.get("canonical_constraint_spec"):
        result["canonical_constraint_spec_sha256"] = site[
            "canonical_constraint_spec"
        ].get("spec_sha256")
    return result


def workstation_bay_geometry(site):
    """Expose source-labeled bay rectangles from the canonical site zones."""

    site = site or {}
    spec = site.get("canonical_constraint_spec") or {}
    instance_constraints = spec.get("instance_constraints") or {}
    bays = []
    for zone in site.get("zones") or []:
        if not isinstance(zone, dict):
            continue
        bay_id = str(zone.get("zone_id") or "")
        if not bay_id.startswith("natural_site/workstation_bay_"):
            continue
        pose = zone.get("pose") or {}
        dimensions = zone.get("dimensions_m") or {}
        try:
            center_x = float(pose["x_m"])
            center_z = float(pose["z_m"])
            width = float(dimensions["width_m"])
            depth = float(dimensions["depth_m"])
        except (KeyError, TypeError, ValueError):
            continue
        match = re.search(r"workstation_bay_row_(\d+)_column_(\d+)$", bay_id)
        if not match:
            continue
        placement_region_id = bay_id + "/placement"
        assigned = sorted(
            instance_id
            for instance_id, constraint in instance_constraints.items()
            if str(constraint.get("workstation_bay_id") or "") == bay_id
        )
        row_alignment_z = next(
            (
                instance_constraints[instance_id].get("workstation_row_alignment_z_m")
                for instance_id in assigned
                if instance_constraints[instance_id].get("workstation_row_alignment_z_m") is not None
            ),
            None,
        )
        bays.append(
            {
                "bay_id": bay_id,
                "label": str(zone.get("zone_name") or ""),
                "row_index": int(match.group(1)),
                "column_index": int(match.group(2)),
                "reserved": str(zone.get("zone_type") or "").casefold() == "forbidden_zone",
                "pose": copy.deepcopy(pose),
                "dimensions_m": {"width_m": width, "depth_m": depth},
                "bounds_m": {
                    "min_x_m": center_x - width / 2.0,
                    "max_x_m": center_x + width / 2.0,
                    "min_z_m": center_z - depth / 2.0,
                    "max_z_m": center_z + depth / 2.0,
                },
                "assigned_instance_ids": assigned,
                "row_alignment_z_m": row_alignment_z,
            }
        )
    return sorted(bays, key=lambda item: (item["row_index"], item["column_index"]))


def non_placeable_zones(site):
    canonical = site.get("canonical_constraint_spec")
    if canonical:
        return [
            {
                "zone_id": item.get("object_id"),
                "zone_type": item.get("object_type"),
                "min_x_m": item.get("min_x_m"),
                "max_x_m": item.get("max_x_m"),
                "min_z_m": item.get("min_z_m"),
                "max_z_m": item.get("max_z_m"),
                "width_m": item.get("width_m"),
                "depth_m": item.get("depth_m"),
            }
            for item in canonical.get("prohibited_regions") or []
        ]
    if site.get("explicit_site"):
        zones = []
        for feature in site.get("site_features") or []:
            feature_type = str(feature.get("feature_type") or "other").lower()
            if feature.get("must_remain_clear") or feature_type in {
                "wall",
                "column",
                "obstacle",
                "forbidden_zone",
                "corridor",
                "entrance",
                "exit",
            }:
                zone = _site_item_zone(
                    feature,
                    zone_id=feature.get("site_feature_id"),
                    zone_type=feature_type,
                )
                if zone:
                    zones.append(zone)
        for item in site.get("zones") or []:
            zone_type = str(item.get("zone_type") or "").lower()
            if zone_type in {
                "wall",
                "aisle",
                "corridor",
                "obstacle",
                "forbidden",
                "forbidden_zone",
                "non_placeable",
                "maintenance_clearance",
            }:
                zone = _site_item_zone(
                    item,
                    zone_id=item.get("zone_id"),
                    zone_type=zone_type,
                )
                if zone:
                    zones.append(zone)
        for item in site.get("embedded_equipment_instances") or []:
            geometry = item.get("geometry") or {}
            zone = _site_item_zone(
                {
                    "pose": item.get("pose") or {},
                    "dimensions_m": geometry.get("dimensions_m") or {},
                },
                zone_id=item.get("instance_id"),
                zone_type="fixed_equipment",
            )
            if zone:
                zones.append(zone)
        return zones
    # Missing geometry is unknown, not permission to invent four walls and a
    # center aisle.  The boundary remains authoritative; no synthetic obstacle
    # is introduced.
    return []


def placement_regions(site):
    canonical = site.get("canonical_constraint_spec")
    if canonical:
        return [
            {
                "zone_id": item.get("region_id"),
                "zone_type": item.get("region_type"),
                "min_x_m": item.get("min_x_m"),
                "max_x_m": item.get("max_x_m"),
                "min_z_m": item.get("min_z_m"),
                "max_z_m": item.get("max_z_m"),
                "width_m": item.get("width_m"),
                "depth_m": item.get("depth_m"),
            }
            for item in canonical.get("allowed_placement_regions") or []
        ]
    if site.get("explicit_site"):
        regions = []
        for item in site.get("zones") or []:
            zone_type = str(item.get("zone_type") or "").lower()
            if zone_type not in {"placement", "placement_region", "equipment_region"}:
                continue
            region = _site_item_zone(
                item,
                zone_id=item.get("zone_id"),
                zone_type="placement_region",
            )
            if region:
                regions.append(region)
        if regions:
            return regions
        return [
            _zone(
                "placement_site_boundary",
                "placement_region",
                -float(site["width_m"]) / 2,
                float(site["width_m"]) / 2,
                -float(site["depth_m"]) / 2,
                float(site["depth_m"]) / 2,
            )
        ]
    center_x = float(site.get("center_x_m") or 0.0)
    center_z = float(site.get("center_z_m") or 0.0)
    width = float(site["width_m"])
    depth = float(site["depth_m"])
    return [
        _zone(
            "placement_site_boundary",
            "placement_region",
            center_x - width / 2,
            center_x + width / 2,
            center_z - depth / 2,
            center_z + depth / 2,
        )
    ]


def _zone(zone_id, zone_type, min_x, max_x, min_z, max_z):
    return {
        "zone_id": zone_id,
        "zone_type": zone_type,
        "min_x_m": float(min_x),
        "max_x_m": float(max_x),
        "min_z_m": float(min_z),
        "max_z_m": float(max_z),
        "width_m": max(0.0, float(max_x) - float(min_x)),
        "depth_m": max(0.0, float(max_z) - float(min_z)),
    }


def _site_item_zone(item, *, zone_id, zone_type):
    pose = item.get("pose") or {}
    dimensions = item.get("dimensions_m") or {}
    try:
        center_x = float(pose.get("x_m") or 0.0)
        center_z = float(pose.get("z_m") or 0.0)
        width = float(dimensions.get("width_m") or 0.0)
        depth = float(dimensions.get("depth_m") or 0.0)
    except (TypeError, ValueError):
        return None
    if width <= GEOMETRY_EPSILON_M or depth <= GEOMETRY_EPSILON_M:
        return None
    return _zone(
        str(zone_id or f"{zone_type}_zone"),
        str(zone_type or "restricted"),
        center_x - width / 2,
        center_x + width / 2,
        center_z - depth / 2,
        center_z + depth / 2,
    )


def objective_value(layout, flp_ir):
    if flp_ir and agv_parameters(flp_ir) is not None:
        return agv_transport_cost(layout, flp_ir)
    placements = {item["instance_id"]: item for item in layout.get("placements", [])}
    placements.update(fixed_flow_endpoints(layout, flp_ir))
    total = 0.0
    for flow in flp_ir.get("flow_relations", []) if flp_ir else []:
        left = placements.get(flow.get("from_instance_id"))
        right = placements.get(flow.get("to_instance_id"))
        if not left or not right:
            continue
        weight = flow.get("flow_weight")
        if weight is None:
            weight = 1.0
        total += float(weight) * _manhattan(left, right)
    return total


def objective_vector(layout, flp_ir):
    """Return the source-declared flow objective values without scalarizing them."""

    values = {
        (AGV_TRANSPORT_OBJECTIVE if agv_parameters(flp_ir or {}) is not None else WEIGHTED_FLOW_DISTANCE_OBJECTIVE): objective_value(layout, flp_ir),
    }
    parameters = _pareto_objective_parameters(flp_ir)
    if parameters is not None:
        if parameters.get("second_objective") == RELOCATION_EFFORT_OBJECTIVE:
            values[RELOCATION_EFFORT_OBJECTIVE] = relocation_effort(layout, parameters)
        else:
            values[HIGH_FLOW_DISTANCE_EXCESS_OBJECTIVE] = (
                high_flow_distance_excess(
                    layout,
                    flp_ir,
                    flow_threshold_batches_per_day=(
                        parameters["flow_threshold_batches_per_day"]
                    ),
                    distance_threshold_m=parameters["distance_threshold_m"],
                )
            )
    return values


def high_flow_distance_excess(
    layout,
    flp_ir,
    *,
    flow_threshold_batches_per_day,
    distance_threshold_m,
):
    """Measure high-flow distance beyond its explicit inclusive thresholds."""

    placements = {
        item["instance_id"]
        for item in layout.get("placements", [])
        if item.get("instance_id")
    }
    placement_by_id = {
        item["instance_id"]: item
        for item in layout.get("placements", [])
        if item.get("instance_id")
    }
    flow_threshold = float(flow_threshold_batches_per_day)
    distance_threshold = float(distance_threshold_m)
    total = 0.0
    for flow in _aggregate_directed_flow_relations(flp_ir):
        from_instance_id = flow["from_instance_id"]
        to_instance_id = flow["to_instance_id"]
        if (
            from_instance_id not in placements
            or to_instance_id not in placements
        ):
            continue
        weight = float(flow["flow_weight"])
        # The threshold is inclusive: a pair at the stated flow is high-flow.
        if weight < flow_threshold:
            continue
        distance = _manhattan(
            placement_by_id[from_instance_id],
            placement_by_id[to_instance_id],
        )
        total += weight * max(0.0, distance - distance_threshold)
    return total


def _pareto_objective_parameters(flp_ir):
    objectives = [
        item
        for item in ((flp_ir or {}).get("objectives") or [])
        if isinstance(item, dict)
    ]
    weighted = [
        item
        for item in objectives
        if item.get("objective_type") in {WEIGHTED_FLOW_DISTANCE_OBJECTIVE, AGV_TRANSPORT_OBJECTIVE}
    ]
    relocation = [item for item in objectives if item.get("objective_type") == RELOCATION_EFFORT_OBJECTIVE]
    if relocation:
        if len(objectives) != 2 or len(weighted) != 1 or len(relocation) != 1:
            return None
        if any(str(item.get("sense") or "minimize").casefold() != "minimize" for item in objectives):
            return None
        parameters = relocation[0].get("parameters") or {}
        validate_relocation_parameters(parameters, [item["instance_id"] for item in (flp_ir or {}).get("equipment_instances") or []])
        return dict(parameters, second_objective=RELOCATION_EFFORT_OBJECTIVE)
    high_flow = [
        item
        for item in objectives
        if item.get("objective_type") == HIGH_FLOW_DISTANCE_EXCESS_OBJECTIVE
    ]
    if (
        len(objectives) != len(PARETO_OBJECTIVE_ORDER)
        or len(weighted) != 1
        or len(high_flow) != 1
        or {
            item.get("objective_type")
            for item in objectives
        }
        != set(PARETO_OBJECTIVE_ORDER)
    ):
        return None
    if any(
        str(item.get("sense") or "minimize").casefold() != "minimize"
        for item in (weighted[0], high_flow[0])
    ):
        return None
    parameters = high_flow[0].get("parameters") or {}
    try:
        flow_threshold = float(parameters["flow_threshold_batches_per_day"])
        distance_threshold = float(parameters["distance_threshold_m"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(flow_threshold)
        or not math.isfinite(distance_threshold)
        or flow_threshold < 0.0
        or distance_threshold < 0.0
    ):
        return None
    return {
        "flow_threshold_batches_per_day": flow_threshold,
        "distance_threshold_m": distance_threshold,
    }


def _contains_relocation_objective(flp_ir):
    return any(item.get("objective_type") == RELOCATION_EFFORT_OBJECTIVE for item in (flp_ir or {}).get("objectives") or [])


def _pareto_order(values):
    first = AGV_TRANSPORT_OBJECTIVE if AGV_TRANSPORT_OBJECTIVE in values else WEIGHTED_FLOW_DISTANCE_OBJECTIVE
    return (first, RELOCATION_EFFORT_OBJECTIVE) if RELOCATION_EFFORT_OBJECTIVE in values else PARETO_OBJECTIVE_ORDER


def _contains_high_flow_distance_excess_objective(flp_ir):
    return any(
        isinstance(item, dict)
        and item.get("objective_type") == HIGH_FLOW_DISTANCE_EXCESS_OBJECTIVE
        for item in ((flp_ir or {}).get("objectives") or [])
    )


def _unsupported_pareto_objective_message():
    return (
        "minimize_high_flow_distance_excess requires exactly the supported "
        "two-objective set: minimize_weighted_flow_distance and "
        "minimize_high_flow_distance_excess."
    )


def _aggregate_directed_flow_relations(flp_ir):
    """Aggregate repeated records only when their directed physical pair matches."""

    aggregated = {}
    for flow in ((flp_ir or {}).get("flow_relations") or []):
        if not isinstance(flow, dict):
            continue
        from_instance_id = flow.get("from_instance_id")
        to_instance_id = flow.get("to_instance_id")
        if not from_instance_id or not to_instance_id:
            continue
        weight = flow.get("flow_weight")
        try:
            numeric_weight = 1.0 if weight is None else float(weight)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_weight):
            continue
        key = (str(from_instance_id), str(to_instance_id))
        item = aggregated.setdefault(
            key,
            {
                "from_instance_id": key[0],
                "to_instance_id": key[1],
                "flow_weight": 0.0,
            },
        )
        item["flow_weight"] += numeric_weight
    return list(aggregated.values())


def _evaluate_pareto_population(
    population,
    flp_ir,
    instances,
    site,
    *,
    generation,
    evaluation_offset,
):
    records = []
    for index, chromosome in enumerate(population):
        layout = _layout_from_genetic_chromosome(
            flp_ir,
            instances,
            chromosome,
            site,
        )
        violations = layout_violations(layout, flp_ir)
        records.append(
            {
                "chromosome": _copy_chromosome(chromosome),
                "layout": layout,
                "objective_values": objective_vector(layout, flp_ir),
                "violations": violations,
                "violation_count": len(violations),
                "constraint_penalty": constraint_penalty(layout, flp_ir),
                "feasible": not violations,
                "generation": int(generation),
                "evaluation_index": int(evaluation_offset + index),
            }
        )
    return records


def _rank_pareto_population(records):
    feasible = [item for item in records if item.get("feasible")]
    fronts = _nondominated_fronts(feasible)
    for rank, front in enumerate(fronts):
        _assign_crowding_distance(front)
        for item in front:
            item["pareto_rank"] = rank
    infeasible = sorted(
        (item for item in records if not item.get("feasible")),
        key=_infeasible_pareto_key,
    )
    for offset, item in enumerate(infeasible):
        item["pareto_rank"] = len(fronts) + offset
        item["crowding_distance"] = 0.0
    return list(records), fronts


def _nondominated_fronts(records):
    remaining = list(records)
    fronts = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                other is not candidate and _pareto_dominates(other, candidate)
                for other in remaining
            )
        ]
        if not front:
            # Numeric inputs are finite by construction. Keep a defensive,
            # deterministic fallback so an unexpected record cannot stall GA.
            front = [min(remaining, key=_infeasible_pareto_key)]
        fronts.append(front)
        front_ids = {id(item) for item in front}
        remaining = [item for item in remaining if id(item) not in front_ids]
    return fronts


def _pareto_dominates(left, right):
    left_values = _pareto_objective_tuple(left)
    right_values = _pareto_objective_tuple(right)
    return all(
        left_value <= right_value + GEOMETRY_EPSILON_M
        for left_value, right_value in zip(left_values, right_values)
    ) and any(
        left_value < right_value - GEOMETRY_EPSILON_M
        for left_value, right_value in zip(left_values, right_values)
    )


def _pareto_objective_tuple(record):
    values = record.get("objective_values") or {}
    return tuple(float(values[objective_type]) for objective_type in _pareto_order(values))


def _assign_crowding_distance(front):
    for item in front:
        item["crowding_distance"] = 0.0
    if len(front) <= 2:
        for item in front:
            item["crowding_distance"] = math.inf
        return
    for objective_type in _pareto_order(front[0]["objective_values"]):
        ordered = sorted(
            front,
            key=lambda item: (
                float((item.get("objective_values") or {})[objective_type]),
                int(item.get("evaluation_index") or 0),
            ),
        )
        first_value = float((ordered[0].get("objective_values") or {})[objective_type])
        last_value = float((ordered[-1].get("objective_values") or {})[objective_type])
        ordered[0]["crowding_distance"] = math.inf
        ordered[-1]["crowding_distance"] = math.inf
        span = last_value - first_value
        if span <= GEOMETRY_EPSILON_M:
            continue
        for index in range(1, len(ordered) - 1):
            if math.isinf(ordered[index].get("crowding_distance") or 0.0):
                continue
            previous_value = float(
                (ordered[index - 1].get("objective_values") or {})[objective_type]
            )
            next_value = float(
                (ordered[index + 1].get("objective_values") or {})[objective_type]
            )
            ordered[index]["crowding_distance"] += (
                next_value - previous_value
            ) / span


def _pareto_tournament_select(records, tournament_size, rng):
    contenders = rng.sample(records, k=min(len(records), tournament_size))
    return min(contenders, key=_pareto_selection_key)


def _pareto_selection_key(record):
    crowding = float(record.get("crowding_distance") or 0.0)
    return (
        0 if record.get("feasible") else 1,
        int(record.get("pareto_rank") or 0),
        -crowding,
        int(record.get("violation_count") or 0),
        float(record.get("constraint_penalty") or 0.0),
        int(record.get("evaluation_index") or 0),
    )


def _infeasible_pareto_key(record):
    return (
        int(record.get("violation_count") or 0),
        float(record.get("constraint_penalty") or 0.0),
        int(record.get("evaluation_index") or 0),
    )


def _select_pareto_survivors(records, population_size):
    ranked, fronts = _rank_pareto_population(records)
    survivors = []
    for front in fronts:
        remaining = population_size - len(survivors)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            survivors.extend(
                sorted(front, key=_pareto_selection_key)
            )
            continue
        survivors.extend(
            sorted(front, key=_pareto_selection_key)[:remaining]
        )
        return survivors
    if len(survivors) < population_size:
        infeasible = sorted(
            (item for item in ranked if not item.get("feasible")),
            key=_infeasible_pareto_key,
        )
        survivors.extend(infeasible[: population_size - len(survivors)])
    return survivors


def _pareto_history_entry(generation, records, front, evaluated_individuals):
    feasible = [item for item in records if item.get("feasible")]
    best_values = {
        objective_type: (
            min(
                float((item.get("objective_values") or {})[objective_type])
                for item in front
            )
            if front
            else None
        )
        for objective_type in _pareto_order(records[0]["objective_values"] if records else {})
    }
    return {
        "generation": int(generation),
        "population_size": len(records),
        "feasible_candidate_count": len(feasible),
        "nondominated_candidate_count": len(front),
        "best_objective_values": best_values,
        "evaluated_individuals": int(evaluated_individuals),
    }


def _pareto_candidate_payloads(front):
    ordered = sorted(
        front,
        key=lambda item: (
            _pareto_objective_tuple(item),
            int(item.get("generation") or 0),
            int(item.get("evaluation_index") or 0),
        ),
    )
    candidates = []
    for index, item in enumerate(ordered, start=1):
        candidates.append(
            {
                "candidate_id": f"pareto_candidate_{index:03d}",
                "layout": copy.deepcopy(item["layout"]),
                "objective_values": copy.deepcopy(item["objective_values"]),
                "violation_count": int(item["violation_count"]),
                "violations": copy.deepcopy(item["violations"]),
                "constraint_penalty": float(item["constraint_penalty"]),
                "generation": int(item["generation"]),
            }
        )
    return candidates


def _select_balanced_pareto_candidate(candidates):
    objective_order = _pareto_order(candidates[0]["objective_values"] if candidates else {})
    policy = {
        "method": "normalized_ideal_distance",
        "description": (
            "Each objective is normalized by its observed retained Pareto-front "
            "range. The candidate closest to the component-wise ideal is selected; "
            "ties use candidate_id. No user-supplied objective weights are applied."
        ),
        "objective_order": list(objective_order),
        "normalization": "observed_pareto_front_range",
        "tie_breaker": "candidate_id",
        "user_weights_applied": False,
    }
    if not candidates:
        policy["status"] = "no_feasible_pareto_candidate"
        return None, policy
    minima = {
        objective_type: min(
            float((item.get("objective_values") or {})[objective_type])
            for item in candidates
        )
        for objective_type in objective_order
    }
    maxima = {
        objective_type: max(
            float((item.get("objective_values") or {})[objective_type])
            for item in candidates
        )
        for objective_type in objective_order
    }
    for item in candidates:
        normalized_components = []
        for objective_type in objective_order:
            span = maxima[objective_type] - minima[objective_type]
            value = float((item.get("objective_values") or {})[objective_type])
            normalized_components.append(
                0.0
                if span <= GEOMETRY_EPSILON_M
                else (value - minima[objective_type]) / span
            )
        item["selection_distance_to_ideal"] = math.sqrt(
            sum(component * component for component in normalized_components)
        )
    policy["status"] = "selected"
    return min(
        candidates,
        key=lambda item: (
            float(item.get("selection_distance_to_ideal") or 0.0),
            str(item.get("candidate_id") or ""),
        ),
    ), policy


def _best_pareto_infeasible_candidate(records):
    if not records:
        raise ValueError("Pareto genetic algorithm produced no candidates.")
    return min(records, key=_pareto_selection_key)


def _pareto_report_fields(
    objective_values,
    pareto_front,
    selected_candidate_id,
    pareto_history,
    solver,
    *,
    seed_layout,
    seed_layout_feasible,
    seed_objective_values,
    generations_completed,
    evaluated_individuals,
    crossover_count,
    mutation_count,
    termination_reason,
    selection_policy=None,
):
    selected_delta_from_seed = (
        {
            objective_type: float(objective_values[objective_type])
            - float(seed_objective_values[objective_type])
            for objective_type in _pareto_order(objective_values)
        }
        if seed_objective_values is not None
        else None
    )
    return {
        "algorithm": "feasible_first_nondominated_sorting_with_crowding_distance",
        "objective_values": copy.deepcopy(objective_values),
        "objective_vector_order": list(_pareto_order(objective_values)),
        "pareto_front": copy.deepcopy(pareto_front),
        "pareto_front_source": "final_survivor_population",
        "selected_candidate_id": selected_candidate_id,
        "selection_policy": copy.deepcopy(selection_policy)
        if selection_policy is not None
        else {
            "method": "normalized_ideal_distance",
            "status": "no_feasible_pareto_candidate",
            "objective_order": list(_pareto_order(objective_values)),
            "user_weights_applied": False,
        },
        "pareto_history": copy.deepcopy(pareto_history),
        "population_size": solver.population_size,
        "generations_requested": solver.generations,
        "generations_completed": int(generations_completed),
        "evaluated_individuals": int(evaluated_individuals),
        "crossover_count": int(crossover_count),
        "mutation_count": int(mutation_count),
        "termination_reason": termination_reason,
        "seed": solver.seed,
        "seed_layout_provided": bool(seed_layout),
        "seed_layout_feasible": bool(seed_layout_feasible),
        "feasible_seed_objective_values": copy.deepcopy(seed_objective_values),
        "selected_objective_deltas_from_feasible_seed": selected_delta_from_seed,
        "weighted_flow_distance_strictly_improved_from_feasible_seed": (
            selected_delta_from_seed.get(WEIGHTED_FLOW_DISTANCE_OBJECTIVE, 0.)
            < -GEOMETRY_EPSILON_M
            if selected_delta_from_seed is not None and WEIGHTED_FLOW_DISTANCE_OBJECTIVE in selected_delta_from_seed
            else None
        ),
        "primary_objective_strictly_improved_from_feasible_seed": (
            selected_delta_from_seed[_pareto_order(objective_values)[0]] < -GEOMETRY_EPSILON_M
            if selected_delta_from_seed is not None else None),
        # The report retains J1 for older consumers, but does not emit an
        # artificial scalar energy for the two user-declared objectives.
        "energy_value": None,
    }


def genetic_energy(layout, flp_ir, *, penalty_weight=1_000_000.0):
    return objective_value(layout, flp_ir) + penalty_weight * constraint_penalty(layout, flp_ir)


def genetic_fitness(energy):
    if not math.isfinite(energy):
        return 0.0
    return 1.0 / (1.0 + max(0.0, float(energy)))


def constraint_penalty(layout, flp_ir=None):
    placements = layout.get("placements", []) if layout else []
    site = layout.get("site", {}) if layout else {}
    return (
        _overlap_penalty(placements)
        + _boundary_penalty(placements, site)
        + _non_placeable_penalty(placements, site)
        + _clearance_penalty(placements, flp_ir)
        + _workstation_bay_clearance_penalty(layout, flp_ir)
        + sum(max(.01,float(v.get("deficit_m",1))) for v in transfer_cell_violations(layout, flp_ir or {}))
        + (len(evaluate_agv_cost(layout, flp_ir)["unreachable"])*1000 if agv_parameters(flp_ir or {}) is not None else 0)
        + (len(agv_access_issues(layout,flp_ir))*1000 if agv_access_required(flp_ir or {}) else 0)
    )


def layout_violations(layout, flp_ir=None):
    violations = transfer_cell_violations(layout, flp_ir or {})
    if agv_access_required(flp_ir or {}):
        violations.extend(dict(violation_type="agv_unreachable_access",instances=[r["vehicle"]],details=r) for r in agv_access_issues(layout,flp_ir))
    if agv_parameters(flp_ir or {}) is not None:
        for row in evaluate_agv_cost(layout, flp_ir)["unreachable"]:
            violations.append(dict(violation_type="agv_unreachable_transfer", instances=[row["vehicle"]], details=row))
    placements = layout.get("placements", []) if layout else []
    site = layout.get("site", {}) if layout else {}
    for index, left in enumerate(placements):
        for right in placements[index + 1 :]:
            if overlaps(left["footprint"], right["footprint"]):
                violations.append({"violation_type": "overlap", "instances": [left["instance_id"], right["instance_id"]]})
    min_x = float(
        site.get("min_x_m")
        if site.get("min_x_m") is not None
        else -float(site.get("width_m") or 0.0) / 2
    )
    max_x = float(
        site.get("max_x_m")
        if site.get("max_x_m") is not None
        else float(site.get("width_m") or 0.0) / 2
    )
    min_z = float(
        site.get("min_z_m")
        if site.get("min_z_m") is not None
        else -float(site.get("depth_m") or 0.0) / 2
    )
    max_z = float(
        site.get("max_z_m")
        if site.get("max_z_m") is not None
        else float(site.get("depth_m") or 0.0) / 2
    )
    for placement in placements:
        footprint = placement["footprint"]
        if (
            footprint["min_x"] < min_x
            or footprint["max_x"] > max_x
            or footprint["min_z"] < min_z
            or footprint["max_z"] > max_z
        ):
            violations.append({"violation_type": "boundary", "instances": [placement["instance_id"]]})
        for zone in site.get("non_placeable_zones", []):
            if _rectangles_overlap(footprint, zone):
                violations.append(
                    {
                        "violation_type": "non_placeable_zone",
                        "instances": [placement["instance_id"]],
                        "zone_id": zone.get("zone_id"),
                        "zone_type": zone.get("zone_type"),
                    }
                )
    clearance_by_instance = _clearance_by_instance(flp_ir)
    for index, left in enumerate(placements):
        for right in placements[index + 1 :]:
            required = max(
                clearance_by_instance.get(left["instance_id"], 0.0),
                clearance_by_instance.get(right["instance_id"], 0.0),
            )
            deficit = _clearance_deficit(left, right, required)
            if deficit > GEOMETRY_EPSILON_M:
                violations.append(
                    {
                        "violation_type": "clearance",
                        "instances": [left["instance_id"], right["instance_id"]],
                        "required_m": required,
                        "deficit_m": deficit,
                    }
                )
    violations.extend(_workstation_bay_issues(layout, flp_ir))
    return violations


def _workstation_bay_issues(layout, flp_ir):
    if not flp_ir:
        return []
    from tool.validation.canonical_constraint_validator import _workstation_service_envelope

    site = (layout or {}).get("site") or {}
    canonical_constraints = (
        (site.get("canonical_constraint_spec") or {}).get("instance_constraints") or {}
    )
    instances = {
        str(item.get("instance_id") or ""): item
        for item in (flp_ir.get("equipment_instances") or [])
    }
    regions = site.get("placement_regions") or []
    issues = []
    for placement in (layout or {}).get("placements") or []:
        instance_id = str(placement.get("instance_id") or "")
        instance = instances.get(instance_id) or {}
        canonical = canonical_constraints.get(instance_id) or {}
        allowed_ids = [
            str(value)
            for value in (instance.get("allowed_regions") or canonical.get("allowed_region_ids") or [])
        ]
        case_clearance = canonical.get("maintenance_clearance_m") or instance.get("maintenance_clearance_m") or {}
        is_case_clearance = case_clearance.get("source") == "user_input"
        is_bay = bool(canonical.get("workstation_bay_id")) or any(
            str(value).startswith("natural_site/workstation_bay_")
            for value in (instance.get("allowed_regions") or [])
        )
        if not is_case_clearance and not is_bay:
            continue
        allowed_regions = [
            region for region in regions
            if not allowed_ids or str(region.get("zone_id") or "") in allowed_ids
        ]
        footprint = placement.get("footprint") or {}
        try:
            body = {
                "min_x_m": float(footprint["min_x"]),
                "max_x_m": float(footprint["max_x"]),
                "min_z_m": float(footprint["min_z"]),
                "max_z_m": float(footprint["max_z"]),
                "width_m": float(footprint["max_x"]) - float(footprint["min_x"]),
                "depth_m": float(footprint["max_z"]) - float(footprint["min_z"]),
            }
        except (KeyError, TypeError, ValueError):
            issues.append({"violation_type": "workstation_bay_clearance", "instances": [instance_id]})
            continue
        front_vector = canonical.get("front_vector_xz")
        if front_vector is None:
            from tool.layout.aisle_facing import model_front_vector

            front_vector = model_front_vector(instance.get("model_asset_id") or placement.get("model_asset_id"))
        envelope = _workstation_service_envelope(
            body,
            float(placement.get("yaw_deg", (placement.get("rotation") or {}).get("yaw_deg") or 0.0)),
            {"maintenance_clearance_m": case_clearance, "front_vector_xz": front_vector},
            allowed_ids,
        )
        if not allowed_regions or not any(_inside_region(envelope, region) for region in allowed_regions):
            issues.append(
                {
                    "violation_type": "workstation_bay_clearance" if is_bay else "service_envelope_region",
                    "instances": [instance_id],
                    "details": {"allowed_region_ids": allowed_ids, "service_envelope": envelope},
                }
            )
        if is_case_clearance:
            boundary = {key: site.get(key) for key in ("min_x_m", "max_x_m", "min_z_m", "max_z_m")}
            if all(value is not None for value in boundary.values()) and not _inside_region(envelope, boundary):
                issues.append({
                    "violation_type": "service_envelope_boundary",
                    "instances": [instance_id],
                    "details": {"service_envelope": envelope, "boundary": boundary},
                })
            for obstacle in site.get("non_placeable_zones") or []:
                if str(obstacle.get("zone_type") or "").casefold() not in {"corridor", "aisle", "forbidden", "forbidden_zone"}:
                    continue
                service_footprint = {
                    "min_x": envelope["min_x_m"],
                    "max_x": envelope["max_x_m"],
                    "min_z": envelope["min_z_m"],
                    "max_z": envelope["max_z_m"],
                }
                body_footprint = {
                    "min_x": body["min_x_m"],
                    "max_x": body["max_x_m"],
                    "min_z": body["min_z_m"],
                    "max_z": body["max_z_m"],
                }
                if _rectangles_overlap(service_footprint, obstacle) and not _rectangles_overlap(body_footprint, obstacle):
                    issues.append({
                        "violation_type": "service_envelope_obstacle",
                        "instances": [instance_id],
                        "details": {"obstacle_id": obstacle.get("zone_id"), "service_envelope": envelope},
                    })
        bay = next(
            (
                item
                for item in ((layout.get("site") or {}).get("workstation_bays") or [])
                if instance_id in (item.get("assigned_instance_ids") or [])
            ),
            None,
        )
        row_alignment_z = (bay or {}).get("row_alignment_z_m")
        if row_alignment_z is not None:
            position = placement.get("position") or placement.get("pose") or {}
            try:
                center_z = float(position["z_m"])
                target_z = float(row_alignment_z)
            except (KeyError, TypeError, ValueError):
                continue
            if abs(center_z - target_z) > GEOMETRY_EPSILON_M:
                issues.append(
                    {
                        "violation_type": "workstation_row_alignment",
                        "instances": [instance_id],
                        "details": {
                            "measured_z_m": center_z,
                            "target_z_m": target_z,
                            "allowed_region_ids": allowed_ids,
                        },
                    }
                )
    return issues


def _workstation_bay_clearance_penalty(layout, flp_ir):
    issues = _workstation_bay_issues(layout, flp_ir)
    penalty = 0.0
    regions = (layout.get("site") or {}).get("placement_regions") or []
    placements = {
        str(item.get("instance_id") or ""): item
        for item in (layout or {}).get("placements") or []
    }
    for issue in issues:
        instance_id = (issue.get("instances") or [None])[0]
        placement = placements.get(str(instance_id)) or {}
        if issue.get("violation_type") == "workstation_row_alignment":
            details = issue.get("details") or {}
            penalty += (
                float(details.get("measured_z_m") or 0.0)
                - float(details.get("target_z_m") or 0.0)
            ) ** 2
            continue
        if issue.get("violation_type") == "service_envelope_obstacle":
            details = issue.get("details") or {}
            envelope = details.get("service_envelope") or {}
            obstacle = next(
                (
                    zone for zone in (layout.get("site") or {}).get("non_placeable_zones") or []
                    if zone.get("zone_id") == details.get("obstacle_id")
                ),
                {},
            )
            x_overlap = min(float(envelope.get("max_x_m") or 0.0), float(obstacle.get("max_x_m") or 0.0)) - max(float(envelope.get("min_x_m") or 0.0), float(obstacle.get("min_x_m") or 0.0))
            z_overlap = min(float(envelope.get("max_z_m") or 0.0), float(obstacle.get("max_z_m") or 0.0)) - max(float(envelope.get("min_z_m") or 0.0), float(obstacle.get("min_z_m") or 0.0))
            penalty += max(0.0, x_overlap) * max(0.0, z_overlap)
            continue
        envelope = (issue.get("details") or {}).get("service_envelope") or {}
        allowed_ids = set((issue.get("details") or {}).get("allowed_region_ids") or [])
        candidates = [region for region in regions if not allowed_ids or region.get("zone_id") in allowed_ids]
        if not candidates:
            penalty += 100.0
            continue
        deficits = []
        for region in candidates:
            dx = max(
                0.0,
                float(region.get("min_x_m") or 0.0) - float(envelope.get("min_x_m") or 0.0),
                float(envelope.get("max_x_m") or 0.0) - float(region.get("max_x_m") or 0.0),
            )
            dz = max(
                0.0,
                float(region.get("min_z_m") or 0.0) - float(envelope.get("min_z_m") or 0.0),
                float(envelope.get("max_z_m") or 0.0) - float(region.get("max_z_m") or 0.0),
            )
            deficits.append(dx * dx + dz * dz)
        penalty += min(deficits, default=100.0)
    return penalty


def _inside_region(footprint, region):
    return (
        footprint["min_x_m"] >= float(region.get("min_x_m") or 0.0) - GEOMETRY_EPSILON_M
        and footprint["max_x_m"] <= float(region.get("max_x_m") or 0.0) + GEOMETRY_EPSILON_M
        and footprint["min_z_m"] >= float(region.get("min_z_m") or 0.0) - GEOMETRY_EPSILON_M
        and footprint["max_z_m"] <= float(region.get("max_z_m") or 0.0) + GEOMETRY_EPSILON_M
    )


def _overlap_penalty(placements):
    penalty = 0.0
    for index, left in enumerate(placements):
        for right in placements[index + 1 :]:
            x_overlap = min(left["footprint"]["max_x"], right["footprint"]["max_x"]) - max(
                left["footprint"]["min_x"],
                right["footprint"]["min_x"],
            )
            z_overlap = min(left["footprint"]["max_z"], right["footprint"]["max_z"]) - max(
                left["footprint"]["min_z"],
                right["footprint"]["min_z"],
            )
            if x_overlap > GEOMETRY_EPSILON_M and z_overlap > GEOMETRY_EPSILON_M:
                penalty += x_overlap * z_overlap
    return penalty


def _boundary_penalty(placements, site):
    min_x = -float(site.get("width_m") or 0.0) / 2
    max_x = float(site.get("width_m") or 0.0) / 2
    min_z = -float(site.get("depth_m") or 0.0) / 2
    max_z = float(site.get("depth_m") or 0.0) / 2
    penalty = 0.0
    for placement in placements:
        footprint = placement["footprint"]
        penalty += max(0.0, min_x - footprint["min_x"]) ** 2
        penalty += max(0.0, footprint["max_x"] - max_x) ** 2
        penalty += max(0.0, min_z - footprint["min_z"]) ** 2
        penalty += max(0.0, footprint["max_z"] - max_z) ** 2
    return penalty


def _non_placeable_penalty(placements, site):
    penalty = 0.0
    for placement in placements:
        footprint = placement["footprint"]
        for zone in site.get("non_placeable_zones", []):
            x_overlap = min(footprint["max_x"], zone["max_x_m"]) - max(footprint["min_x"], zone["min_x_m"])
            z_overlap = min(footprint["max_z"], zone["max_z_m"]) - max(footprint["min_z"], zone["min_z_m"])
            if x_overlap > GEOMETRY_EPSILON_M and z_overlap > GEOMETRY_EPSILON_M:
                penalty += x_overlap * z_overlap
    return penalty


def _clearance_penalty(placements, flp_ir):
    clearance_by_instance = _clearance_by_instance(flp_ir)
    if not clearance_by_instance:
        return 0.0
    penalty = 0.0
    for index, left in enumerate(placements):
        for right in placements[index + 1 :]:
            required = max(
                clearance_by_instance.get(left["instance_id"], 0.0),
                clearance_by_instance.get(right["instance_id"], 0.0),
            )
            deficit = _clearance_deficit(left, right, required)
            penalty += deficit * deficit
    return penalty


def _clearance_deficit(left, right, required):
    if required <= 0:
        return 0.0
    x_gap = max(
        right["footprint"]["min_x"] - left["footprint"]["max_x"],
        left["footprint"]["min_x"] - right["footprint"]["max_x"],
        0.0,
    )
    z_gap = max(
        right["footprint"]["min_z"] - left["footprint"]["max_z"],
        left["footprint"]["min_z"] - right["footprint"]["max_z"],
        0.0,
    )
    actual = math.hypot(x_gap, z_gap) if x_gap > 0 and z_gap > 0 else max(x_gap, z_gap)
    return max(0.0, required - actual)


def _clearance_by_instance(flp_ir):
    clearances = {}
    if not flp_ir:
        return clearances
    for constraint in flp_ir.get("constraints", {}).get("hard", []):
        if constraint.get("constraint_type") != "minimum_clearance":
            continue
        clearances.update((constraint.get("parameters") or {}).get("clearance_by_instance") or {})
    return {instance_id: float(value) for instance_id, value in clearances.items()}


def _initial_genetic_population(flp_ir, instances, seed_layout, site, population_size, rng):
    base_state = (
        _initial_genetic_state(
            flp_ir,
            instances,
            seed_layout,
            site,
            rng,
        )
        if seed_layout
        else _random_genetic_state(instances, site, rng)
    )
    population = [_encode_genetic_state(base_state, instances)]
    for guided_state in _flow_guided_genetic_states(
        base_state,
        instances,
        flp_ir,
        site,
    ):
        if len(population) >= population_size:
            break
        population.append(_encode_genetic_state(guided_state, instances))
    while len(population) < population_size:
        state = _random_genetic_state(instances, site, rng)
        population.append(_encode_genetic_state(state, instances))
    return population


def _flow_guided_genetic_states(base_state, instances, flp_ir, site):
    """Seed one GA population with deterministic objective-directed variants."""

    instance_by_id = {
        item["instance_id"]: item for item in instances
    }
    flows = [
        flow
        for flow in (flp_ir or {}).get("flow_relations") or []
        if flow.get("from_instance_id") in base_state
        and flow.get("to_instance_id") in base_state
    ]
    if not flows:
        return []

    neighbors = {instance_id: [] for instance_id in base_state}
    for flow in flows:
        left_id = flow["from_instance_id"]
        right_id = flow["to_instance_id"]
        weight = max(
            GEOMETRY_EPSILON_M,
            float(flow.get("flow_weight") or 1.0),
        )
        neighbors[left_id].append((right_id, weight))
        neighbors[right_id].append((left_id, weight))

    variants = []
    for alpha in (0.08, 0.16, 0.24, 0.32):
        state = {
            instance_id: dict(item)
            for instance_id, item in base_state.items()
        }
        for instance_id, peers in neighbors.items():
            if not peers or instance_id not in instance_by_id:
                continue
            total_weight = sum(weight for _peer, weight in peers)
            target_x = sum(
                base_state[peer]["x_m"] * weight
                for peer, weight in peers
            ) / total_weight
            target_z = sum(
                base_state[peer]["z_m"] * weight
                for peer, weight in peers
            ) / total_weight
            state[instance_id]["x_m"] += alpha * (
                target_x - base_state[instance_id]["x_m"]
            )
            state[instance_id]["z_m"] += alpha * (
                target_z - base_state[instance_id]["z_m"]
            )
            _clamp_state_item(
                state[instance_id],
                instance_by_id[instance_id],
                site,
            )
        variants.append(state)

    for flow in flows:
        left_id = flow["from_instance_id"]
        right_id = flow["to_instance_id"]
        for moved_id, peer_id in (
            (left_id, right_id),
            (right_id, left_id),
        ):
            state = {
                instance_id: dict(item)
                for instance_id, item in base_state.items()
            }
            state[moved_id]["x_m"] += 0.2 * (
                base_state[peer_id]["x_m"]
                - base_state[moved_id]["x_m"]
            )
            state[moved_id]["z_m"] += 0.2 * (
                base_state[peer_id]["z_m"]
                - base_state[moved_id]["z_m"]
            )
            _clamp_state_item(
                state[moved_id],
                instance_by_id[moved_id],
                site,
            )
            variants.append(state)
    return variants


def _random_genetic_state(instances, site, rng):
    state = {}
    for instance in instances:
        yaw_deg = _random_allowed_yaw(instance, rng)
        width, depth, _height = _oriented_dimensions(instance, yaw_deg)
        bounds = _feasible_center_bounds(
            site,
            width,
            depth,
            instance_id=instance["instance_id"],
            yaw_deg=yaw_deg,
        )
        if bounds:
            selected = rng.choice(bounds)
            x_m = rng.uniform(selected["min_x"], selected["max_x"])
            z_m = rng.uniform(selected["min_z"], selected["max_z"])
        else:
            x_m = 0.0
            z_m = 0.0
        state[instance["instance_id"]] = {
            "x_m": x_m,
            "z_m": z_m,
            "yaw_deg": yaw_deg,
        }
        _clamp_state_item(state[instance["instance_id"]], instance, site)
    return state


def _random_allowed_yaw(instance, rng):
    allowed = instance.get("allowed_orientations_deg") or []
    if allowed:
        normalized = sorted({_normalized_yaw(value) for value in allowed})
        return rng.choice(normalized)
    return rng.choice([0.0, 90.0, 180.0, 270.0])


def _encode_genetic_state(state, instances):
    return [
        (
            float(state[instance["instance_id"]]["x_m"]),
            float(state[instance["instance_id"]]["z_m"]),
            _normalized_yaw(state[instance["instance_id"]]["yaw_deg"]),
        )
        for instance in instances
    ]


def _decode_genetic_chromosome(chromosome, instances):
    return {
        instance["instance_id"]: {
            "x_m": float(gene[0]),
            "z_m": float(gene[1]),
            "yaw_deg": _normalized_yaw(gene[2]),
        }
        for instance, gene in zip(instances, chromosome)
    }


def _layout_from_genetic_chromosome(flp_ir, instances, chromosome, site):
    layout = _layout_from_genetic_state(
        flp_ir,
        instances,
        _decode_genetic_chromosome(chromosome, instances),
        site,
    )
    layout["generation_policy"] = "genetic_algorithm"
    return layout


def _copy_chromosome(chromosome):
    return [tuple(gene) for gene in chromosome]


def _tournament_select(evaluated, tournament_size, rng):
    contenders = rng.sample(evaluated, k=min(len(evaluated), tournament_size))
    return max(contenders, key=lambda item: genetic_fitness(item[0]))[1]


def _uniform_crossover(parent_a, parent_b, rng):
    child_a = []
    child_b = []
    for gene_a, gene_b in zip(parent_a, parent_b):
        child_a.append(tuple(a if rng.random() < 0.5 else b for a, b in zip(gene_a, gene_b)))
        child_b.append(tuple(b if rng.random() < 0.5 else a for a, b in zip(gene_a, gene_b)))
    return child_a, child_b


def _mutate_genetic_chromosome(chromosome, instances, flp_ir, site, move_step, mutation_rate, rng):
    state = _decode_genetic_chromosome(chromosome, instances)
    mutation_count = 0
    for instance in instances:
        if rng.random() >= mutation_rate:
            continue
        mutation_count += 1
        item = state[instance["instance_id"]]
        operation = rng.random()
        if operation < 0.15 and _can_rotate_90(instance):
            allowed = instance.get("allowed_orientations_deg") or [0, 90, 180, 270]
            alternatives = sorted({_normalized_yaw(a) for a in allowed} - {_normalized_yaw(item["yaw_deg"])})
            if alternatives:
                item["yaw_deg"] = rng.choice(alternatives)
        elif operation < 0.35:
            width, depth, _height = _oriented_dimensions(instance, item["yaw_deg"])
            bounds = _feasible_center_bounds(
                site,
                width,
                depth,
                instance_id=instance["instance_id"],
                yaw_deg=item["yaw_deg"],
            )
            if bounds:
                selected = rng.choice(bounds)
                item["x_m"] = rng.uniform(selected["min_x"], selected["max_x"])
                item["z_m"] = rng.uniform(selected["min_z"], selected["max_z"])
        else:
            dx, dz = _neighbor_delta_toward_flow(state, instance, flp_ir, move_step, rng)
            item["x_m"] += dx
            item["z_m"] += dz
        _clamp_state_item(item, instance, site)
    return _encode_genetic_state(state, instances), mutation_count


def _initial_genetic_state(
    flp_ir,
    instances,
    seed_layout,
    site,
    rng,
):
    source_layout = seed_layout or {}
    placements_by_id = {
        placement.get("instance_id"): placement
        for placement in source_layout.get("placements", [])
        if placement.get("instance_id")
    }
    state = {}
    for instance in instances:
        instance_id = instance["instance_id"]
        placement = placements_by_id.get(instance_id)
        if placement:
            position = placement.get("position") or {}
            yaw_deg = float(placement.get("yaw_deg", (placement.get("rotation") or {}).get("yaw_deg") or 0.0))
            state[instance_id] = {
                "x_m": float(position.get("x_m") or 0.0),
                "z_m": float(position.get("z_m") or 0.0),
                "yaw_deg": _normalized_yaw(yaw_deg),
            }
        else:
            state[instance_id] = _random_genetic_state(
                [instance],
                site,
                rng,
            )[instance_id]
        _clamp_state_item(state[instance_id], instance, site)
    return state


def _layout_from_genetic_state(flp_ir, instances, state, site):
    placements = []
    for instance in instances:
        item = state[instance["instance_id"]]
        width, depth, height = _oriented_dimensions(instance, item["yaw_deg"])
        placements.append(_placement(instance, width, depth, height, item["x_m"], item["z_m"], yaw_deg=item["yaw_deg"]))
    layout = align_transfer_cells({
        "layout_id": "ROUGH_LAYOUT_001",
        "layout_type": "rough_design",
        "site": layout_site(site),
        "placements": placements,
        "generation_policy": "genetic_algorithm",
    }, flp_ir)
    layout["site"]["workcell_areas"] = workcell_regions(layout,flp_ir)
    return layout


def _neighbor_delta_toward_flow(state, instance, flp_ir, move_step, rng):
    instance_id = instance["instance_id"]
    peers = []
    for flow in flp_ir.get("flow_relations", []) if flp_ir else []:
        if flow.get("from_instance_id") == instance_id and flow.get("to_instance_id") in state:
            peers.append(state[flow["to_instance_id"]])
        elif flow.get("to_instance_id") == instance_id and flow.get("from_instance_id") in state:
            peers.append(state[flow["from_instance_id"]])
    if peers and rng.random() < 0.65:
        peer = rng.choice(peers)
        current = state[instance_id]
        dx = _signed_step(peer["x_m"] - current["x_m"], move_step, rng)
        dz = _signed_step(peer["z_m"] - current["z_m"], move_step, rng)
        return dx, dz
    return rng.uniform(-move_step, move_step), rng.uniform(-move_step, move_step)


def _signed_step(delta, move_step, rng):
    if abs(delta) <= GEOMETRY_EPSILON_M:
        return rng.uniform(-move_step, move_step)
    return math.copysign(rng.uniform(0.0, move_step), delta)


def _clamp_state_item(item, instance, site):
    width, depth, _height = _oriented_dimensions(instance, item["yaw_deg"])
    bounds = _feasible_center_bounds(
        site,
        width,
        depth,
        instance_id=instance["instance_id"],
        yaw_deg=item["yaw_deg"],
    )
    if not bounds:
        min_x = float(
            site.get("min_x_m")
            if site.get("min_x_m") is not None
            else -site["width_m"] / 2
        ) + width / 2
        max_x = float(
            site.get("max_x_m")
            if site.get("max_x_m") is not None
            else site["width_m"] / 2
        ) - width / 2
        min_z = float(
            site.get("min_z_m")
            if site.get("min_z_m") is not None
            else -site["depth_m"] / 2
        ) + depth / 2
        max_z = float(
            site.get("max_z_m")
            if site.get("max_z_m") is not None
            else site["depth_m"] / 2
        ) - depth / 2
        item["x_m"] = min(max(item["x_m"], min_x), max_x)
        item["z_m"] = min(max(item["z_m"], min_z), max_z)
        return

    best = min(
        bounds,
        key=lambda bound: _squared_distance_to_bounds(item["x_m"], item["z_m"], bound),
    )
    item["x_m"] = min(max(item["x_m"], best["min_x"]), best["max_x"])
    item["z_m"] = min(max(item["z_m"], best["min_z"]), best["max_z"])


def _feasible_center_bounds(
    site,
    width,
    depth,
    *,
    instance_id=None,
    yaw_deg=0.0,
):
    cell_bounds = _transfer_cell_center_bounds(site, instance_id, yaw_deg)
    if cell_bounds is not None:
        return cell_bounds
    bounds = []
    regions = placement_regions(site)
    constraint = (
        (
            (site.get("canonical_constraint_spec") or {}).get(
                "instance_constraints"
            )
            or {}
        ).get(instance_id)
        or {}
    )
    allowed_region_ids = set(
        constraint.get("allowed_region_ids") or []
    )
    if allowed_region_ids:
        regions = [
            region
            for region in regions
            if region.get("zone_id") in allowed_region_ids
        ]
    margins = _workstation_service_margins(constraint, allowed_region_ids, yaw_deg)
    row_alignment_z = constraint.get("workstation_row_alignment_z_m")
    for region in regions:
        min_x = region["min_x_m"] + margins["left_m"] + width / 2
        max_x = region["max_x_m"] - margins["right_m"] - width / 2
        min_z = region["min_z_m"] + margins["south_m"] + depth / 2
        max_z = region["max_z_m"] - margins["north_m"] - depth / 2
        if row_alignment_z is not None:
            if float(row_alignment_z) < min_z - GEOMETRY_EPSILON_M or float(row_alignment_z) > max_z + GEOMETRY_EPSILON_M:
                continue
            min_z = max_z = float(row_alignment_z)
        if min_x <= max_x + GEOMETRY_EPSILON_M and min_z <= max_z + GEOMETRY_EPSILON_M:
            bounds.append({"min_x": min_x, "max_x": max_x, "min_z": min_z, "max_z": max_z})
    return bounds


def _transfer_cell_center_bounds(site, instance_id, yaw_deg):
    """Bound a conveyor center by the full connected cell footprint and access strip."""

    spec = (site or {}).get("canonical_constraint_spec") or {}
    problem = spec.get("transfer_problem") or {}
    cell = next(
        (
            item
            for item in transfer_cells(problem)
            if str(item.get("conveyor_instance_id") or "") == str(instance_id or "")
        ),
        None,
    )
    if cell is None:
        return None

    normalized_yaw = _normalized_yaw(yaw_deg)
    cache = (site or {}).setdefault("_transfer_cell_center_bounds_cache", {})
    cache_key = (str(instance_id), round(normalized_yaw, 8))
    if cache_key in cache:
        return copy.deepcopy(cache[cache_key])

    if not math.isclose(normalized_yaw / 90.0, round(normalized_yaw / 90.0), abs_tol=1e-8):
        cache[cache_key] = []
        return []

    facts = {
        str(item.get("instance_id") or ""): item
        for item in problem.get("equipment_instances") or []
    }
    ids = [
        str(cell.get(key) or "")
        for key in ("conveyor_instance_id", "input_table_instance_id", "output_table_instance_id")
    ]
    if any(not key or key not in facts for key in ids):
        cache[cache_key] = []
        return []

    conveyor_id, input_id, output_id = ids
    conveyor_geometry = (facts[conveyor_id].get("geometry") or {}).get("dimensions_m") or {}
    try:
        conveyor_length = float(conveyor_geometry["width_m"])
        end_gap = float(cell.get("end_gap_m") or 0.0)
        strip = float(cell.get("operator_strip_m") or 0.0)
    except (KeyError, TypeError, ValueError):
        cache[cache_key] = []
        return []

    angle = math.radians(normalized_yaw)
    ux, uz = round(math.cos(angle)), round(-math.sin(angle))
    relative_centers = {conveyor_id: (0.0, 0.0)}
    for table_id, sign in ((input_id, -1.0), (output_id, 1.0)):
        table_dimensions = (facts[table_id].get("geometry") or {}).get("dimensions_m") or {}
        try:
            table_length = float(table_dimensions["width_m"])
        except (KeyError, TypeError, ValueError):
            cache[cache_key] = []
            return []
        distance = conveyor_length / 2.0 + end_gap + table_length / 2.0
        relative_centers[table_id] = (sign * distance * ux, sign * distance * uz)

    rows = []
    for member_id in ids:
        fact = facts[member_id]
        constraint = (spec.get("instance_constraints") or {}).get(member_id) or {}
        try:
            width, depth, _height = _oriented_dimensions(fact, normalized_yaw)
        except (KeyError, TypeError, ValueError):
            cache[cache_key] = []
            return []
        margins = _workstation_service_margins(
            constraint,
            set(constraint.get("allowed_region_ids") or []),
            normalized_yaw,
        )
        offset_x, offset_z = relative_centers[member_id]
        rows.append(
            {
                "id": member_id,
                "offset_x": offset_x,
                "offset_z": offset_z,
                "min_x_rel": offset_x - width / 2.0 - margins["left_m"],
                "max_x_rel": offset_x + width / 2.0 + margins["right_m"],
                "min_z_rel": offset_z - depth / 2.0 - margins["south_m"],
                "max_z_rel": offset_z + depth / 2.0 + margins["north_m"],
                "constraint": constraint,
                "fact": fact,
            }
        )

    # The operator strip sits south of the cell body, inside the same usable
    # region. Table/conveyor envelopes already include both end gaps.
    cell_min_x_rel = min(item["min_x_rel"] for item in rows)
    cell_max_x_rel = max(item["max_x_rel"] for item in rows)
    cell_min_z_rel = min(item["min_z_rel"] for item in rows) - strip
    cell_max_z_rel = max(item["max_z_rel"] for item in rows)

    regions = placement_regions(site)
    regions_by_id = {str(item.get("zone_id") or ""): item for item in regions}
    region_choices = []
    for row in rows:
        allowed_ids = set(row["constraint"].get("allowed_region_ids") or [])
        choices = [
            item
            for item in regions
            if not allowed_ids or str(item.get("zone_id") or "") in allowed_ids
        ]
        if not choices:
            cache[cache_key] = []
            return []
        region_choices.append(choices)

    # Intersect each member's permitted region set. This supports both one
    # shared region ID and separately named regions that overlap geometrically.
    common_regions = [None]
    for choices in region_choices:
        intersections = []
        for current in common_regions:
            for region in choices:
                if current is None:
                    candidate = {
                        key: float(region[f"{key}_m"])
                        for key in ("min_x", "max_x", "min_z", "max_z")
                    }
                else:
                    candidate = {
                        "min_x": max(current["min_x"], float(region["min_x_m"])),
                        "max_x": min(current["max_x"], float(region["max_x_m"])),
                        "min_z": max(current["min_z"], float(region["min_z_m"])),
                        "max_z": min(current["max_z"], float(region["max_z_m"])),
                    }
                if (
                    candidate["min_x"] <= candidate["max_x"] + GEOMETRY_EPSILON_M
                    and candidate["min_z"] <= candidate["max_z"] + GEOMETRY_EPSILON_M
                ):
                    intersections.append(candidate)
        deduplicated = {}
        for candidate in intersections:
            key = tuple(round(candidate[name], 8) for name in ("min_x", "max_x", "min_z", "max_z"))
            deduplicated[key] = candidate
        common_regions = list(deduplicated.values())
        if not common_regions:
            cache[cache_key] = []
            return []

    result = []
    for region in common_regions:
        candidate = {
            "min_x": region["min_x"] - cell_min_x_rel,
            "max_x": region["max_x"] - cell_max_x_rel,
            "min_z": region["min_z"] - cell_min_z_rel,
            "max_z": region["max_z"] - cell_max_z_rel,
        }
        if candidate["min_x"] > candidate["max_x"] + GEOMETRY_EPSILON_M:
            continue
        if candidate["min_z"] > candidate["max_z"] + GEOMETRY_EPSILON_M:
            continue

        # A source-locked cell member anchors the conveyor's center. Respect a
        # captured fixed or initial pose instead of moving that endpoint.
        for row in rows:
            if not (row["constraint"].get("locked") or row["fact"].get("locked")):
                continue
            pose = row["fact"].get("fixed_pose") or row["fact"].get("initial_pose") or {}
            pose = pose.get("pose") or pose
            try:
                locked_x = float(pose["x_m"]) - row["offset_x"]
                locked_z = float(pose["z_m"]) - row["offset_z"]
            except (KeyError, TypeError, ValueError):
                continue
            candidate["min_x"] = max(candidate["min_x"], locked_x)
            candidate["max_x"] = min(candidate["max_x"], locked_x)
            candidate["min_z"] = max(candidate["min_z"], locked_z)
            candidate["max_z"] = min(candidate["max_z"], locked_z)
        if (
            candidate["min_x"] <= candidate["max_x"] + GEOMETRY_EPSILON_M
            and candidate["min_z"] <= candidate["max_z"] + GEOMETRY_EPSILON_M
        ):
            result.append(candidate)

    cache[cache_key] = copy.deepcopy(result)
    return result


def _workstation_service_margins(constraint, allowed_region_ids, yaw_deg):
    clearance = (constraint or {}).get("maintenance_clearance_m") or {}
    if clearance.get("source") != "user_input":
        return {"left_m": 0.0, "right_m": 0.0, "south_m": 0.0, "north_m": 0.0}
    from tool.layout.aisle_facing import case_clearance_margins

    return case_clearance_margins(
        clearance,
        (constraint or {}).get("front_vector_xz"),
        yaw_deg,
    )


def _squared_distance_to_bounds(x, z, bounds):
    clamped_x = min(max(x, bounds["min_x"]), bounds["max_x"])
    clamped_z = min(max(z, bounds["min_z"]), bounds["max_z"])
    return (x - clamped_x) ** 2 + (z - clamped_z) ** 2


def _oriented_dimensions(instance, yaw_deg):
    dimensions = instance["geometry"]["dimensions_m"]
    width = float(dimensions["width_m"])
    depth = float(dimensions["depth_m"])
    height = float(dimensions.get("height_m") or 0.1)
    yaw = _normalized_yaw(yaw_deg)
    if yaw % 180 == 90.0:
        return depth, width, height
    if yaw % 180 != 0.0:
        cosine,sine=abs(math.cos(math.radians(yaw))),abs(math.sin(math.radians(yaw)))
        return width*cosine+depth*sine,width*sine+depth*cosine,height
    return width, depth, height


def _normalized_yaw(yaw_deg):
    # Opposite machine faces have identical footprints but different access
    # directions. Do not collapse the full pose into a footprint-axis class.
    return float(yaw_deg) % 360.0


def _can_rotate_90(instance):
    allowed = instance.get("allowed_orientations_deg") or []
    if not allowed:
        return True
    normalized = {_normalized_yaw(value) for value in allowed}
    return len(normalized) > 1


def overlaps(left, right):
    return (
        left["min_x"] < right["max_x"] - GEOMETRY_EPSILON_M
        and left["max_x"] > right["min_x"] + GEOMETRY_EPSILON_M
        and left["min_z"] < right["max_z"] - GEOMETRY_EPSILON_M
        and left["max_z"] > right["min_z"] + GEOMETRY_EPSILON_M
    )


def _rectangles_overlap(footprint, zone):
    return (
        footprint["min_x"] < zone["max_x_m"] - GEOMETRY_EPSILON_M
        and footprint["max_x"] > zone["min_x_m"] + GEOMETRY_EPSILON_M
        and footprint["min_z"] < zone["max_z_m"] - GEOMETRY_EPSILON_M
        and footprint["max_z"] > zone["min_z_m"] + GEOMETRY_EPSILON_M
    )


def _placement(instance, width, depth, height, center_x, center_z, *, yaw_deg=0.0):
    return {
        "instance_id": instance["instance_id"],
        "model_asset_id": instance["model_asset_id"],
        "equipment_type": instance.get("role") or instance.get("model_asset_id"),
        "position": {"x_m": center_x, "y_m": 0.0, "z_m": center_z},
        "rotation": {"yaw_deg": yaw_deg},
        "yaw_deg": yaw_deg,
        "dimensions_m": {"width_m": width, "height_m": height, "depth_m": depth},
        "footprint": {
            "width_m": width,
            "depth_m": depth,
            "height_m": height,
            "min_x": center_x - width / 2,
            "max_x": center_x + width / 2,
            "min_z": center_z - depth / 2,
            "max_z": center_z + depth / 2,
        },
    }


def _manhattan(left, right):
    return abs(left["position"]["x_m"] - right["position"]["x_m"]) + abs(
        left["position"]["z_m"] - right["position"]["z_m"]
    )
