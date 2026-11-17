"""Open-loop execution of a user-authored ConstraintRepair contract."""

from __future__ import annotations

import copy
import time

from tool.validation.canonical_constraint_validator import CanonicalConstraintValidator
from tool.layout.layout_solvers import solve_canonical_feasibility


class ViolationResolutionExecutor:
    """Resolve one controlled violation within the user-authorized scope."""

    def __init__(self):
        self.validator = CanonicalConstraintValidator()

    def execute(
        self,
        layout,
        flp_ir,
        canonical_spec,
        task_contract,
        *,
        solver_options=None,
        time_limit_s=5.0,
    ):
        started = time.perf_counter()
        contract = copy.deepcopy(task_contract or {})
        movable = set(contract.get("movable_instance_ids") or [])
        preserve = set(contract.get("preserve_instance_ids") or [])
        controlled = copy.deepcopy(contract.get("controlled_violation"))
        allowed = list(contract.get("allowed_repair_operations") or [])
        invalid_reason = None
        if contract.get("task_type") != "ConstraintRepair":
            invalid_reason = "task_type_must_be_ConstraintRepair"
        elif not controlled:
            invalid_reason = "controlled_violation_is_required"
        elif not movable:
            invalid_reason = "movable_instance_ids_are_required"
        elif movable & preserve:
            invalid_reason = "movable_and_preserve_scopes_overlap"
        elif not allowed:
            invalid_reason = "allowed_repair_operations_are_required"

        before_validation = self.validator.validate(
            layout,
            canonical_spec,
            reference_layout=layout,
            expected_instance_ids=contract.get("expected_instance_ids"),
            preserve_instance_ids=contract.get("preserve_instance_ids"),
        )
        if invalid_reason is None and not _controlled_violation_present(
            controlled,
            before_validation.get("violations") or [],
        ):
            invalid_reason = "controlled_violation_not_present"
        if invalid_reason:
            return self._result(
                "INVALID_REPAIR_CONTRACT",
                layout,
                contract,
                [],
                allowed,
                started,
                invalid_reason=invalid_reason,
                validation=before_validation,
            )

        execution_flp_ir = copy.deepcopy(flp_ir)
        execution_spec = copy.deepcopy(canonical_spec)
        if _translation_only_operations(allowed):
            _preserve_parent_orientations(
                execution_flp_ir,
                execution_spec,
                layout,
                movable,
            )
        candidate = solve_canonical_feasibility(
            execution_flp_ir,
            execution_spec,
            seed_layout=layout,
            reference_layout=layout,
            movable_instance_ids=sorted(movable),
            preserve_instance_ids=sorted(preserve),
            scope="user_constraint_repair",
            optimize_objective=False,
            time_limit_s=time_limit_s,
            solver_options=solver_options,
        )
        if candidate is None:
            return self._result(
                "REPAIR_UNSAT",
                layout,
                contract,
                [],
                allowed,
                started,
                invalid_reason=None,
                validation=before_validation,
            )

        candidate = _restore_out_of_scope(layout, candidate, movable)
        refinement = None
        if any(o.get("objective_type")=="minimize_layout_change" for o in flp_ir.get("objectives") or []):
            candidate,refinement=_reduce_repair_translation(candidate,layout,execution_spec,contract,self.validator)
        changed = _changed_instance_ids(layout, candidate)
        after_validation = self.validator.validate(
            candidate,
            execution_spec,
            reference_layout=layout,
            expected_instance_ids=contract.get("expected_instance_ids"),
            preserve_instance_ids=contract.get("preserve_instance_ids"),
        )
        controlled_remains = _controlled_violation_present(
            controlled,
            after_validation.get("violations") or [],
        )
        status = (
            "REPAIR_SUCCESS"
            if (
                _full_hard_validation_passed(after_validation)
                and not controlled_remains
                and set(changed) <= movable
            )
            else "REPAIR_UNSAT"
        )
        if status == "REPAIR_SUCCESS":
            candidate["known_invalid"] = False
            candidate["known_violations"] = [
                item
                for item in candidate.get("known_violations") or []
                if not _same_controlled_violation(controlled, item)
            ]
        result=self._result(
            status,
            candidate,
            contract,
            changed,
            allowed,
            started,
            invalid_reason=None,
            validation=after_validation,
        )
        result["translation_refinement"]=refinement
        return result

    @staticmethod
    def _result(
        status,
        layout,
        contract,
        changed,
        allowed,
        started,
        *,
        invalid_reason,
        validation,
    ):
        return {
            "schema_version": "violation-resolution-execution-v1",
            "status": status,
            "layout": copy.deepcopy(layout),
            "controlled_violation": copy.deepcopy(
                contract.get("controlled_violation")
            ),
            "allowed_repair_operations": list(allowed),
            "changed_instance_ids": list(changed),
            "validation": copy.deepcopy(validation),
            "invalid_reason": invalid_reason,
            "runtime_s": time.perf_counter() - started,
            "trace": {
                "execution_type": "user_constraint_repair",
                "trigger_source": "user_constraint_repair_request",
                "task_type": "ConstraintRepair",
                "movable_instance_ids": copy.deepcopy(
                    contract.get("movable_instance_ids") or []
                ),
                "preserve_instance_ids": copy.deepcopy(
                    contract.get("preserve_instance_ids") or []
                ),
                "changed_instance_ids": list(changed),
                "deviated_from_user_directive": False,
                "final_status": status,
            },
        }


def _reduce_repair_translation(candidate, reference, spec, contract, validator):
    """Reduce GA translation within the same scope and all existing constraints.

    Try the accepted parent coordinate, then approach its nearest feasible
    point from the current coordinate. This is a bounded local refinement,
    not a proof of globally minimum relocation.
    """
    result=copy.deepcopy(candidate)
    old={p["instance_id"]:p for p in reference.get("placements") or []}
    movable=set(contract.get("movable_instance_ids") or [])
    def movement(layout):
        return sum(abs(p["position"][a]-old[p["instance_id"]]["position"][a]) for p in layout["placements"] if p["instance_id"] in movable for a in ("x_m","z_m"))
    def valid(layout):
        return validator.validate(layout,spec,reference_layout=reference,expected_instance_ids=contract.get("expected_instance_ids"),preserve_instance_ids=contract.get("preserve_instance_ids"))["status"]=="VALID"
    if not valid(result):return result,dict(applied=False,reason="candidate_not_feasible")
    before=movement(result);evaluations=0
    indices=[i for i,p in enumerate(result["placements"]) if p["instance_id"] in movable]
    for ordinal,left in enumerate(indices):
        for right in indices[ordinal+1:]:
            for axis,lo,hi in (("x_m","min_x","max_x"),("z_m","min_z","max_z")):
                draft=copy.deepcopy(result)
                for index in (left,right):
                    p=draft["placements"][index];value=old[p["instance_id"]]["position"][axis];delta=value-p["position"][axis]
                    p["position"][axis]=value;p["footprint"][lo]+=delta;p["footprint"][hi]+=delta
                if movement(draft)>=movement(result)-1e-6:continue
                evaluations+=1
                if valid(draft):result=draft
    for _ in range(2):
        start=movement(result)
        for index,item in enumerate(result["placements"]):
            key=item["instance_id"]
            if key not in movable:continue
            for axis,lo,hi in (("x_m","min_x","max_x"),("z_m","min_z","max_z")):
                original=old[key]["position"][axis];current=result["placements"][index]["position"][axis]
                if abs(current-original)<1e-6:continue
                def trial(value):
                    draft=copy.deepcopy(result);p=draft["placements"][index];delta=value-p["position"][axis]
                    p["position"][axis]=value;p["footprint"][lo]+=delta;p["footprint"][hi]+=delta
                    return draft
                exact=trial(original);evaluations+=1
                if valid(exact):result=exact;continue
                infeasible,feasible=original,current
                best=result
                for _ in range(12):
                    middle=(infeasible+feasible)/2;draft=trial(middle);evaluations+=1
                    if valid(draft):feasible=middle;best=draft
                    else:infeasible=middle
                result=best
        if start-movement(result)<1e-6:break
    return result,dict(applied=True,translation_before_m=before,translation_after_m=movement(result),candidate_checks=evaluations,
        method="Constraint-preserving coordinate reversion and bounded line search; no global minimum claim")


def _controlled_violation_present(controlled, violations):
    return any(_same_controlled_violation(controlled, item) for item in violations)


def _same_controlled_violation(controlled, violation):
    controlled_type = str(
        controlled.get("constraint_type") or controlled.get("code") or ""
    )
    violation_code = str(
        violation.get("code") or violation.get("constraint_type") or ""
    )
    code_by_type = {
        "no_overlap": {"EQUIPMENT_OVERLAP", "no_overlap"},
        "inside_boundary": {"OUTSIDE_BOUNDARY", "inside_boundary"},
        "forbidden_zone_avoidance": {
            "FORBIDDEN_ZONE_INTERSECTION",
            "PROHIBITED_REGION_INTERSECTION",
            "forbidden_zone_avoidance",
        },
        "corridor_preservation": {
            "CORRIDOR_OBSTRUCTION",
            "ENTRANCE_EXIT_OBSTRUCTION",
            "FORBIDDEN_ZONE_INTERSECTION",
            "OUTSIDE_ALLOWED_REGION",
            "SERVICE_ENVELOPE_INTERSECTION",
            "SERVICE_ENVELOPE_OUTSIDE_BOUNDARY",
            "WORKSTATION_ROW_ALIGNMENT",
            "corridor_preservation",
        },
        "entrance_exit_unblocked": {
            "ENTRANCE_EXIT_OBSTRUCTION",
            "entrance_exit_unblocked",
        },
        "minimum_clearance": {
            "CLEARANCE_DEFICIT",
            "minimum_clearance",
        },
        "safety_clearance": {
            "CLEARANCE_DEFICIT",
            "safety_clearance",
        },
        "region_membership": {
            "OUTSIDE_ALLOWED_REGION",
            "region_membership",
        },
        "allowed_orientation": {
            "ORIENTATION_NOT_ALLOWED",
            "allowed_orientation",
        },
    }
    if violation_code not in code_by_type.get(controlled_type, {controlled_type}):
        return False
    controlled_ids = set(
        controlled.get("instance_ids")
        or controlled.get("derived_obstructing_instance_ids")
        or []
    )
    violation_ids = set(violation.get("instance_ids") or [])
    return not controlled_ids or bool(controlled_ids & violation_ids)


def _restore_out_of_scope(reference, candidate, movable):
    result = copy.deepcopy(candidate)
    reference_by_id = {
        item.get("instance_id"): item
        for item in (reference or {}).get("placements") or []
    }
    candidate_by_id = {
        item.get("instance_id"): item
        for item in result.get("placements") or []
    }
    for instance_id, placement in reference_by_id.items():
        if instance_id not in movable:
            candidate_by_id[instance_id] = copy.deepcopy(placement)
    result["placements"] = [
        candidate_by_id[instance_id]
        for instance_id in reference_by_id
        if instance_id in candidate_by_id
    ]
    return result


def _changed_instance_ids(before, after):
    before_by_id = {
        item.get("instance_id"): item
        for item in (before or {}).get("placements") or []
    }
    after_by_id = {
        item.get("instance_id"): item
        for item in (after or {}).get("placements") or []
    }
    return sorted(
        instance_id
        for instance_id in set(before_by_id) | set(after_by_id)
        if before_by_id.get(instance_id) != after_by_id.get(instance_id)
    )


def _translation_only_operations(allowed):
    translation_operations = {
        "move",
        "translate",
        "translation",
        "minimum_translation",
        "minimum_translation_exit_obstacle",
        "minimum_translation_remove_overlap",
        "minimum_translation_restore_clearance",
    }
    normalized = {str(item or "").strip().casefold() for item in allowed}
    return bool(normalized) and normalized <= translation_operations


def _preserve_parent_orientations(flp_ir, canonical_spec, layout, movable):
    parent_by_id = {
        item.get("instance_id"): item
        for item in (layout or {}).get("placements") or []
        if item.get("instance_id")
    }
    yaw_by_id = {
        instance_id: _placement_yaw(parent_by_id.get(instance_id))
        for instance_id in movable
        if instance_id in parent_by_id
    }
    for instance in (flp_ir or {}).get("equipment_instances") or []:
        instance_id = instance.get("instance_id")
        if instance_id in yaw_by_id:
            instance["allowed_orientations_deg"] = [yaw_by_id[instance_id]]
    constraints = (canonical_spec or {}).setdefault(
        "instance_constraints",
        {},
    )
    for instance_id, yaw in yaw_by_id.items():
        constraint = constraints.setdefault(instance_id, {})
        constraint["allowed_orientations_deg"] = [yaw]


def _placement_yaw(placement):
    placement = placement or {}
    values = (
        placement.get("yaw_deg"),
        (placement.get("rotation") or {}).get("yaw_deg"),
        (placement.get("pose") or {}).get("yaw_deg"),
    )
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _full_hard_validation_passed(validation):
    report = validation or {}
    return (
        report.get("status") == "VALID"
        and report.get("hard_constraints_satisfied") is True
        and int(report.get("violation_count") or 0) == 0
        and not (report.get("violations") or [])
    )
