"""First-class contracts for layout generation and existing-layout transformations."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field


SCHEMA_VERSION = "task-contract-v1"
POSE_TOLERANCE_M = 1e-6
EXISTING_LAYOUT_TRANSFORMATION_TASKS = frozenset(
    {"DirectiveEdit", "ConstraintRepair"}
)


@dataclass
class TaskContract:
    task_type: str
    complete: bool
    task_family: str | None = None
    expected_instance_ids: list[str] = field(default_factory=list)
    base_layout: dict | None = None
    target_instance_ids: list[str] = field(default_factory=list)
    movable_instance_ids: list[str] = field(default_factory=list)
    preserve_instance_ids: list[str] = field(default_factory=list)
    locked_instance_ids: list[str] = field(default_factory=list)
    edit_operations: list[dict] = field(default_factory=list)
    controlled_violation: dict | None = None
    allowed_repair_operations: list[str] = field(default_factory=list)
    edit_relaxation_policy: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    objective_required: bool = False
    baseline_objective_value: float | None = None
    objective_flow_relations: list[dict] = field(default_factory=list)
    strict_objective_improvement: bool = False
    metadata: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        return asdict(self)


class TaskContractCompiler:
    """Compile task semantics and authoritative state into one executable contract."""

    _TASK_FROM_INTENT = {
        "generate_feasible_layout": "Generate",
        "optimize_layout_from_scratch": "Generate",
        "optimize_existing_layout": "Optimize",
        "edit_layout": "DirectiveEdit",
        "constraint_repair": "ConstraintRepair",
    }

    def compile(self, intent=None, semantic_intent=None, authoritative_context=None):
        intent = intent or {}
        semantic_intent = semantic_intent or {}
        context = authoritative_context or {}
        has_authoritative_site = bool(context.get("site_snapshot"))
        task_type = context.get("task_type") or semantic_intent.get(
            "task_contract_type"
        )
        task_type = task_type or self._TASK_FROM_INTENT.get(
            intent.get("intent_type") or semantic_intent.get("intent_type"),
            "Generate",
        )
        task_type = {
            "Edit": "DirectiveEdit",
            "Repair": "ConstraintRepair",
        }.get(task_type, task_type)
        if task_type not in {
            "Generate",
            "Optimize",
            "DirectiveEdit",
            "ConstraintRepair",
        }:
            task_type = "Generate"

        base_layout = copy.deepcopy(context.get("authoritative_layout")) or None
        base_flp_ir = context.get("authoritative_flp_ir") or {}
        expected = _expected_instance_ids(context, base_flp_ir, base_layout)
        scope = base_flp_ir.get("optimization_scope") or {}
        locked = _unique(
            scope.get("locked_instances")
            or [
                item.get("instance_id")
                for item in base_flp_ir.get("equipment_instances") or []
                if item.get("locked")
            ]
        )
        movable = _unique(
            scope.get("movable_instances")
            or [item for item in expected if item not in locked]
        )
        selected = _unique(
            semantic_intent.get("selected_instance_ids")
            or context.get("selected_instance_ids")
            or []
        )

        if task_type == "DirectiveEdit":
            operations = copy.deepcopy(semantic_intent.get("edit_operations") or [])
            next_added = max([int(str(i).rsplit("_",1)[-1]) for i in expected if str(i).rsplit("_",1)[-1].isdigit()] or [0])+1
            added_ids=[]
            for operation in operations:
                if operation.get("op_type")=="add":
                    operation["instance_id"]=f"equipment_{next_added:03d}"
                    quantity=int(operation.get("quantity") or 1)
                    added_ids.extend(f"equipment_{i:03d}" for i in range(next_added,next_added+quantity))
                    next_added+=quantity
            targets = _unique(
                [
                    operation.get("instance_id")
                    or operation.get("target_instance_id")
                    for operation in operations
                ]
                + selected
            )
            operations = _bind_selected_targets(operations, selected)
            missing = _edit_missing_fields(operations, targets)
            if not has_authoritative_site:
                missing.insert(0, "site_snapshot")
            deleted = {
                operation.get("instance_id") or operation.get("target_instance_id")
                for operation in operations
                if str(
                    operation.get("op_type")
                    or operation.get("operation_type")
                    or operation.get("type")
                    or ""
                ).casefold()
                in {"delete", "remove"}
            }
            expected_after_edit = [item for item in expected if item not in deleted]+added_ids
            preserve = [item for item in expected if item not in targets]
            relaxation = (
                semantic_intent.get("edit_relaxation_policy")
                or (semantic_intent.get("edit_policy") or {}).get(
                    "relaxation_strategy"
                )
            )
            return TaskContract(
                task_type=task_type,
                complete=bool(base_layout and not missing),
                task_family="ExistingLayoutTransformation",
                expected_instance_ids=expected_after_edit,
                base_layout=base_layout,
                target_instance_ids=targets,
                movable_instance_ids=targets,
                preserve_instance_ids=preserve,
                locked_instance_ids=preserve,
                edit_operations=operations,
                edit_relaxation_policy=(
                    str(relaxation) if relaxation else None
                ),
                missing_fields=missing,
                objective_required=False,
                metadata={"selected_instance_ids": selected},
            )

        if task_type == "ConstraintRepair":
            known = copy.deepcopy(
                semantic_intent.get("controlled_initial_violations")
                or (base_layout or {}).get("known_violations")
                or []
            )
            controlled = copy.deepcopy(
                semantic_intent.get("controlled_violation")
            )
            if controlled is None and len(known) == 1:
                controlled = known[0]
            controlled_ids = _unique((controlled or {}).get("instance_ids") or [])
            repair_scope = semantic_intent.get("repair_scope") or {}
            requested_movable = _unique(
                repair_scope.get("movable_instance_ids")
                or semantic_intent.get("movable_instance_ids")
                or controlled_ids
            )
            movable = [item for item in movable if item in requested_movable]
            requested_preserve = _unique(
                repair_scope.get("preserve_instance_ids")
                or semantic_intent.get("preserve_instance_ids")
            )
            preserve = (
                requested_preserve
                if requested_preserve
                else [item for item in expected if item not in movable]
            )
            allowed_operations = _unique(
                semantic_intent.get("allowed_repair_operations")
                or (controlled or {}).get("allowed_repair_operations")
                or _default_repair_operations(controlled)
            )
            missing = []
            if not has_authoritative_site:
                missing.append("site_snapshot")
            if not base_layout:
                missing.append("base_layout")
            if not controlled:
                missing.append("controlled_violation")
            if not movable:
                missing.append("movable_instance_ids")
            if not allowed_operations:
                missing.append("allowed_repair_operations")
            return TaskContract(
                task_type=task_type,
                complete=not missing,
                task_family="ExistingLayoutTransformation",
                expected_instance_ids=expected,
                base_layout=base_layout,
                target_instance_ids=controlled_ids,
                movable_instance_ids=movable,
                preserve_instance_ids=preserve,
                locked_instance_ids=_unique(locked + preserve),
                controlled_violation=controlled,
                allowed_repair_operations=allowed_operations,
                missing_fields=missing,
                objective_required=False,
            )

        if task_type == "Optimize":
            missing = []
            if not has_authoritative_site:
                missing.append("site_snapshot")
            if not base_layout:
                missing.append("base_layout")
            return TaskContract(
                task_type=task_type,
                complete=not missing,
                expected_instance_ids=expected,
                base_layout=base_layout,
                movable_instance_ids=movable,
                preserve_instance_ids=locked,
                locked_instance_ids=locked,
                missing_fields=missing,
                objective_required=True,
                strict_objective_improvement=True,
            )

        missing = []
        if not has_authoritative_site:
            missing.append("site_snapshot")
        if not expected:
            missing.append("equipment_instances")
        return TaskContract(
            task_type="Generate",
            complete=not missing,
            expected_instance_ids=expected,
            base_layout=None,
            movable_instance_ids=expected,
            preserve_instance_ids=[],
            locked_instance_ids=[],
            missing_fields=missing,
            objective_required=False,
        )


def validate_task_contract(layout, contract, *, tolerance=POSE_TOLERANCE_M):
    contract = _contract_dict(contract)
    task_type = contract.get("task_type")
    expected = set(contract.get("expected_instance_ids") or [])
    placements = list((layout or {}).get("placements") or [])
    by_id = {item.get("instance_id"): item for item in placements}
    actual = [item.get("instance_id") for item in placements if item.get("instance_id")]
    errors = []

    missing = sorted(expected - set(actual))
    duplicates = sorted(
        instance_id for instance_id in set(actual) if actual.count(instance_id) > 1
    )
    unexpected = sorted(set(actual) - expected) if expected else []
    if missing:
        errors.append(
            _contract_error("MISSING_INSTANCE", missing, "restore_missing_instance")
        )
    if duplicates:
        errors.append(
            _contract_error("DUPLICATE_INSTANCE", duplicates, "remove_duplicate_instance")
        )
    if unexpected:
        errors.append(
            _contract_error("UNEXPECTED_INSTANCE", unexpected, "remove_unexpected_instance")
        )

    base_by_id = {
        item.get("instance_id"): item
        for item in ((contract.get("base_layout") or {}).get("placements") or [])
    }
    for instance_id in contract.get("preserve_instance_ids") or []:
        if instance_id in by_id and instance_id in base_by_id and not _same_placement(
            by_id[instance_id], base_by_id[instance_id], tolerance
        ):
            errors.append(
                _contract_error(
                    "PRESERVE_INSTANCE_CHANGED",
                    [instance_id],
                    "restore_preserved_instance",
                    preserve_instance_ids=contract.get("preserve_instance_ids") or [],
                )
            )

    satisfaction = 1.0
    if task_type == "DirectiveEdit" and contract.get("complete"):
        for operation in contract.get("edit_operations") or []:
            error = _validate_edit_operation(
                operation, by_id, base_by_id, tolerance=tolerance
            )
            if error:
                errors.append(error)
        target_count = max(1, len(contract.get("target_instance_ids") or []))
        target_errors = sum(
            1
            for item in errors
            if item["code"].startswith("EDIT_")
        )
        satisfaction = max(0.0, 1.0 - target_errors / target_count)

    if task_type == "ConstraintRepair" and contract.get("controlled_violation"):
        controlled = contract["controlled_violation"]
        controlled_ids = controlled.get("instance_ids") or []
        if (
            controlled.get("constraint_type") == "no_overlap"
            and len(controlled_ids) == 2
            and all(item in by_id for item in controlled_ids)
            and _footprints_overlap(
                by_id[controlled_ids[0]].get("footprint") or {},
                by_id[controlled_ids[1]].get("footprint") or {},
            )
        ):
            errors.append(
                _contract_error(
                    "CONTROLLED_VIOLATION_NOT_REPAIRED",
                    controlled_ids,
                    "minimum_translation_remove_overlap",
                    preserve_instance_ids=contract.get("preserve_instance_ids") or [],
                )
            )
            satisfaction = 0.0

    if task_type == "Optimize" and contract.get("complete"):
        baseline = contract.get("baseline_objective_value")
        relations = contract.get("objective_flow_relations") or []
        objective_problem = (contract.get("metadata") or {}).get("objective_problem") or {}
        if any(o.get("objective_type") == "minimize_agv_transport_cost" for o in objective_problem.get("objectives") or []):
            from tool.logistics.agv_transport_cost import agv_transport_cost
            candidate_objective = agv_transport_cost(layout, objective_problem)
        else:
            candidate_objective = _layout_objective(layout, relations)
        moved = [
            instance_id
            for instance_id in contract.get("movable_instance_ids") or []
            if instance_id in by_id
            and instance_id in base_by_id
            and not _same_placement(
                by_id[instance_id], base_by_id[instance_id], tolerance
            )
        ]
        if not moved:
            errors.append(
                _contract_error(
                    "OPTIMIZE_MOVABLE_PLACEMENTS_UNCHANGED",
                    contract.get("movable_instance_ids") or [],
                    "canonical_objective_optimize",
                    measured_value=[],
                    target_value="at_least_one_movable_instance_changed",
                )
            )
        if (
            baseline is None
            or candidate_objective is None
            or not (
                candidate_objective
                < float(baseline) - max(tolerance, 1e-9)
            )
        ):
            errors.append(
                _contract_error(
                    "OPTIMIZE_OBJECTIVE_NOT_IMPROVED",
                    contract.get("movable_instance_ids") or [],
                    "canonical_objective_optimize",
                    measured_value=candidate_objective,
                    target_value={
                        "strictly_less_than": baseline,
                    },
                )
            )
        if errors:
            satisfaction = max(
                0.0,
                1.0
                - sum(
                    item["code"].startswith("OPTIMIZE_")
                    for item in errors
                )
                / 2.0,
            )

    if errors and task_type not in {"DirectiveEdit", "ConstraintRepair"}:
        satisfaction = max(0.0, 1.0 - len(errors) / max(1, len(expected)))
    return {
        "schema_version": "task-contract-validation-v1",
        "status": "VALID" if not errors else "INVALID",
        "task_type": task_type,
        "contract_complete": bool(contract.get("complete")),
        "satisfaction": satisfaction,
        "errors": errors,
        "violation_count": len(errors),
    }


def _contract_dict(contract):
    if isinstance(contract, TaskContract):
        return contract.to_dict()
    return contract or {}


def _expected_instance_ids(context, flp_ir, layout):
    directory = context.get("instance_directory") or []
    values = [item.get("instance_id") for item in directory]
    if not any(values):
        values = [
            item.get("instance_id")
            for item in (flp_ir.get("equipment_instances") or [])
        ]
    if not any(values):
        values = [
            item.get("instance_id") for item in ((layout or {}).get("placements") or [])
        ]
    return _unique(values)


def _bind_selected_targets(operations, selected):
    result = []
    for operation in operations:
        item = copy.deepcopy(operation)
        if (
            not item.get("instance_id")
            and not item.get("target_instance_id")
            and len(selected) == 1
        ):
            item["instance_id"] = selected[0]
        result.append(item)
    return result


def _edit_missing_fields(operations, targets):
    missing = []
    if not targets:
        missing.append("edit_target")
    if not operations:
        missing.append("edit_operations")
    for operation in operations:
        operation_type = str(
            operation.get("op_type")
            or operation.get("operation_type")
            or operation.get("type")
            or ""
        ).casefold()
        if operation_type == "move":
            delta = operation.get("delta_m") or operation.get("translation_m") or {}
            move_mode = str(operation.get("move_mode") or "").casefold()
            if move_mode == "relative":
                values = [
                    delta.get(key)
                    for key in ("x_m", "z_m", "dx_m", "dz_m", "x", "z")
                ] + [
                    operation.get(key)
                    for key in ("dx_m", "dz_m")
                ]
                move_is_specified = any(
                    _finite(value) and abs(float(value)) > 1e-12
                    for value in values
                )
            else:
                move_is_specified = any(
                    _finite(delta.get(key))
                    for key in ("x_m", "z_m", "dx_m", "dz_m", "x", "z")
                ) or any(
                    _finite(operation.get(key))
                    for key in ("x_m", "z_m", "dx_m", "dz_m")
                )
            if not move_is_specified:
                distance = operation.get("distance_m")
                if not _finite(distance):
                    missing.append("move_distance")
    return _unique(missing)


def _validate_edit_operation(operation, by_id, base_by_id, *, tolerance):
    instance_id = operation.get("instance_id") or operation.get("target_instance_id")
    operation_type = str(
        operation.get("op_type")
        or operation.get("operation_type")
        or operation.get("type")
        or ""
    ).casefold()
    if operation_type in {"delete", "remove"}:
        if instance_id in by_id:
            return _contract_error(
                "EDIT_TARGET_NOT_DELETED",
                [instance_id],
                "restore_exact_edit_result",
            )
        return None
    if operation_type == "add":
        if instance_id not in by_id or instance_id in base_by_id:
            return _contract_error("EDIT_ADDED_INSTANCE_MISSING", [instance_id], "restore_exact_edit_result")
        if by_id[instance_id].get("model_asset_id") != operation.get("new_model_asset_id"):
            return _contract_error("EDIT_ADDED_ASSET_MISMATCH", [instance_id], "restore_exact_edit_result")
        return None
    if instance_id not in by_id or instance_id not in base_by_id:
        return _contract_error(
            "EDIT_TARGET_MISSING", [instance_id], "restore_missing_instance"
        )
    if operation_type == "rotate":
        actual_yaw = float(
            by_id[instance_id].get("yaw_deg")
            or (by_id[instance_id].get("rotation") or {}).get("yaw_deg")
            or 0.0
        ) % 360.0
        base_yaw = float(
            base_by_id[instance_id].get("yaw_deg")
            or (base_by_id[instance_id].get("rotation") or {}).get("yaw_deg")
            or 0.0
        )
        target = operation.get("target_yaw_deg")
        if target is None:
            target = base_yaw + float(operation.get("delta_yaw_deg") or 0.0)
        target = float(target) % 360.0
        if not math.isclose(actual_yaw, target, abs_tol=tolerance):
            return _contract_error(
                "EDIT_TARGET_ROTATION_MISMATCH",
                [instance_id],
                "restore_exact_edit_result",
                measured_value=actual_yaw,
                target_value=target,
            )
        return None
    if operation_type == "replace":
        target_asset = operation.get("new_model_asset_id")
        if by_id[instance_id].get("model_asset_id") != target_asset:
            return _contract_error(
                "EDIT_TARGET_REPLACEMENT_MISMATCH",
                [instance_id],
                "restore_exact_edit_result",
                measured_value=by_id[instance_id].get("model_asset_id"),
                target_value=target_asset,
            )
        return None
    if operation_type != "move":
        return None
    actual_pose = (
        by_id[instance_id].get("position")
        or by_id[instance_id].get("pose")
        or {}
    )
    base_pose = (
        base_by_id[instance_id].get("position")
        or base_by_id[instance_id].get("pose")
        or {}
    )
    move_mode = str(operation.get("move_mode") or "").casefold()
    if move_mode == "absolute":
        expected_x = _first_finite(operation.get("x_m"))
        expected_z = _first_finite(operation.get("z_m"))
        actual_x = float(actual_pose.get("x_m") or 0.0)
        actual_z = float(actual_pose.get("z_m") or 0.0)
        matches = (
            (expected_x is None or math.isclose(actual_x, expected_x, abs_tol=tolerance))
            and (expected_z is None or math.isclose(actual_z, expected_z, abs_tol=tolerance))
        )
        measured = {"x_m": actual_x, "z_m": actual_z}
        target = {"x_m": expected_x, "z_m": expected_z}
    else:
        expected_dx, expected_dz = _operation_delta(operation)
        actual_dx = float(actual_pose.get("x_m") or 0.0) - float(
            base_pose.get("x_m") or 0.0
        )
        actual_dz = float(actual_pose.get("z_m") or 0.0) - float(
            base_pose.get("z_m") or 0.0
        )
        matches = (
            math.isclose(actual_dx, expected_dx, abs_tol=tolerance)
            and math.isclose(actual_dz, expected_dz, abs_tol=tolerance)
        )
        measured = {"dx_m": actual_dx, "dz_m": actual_dz}
        target = {"dx_m": expected_dx, "dz_m": expected_dz}
    if not matches:
        return _contract_error(
            "EDIT_TARGET_POSE_MISMATCH",
            [instance_id],
            "restore_exact_edit_target_pose",
            measured_value=measured,
            target_value=target,
        )
    return None


def _default_repair_operations(controlled):
    constraint_type = str((controlled or {}).get("constraint_type") or "")
    return {
        "no_overlap": ["minimum_translation_remove_overlap", "local_ga_resolve"],
        "inside_boundary": ["project_inside_boundary", "local_ga_resolve"],
        "forbidden_zone_avoidance": [
            "minimum_translation_exit_obstacle",
            "local_ga_resolve",
        ],
        "corridor_preservation": [
            "minimum_translation_exit_obstacle",
            "local_ga_resolve",
        ],
    }.get(constraint_type, ["local_ga_resolve"] if constraint_type else [])


def _operation_delta(operation):
    delta = operation.get("delta_m") or operation.get("translation_m") or {}
    dx = _first_finite(
        delta.get("x_m"),
        delta.get("dx_m"),
        delta.get("x"),
        operation.get("dx_m"),
    )
    dz = _first_finite(
        delta.get("z_m"),
        delta.get("dz_m"),
        delta.get("z"),
        operation.get("dz_m"),
    )
    if dx is not None or dz is not None:
        return float(dx or 0.0), float(dz or 0.0)
    distance = float(operation.get("distance_m") or 0.0)
    direction = str(operation.get("direction") or "").casefold()
    return {
        "right": (distance, 0.0),
        "left": (-distance, 0.0),
        "forward": (0.0, distance),
        "north": (0.0, distance),
        "back": (0.0, -distance),
        "backward": (0.0, -distance),
        "south": (0.0, -distance),
    }.get(direction, (distance, 0.0))


def _contract_error(
    code,
    instance_ids,
    operation,
    *,
    preserve_instance_ids=None,
    measured_value=None,
    target_value=None,
):
    return {
        "code": code,
        "instance_ids": [item for item in instance_ids if item],
        "obstacle_id": None,
        "measured_value": measured_value,
        "target_value": target_value,
        "minimum_repair_displacement": None,
        "allowed_repair_operations": [operation],
        "preserve_instance_ids": list(preserve_instance_ids or []),
    }


def _same_placement(left, right, tolerance):
    # Footprint is derived from position + dimensions + yaw and is not an
    # independent preservation truth.
    left_keys = set(left) - {"footprint"}
    right_keys = set(right) - {"footprint"}
    if left_keys != right_keys:
        return False
    for key in left_keys:
        if key in {"pose", "position", "rotation", "dimensions_m"}:
            if not _numeric_mapping_equal(left[key], right[key], tolerance):
                return False
        elif key == "yaw_deg":
            if not math.isclose(
                float(left[key]),
                float(right[key]),
                abs_tol=tolerance,
            ):
                return False
        elif left[key] != right[key]:
            return False
    return True


def _numeric_mapping_equal(left, right, tolerance):
    if set(left or {}) != set(right or {}):
        return False
    for key, value in (left or {}).items():
        other = (right or {}).get(key)
        if isinstance(value, (int, float)) and isinstance(other, (int, float)):
            if not math.isclose(float(value), float(other), abs_tol=tolerance):
                return False
        elif value != other:
            return False
    return True


def _footprints_overlap(left, right):
    try:
        return (
            min(float(left["max_x"]), float(right["max_x"]))
            - max(float(left["min_x"]), float(right["min_x"]))
            > 1e-9
            and min(float(left["max_z"]), float(right["max_z"]))
            - max(float(left["min_z"]), float(right["min_z"]))
            > 1e-9
        )
    except (KeyError, TypeError, ValueError):
        return False


def _layout_objective(layout, relations):
    from tool.logistics.facility_logistics import fixed_flow_endpoints
    by_id = {
        item.get("instance_id"): item
        for item in ((layout or {}).get("placements") or [])
    }
    by_id.update(fixed_flow_endpoints(layout,None))
    total = 0.0
    used = False
    for relation in relations or []:
        left = by_id.get(relation.get("from_instance_id"))
        right = by_id.get(relation.get("to_instance_id"))
        if left is None or right is None:
            return None
        left_pose = left.get("position") or left.get("pose") or {}
        right_pose = right.get("position") or right.get("pose") or {}
        if not all(
            _finite(value)
            for value in (
                left_pose.get("x_m"),
                left_pose.get("z_m"),
                right_pose.get("x_m"),
                right_pose.get("z_m"),
            )
        ):
            return None
        weight = float(
            relation.get("flow_weight")
            or relation.get("weight")
            or 1.0
        )
        total += weight * (
            abs(float(left_pose["x_m"]) - float(right_pose["x_m"]))
            + abs(float(left_pose["z_m"]) - float(right_pose["z_m"]))
        )
        used = True
    return total if used else None


def _unique(values):
    result = []
    for value in values or []:
        if value is None:
            continue
        value = str(value)
        if value and value not in result:
            result.append(value)
    return result


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _first_finite(*values):
    for value in values:
        if _finite(value):
            return float(value)
    return None
