"""Internal recovery driven exclusively by Validator TypedViolation feedback."""

from __future__ import annotations

import copy
import math
import time

from tool.validation.canonical_constraint_validator import CanonicalConstraintValidator
from tool.representations.task_contracts import validate_task_contract


EPSILON_M = 1e-8


class ValidationDrivenRecovery:
    """Apply executable Validator feedback before escalating to a GA solve."""

    def __init__(self):
        self.validator = CanonicalConstraintValidator()

    def recover(
        self,
        layout,
        flp_ir,
        canonical_spec,
        typed_violations,
        task_contract,
        *,
        reference_layout=None,
        solver_options=None,
    ):
        start = time.perf_counter()
        contract = (
            task_contract.to_dict()
            if hasattr(task_contract, "to_dict")
            else copy.deepcopy(task_contract or {})
        )
        reference = (
            copy.deepcopy(reference_layout)
            or copy.deepcopy(contract.get("base_layout"))
            or None
        )
        candidate = copy.deepcopy(layout or reference or {})
        candidate.setdefault("placements", [])
        before = copy.deepcopy(candidate)
        actions = []
        preserve = set(contract.get("preserve_instance_ids") or [])
        movable = set(contract.get("movable_instance_ids") or [])
        if not movable:
            movable = set(canonical_spec.get("movable_instance_ids") or [])
        task_type = contract.get("task_type")
        if (
            task_type == "DirectiveEdit"
            and contract.get("edit_relaxation_policy")
            not in {"nearest_feasible"}
        ):
            return {
                "schema_version": "validation-driven-recovery-result-v1",
                "status": "RECOVERY_NOT_ALLOWED",
                "layout": candidate,
                "actions": [],
                "changed_instance_ids": [],
                "candidate_changed": False,
                "validation": self._validate(
                    candidate, canonical_spec, contract, reference
                ),
                "runtime_s": time.perf_counter() - start,
                "trace": _recovery_trace(
                    contract,
                    [],
                    "RECOVERY_NOT_ALLOWED",
                ),
            }

        self._restore_instance_set(
            candidate,
            flp_ir,
            canonical_spec,
            typed_violations,
            reference,
            contract,
            actions,
        )
        self._restore_preserved(candidate, reference, preserve, actions)

        for violation in typed_violations or []:
            self._apply_violation(
                candidate,
                canonical_spec,
                violation,
                movable,
                preserve,
                actions,
            )

        validation = self._validate(
            candidate, canonical_spec, contract, reference
        )
        if not validation["valid"]:
            local_ids = _unique(
                instance_id
                for item in validation["violations"]
                for instance_id in item.get("instance_ids") or []
                if instance_id in movable
            )
            solved = self._feasibility_solve(
                candidate,
                flp_ir,
                canonical_spec,
                contract,
                reference,
                movable_instance_ids=local_ids,
                preserve_instance_ids=preserve | (movable - set(local_ids)),
                scope="local",
                solver_options=solver_options,
            )
            if solved is not None:
                candidate = solved
                actions.append(
                    {
                        "operation": "local_ga_resolve",
                        "instance_ids": local_ids,
                    }
                )
                validation = self._validate(
                    candidate, canonical_spec, contract, reference
                )

        if not validation["valid"] and task_type not in {
            "DirectiveEdit",
            "ConstraintRepair",
        }:
            solved = self._feasibility_solve(
                candidate,
                flp_ir,
                canonical_spec,
                contract,
                reference,
                movable_instance_ids=sorted(movable),
                preserve_instance_ids=preserve,
                scope="global",
                solver_options=solver_options,
            )
            if solved is not None:
                candidate = solved
                actions.append(
                    {
                        "operation": "global_ga_solve",
                        "instance_ids": sorted(movable),
                    }
                )
                validation = self._validate(
                    candidate, canonical_spec, contract, reference
                )

        if task_type == "ConstraintRepair" and validation["valid"]:
            candidate["known_invalid"] = False
            candidate["known_violations"] = []
        changed_ids = _changed_instance_ids(before, candidate)
        return {
            "schema_version": "validation-driven-recovery-result-v1",
            "status": (
                "RECOVERY_SUCCESS"
                if validation["valid"]
                else "RECOVERY_INCOMPLETE"
            ),
            "layout": candidate,
            "actions": actions,
            "changed_instance_ids": changed_ids,
            "candidate_changed": bool(changed_ids or candidate != before),
            "validation": validation,
            "runtime_s": time.perf_counter() - start,
            "trace": _recovery_trace(
                contract,
                changed_ids,
                (
                    "RECOVERY_SUCCESS"
                    if validation["valid"]
                    else "RECOVERY_INCOMPLETE"
                ),
            ),
        }

    def _validate(self, layout, spec, contract, reference):
        hard = self.validator.validate(
            layout,
            spec,
            reference_layout=reference,
            expected_instance_ids=contract.get("expected_instance_ids"),
            preserve_instance_ids=contract.get("preserve_instance_ids"),
        )
        validation_contract = contract
        if (
            contract.get("task_type") == "DirectiveEdit"
            and contract.get("edit_relaxation_policy")
            == "nearest_feasible"
        ):
            validation_contract = copy.deepcopy(contract)
            validation_contract["edit_operations"] = []
        task = validate_task_contract(layout, validation_contract)
        return {
            "valid": hard["status"] == "VALID" and task["status"] == "VALID",
            "hard_validation": hard,
            "task_contract_validation": task,
            "violations": [
                *(hard.get("violations") or []),
                *(task.get("errors") or []),
            ],
        }

    @staticmethod
    def _restore_instance_set(
        candidate,
        flp_ir,
        spec,
        violations,
        reference,
        contract,
        actions,
    ):
        expected = set(contract.get("expected_instance_ids") or [])
        placements = candidate["placements"]
        seen = set()
        kept = []
        for placement in placements:
            instance_id = placement.get("instance_id")
            if instance_id in seen or (expected and instance_id not in expected):
                actions.append(
                    {
                        "operation": (
                            "remove_duplicate_instance"
                            if instance_id in seen
                            else "remove_unexpected_instance"
                        ),
                        "instance_ids": [instance_id],
                    }
                )
                continue
            seen.add(instance_id)
            kept.append(placement)
        candidate["placements"] = kept
        reference_by_id = {
            item.get("instance_id"): item
            for item in ((reference or {}).get("placements") or [])
        }
        flp_by_id = {
            item.get("instance_id"): item
            for item in (flp_ir or {}).get("equipment_instances") or []
        }
        for instance_id in sorted(expected - seen):
            if instance_id in reference_by_id:
                placement = copy.deepcopy(reference_by_id[instance_id])
            else:
                placement = _placement_from_flp(
                    flp_by_id.get(instance_id) or {},
                    spec,
                )
            if placement:
                candidate["placements"].append(placement)
                actions.append(
                    {
                        "operation": "restore_missing_instance",
                        "instance_ids": [instance_id],
                    }
                )

    @staticmethod
    def _restore_preserved(candidate, reference, preserve, actions):
        if not reference:
            return
        reference_by_id = {
            item.get("instance_id"): item
            for item in reference.get("placements") or []
        }
        for index, placement in enumerate(candidate.get("placements") or []):
            instance_id = placement.get("instance_id")
            target = reference_by_id.get(instance_id)
            if instance_id in preserve and target is not None and placement != target:
                candidate["placements"][index] = copy.deepcopy(target)
                actions.append(
                    {
                        "operation": "restore_preserved_instance",
                        "instance_ids": [instance_id],
                    }
                )

    def _apply_violation(
        self,
        candidate,
        spec,
        violation,
        movable,
        preserve,
        actions,
    ):
        code = str(violation.get("code") or violation.get("violation_type") or "")
        instance_ids = _unique(
            violation.get("instance_ids") or violation.get("instances") or []
        )
        by_id = {
            item.get("instance_id"): item
            for item in candidate.get("placements") or []
        }
        if code == "ORIENTATION_NOT_ALLOWED":
            instance_id = next(
                (item for item in instance_ids if item in movable and item not in preserve),
                None,
            )
            if instance_id and instance_id in by_id:
                _set_yaw(by_id[instance_id], float(violation.get("target_value") or 0.0))
                actions.append(
                    {
                        "operation": "set_allowed_orientation",
                        "instance_ids": [instance_id],
                    }
                )
            return
        if code in {
            "OUTSIDE_BOUNDARY",
            "OUTSIDE_ALLOWED_REGION",
            "WALL_INTERSECTION",
            "CORRIDOR_OBSTRUCTION",
            "ENTRANCE_EXIT_OBSTRUCTION",
            "COLUMN_INTERSECTION",
            "FORBIDDEN_ZONE_INTERSECTION",
            "FIXED_EQUIPMENT_INTERSECTION",
            "PROHIBITED_REGION_INTERSECTION",
        }:
            instance_id = next(
                (item for item in instance_ids if item in movable and item not in preserve),
                None,
            )
            if instance_id and instance_id in by_id:
                displacement = violation.get("minimum_repair_displacement") or {}
                dx = float(displacement.get("dx_m") or 0.0)
                dz = float(displacement.get("dz_m") or 0.0)
                if dx or dz:
                    _translate_placement(by_id[instance_id], dx, dz)
                    actions.append(
                        {
                            "operation": (
                                "project_inside_boundary"
                                if code == "OUTSIDE_BOUNDARY"
                                else "minimum_translation_exit_obstacle"
                            ),
                            "instance_ids": [instance_id],
                            "dx_m": dx,
                            "dz_m": dz,
                        }
                    )
            return
        if code in {"EQUIPMENT_OVERLAP", "CLEARANCE_DEFICIT"} and len(instance_ids) >= 2:
            left_id, right_id = instance_ids[:2]
            if (
                code == "EQUIPMENT_OVERLAP"
                and left_id in movable
                and right_id in movable
                and left_id not in preserve
                and right_id not in preserve
            ):
                split = self._try_split_overlap(
                    candidate,
                    spec,
                    left_id,
                    right_id,
                )
                if split is not None:
                    candidate.clear()
                    candidate.update(split["layout"])
                    actions.append(
                        {
                            "operation": "minimum_split_translation_remove_overlap",
                            "instance_ids": [left_id, right_id],
                            "displacements": split["displacements"],
                        }
                    )
                    return
            movable_id = (
                right_id
                if right_id in movable and right_id not in preserve
                else (
                    left_id
                    if left_id in movable and left_id not in preserve
                    else None
                )
            )
            fixed_id = left_id if movable_id == right_id else right_id
            if movable_id in by_id and fixed_id in by_id:
                displacement = _minimum_separation_for_placements(
                    by_id[fixed_id], by_id[movable_id]
                )
                _translate_placement(
                    by_id[movable_id],
                    displacement["dx_m"],
                    displacement["dz_m"],
                )
                actions.append(
                    {
                        "operation": (
                            "minimum_translation_remove_overlap"
                            if code == "EQUIPMENT_OVERLAP"
                            else "minimum_translation_restore_clearance"
                        ),
                        "instance_ids": [movable_id],
                        "against_instance_id": fixed_id,
                        **displacement,
                    }
                )

    def _try_split_overlap(self, candidate, spec, left_id, right_id):
        by_id = {
            item.get("instance_id"): item
            for item in candidate.get("placements") or []
        }
        left = _placement_footprint(by_id.get(left_id) or {})
        right = _placement_footprint(by_id.get(right_id) or {})
        if not left or not right or not _overlap(left, right):
            return None
        overlap_x = min(left["max_x"], right["max_x"]) - max(
            left["min_x"], right["min_x"]
        )
        overlap_z = min(left["max_z"], right["max_z"]) - max(
            left["min_z"], right["min_z"]
        )
        options = []
        for axis, amount in (("x", overlap_x), ("z", overlap_z)):
            separation = float(amount) + EPSILON_M
            for sign in (-1.0, 1.0):
                for left_share in (0.5, 1.0, 0.0):
                    left_amount = separation * left_share
                    right_amount = separation * (1.0 - left_share)
                    left_delta = {
                        "dx_m": (
                            sign * left_amount if axis == "x" else 0.0
                        ),
                        "dz_m": (
                            sign * left_amount if axis == "z" else 0.0
                        ),
                    }
                    right_delta = {
                        "dx_m": (
                            -sign * right_amount if axis == "x" else 0.0
                        ),
                        "dz_m": (
                            -sign * right_amount if axis == "z" else 0.0
                        ),
                    }
                    options.append(
                        (amount, axis, left_delta, right_delta)
                    )
        for _amount, _axis, left_delta, right_delta in sorted(
            options,
            key=lambda item: (item[0], item[1]),
        ):
            trial = copy.deepcopy(candidate)
            trial_by_id = {
                item.get("instance_id"): item
                for item in trial.get("placements") or []
            }
            _translate_placement(
                trial_by_id[left_id],
                left_delta["dx_m"],
                left_delta["dz_m"],
            )
            _translate_placement(
                trial_by_id[right_id],
                right_delta["dx_m"],
                right_delta["dz_m"],
            )
            if self.validator.validate(trial, spec)["status"] == "VALID":
                return {
                    "layout": trial,
                    "displacements": {
                        left_id: left_delta,
                        right_id: right_delta,
                    },
                }
        return None

    @staticmethod
    def _feasibility_solve(
        candidate,
        flp_ir,
        spec,
        contract,
        reference,
        *,
        movable_instance_ids,
        preserve_instance_ids,
        scope,
        solver_options,
    ):
        if not movable_instance_ids:
            return None
        try:
            from tool.layout.layout_solvers import solve_canonical_feasibility

            return solve_canonical_feasibility(
                flp_ir,
                spec,
                seed_layout=candidate,
                reference_layout=reference,
                movable_instance_ids=movable_instance_ids,
                preserve_instance_ids=sorted(preserve_instance_ids),
                scope=scope,
                optimize_objective=contract.get("task_type") == "Optimize",
                objective_upper_bound=contract.get(
                    "baseline_objective_value"
                ),
                stop_after_first_solution=(
                    contract.get("task_type") == "Optimize"
                ),
                solver_options=solver_options,
            )
        except (ImportError, RuntimeError, ValueError):
            return None


def _placement_from_flp(instance, spec):
    instance_id = instance.get("instance_id")
    if not instance_id:
        return None
    dimensions = copy.deepcopy((instance.get("geometry") or {}).get("dimensions_m") or {})
    pose = instance.get("initial_pose") or instance.get("fixed_pose") or {}
    x = float(pose.get("x_m") or 0.0)
    z = float(pose.get("z_m") or 0.0)
    width = float(dimensions.get("width_m") or 0.0)
    depth = float(dimensions.get("depth_m") or 0.0)
    return {
        "instance_id": instance_id,
        "model_asset_id": instance.get("model_asset_id"),
        "dimensions_m": dimensions,
        "position": {"x_m": x, "y_m": float(pose.get("y_m") or 0.0), "z_m": z},
        "rotation": {"yaw_deg": float(pose.get("yaw_deg") or 0.0)},
        "yaw_deg": float(pose.get("yaw_deg") or 0.0),
        "footprint": {
            "min_x": x - width / 2.0,
            "max_x": x + width / 2.0,
            "min_z": z - depth / 2.0,
            "max_z": z + depth / 2.0,
        },
    }


def _translate_placement(placement, dx, dz):
    position = placement.get("position")
    if not isinstance(position, dict):
        position = placement.setdefault("pose", {})
    position["x_m"] = float(position.get("x_m") or 0.0) + float(dx)
    position["z_m"] = float(position.get("z_m") or 0.0) + float(dz)
    footprint = placement.get("footprint")
    if isinstance(footprint, dict):
        for key in ("min_x", "max_x"):
            if key in footprint:
                footprint[key] = float(footprint[key]) + float(dx)
        for key in ("min_z", "max_z"):
            if key in footprint:
                footprint[key] = float(footprint[key]) + float(dz)


def _set_position(placement, x, z):
    position = placement.get("position")
    if not isinstance(position, dict):
        position = placement.setdefault("pose", {})
    old_x = float(position.get("x_m") or 0.0)
    old_z = float(position.get("z_m") or 0.0)
    _translate_placement(placement, x - old_x, z - old_z)


def _set_yaw(placement, yaw):
    old_yaw = float(
        placement.get("yaw_deg")
        or (placement.get("rotation") or {}).get("yaw_deg")
        or 0.0
    )
    placement["yaw_deg"] = float(yaw)
    placement.setdefault("rotation", {})["yaw_deg"] = float(yaw)
    dimensions = placement.get("dimensions_m") or {}
    position = placement.get("position") or placement.get("pose") or {}
    width = float(dimensions.get("width_m") or 0.0)
    depth = float(dimensions.get("depth_m") or 0.0)
    if (int(round(old_yaw)) % 180 == 90) != (
        int(round(float(yaw))) % 180 == 90
    ):
        width, depth = depth, width
    if isinstance(dimensions, dict):
        dimensions["width_m"] = width
        dimensions["depth_m"] = depth
    x = float(position.get("x_m") or 0.0)
    z = float(position.get("z_m") or 0.0)
    placement["footprint"] = {
        "width_m": width,
        "depth_m": depth,
        "min_x": x - width / 2.0,
        "max_x": x + width / 2.0,
        "min_z": z - depth / 2.0,
        "max_z": z + depth / 2.0,
    }


def _minimum_separation_for_placements(fixed, movable):
    left = _placement_footprint(fixed)
    right = _placement_footprint(movable)
    options = [
        (left["min_x"] - right["max_x"], 0.0),
        (left["max_x"] - right["min_x"], 0.0),
        (0.0, left["min_z"] - right["max_z"]),
        (0.0, left["max_z"] - right["min_z"]),
    ]
    dx, dz = min(
        options,
        key=lambda item: (
            abs(item[0]) + abs(item[1]),
            0 if item[0] else 1,
            item[0],
            item[1],
        ),
    )
    if dx:
        dx += math.copysign(EPSILON_M, dx)
    if dz:
        dz += math.copysign(EPSILON_M, dz)
    return {"dx_m": dx, "dz_m": dz, "distance_m": math.hypot(dx, dz)}


def _placement_footprint(placement):
    footprint = placement.get("footprint") or {}
    try:
        return {
            "min_x": float(footprint["min_x"]),
            "max_x": float(footprint["max_x"]),
            "min_z": float(footprint["min_z"]),
            "max_z": float(footprint["max_z"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _spec_rect(item):
    try:
        return {
            "min_x": float(item["min_x_m"]),
            "max_x": float(item["max_x_m"]),
            "min_z": float(item["min_z_m"]),
            "max_z": float(item["max_z_m"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _overlap(left, right):
    return (
        min(left["max_x"], right["max_x"])
        - max(left["min_x"], right["min_x"])
        > EPSILON_M
        and min(left["max_z"], right["max_z"])
        - max(left["min_z"], right["min_z"])
        > EPSILON_M
    )


def _changed_instance_ids(before, after):
    before_by_id = {
        item.get("instance_id"): item for item in before.get("placements") or []
    }
    after_by_id = {
        item.get("instance_id"): item for item in after.get("placements") or []
    }
    return sorted(
        instance_id
        for instance_id in set(before_by_id) | set(after_by_id)
        if before_by_id.get(instance_id) != after_by_id.get(instance_id)
    )


def _unique(values):
    result = []
    for value in values or []:
        if value is None:
            continue
        value = str(value)
        if value not in result:
            result.append(value)
    return result


def _recovery_trace(contract, changed_instance_ids, final_status):
    return {
        "execution_type": "validation_driven_recovery",
        "trigger_source": "validator_typed_violation",
        "task_type": contract.get("task_type"),
        "movable_instance_ids": copy.deepcopy(
            contract.get("movable_instance_ids") or []
        ),
        "preserve_instance_ids": copy.deepcopy(
            contract.get("preserve_instance_ids") or []
        ),
        "changed_instance_ids": list(changed_instance_ids),
        "deviated_from_user_directive": (
            contract.get("task_type") == "DirectiveEdit"
        ),
        "final_status": final_status,
    }
