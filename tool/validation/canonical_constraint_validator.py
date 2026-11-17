"""Canonical hard-constraint validation with executable typed violations."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field


EPSILON_M = 1e-9
SCHEMA_VERSION = "typed-violation-v1"


@dataclass
class TypedViolation:
    code: str
    instance_ids: list[str] = field(default_factory=list)
    obstacle_id: str | None = None
    measured_value: object = None
    target_value: object = None
    minimum_repair_displacement: dict | None = None
    allowed_repair_operations: list[str] = field(default_factory=list)
    preserve_instance_ids: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        return asdict(self)


class CanonicalConstraintValidator:
    """Validate a layout against exactly the geometry compiled for the solver."""

    def validate(
        self,
        layout,
        canonical_spec,
        *,
        reference_layout=None,
        expected_instance_ids=None,
        preserve_instance_ids=None,
    ):
        spec = canonical_spec or {}
        layout = layout or {}
        preserve = _unique(
            preserve_instance_ids or spec.get("locked_instance_ids") or []
        )
        expected = set(
            expected_instance_ids
            or (spec.get("instance_constraints") or {}).keys()
        )
        placements = list(layout.get("placements") or [])
        actual_ids = [
            item.get("instance_id") for item in placements if item.get("instance_id")
        ]
        violations = []

        for instance_id in sorted(expected - set(actual_ids)):
            violations.append(
                TypedViolation(
                    code="MISSING_INSTANCE",
                    instance_ids=[instance_id],
                    allowed_repair_operations=[
                        "restore_missing_instance",
                        "local_ga_resolve",
                        "global_feasibility_solve",
                    ],
                    preserve_instance_ids=preserve,
                )
            )
        for instance_id in sorted(
            item for item in set(actual_ids) if actual_ids.count(item) > 1
        ):
            violations.append(
                TypedViolation(
                    code="DUPLICATE_INSTANCE",
                    instance_ids=[instance_id],
                    measured_value=actual_ids.count(instance_id),
                    target_value=1,
                    allowed_repair_operations=["remove_duplicate_instance"],
                    preserve_instance_ids=preserve,
                )
            )
        for instance_id in sorted(set(actual_ids) - expected) if expected else []:
            violations.append(
                TypedViolation(
                    code="UNEXPECTED_INSTANCE",
                    instance_ids=[instance_id],
                    allowed_repair_operations=["remove_unexpected_instance"],
                    preserve_instance_ids=preserve,
                )
            )

        by_id = {}
        for placement in placements:
            instance_id = placement.get("instance_id")
            if instance_id and instance_id not in by_id:
                by_id[instance_id] = placement
        canonical_footprints = {}
        for instance_id, placement in by_id.items():
            constraint = (spec.get("instance_constraints") or {}).get(
                instance_id
            ) or {}
            footprint = _canonical_footprint(placement, constraint)
            if not footprint:
                violations.append(
                    TypedViolation(
                        code="INVALID_POSITION_DATA",
                        instance_ids=[instance_id],
                        allowed_repair_operations=[
                            "restore_missing_position",
                            "global_feasibility_solve",
                        ],
                        preserve_instance_ids=preserve,
                    )
                )
                continue
            canonical_footprints[instance_id] = footprint
            violations.extend(
                self._validate_placement(
                    instance_id,
                    placement,
                    footprint,
                    constraint,
                    spec,
                    preserve,
                )
            )

        ordered = sorted(canonical_footprints)
        for index, left_id in enumerate(ordered):
            for right_id in ordered[index + 1 :]:
                left = canonical_footprints[left_id]
                right = canonical_footprints[right_id]
                overlap = _overlap_measure(left, right)
                required = max(
                    float(
                        (
                            (spec.get("instance_constraints") or {}).get(left_id)
                            or {}
                        ).get("minimum_clearance_m")
                        or 0.0
                    ),
                    float(
                        (
                            (spec.get("instance_constraints") or {}).get(right_id)
                            or {}
                        ).get("minimum_clearance_m")
                        or 0.0
                    ),
                )
                if overlap["x_m"] > EPSILON_M and overlap["z_m"] > EPSILON_M:
                    violations.append(
                        TypedViolation(
                            code="EQUIPMENT_OVERLAP",
                            instance_ids=[left_id, right_id],
                            measured_value={
                                "overlap_x_m": overlap["x_m"],
                                "overlap_z_m": overlap["z_m"],
                                "overlap_area_m2": overlap["x_m"]
                                * overlap["z_m"],
                            },
                            target_value={"overlap_area_m2": 0.0},
                            minimum_repair_displacement=_minimum_separation(
                                left, right
                            ),
                            allowed_repair_operations=[
                                "minimum_translation_remove_overlap",
                                "local_ga_resolve",
                                "global_feasibility_solve",
                            ],
                            preserve_instance_ids=preserve,
                        )
                    )
                elif required > EPSILON_M:
                    gap = _rectangle_gap(left, right)
                    if gap + EPSILON_M < required:
                        violations.append(
                            TypedViolation(
                                code="CLEARANCE_DEFICIT",
                                instance_ids=[left_id, right_id],
                                measured_value=gap,
                                target_value=required,
                                minimum_repair_displacement={
                                    "dx_m": required - gap,
                                    "dz_m": 0.0,
                                    "distance_m": required - gap,
                                },
                                allowed_repair_operations=[
                                    "minimum_translation_restore_clearance",
                                    "local_ga_resolve",
                                ],
                                preserve_instance_ids=preserve,
                            )
                        )

        reference_by_id = {
            item.get("instance_id"): item
            for item in ((reference_layout or {}).get("placements") or [])
        }
        for instance_id in preserve:
            if (
                instance_id in by_id
                and instance_id in reference_by_id
                and not _placement_state_equal(
                    by_id[instance_id], reference_by_id[instance_id]
                )
            ):
                violations.append(
                    TypedViolation(
                        code="LOCKED_INSTANCE_MOVED",
                        instance_ids=[instance_id],
                        measured_value=_placement_state(by_id[instance_id]),
                        target_value=_placement_state(reference_by_id[instance_id]),
                        allowed_repair_operations=["restore_preserved_instance"],
                        preserve_instance_ids=preserve,
                    )
                )

        from tool.logistics.transfer_workcell import transfer_cell_violations
        for issue in transfer_cell_violations(layout, spec.get("transfer_problem") or {}):
            violations.append(TypedViolation(code="TRANSFER_CELL_GEOMETRY",instance_ids=issue["instances"],measured_value=issue,
                target_value="aligned end tables and unobstructed operator area",allowed_repair_operations=["local_ga_resolve"],preserve_instance_ids=preserve))
        transport = spec.get("transport_problem") or {}
        if not transport.get("objectives") and not violations:
            from tool.logistics.agv_transport_cost import agv_access_required, agv_access_issues
            if agv_access_required(transport):
                for issue in agv_access_issues(layout,transport):
                    violations.append(TypedViolation(code="AGV_TRANSFER_UNREACHABLE",instance_ids=[issue["vehicle"]],measured_value=issue,
                        target_value="body-clearance path to every assigned external handoff",allowed_repair_operations=["local_ga_resolve"],preserve_instance_ids=preserve))
        if transport.get("objectives") and not violations:
            from tool.logistics.agv_transport_cost import evaluate_agv_cost
            for issue in evaluate_agv_cost(layout, transport)["unreachable"]:
                violations.append(TypedViolation(code="AGV_TRANSFER_UNREACHABLE", instance_ids=[issue["vehicle"]],
                    measured_value=issue, target_value="body-clearance path to every assigned external handoff",
                    allowed_repair_operations=["local_ga_resolve"], preserve_instance_ids=preserve))
        serialized = [item.to_dict() for item in violations]
        return {
            "schema_version": "canonical-hard-validation-v1",
            "status": "VALID" if not serialized else "INVALID",
            "canonical_spec_sha256": spec.get("spec_sha256"),
            "violations": serialized,
            "issues": serialized,
            "violation_count": len(serialized),
            "hard_constraints_satisfied": not serialized,
        }

    def _validate_placement(
        self,
        instance_id,
        placement,
        footprint,
        constraint,
        spec,
        preserve,
    ):
        violations = []
        boundary = spec.get("boundary") or {}
        if not _inside(footprint, boundary):
            displacement = _project_inside(footprint, boundary)
            violations.append(
                TypedViolation(
                    code="OUTSIDE_BOUNDARY",
                    instance_ids=[instance_id],
                    obstacle_id=boundary.get("object_id"),
                    measured_value=copy.deepcopy(footprint),
                    target_value={
                        "min_x_m": boundary.get("min_x_m"),
                        "max_x_m": boundary.get("max_x_m"),
                        "min_z_m": boundary.get("min_z_m"),
                        "max_z_m": boundary.get("max_z_m"),
                    },
                    minimum_repair_displacement=displacement,
                    allowed_repair_operations=[
                        "project_inside_boundary",
                        "local_ga_resolve",
                    ],
                    preserve_instance_ids=preserve,
                )
            )

        allowed_ids = set(constraint.get("allowed_region_ids") or [])
        allowed_regions = [
            item
            for item in spec.get("allowed_placement_regions") or []
            if not allowed_ids or item.get("region_id") in allowed_ids
        ]
        region_footprint = _workstation_service_envelope(
            footprint,
            _yaw(placement),
            constraint,
            allowed_ids,
        )
        has_case_clearance = (
            (constraint.get("maintenance_clearance_m") or {}).get("source") == "user_input"
        )
        if has_case_clearance and not _inside(region_footprint, boundary):
            violations.append(
                TypedViolation(
                    code="SERVICE_ENVELOPE_OUTSIDE_BOUNDARY",
                    instance_ids=[instance_id],
                    obstacle_id=boundary.get("object_id"),
                    measured_value=copy.deepcopy(region_footprint),
                    target_value={
                        "min_x_m": boundary.get("min_x_m"),
                        "max_x_m": boundary.get("max_x_m"),
                        "min_z_m": boundary.get("min_z_m"),
                        "max_z_m": boundary.get("max_z_m"),
                    },
                    allowed_repair_operations=["project_inside_boundary", "local_ga_resolve"],
                    preserve_instance_ids=preserve,
                )
            )
        if allowed_regions and not any(
            _inside(region_footprint, region) for region in allowed_regions
        ):
            displacement = min(
                (
                    _project_inside(region_footprint, region)
                    for region in allowed_regions
                    if _can_fit(region_footprint, region)
                ),
                key=lambda item: item["distance_m"],
                default=None,
            )
            violations.append(
                TypedViolation(
                    code="OUTSIDE_ALLOWED_REGION",
                    instance_ids=[instance_id],
                    measured_value=copy.deepcopy(region_footprint),
                    target_value=sorted(allowed_ids),
                    minimum_repair_displacement=displacement,
                    allowed_repair_operations=[
                        "project_to_allowed_region",
                        "local_ga_resolve",
                        "global_feasibility_solve",
                    ],
                    preserve_instance_ids=preserve,
                )
            )

        row_alignment_z = constraint.get("workstation_row_alignment_z_m")
        if row_alignment_z is not None:
            position = placement.get("position") or placement.get("pose") or {}
            try:
                center_z = float(position["z_m"])
                target_z = float(row_alignment_z)
            except (KeyError, TypeError, ValueError):
                center_z = target_z = None
            if center_z is not None and abs(center_z - target_z) > EPSILON_M:
                violations.append(
                    TypedViolation(
                        code="WORKSTATION_ROW_ALIGNMENT",
                        instance_ids=[instance_id],
                        measured_value=center_z,
                        target_value=target_z,
                        minimum_repair_displacement={
                            "dx_m": 0.0,
                            "dz_m": target_z - center_z,
                            "distance_m": abs(target_z - center_z),
                        },
                        allowed_repair_operations=["align_with_workstation_row"],
                        preserve_instance_ids=preserve,
                    )
                )

        for obstacle in spec.get("prohibited_regions") or []:
            overlap = _overlap_measure(footprint, obstacle)
            obstacle_type = obstacle.get("object_type")
            if (
                has_case_clearance
                and obstacle_type in {"corridor", "aisle", "forbidden", "forbidden_zone"}
                and (overlap["x_m"] <= EPSILON_M or overlap["z_m"] <= EPSILON_M)
            ):
                service_overlap = _overlap_measure(region_footprint, obstacle)
                if service_overlap["x_m"] > EPSILON_M and service_overlap["z_m"] > EPSILON_M:
                    violations.append(
                        TypedViolation(
                            code="SERVICE_ENVELOPE_INTERSECTION",
                            instance_ids=[instance_id],
                            obstacle_id=obstacle.get("object_id"),
                            measured_value={
                                "overlap_x_m": service_overlap["x_m"],
                                "overlap_z_m": service_overlap["z_m"],
                            },
                            target_value={"overlap_area_m2": 0.0},
                            allowed_repair_operations=["project_to_allowed_region", "local_ga_resolve"],
                            preserve_instance_ids=preserve,
                            details={"obstacle_type": obstacle_type},
                        )
                    )
            if overlap["x_m"] <= EPSILON_M or overlap["z_m"] <= EPSILON_M:
                continue
            code = {
                "wall": "WALL_INTERSECTION",
                "corridor": "CORRIDOR_OBSTRUCTION",
                "aisle": "CORRIDOR_OBSTRUCTION",
                "entrance": "ENTRANCE_EXIT_OBSTRUCTION",
                "exit": "ENTRANCE_EXIT_OBSTRUCTION",
                "column": "COLUMN_INTERSECTION",
                "forbidden": "FORBIDDEN_ZONE_INTERSECTION",
                "forbidden_zone": "FORBIDDEN_ZONE_INTERSECTION",
                "fixed_equipment": "FIXED_EQUIPMENT_INTERSECTION",
            }.get(obstacle_type, "PROHIBITED_REGION_INTERSECTION")
            violations.append(
                TypedViolation(
                    code=code,
                    instance_ids=[instance_id],
                    obstacle_id=obstacle.get("object_id"),
                    measured_value={
                        "overlap_x_m": overlap["x_m"],
                        "overlap_z_m": overlap["z_m"],
                    },
                    target_value={"overlap_area_m2": 0.0},
                    minimum_repair_displacement=_minimum_separation(
                        obstacle, footprint
                    ),
                    allowed_repair_operations=[
                        "minimum_translation_exit_obstacle",
                        "project_to_allowed_region",
                        "local_ga_resolve",
                    ],
                    preserve_instance_ids=preserve,
                    details={"obstacle_type": obstacle_type},
                )
            )

        allowed_orientations = constraint.get("allowed_orientations_deg") or []
        yaw = _yaw(placement)
        if allowed_orientations and not any(
            _same_angle(yaw, item) for item in allowed_orientations
        ):
            target = min(
                allowed_orientations,
                key=lambda item: _angle_distance(yaw, float(item)),
            )
            violations.append(
                TypedViolation(
                    code="ORIENTATION_NOT_ALLOWED",
                    instance_ids=[instance_id],
                    measured_value=yaw,
                    target_value=float(target),
                    allowed_repair_operations=["set_allowed_orientation"],
                    preserve_instance_ids=preserve,
                )
            )
        return violations


def _canonical_footprint(placement, constraint):
    position = placement.get("position") or placement.get("pose") or {}
    try:
        center_x = float(position["x_m"])
        center_z = float(position["z_m"])
    except (KeyError, TypeError, ValueError):
        return None
    geometry = constraint.get("geometry") or {}
    dimensions = geometry.get("dimensions_m") or placement.get("dimensions_m") or {}
    try:
        width = float(dimensions["width_m"])
        depth = float(dimensions["depth_m"])
    except (KeyError, TypeError, ValueError):
        footprint = placement.get("footprint") or {}
        try:
            width = float(footprint["max_x"]) - float(footprint["min_x"])
            depth = float(footprint["max_z"]) - float(footprint["min_z"])
        except (KeyError, TypeError, ValueError):
            return None
    if int(round(_yaw(placement))) % 180 == 90:
        width, depth = depth, width
    return {
        "min_x_m": center_x - width / 2.0,
        "max_x_m": center_x + width / 2.0,
        "min_z_m": center_z - depth / 2.0,
        "max_z_m": center_z + depth / 2.0,
        "width_m": width,
        "depth_m": depth,
    }


def _workstation_service_envelope(footprint, yaw, constraint, allowed_region_ids):
    """Expand a footprint by an explicit user case service allowance."""

    clearance = constraint.get("maintenance_clearance_m") or {}
    if clearance.get("source") != "user_input":
        return footprint
    from tool.layout.aisle_facing import case_clearance_margins

    margins = case_clearance_margins(
        clearance,
        constraint.get("front_vector_xz"),
        yaw,
    )
    margin_left = margins["left_m"]
    margin_right = margins["right_m"]
    margin_south = margins["south_m"]
    margin_north = margins["north_m"]
    return {
        "min_x_m": footprint["min_x_m"] - margin_left,
        "max_x_m": footprint["max_x_m"] + margin_right,
        "min_z_m": footprint["min_z_m"] - margin_south,
        "max_z_m": footprint["max_z_m"] + margin_north,
        "width_m": footprint["width_m"] + margin_left + margin_right,
        "depth_m": footprint["depth_m"] + margin_south + margin_north,
    }


def _inside(footprint, region):
    return (
        footprint["min_x_m"] >= float(region.get("min_x_m") or 0.0) - EPSILON_M
        and footprint["max_x_m"]
        <= float(region.get("max_x_m") or 0.0) + EPSILON_M
        and footprint["min_z_m"] >= float(region.get("min_z_m") or 0.0) - EPSILON_M
        and footprint["max_z_m"]
        <= float(region.get("max_z_m") or 0.0) + EPSILON_M
    )


def _can_fit(footprint, region):
    return (
        footprint["width_m"]
        <= float(region.get("max_x_m") or 0.0)
        - float(region.get("min_x_m") or 0.0)
        + EPSILON_M
        and footprint["depth_m"]
        <= float(region.get("max_z_m") or 0.0)
        - float(region.get("min_z_m") or 0.0)
        + EPSILON_M
    )


def _project_inside(footprint, region):
    dx = 0.0
    dz = 0.0
    if footprint["min_x_m"] < float(region.get("min_x_m") or 0.0):
        dx = float(region.get("min_x_m") or 0.0) - footprint["min_x_m"]
    elif footprint["max_x_m"] > float(region.get("max_x_m") or 0.0):
        dx = float(region.get("max_x_m") or 0.0) - footprint["max_x_m"]
    if footprint["min_z_m"] < float(region.get("min_z_m") or 0.0):
        dz = float(region.get("min_z_m") or 0.0) - footprint["min_z_m"]
    elif footprint["max_z_m"] > float(region.get("max_z_m") or 0.0):
        dz = float(region.get("max_z_m") or 0.0) - footprint["max_z_m"]
    return {"dx_m": dx, "dz_m": dz, "distance_m": math.hypot(dx, dz)}


def _overlap_measure(left, right):
    return {
        "x_m": min(left["max_x_m"], right["max_x_m"])
        - max(left["min_x_m"], right["min_x_m"]),
        "z_m": min(left["max_z_m"], right["max_z_m"])
        - max(left["min_z_m"], right["min_z_m"]),
    }


def _minimum_separation(fixed, movable):
    options = [
        (fixed["min_x_m"] - movable["max_x_m"], 0.0),
        (fixed["max_x_m"] - movable["min_x_m"], 0.0),
        (0.0, fixed["min_z_m"] - movable["max_z_m"]),
        (0.0, fixed["max_z_m"] - movable["min_z_m"]),
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
        dx += math.copysign(EPSILON_M * 10.0, dx)
    if dz:
        dz += math.copysign(EPSILON_M * 10.0, dz)
    return {"dx_m": dx, "dz_m": dz, "distance_m": math.hypot(dx, dz)}


def _rectangle_gap(left, right):
    gap_x = max(
        0.0,
        max(left["min_x_m"], right["min_x_m"])
        - min(left["max_x_m"], right["max_x_m"]),
    )
    gap_z = max(
        0.0,
        max(left["min_z_m"], right["min_z_m"])
        - min(left["max_z_m"], right["max_z_m"]),
    )
    if gap_x and gap_z:
        return math.hypot(gap_x, gap_z)
    return max(gap_x, gap_z)


def _yaw(placement):
    return float(
        placement.get("yaw_deg")
        or (placement.get("rotation") or {}).get("yaw_deg")
        or (placement.get("pose") or {}).get("yaw_deg")
        or 0.0
    )


def _same_angle(left, right):
    return _angle_distance(float(left), float(right)) <= 1e-6


def _angle_distance(left, right):
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _placement_state(placement):
    return {
        "position": copy.deepcopy(
            placement.get("position") or placement.get("pose") or {}
        ),
        "yaw_deg": _yaw(placement),
        "dimensions_m": copy.deepcopy(placement.get("dimensions_m") or {}),
        "model_asset_id": placement.get("model_asset_id"),
        "instance_id": placement.get("instance_id"),
    }


def _placement_state_equal(left, right):
    return _values_equal_with_tolerance(
        _placement_state(left),
        _placement_state(right),
        1e-6,
    )


def _values_equal_with_tolerance(left, right, tolerance):
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left),
            float(right),
            abs_tol=tolerance,
            rel_tol=0.0,
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _values_equal_with_tolerance(left[key], right[key], tolerance)
            for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_equal_with_tolerance(a, b, tolerance)
            for a, b in zip(left, right)
        )
    return left == right


def _unique(values):
    result = []
    for value in values or []:
        if value is None:
            continue
        value = str(value)
        if value not in result:
            result.append(value)
    return result
