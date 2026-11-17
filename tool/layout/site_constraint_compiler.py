"""Compile site and task state into one canonical hard-constraint spec."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from tool.layout.aisle_facing import model_front_vector


SCHEMA_VERSION = "canonical-constraint-spec-v1"
EPSILON_M = 1e-9


class CanonicalConstraintCompiler:
    """Create the geometry contract consumed by solver and validator."""

    PROHIBITED_FEATURE_TYPES = {
        "wall",
        "corridor",
        "aisle",
        "entrance",
        "exit",
        "column",
        "obstacle",
        "forbidden",
        "forbidden_zone",
        "non_placeable",
    }

    def compile(
        self,
        site_snapshot,
        *,
        flp_ir=None,
        task_contract=None,
        reference_layout=None,
    ):
        snapshot = copy.deepcopy(site_snapshot or {})
        flp_ir = flp_ir or {}
        contract = _contract_dict(task_contract)
        boundary = self._boundary(snapshot, flp_ir, reference_layout)
        placement_regions = self._placement_regions(snapshot, boundary)
        feature_groups = {
            "walls": [],
            "corridors": [],
            "entrances_exits": [],
            "columns": [],
            "forbidden_zones": [],
            "other_obstacles": [],
        }
        prohibited = []
        for feature in snapshot.get("site_features") or []:
            item = _rect_from_item(
                feature,
                object_id=feature.get("site_feature_id"),
                object_type=feature.get("feature_type"),
            )
            if not item:
                continue
            feature_type = item["object_type"]
            group = _feature_group(feature_type)
            feature_groups[group].append(item)
            if (
                feature.get("must_remain_clear")
                or feature_type in self.PROHIBITED_FEATURE_TYPES
            ):
                prohibited.append(item)
        for zone in snapshot.get("zones") or []:
            zone_type = str(zone.get("zone_type") or "").casefold()
            if zone_type in {"placement", "placement_region", "equipment_region"}:
                continue
            item = _rect_from_item(
                zone,
                object_id=zone.get("zone_id"),
                object_type=zone_type,
            )
            if item and zone_type in self.PROHIBITED_FEATURE_TYPES:
                feature_groups[_feature_group(zone_type)].append(item)
                prohibited.append(item)

        fixed_equipment = []
        for equipment in snapshot.get("embedded_equipment_instances") or []:
            item = _fixed_equipment_rect(equipment)
            if item:
                fixed_equipment.append(item)
                prohibited.append(item)

        scope = flp_ir.get("optimization_scope") or {}
        locked = _unique(
            contract.get("locked_instance_ids")
            or scope.get("locked_instances")
            or [
                item.get("instance_id")
                for item in flp_ir.get("equipment_instances") or []
                if item.get("locked")
            ]
        )
        movable = _unique(
            contract.get("movable_instance_ids")
            or scope.get("movable_instances")
            or [
                item.get("instance_id")
                for item in flp_ir.get("equipment_instances") or []
                if item.get("instance_id") not in locked
            ]
        )
        instance_constraints = {}
        for instance in flp_ir.get("equipment_instances") or []:
            instance_id = instance.get("instance_id")
            if not instance_id:
                continue
            instance_constraints[instance_id] = {
                "instance_id": instance_id,
                "allowed_region_ids": _unique(
                    instance.get("allowed_regions")
                    or [item["region_id"] for item in placement_regions]
                ),
                "allowed_orientations_deg": [
                    float(item)
                    for item in (
                        instance.get("allowed_orientations_deg") or []
                    )
                ],
                "locked": instance_id in locked or bool(instance.get("locked")),
                "movable": instance_id in movable and instance_id not in locked,
                "model_asset_id": instance.get("model_asset_id"),
                "front_vector_xz": model_front_vector(instance.get("model_asset_id")),
                "minimum_clearance_m": _instance_clearance(instance),
                "geometry": copy.deepcopy(instance.get("geometry") or {}),
                "maintenance_clearance_m": copy.deepcopy(instance.get("maintenance_clearance_m")),
                "safety_clearance_m": copy.deepcopy(instance.get("safety_clearance_m")),
            }
        for instance_id, row_spec in _workstation_row_alignments(
            snapshot, instance_constraints, flp_ir=flp_ir
        ).items():
            instance_constraints[instance_id].update(row_spec)
        clearance_constraints = _clearance_constraints(flp_ir)
        spec = {
            "schema_version": SCHEMA_VERSION,
            "site_id": snapshot.get("site_id")
            or ((flp_ir.get("site_ref") or {}).get("site_id"))
            or boundary["object_id"],
            "coordinate_frame_id": boundary.get("coordinate_frame_id"),
            "boundary": boundary,
            "allowed_placement_regions": placement_regions,
            **feature_groups,
            "fixed_equipment": fixed_equipment,
            "site_features": copy.deepcopy(snapshot.get("site_features") or []),
            "zones": copy.deepcopy(snapshot.get("zones") or []),
            "fixed_equipment_instances": copy.deepcopy(
                snapshot.get("embedded_equipment_instances") or []
            ),
            "prohibited_regions": _deduplicate_rects(prohibited),
            "instance_constraints": instance_constraints,
            "locked_instance_ids": locked,
            "movable_instance_ids": movable,
            "clearance_constraints": clearance_constraints,
            "transport_problem": copy.deepcopy({
                "objectives": [o for o in flp_ir.get("objectives") or [] if o.get("objective_type") == "minimize_agv_transport_cost"],
                "flow_relations": flp_ir.get("flow_relations") or [],
                "site_ref": flp_ir.get("site_ref") or {},
                "constraints": flp_ir.get("constraints") or {},
            }),
            "transfer_problem": copy.deepcopy({"equipment_instances":flp_ir.get("equipment_instances") or [], "constraints":flp_ir.get("constraints") or {}}),
            "source": {
                "site_snapshot_present": bool(snapshot),
                "site_snapshot_schema_version": snapshot.get("schema_version"),
                "task_contract_type": contract.get("task_type"),
            },
        }
        spec["spec_sha256"] = _stable_sha256(spec)
        return spec

    @staticmethod
    def _boundary(snapshot, flp_ir, reference_layout):
        boundary = snapshot.get("boundary") or {}
        item = _rect_from_item(
            boundary,
            object_id=snapshot.get("site_id")
            or ((flp_ir.get("site_ref") or {}).get("site_id"))
            or "site_boundary",
            object_type="boundary",
        )
        if item:
            return item
        layout_site = (reference_layout or {}).get("site") or {}
        dimensions = (
            layout_site.get("dimensions_m")
            or (flp_ir.get("site_ref") or {}).get("default_dimensions_m")
            or {}
        )
        width_value = dimensions.get("width_m") or layout_site.get("width_m")
        depth_value = dimensions.get("depth_m") or layout_site.get("depth_m")
        if (
            not _finite(width_value)
            or not _finite(depth_value)
            or float(width_value) <= 0.0
            or float(depth_value) <= 0.0
        ):
            raise ValueError(
                "CanonicalConstraintCompiler requires an authoritative boundary."
            )
        width = float(width_value)
        depth = float(depth_value)
        return _rect(
            "site_boundary",
            "boundary",
            0.0,
            0.0,
            width,
            depth,
            coordinate_frame_id=None,
            yaw_deg=0.0,
        )

    @staticmethod
    def _placement_regions(snapshot, boundary):
        result = []
        for zone in snapshot.get("zones") or []:
            if str(zone.get("zone_type") or "").casefold() not in {
                "placement",
                "placement_region",
                "equipment_region",
            }:
                continue
            item = _rect_from_item(
                zone,
                object_id=zone.get("zone_id"),
                object_type="placement_region",
            )
            if item:
                item["region_id"] = item.pop("object_id")
                item["region_type"] = item.pop("object_type")
                item["allowed_categories"] = copy.deepcopy(
                    zone.get("allowed_categories") or []
                )
                item["forbidden_categories"] = copy.deepcopy(
                    zone.get("forbidden_categories") or []
                )
                result.append(item)
        if result:
            return result
        fallback = copy.deepcopy(boundary)
        fallback["region_id"] = "placement_site_boundary"
        fallback["region_type"] = "placement_region"
        fallback.pop("object_id", None)
        fallback.pop("object_type", None)
        fallback["allowed_categories"] = []
        fallback["forbidden_categories"] = []
        return [fallback]


def canonical_site_from_spec(spec):
    """Adapt a canonical spec to the legacy solver site representation."""

    boundary = (spec or {}).get("boundary") or {}
    return {
        "site_id": (spec or {}).get("site_id") or boundary.get("object_id"),
        "width_m": float(boundary.get("width_m") or 0.0),
        "depth_m": float(boundary.get("depth_m") or 0.0),
        "center_x_m": float(boundary.get("center_x_m") or 0.0),
        "center_z_m": float(boundary.get("center_z_m") or 0.0),
        "min_x_m": float(boundary.get("min_x_m") or 0.0),
        "max_x_m": float(boundary.get("max_x_m") or 0.0),
        "min_z_m": float(boundary.get("min_z_m") or 0.0),
        "max_z_m": float(boundary.get("max_z_m") or 0.0),
        "explicit_site": True,
        "canonical_constraint_spec": copy.deepcopy(spec),
        "site_features": copy.deepcopy((spec or {}).get("site_features") or []),
        "zones": copy.deepcopy((spec or {}).get("zones") or []),
        "embedded_equipment_instances": copy.deepcopy(
            (spec or {}).get("fixed_equipment_instances") or []
        ),
    }


def _contract_dict(contract):
    if hasattr(contract, "to_dict"):
        return contract.to_dict()
    return contract or {}


def _rect_from_item(item, *, object_id, object_type):
    item = item or {}
    pose = item.get("pose") or {}
    dimensions = item.get("dimensions_m") or {}
    try:
        width = float(dimensions.get("width_m") or 0.0)
        depth = float(dimensions.get("depth_m") or 0.0)
        center_x = float(pose.get("x_m") or 0.0)
        center_z = float(pose.get("z_m") or 0.0)
        yaw = float(pose.get("yaw_deg") or 0.0)
    except (TypeError, ValueError):
        return None
    if width <= EPSILON_M or depth <= EPSILON_M:
        return None
    return _rect(
        str(object_id or object_type or "site_object"),
        str(object_type or "obstacle").casefold(),
        center_x,
        center_z,
        width,
        depth,
        coordinate_frame_id=pose.get("coordinate_frame_id"),
        yaw_deg=yaw,
    )


def _fixed_equipment_rect(equipment):
    geometry = equipment.get("geometry") or {}
    return _rect_from_item(
        {
            "pose": equipment.get("pose") or {},
            "dimensions_m": geometry.get("dimensions_m")
            or (geometry.get("footprint") or {}),
        },
        object_id=equipment.get("instance_id"),
        object_type="fixed_equipment",
    )


def _rect(
    object_id,
    object_type,
    center_x,
    center_z,
    width,
    depth,
    *,
    coordinate_frame_id,
    yaw_deg,
):
    normalized_yaw = yaw_deg % 360.0
    if math.isclose(normalized_yaw % 180.0, 90.0, abs_tol=1e-6):
        width, depth = depth, width
    half_width = width / 2.0
    half_depth = depth / 2.0
    return {
        "object_id": object_id,
        "object_type": object_type,
        "coordinate_frame_id": coordinate_frame_id,
        "center_x_m": center_x,
        "center_z_m": center_z,
        "width_m": width,
        "depth_m": depth,
        "yaw_deg": yaw_deg,
        "min_x_m": center_x - half_width,
        "max_x_m": center_x + half_width,
        "min_z_m": center_z - half_depth,
        "max_z_m": center_z + half_depth,
    }


def _feature_group(feature_type):
    return {
        "wall": "walls",
        "corridor": "corridors",
        "aisle": "corridors",
        "entrance": "entrances_exits",
        "exit": "entrances_exits",
        "column": "columns",
        "forbidden": "forbidden_zones",
        "forbidden_zone": "forbidden_zones",
        "non_placeable": "forbidden_zones",
    }.get(feature_type, "other_obstacles")


def _instance_clearance(instance):
    values = []
    for key in ("safety_clearance_m", "maintenance_clearance_m"):
        clearance = instance.get(key)
        if isinstance(clearance, (int, float)):
            values.append(float(clearance))
        elif isinstance(clearance, dict):
            for field, value in clearance.items():
                if field.endswith("_m") and _finite(value):
                    values.append(float(value))
    return max(values, default=0.0)


def _workstation_row_alignments(snapshot, instance_constraints, *, flp_ir=None):
    """Derive one feasible east-west centerline for each source-defined row."""

    from tool.layout.aisle_facing import case_clearance_margins, model_front_vector, preferred_yaw_for_direction

    zones = {
        str(item.get("zone_id") or ""): item
        for item in (snapshot.get("zones") or [])
        if isinstance(item, dict)
    }
    explicitly_assigned_regions = {
        str(item.get("instance_id") or ""): [
            str(value) for value in item.get("allowed_regions") or []
        ]
        for item in (flp_ir or {}).get("equipment_instances") or []
        if item.get("instance_id")
    }
    members_by_row = {}
    for instance_id, constraint in (instance_constraints or {}).items():
        # Canonical constraints default unassigned equipment to every legal
        # placement region. A workstation bay appearing in that fallback list
        # does not make the equipment a member of the row.
        explicit_regions = explicitly_assigned_regions.get(str(instance_id)) or []
        region_id = next(
            (
                value
                for value in explicit_regions
                if str(value).startswith("natural_site/workstation_bay_")
                and str(value).endswith("/placement")
            ),
            None,
        )
        if not region_id:
            continue
        bay_id = str(region_id)[: -len("/placement")]
        match = re.search(r"workstation_bay_row_(\d+)_column_\d+$", bay_id)
        bay = zones.get(bay_id)
        if not match or not bay:
            continue
        pose = bay.get("pose") or {}
        dimensions = bay.get("dimensions_m") or {}
        geometry = (constraint.get("geometry") or {}).get("dimensions_m") or {}
        try:
            center_z = float(pose["z_m"])
            band_depth = float(dimensions["depth_m"])
            body_width = float(geometry["width_m"])
            body_depth = float(geometry["depth_m"])
        except (KeyError, TypeError, ValueError):
            continue
        clearance = constraint.get("maintenance_clearance_m") or {}
        front_vector = model_front_vector(constraint.get("model_asset_id")) or (0.0, 1.0)
        preferred_yaw = preferred_yaw_for_direction(
            constraint.get("model_asset_id"),
            0.0,
            1.0,
            allowed_orientations=constraint.get("allowed_orientations_deg") or None,
        )
        yaw = 0.0 if preferred_yaw is None else float(preferred_yaw)
        body_depth_in_row = body_width if int(round(yaw)) % 180 == 90 else body_depth
        if clearance.get("source") == "user_input":
            margins = case_clearance_margins(clearance, front_vector, yaw)
        else:
            margins = {"south_m": 0.0, "north_m": 0.0}
        min_z = center_z - band_depth / 2.0
        max_z = center_z + band_depth / 2.0
        members_by_row.setdefault(int(match.group(1)), []).append(
            {
                "instance_id": instance_id,
                "bay_id": bay_id,
                "minimum_center_z_m": min_z + margins["south_m"] + body_depth_in_row / 2.0,
                "maximum_center_z_m": max_z - margins["north_m"] - body_depth_in_row / 2.0,
            }
        )

    result = {}
    for row_index, members in members_by_row.items():
        lower = max(item["minimum_center_z_m"] for item in members)
        upper = min(item["maximum_center_z_m"] for item in members)
        if lower > upper + EPSILON_M:
            # The natural-language geometry compiler reports the fit failure;
            # leave no alignment preference if an externally built snapshot
            # does not contain a common feasible line.
            continue
        line = (lower + upper) / 2.0
        for item in members:
            result[item["instance_id"]] = {
                "workstation_row_index": row_index,
                "workstation_bay_id": item["bay_id"],
                "workstation_row_alignment_z_m": line,
            }
    return result


def _clearance_constraints(flp_ir):
    result = []
    for constraint in (flp_ir.get("constraints") or {}).get("hard") or []:
        if str(constraint.get("constraint_type") or "").casefold() not in {
            "minimum_clearance",
            "clearance",
            "safety_clearance",
            "maintenance_clearance",
        }:
            continue
        parameters = constraint.get("parameters") or {}
        value = next(
            (
                float(parameters[key])
                for key in (
                    "minimum_clearance_m",
                    "clearance_m",
                    "distance_m",
                )
                if _finite(parameters.get(key))
            ),
            0.0,
        )
        result.append(
            {
                "constraint_id": constraint.get("constraint_id"),
                "instance_ids": [
                    item.get("target_id")
                    for item in constraint.get("target_refs") or []
                    if item.get("target_id") not in {None, "all"}
                ],
                "minimum_clearance_m": value,
            }
        )
    return result


def _deduplicate_rects(items):
    result = []
    seen = set()
    for item in items:
        key = (
            item.get("object_id"),
            item.get("min_x_m"),
            item.get("max_x_m"),
            item.get("min_z_m"),
            item.get("max_z_m"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


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


def _stable_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
