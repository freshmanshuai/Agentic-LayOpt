import copy
import json
import math
import re
from pathlib import Path

from jsonschema import Draft202012Validator

from tool.representations.flp_ir_layer import SCHEMA_VERSION, _schema_registry
from tool.layout.layout_solvers import non_placeable_zones, placement_regions


TARGETED_OPERATIONS = {"move", "rotate", "remove", "replace"}
SUPPORTED_OPERATIONS = TARGETED_OPERATIONS | {"add", "resize_site"}


def compile_layout_edit_ir(intent, semantic_intent, scene_context, adapter):
    base_layout = scene_context.get("last_layout") or scene_context.get("seed_layout") or {}
    base_layout_id = base_layout.get("layout_id") or intent.get("active_layout_id") or ""
    selected_ids = list(scene_context.get("selected_instance_ids") or [])
    operations = []
    unresolved = []
    next_added_number = max([int(str(p.get("instance_id", "")).rsplit("_",1)[-1]) for p in base_layout.get("placements") or []
                             if str(p.get("instance_id", "")).rsplit("_",1)[-1].isdigit()] or [0])+1
    for index, raw in enumerate(semantic_intent.get("edit_operations") or [], start=1):
        op_type = raw.get("op_type")
        if op_type not in SUPPORTED_OPERATIONS:
            _add_unresolved(
                unresolved,
                f"unsupported_edit_operation_{index:03d}",
                f"operations[{index - 1}].op_type",
                f"Unsupported layout edit operation: {op_type or 'missing'}.",
                item_type="unsupported_requirement",
            )
            continue
        instance_id = _resolve_instance_id(raw, base_layout, selected_ids)
        if op_type == "add":
            instance_id = f"equipment_{next_added_number:03d}"
            next_added_number += int(raw.get("quantity") or 1)
        if op_type in TARGETED_OPERATIONS and not instance_id:
            _add_unresolved(
                unresolved,
                f"missing_edit_target_{index:03d}",
                f"operations[{index - 1}].instance_id",
                "The edit target is missing or ambiguous; select one equipment instance.",
            )
        new_model_asset_id = raw.get("new_model_asset_id")
        if op_type in {"add", "replace"} and new_model_asset_id not in adapter.assets_by_id:
            _add_unresolved(
                unresolved,
                f"missing_edit_asset_{index:03d}",
                f"operations[{index - 1}].parameters.new_model_asset_id",
                "Add and replace operations require a valid new_model_asset_id.",
                candidates=[
                    item.get("model_asset_id") if isinstance(item, dict) else str(item)
                    for item in adapter.find_candidates(new_model_asset_id or raw.get("surface_text") or "")[:5]
                    if (item.get("model_asset_id") if isinstance(item, dict) else item)
                ],
            )
        parameters = _operation_parameters(raw)
        if op_type == "add":
            role=re.sub(r"^(?:one|a|an|new|additional)\s+","",str(raw.get("surface_text") or ""),flags=re.I)
            parameters["role"] = re.sub(r"\s+(?:labeled|labelled)\s+([A-Za-z])$",r" \1",role,flags=re.I)
            source=str(raw.get("source_text") or "")
            near=source.lower().split(" near ",1)
            if len(near)==2:
                candidates=[entry for entry in scene_context.get("instance_directory") or []
                            if str(entry.get("stable_alias") or "").strip() and
                            str(entry["stable_alias"]).lower() in near[1]]
                if len(candidates)==1:
                    parameters["reference_instance_id"]=candidates[0]["instance_id"]
                    parameters["placement_policy"]="near_instance"
        _validate_required_parameters(op_type, parameters, unresolved, index)
        target_refs = []
        if instance_id:
            target_refs.append({"target_type": "equipment_instance", "target_id": instance_id})
        elif op_type == "resize_site":
            target_refs.append({"target_type": "site", "target_id": base_layout.get("site", {}).get("site_id") or "site"})
        operations.append(
            {
                "operation_id": f"edit_operation_{index:03d}",
                "op_type": op_type,
                "instance_id": instance_id,
                "target_refs": target_refs,
                "parameters": parameters,
                "preserve": _preserve_fields(op_type),
                "requires_confirmation": False,
            }
        )

    if not base_layout_id:
        _add_unresolved(
            unresolved,
            "missing_active_layout",
            "base_layout_id",
            "Layout editing requires an active layout in the current session.",
        )
    if not operations and not unresolved:
        _add_unresolved(
            unresolved,
            "missing_edit_operations",
            "operations",
            "No executable layout edit operation was extracted from the request.",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "edit_id": "EDIT_001",
        "project_id": intent.get("project_id") or "PROJECT_001",
        "base_layout_id": base_layout_id,
        "edit_type": "multi_step" if len(operations) > 1 else "single_step",
        "operations": operations,
        "post_edit_policy": {
            "dry_run_first": True,
            "check_hard_constraints": True,
            "allow_auto_repair": False,
            "repair_scope": "selected_instances",
            "max_repair_radius_m": None,
            "max_moved_instances": len(operations),
        },
        "provenance": [
            {
                "field_path": "operations",
                "source_type": "user_input",
                "source_id": intent.get("intent_id") or "INTENT_001",
                "source_field": "user_message",
                "confidence": float(intent.get("confidence") or 0.0),
                "note": "Edit operations extracted by IntentUnderstandingAgent.",
            }
        ],
        "unresolved_items": unresolved,
    }


def validate_layout_edit_ir(edit_ir, base_layout, adapter, schemas_dir):
    errors = []
    schema_path = Path(schemas_dir) / "task" / "layout_edit_ir.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, registry=_schema_registry(Path(schemas_dir)))
    for error in sorted(validator.iter_errors(edit_ir), key=lambda item: list(item.path)):
        errors.append(_error("schema_error", error.message, ".".join(str(part) for part in error.path)))
    if not base_layout or edit_ir.get("base_layout_id") != base_layout.get("layout_id"):
        errors.append(_error("missing_active_layout_ref", "Layout Edit IR base_layout_id does not match the active layout.", "base_layout_id"))
    for item in edit_ir.get("unresolved_items") or []:
        error_type = "missing_asset_reference" if "asset" in item.get("item_id", "") else "missing_required_field"
        errors.append(_error(error_type, item.get("description"), item.get("field_path")))
    instance_ids = {item.get("instance_id") for item in (base_layout or {}).get("placements") or []}
    for index, operation in enumerate(edit_ir.get("operations") or []):
        if operation.get("op_type") in TARGETED_OPERATIONS and operation.get("instance_id") not in instance_ids:
            errors.append(_error("missing_required_field", "Edit target does not exist in the active layout.", f"operations[{index}].instance_id"))
        asset_id = (operation.get("parameters") or {}).get("new_model_asset_id")
        if asset_id and asset_id not in adapter.assets_by_id:
            errors.append(_error("missing_asset_reference", f"Unknown model_asset_id: {asset_id}", f"operations[{index}].parameters.new_model_asset_id"))
    if not edit_ir.get("operations") and not errors:
        errors.append(_error("missing_required_field", "Layout Edit IR requires at least one operation.", "operations"))
    if not errors:
        status = "VALID_EDIT_IR"
    elif any(error["error_type"] in {"missing_active_layout_ref", "missing_asset_reference", "missing_required_field"} for error in errors):
        status = "NEED_CLARIFICATION"
    else:
        status = "INVALID_TASK"
    return {"status": status, "errors": errors, "warnings": []}


def apply_layout_edit_ir(edit_ir, base_layout, base_flp_ir, adapter):
    layout = copy.deepcopy(base_layout)
    flp_ir = copy.deepcopy(base_flp_ir)
    changed_instances = []
    for operation in edit_ir["operations"]:
        op_type = operation["op_type"]
        if op_type == "move":
            changed_instances.append(_apply_move(layout, operation))
        elif op_type == "rotate":
            changed_instances.append(_apply_rotate(layout, flp_ir, operation))
        elif op_type == "resize_site":
            _apply_resize_site(layout, flp_ir, operation)
        elif op_type == "add":
            changed_instances.extend(_apply_add(layout, flp_ir, operation, adapter))
        elif op_type == "remove":
            changed_instances.append(_apply_remove(layout, flp_ir, operation))
        elif op_type == "replace":
            changed_instances.append(_apply_replace(layout, flp_ir, operation, adapter))

    parent_layout_id = base_layout["layout_id"]
    layout["layout_id"] = f"{parent_layout_id}_edit"
    layout["parent_layout_id"] = parent_layout_id
    layout["source_edit_id"] = edit_ir["edit_id"]
    layout["revision_id"] = layout["layout_id"]
    layout["generation_policy"] = "layout_edit_ir"
    if any(op["op_type"] in {"add","remove","replace"} for op in edit_ir["operations"]):
        # J2 is defined for relocating the original inventory. Purchase,
        # disposal and replacement have no supplied cost model in this case.
        removed = [o for o in flp_ir.get("objectives") or [] if o.get("objective_type")=="minimize_relocation_effort"]
        if removed:
            flp_ir["objectives"]=[o for o in flp_ir["objectives"] if o not in removed]
            flp_ir.setdefault("provenance",[]).append(dict(field_path="objectives",source_type="user_input",source_id=edit_ir["edit_id"],source_field="operations",confidence=1.,note="Relocation-only J2 is no longer comparable after inventory changes; acquisition/disposal/replacement costs were not provided. The accepted Optimize result retains its original J2."))
    _synchronize_flp_ir(flp_ir, layout, edit_ir)
    layout_diff = {
        "schema_version": SCHEMA_VERSION,
        "layout_diff_id": f"DIFF_{edit_ir['edit_id']}",
        "from_layout_id": parent_layout_id,
        "to_layout_id": layout["layout_id"],
        "changed_instances": changed_instances,
        "changed_constraints": [],
        "changed_objectives": [],
        "provenance": copy.deepcopy(edit_ir["provenance"]),
    }
    return {"rough_layout": layout, "flp_ir": flp_ir, "layout_diff": layout_diff}


def _operation_parameters(raw):
    move_mode = raw.get("move_mode")
    target_pose = None
    if any(raw.get(key) is not None for key in ("x_m", "y_m", "z_m")):
        target_pose = _pose(raw.get("x_m"), raw.get("y_m"), raw.get("z_m"), raw.get("target_yaw_deg"))
        move_mode = move_mode or "absolute"
    delta_pose = None
    if any(raw.get(key) is not None for key in ("dx_m", "dy_m", "dz_m")):
        delta_pose = _delta_pose(raw.get("dx_m"), raw.get("dy_m"), raw.get("dz_m"), raw.get("delta_yaw_deg"))
        move_mode = move_mode or "relative"
    return {
        "move_mode": move_mode,
        "target_pose": target_pose,
        "delta_pose": delta_pose,
        "rotation_mode": raw.get("rotation_mode"),
        "target_yaw_deg": _number_or_none(raw.get("target_yaw_deg")),
        "delta_yaw_deg": _number_or_none(raw.get("delta_yaw_deg")),
        "new_library_id": raw.get("new_library_id"),
        "new_model_asset_id": raw.get("new_model_asset_id"),
        "quantity": int(_number_or_none(raw.get("quantity")) or 1),
        "placement_policy": (
            raw.get("placement_policy")
            or ("explicit_pose" if target_pose else "auto_place")
        )
        if raw.get("op_type") == "add"
        else None,
        "target_site_feature_id": raw.get("target_site_feature_id"),
        "target_zone_id": raw.get("target_zone_id"),
        "site_width_m": _number_or_none(raw.get("site_width_m")),
        "site_depth_m": _number_or_none(raw.get("site_depth_m")),
        "constraint_update": None,
        "objective_update": None,
    }


def _validate_required_parameters(op_type, parameters, unresolved, index):
    missing = None
    if op_type == "move" and not (parameters["target_pose"] or parameters["delta_pose"]):
        missing = "Move requires an absolute target position or a relative position delta."
    elif op_type == "rotate" and parameters["target_yaw_deg"] is None and parameters["delta_yaw_deg"] is None:
        missing = "Rotate requires target_yaw_deg or delta_yaw_deg."
    elif op_type == "resize_site" and not (parameters["site_width_m"] and parameters["site_depth_m"]):
        missing = "Site modification requires positive site_width_m and site_depth_m."
    elif op_type == "add" and parameters["quantity"] < 1:
        missing = "Add requires a positive equipment quantity."
    if missing:
        _add_unresolved(unresolved, f"missing_edit_parameters_{index:03d}", f"operations[{index - 1}].parameters", missing)


def _resolve_instance_id(raw, layout, selected_ids):
    placements = layout.get("placements") or []
    instance_id = raw.get("instance_id")
    if instance_id and any(item.get("instance_id") == instance_id for item in placements):
        return instance_id
    model_asset_id = raw.get("target_model_asset_id")
    if model_asset_id:
        matches = [item["instance_id"] for item in placements if item.get("model_asset_id") == model_asset_id]
        if len(matches) == 1:
            return matches[0]
    if len(selected_ids) == 1 and any(item.get("instance_id") == selected_ids[0] for item in placements):
        return selected_ids[0]
    return None


def _apply_move(layout, operation):
    placement = _placement(layout, operation["instance_id"])
    before = copy.deepcopy(placement)
    parameters = operation["parameters"]
    if parameters["move_mode"] == "absolute":
        pose = parameters["target_pose"] or {}
        for axis in ("x_m", "y_m", "z_m"):
            if pose.get(axis) is not None:
                placement["position"][axis] = float(pose[axis])
    else:
        delta = parameters["delta_pose"] or {}
        placement["position"]["x_m"] += float(delta.get("dx_m") or 0.0)
        placement["position"]["y_m"] += float(delta.get("dy_m") or 0.0)
        placement["position"]["z_m"] += float(delta.get("dz_m") or 0.0)
    _refresh_footprint(placement)
    return _diff_entry(before, placement, "moved")


def _apply_rotate(layout, flp_ir, operation):
    placement = _placement(layout, operation["instance_id"])
    before = copy.deepcopy(placement)
    parameters = operation["parameters"]
    current = float(placement.get("yaw_deg") or (placement.get("rotation") or {}).get("yaw_deg") or 0.0)
    yaw = parameters["target_yaw_deg"] if parameters["target_yaw_deg"] is not None else current + float(parameters["delta_yaw_deg"])
    placement["yaw_deg"] = yaw % 360.0
    placement.setdefault("rotation", {})["yaw_deg"] = placement["yaw_deg"]
    instance = _instance(flp_ir, operation["instance_id"])
    canonical = instance["geometry"]["dimensions_m"]
    width, depth = _oriented_dimensions(canonical["width_m"], canonical["depth_m"], placement["yaw_deg"])
    placement["dimensions_m"].update({"width_m": width, "depth_m": depth})
    placement["footprint"].update({"width_m": width, "depth_m": depth})
    _refresh_footprint(placement)
    return _diff_entry(before, placement, "rotated")


def _apply_resize_site(layout, flp_ir, operation):
    width = float(operation["parameters"]["site_width_m"])
    depth = float(operation["parameters"]["site_depth_m"])
    site = layout["site"]
    site.update({"width_m": width, "depth_m": depth})
    site.setdefault("dimensions_m", {}).update({"width_m": width, "depth_m": depth})
    site["non_placeable_zones"] = non_placeable_zones(site)
    site["placement_regions"] = placement_regions(site)
    flp_ir["site_ref"]["default_dimensions_m"] = {"width_m": width, "depth_m": depth}


def _apply_add(layout, flp_ir, operation, adapter):
    parameters = operation["parameters"]
    fact = adapter.asset_fact(parameters["new_model_asset_id"])
    changes = []
    for ordinal in range(parameters["quantity"]):
        reserved = operation.get("instance_id")
        instance_id = (f"equipment_{int(reserved.rsplit('_',1)[-1])+ordinal:03d}" if reserved else _next_instance_id(flp_ir))
        if any(p["instance_id"] == instance_id for p in layout["placements"]):
            raise ValueError(f"Added instance identifier already exists: {instance_id}.")
        instance = _new_instance(instance_id, fact, flp_ir)
        instance["role"] = parameters.get("role") or fact.get("canonical_name")
        _inherit_case_service_requirements(instance, fact, flp_ir)
        # Make the accepted case requirement and inherited area available to
        # the automatic pose search, so it tests the service envelope before
        # choosing a position.
        flp_ir["equipment_instances"].append(instance)
        _refresh_equipment_clearance_constraint(flp_ir, instance)
        placement = _new_placement(instance_id, fact, parameters.get("target_pose"))
        if parameters.get("placement_policy") != "explicit_pose" or not parameters.get("target_pose"):
            reference=next((p for p in layout["placements"] if p["instance_id"]==parameters.get("reference_instance_id")),None)
            pose = _find_available_pose(layout, placement["footprint"], preferred_pose=(reference or {}).get("position"),new_placement=placement,problem=flp_ir)
            if pose is None:
                raise ValueError(f"No available placement region for {instance_id}.")
            placement["position"].update(pose)
            _refresh_footprint(placement)
        layout["placements"].append(placement)
        changes.append(_diff_entry(None, placement, "added"))
    return changes


def _apply_remove(layout, flp_ir, operation):
    placement = _placement(layout, operation["instance_id"])
    before = copy.deepcopy(placement)
    instance_id = operation["instance_id"]
    layout["placements"] = [item for item in layout["placements"] if item.get("instance_id") != instance_id]
    flp_ir["equipment_instances"] = [item for item in flp_ir["equipment_instances"] if item.get("instance_id") != instance_id]
    flp_ir["flow_relations"] = [item for item in flp_ir.get("flow_relations", []) if instance_id not in {item.get("from_instance_id"), item.get("to_instance_id")}]
    return _diff_entry(before, None, "removed")


def _apply_replace(layout, flp_ir, operation, adapter):
    placement = _placement(layout, operation["instance_id"])
    before = copy.deepcopy(placement)
    fact = adapter.asset_fact(operation["parameters"]["new_model_asset_id"])
    instance = _instance(flp_ir, operation["instance_id"])
    replacement = _new_instance(instance["instance_id"], fact, flp_ir)
    replacement["role"] = instance.get("role")
    inherited_regions = copy.deepcopy(instance.get("allowed_regions") or [])
    _inherit_case_service_requirements(
        replacement,
        fact,
        flp_ir,
        preferred_instance=instance,
    )
    if inherited_regions:
        replacement["allowed_regions"] = inherited_regions
    if instance.get("model_asset_id")==fact["model_asset_id"]:
        replacement["quantity_index"] = instance.get("quantity_index")
    flp_ir["equipment_instances"][flp_ir["equipment_instances"].index(instance)] = replacement
    _refresh_equipment_clearance_constraint(flp_ir, replacement)
    placement["model_asset_id"] = fact["model_asset_id"]
    placement["equipment_type"] = fact.get("equipment_type") or fact["model_asset_id"]
    canonical = fact["geometry"]["dimensions_m"]
    width, depth = _oriented_dimensions(canonical["width_m"], canonical["depth_m"], placement.get("yaw_deg") or 0.0)
    placement["dimensions_m"] = {"width_m": width, "height_m": canonical["height_m"], "depth_m": depth}
    placement["footprint"].update({"width_m": width, "depth_m": depth, "height_m": canonical["height_m"]})
    _refresh_footprint(placement)
    return _diff_entry(before, placement, "replaced")


def _new_instance(instance_id, fact, flp_ir):
    planning = fact.get("planning_fields") or {}
    same_asset_max = max([int(item.get("quantity_index") or 0) for item in flp_ir.get("equipment_instances") or [] if item.get("model_asset_id") == fact["model_asset_id"]] or [0])
    return {
        "instance_id": instance_id,
        "model_asset_id": fact["model_asset_id"],
        "library_id": fact.get("library_id") or "library/production_model_library",
        "quantity_index": same_asset_max + 1,
        "role": None,
        "locked": False,
        "initial_pose": None,
        "fixed_pose": None,
        "allowed_regions": [],
        "required_ports": [],
        "geometry": copy.deepcopy(fact["geometry"]),
        "maintenance_clearance_m": copy.deepcopy(planning.get("maintenance_clearance_m") or _empty_clearance()),
        "safety_clearance_m": copy.deepcopy(planning.get("safety_clearance_m") or _empty_clearance()),
        "allowed_orientations_deg": copy.deepcopy(planning.get("allowed_orientations_deg") or []),
    }


def _inherit_case_service_requirements(instance, fact, flp_ir, *, preferred_instance=None):
    """Carry accepted user case clearances and area constraints to an Edit asset.

    Same-model peers are the strongest match. For a cross-model replacement,
    an accepted clearance provenance record can also match the new RAG asset's
    engineering name/type to the source-stated equipment group. RAG facts are
    read only; this only updates the working FLP instance.
    """
    peers = list((flp_ir or {}).get("equipment_instances") or [])
    if preferred_instance and all(
        item.get("instance_id") != preferred_instance.get("instance_id")
        for item in peers
    ):
        peers.append(preferred_instance)
    provenance_by_instance = _case_clearance_provenance_by_instance(flp_ir)
    # A replacement keeps the old instance role for route identity. Do not let
    # that old name make a different RAG asset inherit the wrong clearance
    # group; cross-model matches must come from the replacement asset itself.
    target_terms = _asset_requirement_terms(
        fact,
        None if preferred_instance is not None else instance.get("role"),
    )

    preferred_peer = None
    if preferred_instance and _has_user_case_clearance(preferred_instance):
        if preferred_instance.get("model_asset_id") == fact.get("model_asset_id") or _instance_matches_case_requirement(
            preferred_instance,
            target_terms,
            provenance_by_instance,
        ):
            # A replacement occupies the same accepted role and region. Keep
            # its source requirement as primary even if another instance of
            # the new model serves a different area.
            preferred_peer = preferred_instance
    same_model = [
        item for item in peers
        if item.get("model_asset_id") == fact.get("model_asset_id")
        and _has_user_case_clearance(item)
    ]
    matching_group = []
    if not same_model and target_terms:
        for item in peers:
            if not _has_user_case_clearance(item):
                continue
            records = provenance_by_instance.get(str(item.get("instance_id") or "")) or []
            if any(
                _requirement_terms_overlap(
                    target_terms,
                    _clearance_requirement_terms(record.get("source_field")),
                )
                for record in records
            ):
                matching_group.append(item)
    candidates = [preferred_peer] if preferred_peer else same_model or matching_group
    if not candidates:
        return

    for field_name in ("maintenance_clearance_m", "safety_clearance_m"):
        values = [
            (item.get(field_name) or {}, item)
            for item in candidates
            if (item.get(field_name) or {}).get("source") == "user_input"
        ]
        unique_values = {
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value, _item in values
        }
        if len(unique_values) > 1:
            raise ValueError(
                "The added or replacement equipment matches conflicting accepted case service requirements."
            )
        if not values:
            continue
        clearance, source_instance = values[0]
        instance[field_name] = copy.deepcopy(clearance)
        source_record = next(
            (
                item for item in provenance_by_instance.get(str(source_instance.get("instance_id") or ""), [])
                if str(item.get("field_path") or "").endswith("." + field_name)
            ),
            None,
        )
        if source_record:
            _append_propagated_clearance_provenance(
                flp_ir,
                instance["instance_id"],
                field_name,
                source_record,
            )

    if not instance.get("allowed_regions"):
        region_sets = {
            tuple(str(value) for value in item.get("allowed_regions") or [])
            for item in candidates
            if item.get("allowed_regions")
        }
        if len(region_sets) > 1:
            raise ValueError(
                "The added equipment matches accepted peers in different placement regions."
            )
        if region_sets:
            instance["allowed_regions"] = list(next(iter(region_sets)))


def _has_user_case_clearance(instance):
    return any(
        (instance.get(field_name) or {}).get("source") == "user_input"
        for field_name in ("maintenance_clearance_m", "safety_clearance_m")
    )


def _instance_matches_case_requirement(instance, target_terms, provenance_by_instance):
    records = provenance_by_instance.get(str(instance.get("instance_id") or "")) or []
    return any(
        _requirement_terms_overlap(
            target_terms,
            _clearance_requirement_terms(record.get("source_field")),
        )
        for record in records
    )


def _case_clearance_provenance_by_instance(flp_ir):
    result = {}
    pattern = re.compile(
        r"^equipment_instances\[(?P<instance_id>[^\]]+)\]\.(?:maintenance_clearance_m|safety_clearance_m)$"
    )
    for item in (flp_ir or {}).get("provenance") or []:
        match = pattern.match(str(item.get("field_path") or ""))
        if match:
            result.setdefault(match.group("instance_id"), []).append(item)
    return result


def _asset_requirement_terms(fact, role):
    values = [
        role,
        fact.get("canonical_name"),
        fact.get("asset_name"),
        fact.get("equipment_type"),
        fact.get("category"),
        *(fact.get("aliases") or []),
    ]
    return _case_terms(" ".join(str(value or "") for value in values))


def _clearance_requirement_terms(source_field):
    text = str(source_field or "").strip().lstrip("-* ")
    if ":" in text:
        text = text.split(":", 1)[0]
    else:
        match = re.search(r"\bfor\s+(?P<target>[^,.;]+)", text, re.I)
        if match:
            text = match.group("target")
    return _case_terms(text)


def _case_terms(value):
    ignored = {
        "a", "an", "the", "and", "or", "of", "for", "one", "two", "three",
        "four", "five", "six", "seven", "eight", "nine", "ten", "equipment",
        "unit", "units", "machine", "machines", "each", "service", "front", "rear",
        "clearance", "space", "spaces", "m", "per",
    }
    result = set()
    for token in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
        if token in ignored:
            continue
        result.add(token[:-1] if len(token) > 4 and token.endswith("s") else token)
    return result


def _requirement_terms_overlap(asset_terms, requirement_terms):
    shared = set(asset_terms or []) & set(requirement_terms or [])
    return bool(shared)


def _append_propagated_clearance_provenance(flp_ir, instance_id, field_name, source_record):
    field_path = f"equipment_instances[{instance_id}].{field_name}"
    record = {
        "field_path": field_path,
        "source_type": source_record.get("source_type") or "user_input",
        "source_id": source_record.get("source_id"),
        "source_field": source_record.get("source_field"),
        "confidence": source_record.get("confidence", 1.0),
        "note": "Propagated an accepted case service requirement to an Edit asset of the same source-defined equipment group; not manufacturer clearance data.",
    }
    provenance = flp_ir.setdefault("provenance", [])
    if not any(
        item.get("field_path") == field_path
        and item.get("source_field") == record["source_field"]
        for item in provenance
    ):
        provenance.append(record)


def _refresh_equipment_clearance_constraint(flp_ir, instance):
    """Keep Edit auto-placement's pairwise clearance map aligned with instances."""
    values = []
    for field_name in ("maintenance_clearance_m", "safety_clearance_m"):
        clearance = instance.get(field_name) or {}
        for side in ("front_m", "rear_m", "left_m", "right_m"):
            value = clearance.get(side)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    required = max(values or [0.0])
    hard = flp_ir.setdefault("constraints", {}).setdefault("hard", [])
    computed = next(
        (
            item for item in hard
            if item.get("constraint_type") == "minimum_clearance"
            and "equipment_instances.clearance" in (item.get("evidence_refs") or [])
        ),
        None,
    )
    if computed is None and required > 0.0:
        computed = {
            "constraint_id": f"constraint_{len(hard) + 1:03d}",
            "constraint_type": "minimum_clearance",
            "target_refs": [{"target_type": "all_equipment", "target_id": "all"}],
            "parameters": {"clearance_by_instance": {}},
            "source": "computed",
            "evidence_refs": ["equipment_instances.clearance"],
        }
        hard.append(computed)
    if computed is not None:
        clearance_by_instance = computed.setdefault("parameters", {}).setdefault("clearance_by_instance", {})
        if required > 0.0:
            clearance_by_instance[instance["instance_id"]] = required
        else:
            clearance_by_instance.pop(instance["instance_id"], None)


def _new_placement(instance_id, fact, target_pose=None):
    dimensions = fact["geometry"]["dimensions_m"]
    pose = target_pose or _pose(0.0, 0.0, 0.0, 0.0)
    placement = {
        "instance_id": instance_id,
        "model_asset_id": fact["model_asset_id"],
        "equipment_type": fact.get("equipment_type") or fact["model_asset_id"],
        "position": {"x_m": float(pose.get("x_m") or 0.0), "y_m": float(pose.get("y_m") or 0.0), "z_m": float(pose.get("z_m") or 0.0)},
        "rotation": {"yaw_deg": float(pose.get("yaw_deg") or 0.0)},
        "yaw_deg": float(pose.get("yaw_deg") or 0.0),
        "dimensions_m": {"width_m": float(dimensions["width_m"]), "height_m": float(dimensions["height_m"]), "depth_m": float(dimensions["depth_m"])},
        "footprint": {"width_m": float(dimensions["width_m"]), "depth_m": float(dimensions["depth_m"]), "height_m": float(dimensions["height_m"])},
    }
    _refresh_footprint(placement)
    return placement


def _find_available_pose(layout, footprint, preferred_pose=None, new_placement=None, problem=None):
    width = float(footprint["width_m"])
    depth = float(footprint["depth_m"])
    proposals=[]
    zones=layout["site"].get("non_placeable_zones") or []
    for region in layout["site"].get("placement_regions") or placement_regions(layout["site"]):
        if preferred_pose and not (region["min_x_m"]<=preferred_pose["x_m"]<=region["max_x_m"] and region["min_z_m"]<=preferred_pose["z_m"]<=region["max_z_m"]):
            continue
        min_x = region["min_x_m"] + width / 2
        max_x = region["max_x_m"] - width / 2
        min_z = region["min_z_m"] + depth / 2
        max_z = region["max_z_m"] - depth / 2
        x = min_x
        while x <= max_x + 1e-9:
            z = min_z
            while z <= max_z + 1e-9:
                candidate = {"min_x": x - width / 2, "max_x": x + width / 2, "min_z": z - depth / 2, "max_z": z + depth / 2}
                if not any(_overlaps(candidate, item["footprint"]) for item in layout.get("placements") or []) and not any(
                    _overlaps(candidate,dict(min_x=q["min_x_m"],max_x=q["max_x_m"],min_z=q["min_z_m"],max_z=q["max_z_m"])) for q in zones):
                    if preferred_pose is None and new_placement is None: return {"x_m": x, "y_m": 0.0, "z_m": z}
                    proposals.append((math.hypot(x-preferred_pose["x_m"],z-preferred_pose["z_m"]) if preferred_pose else len(proposals),x,z))
                z += max(0.25, depth / 2)
            x += max(0.25, width / 2)
    for _,x,z in sorted(proposals):
        if new_placement is not None and problem is not None:
            from tool.layout.layout_solvers import layout_violations
            item=copy.deepcopy(new_placement);item["position"].update(x_m=x,z_m=z);_refresh_footprint(item)
            draft=dict(layout,placements=list(layout["placements"])+[item])
            if layout_violations(draft,problem):continue
        return dict(x_m=x,y_m=0.,z_m=z)
    return None


def _synchronize_flp_ir(flp_ir, layout, edit_ir):
    instance_ids = [item["instance_id"] for item in flp_ir.get("equipment_instances") or []]
    flp_ir.setdefault("optimization_scope", {})["movable_instances"] = instance_ids
    flp_ir["optimization_scope"]["locked_instances"] = [item["instance_id"] for item in flp_ir["equipment_instances"] if item.get("locked")]
    flp_ir["optimization_scope"]["base_layout_id"] = layout["layout_id"]
    by_id = {str(item.get("instance_id")): item for item in flp_ir.get("equipment_instances") or []}
    clearance_path = re.compile(
        r"^equipment_instances\[(?P<instance_id>[^\]]+)\]\.(?P<field>maintenance_clearance_m|safety_clearance_m)$"
    )
    provenance = []
    for item in flp_ir.get("provenance") or []:
        field_path = str(item.get("field_path") or "")
        if not field_path.startswith("equipment_instances["):
            provenance.append(item)
            continue
        match = clearance_path.match(field_path)
        instance = by_id.get(match.group("instance_id")) if match else None
        if instance and instance.get(match.group("field")):
            # Geometry provenance is refreshed below. Keep accepted clearance
            # evidence for surviving instances, including newly propagated
            # source-backed case requirements.
            provenance.append(item)
    for index, instance in enumerate(flp_ir["equipment_instances"]):
        provenance.append({"field_path": f"equipment_instances[{index}].geometry", "source_type": "model_extraction", "source_id": instance["model_asset_id"], "source_field": "geometry", "confidence": 0.75, "note": None})
    provenance.append({"field_path": "layout_edit_ir", "source_type": "user_input", "source_id": edit_ir["edit_id"], "source_field": "operations", "confidence": 1.0, "note": "Applied Layout Edit IR to the active layout."})
    flp_ir["provenance"] = provenance


def _diff_entry(before, after, change_type):
    source = before or after
    from_pose = _placement_pose(before) if before else None
    to_pose = _placement_pose(after) if after else None
    movement = None
    yaw_change = None
    if from_pose and to_pose:
        movement = math.hypot(to_pose["x_m"] - from_pose["x_m"], to_pose["z_m"] - from_pose["z_m"])
        yaw_change = (to_pose["yaw_deg"] - from_pose["yaw_deg"]) % 360.0
    return {
        "instance_id": source["instance_id"],
        "change_type": change_type,
        "from_pose": from_pose,
        "to_pose": to_pose,
        "movement_distance_m": movement,
        "yaw_change_deg": yaw_change,
        "from_library_id": None,
        "to_library_id": None,
        "from_model_asset_id": before.get("model_asset_id") if before else None,
        "to_model_asset_id": after.get("model_asset_id") if after else None,
    }


def _placement_pose(placement):
    return _pose(placement["position"].get("x_m"), placement["position"].get("y_m"), placement["position"].get("z_m"), placement.get("yaw_deg") or (placement.get("rotation") or {}).get("yaw_deg"))


def _refresh_footprint(placement):
    x = float(placement["position"]["x_m"])
    z = float(placement["position"]["z_m"])
    footprint = placement["footprint"]
    width = footprint.get("width_m")
    depth = footprint.get("depth_m")
    if width is None and all(
        footprint.get(key) is not None for key in ("min_x", "max_x")
    ):
        width = float(footprint["max_x"]) - float(footprint["min_x"])
    if depth is None and all(
        footprint.get(key) is not None for key in ("min_z", "max_z")
    ):
        depth = float(footprint["max_z"]) - float(footprint["min_z"])
    dimensions = placement.get("dimensions_m") or {}
    width = float(width if width is not None else dimensions["width_m"])
    depth = float(depth if depth is not None else dimensions["depth_m"])
    # Preserve the footprint schema of an existing placement.  Adding derived
    # width/depth fields during a pure move makes an otherwise exact edit look
    # like a metadata mutation to the independent edit-contract evaluator.
    if "width_m" in footprint:
        footprint["width_m"] = width
    if "depth_m" in footprint:
        footprint["depth_m"] = depth
    footprint.update(
        {
            "min_x": x - width / 2,
            "max_x": x + width / 2,
            "min_z": z - depth / 2,
            "max_z": z + depth / 2,
        }
    )


def _placement(layout, instance_id):
    for item in layout["placements"]:
        if item.get("instance_id") == instance_id:
            return item
    raise ValueError(f"Unknown layout instance: {instance_id}.")


def _instance(flp_ir, instance_id):
    for item in flp_ir["equipment_instances"]:
        if item.get("instance_id") == instance_id:
            return item
    raise ValueError(f"Unknown FLP-IR equipment instance: {instance_id}.")


def _next_instance_id(flp_ir):
    values = []
    for item in flp_ir.get("equipment_instances") or []:
        suffix = str(item.get("instance_id") or "").rsplit("_", 1)[-1]
        if suffix.isdigit():
            values.append(int(suffix))
    return f"equipment_{max(values, default=0) + 1:03d}"


def _oriented_dimensions(width, depth, yaw):
    return (float(depth), float(width)) if round(float(yaw) % 180.0, 6) == 90.0 else (float(width), float(depth))


def _overlaps(left, right):
    return left["min_x"] < right["max_x"] - 1e-9 and left["max_x"] > right["min_x"] + 1e-9 and left["min_z"] < right["max_z"] - 1e-9 and left["max_z"] > right["min_z"] + 1e-9


def _pose(x, y, z, yaw):
    return {"x_m": _number_or_none(x), "y_m": _number_or_none(y), "z_m": _number_or_none(z), "roll_deg": None, "pitch_deg": None, "yaw_deg": _number_or_none(yaw), "coordinate_frame_id": None}


def _delta_pose(x, y, z, yaw):
    return {"dx_m": _number_or_none(x), "dy_m": _number_or_none(y), "dz_m": _number_or_none(z), "droll_deg": None, "dpitch_deg": None, "dyaw_deg": _number_or_none(yaw)}


def _number_or_none(value):
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _preserve_fields(op_type):
    if op_type == "move":
        return ["yaw_deg", "flow_relations", "instance_id", "constraints", "objectives"]
    if op_type == "rotate":
        return ["position", "flow_relations", "instance_id", "constraints", "objectives"]
    if op_type == "replace":
        return ["pose", "flow_relations", "instance_id", "constraints", "objectives"]
    return ["constraints", "objectives"]


def _empty_clearance():
    return {"front_m": None, "rear_m": None, "left_m": None, "right_m": None, "top_m": None, "bottom_m": None, "source": "computed", "confidence": None}


def _add_unresolved(items, item_id, field_path, description, *, item_type="missing_field", candidates=None):
    items.append({"item_id": item_id, "item_type": item_type, "field_path": field_path, "description": description, "severity": "blocking", "candidates": candidates or [], "resolution_policy": "ask_user"})


def _error(error_type, message, field_path):
    return {"error_type": error_type, "message": message, "field_path": field_path}
